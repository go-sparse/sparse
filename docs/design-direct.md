# Design: sparse direct solvers

Status: design, nothing implemented yet. Scope: sparse triangular solve, sparse
LU with threshold partial pivoting, `spsolve`, `splu`/`spilu` factor objects,
`factorized`, `inv`, `onenormest`, `expm` and `expm_multiply` of
`scipy.sparse.linalg` 1.18.1, an updatable basis factorization for a simplex
method, and MATLAB sparse `lu`/`mldivide` as the delta. Consumers (go-scikit
roadmap, COVERAGE.md line 38): optimize MILP and conic (a HiGHS-class simplex
needs an updatable sparse LU), sparse Jacobians in integrate and optimize;
line 36: sparse `expm` for signal.

## 1. Sources

Opened for this design (cite only these):

| # | Source | What was read |
|---|---|---|
| S1 | SciPy v1.18.1 `scipy/sparse/linalg/_dsolve/SuperLU/README.scipy` | SuperLU is vendored from github.com/xiaoyeli/superlu at commit `53afbff31d4a5340bb71aa15cf8a9186a7049c99`, all of `SRC/*.[ch]` except `mc64ad.c` ("license incompatible"), plus `scipychanges.patch` and `scipychanges_blas.patch` |
| S2 | same dir, `License.txt` | BSD-3-Clause, (c) 2003 The Regents of the University of California (LBNL); `colamd.c` and `mmd.c` carry the same header |
| S3 | `_dsolve/meson.build`, `scipy_slu_config.h` | SuperLU is compiled with `-DUSE_VENDOR_BLAS=1` and linked to SciPy's BLAS (OpenBLAS in wheels); `SCIPY_FIX 1`; `slu_blasint` is `int` unless ILP64 |
| S4 | `SRC/dgstrf.c`, `dpivotL.c`, `dcolumn_bmod.c`, `dpanel_bmod.c`, `dsnode_bmod.c`, `dgstrs.c`, `dgssv.c`, `sp_preorder.c`, `get_perm_c.c`, `sp_ienv.c`, `util.c`, `dmemory.c`, `dgsitrf.c`, `ilu_dpivotL.c` (headers and the lines cited below) | algorithm, defaults, BLAS call sites, memory policy |
| S5 | `_dsolve/linsolve.py`, `_superluobject.c`, `_superlumodule.c` | `spsolve`, `splu`, `spilu`, `factorized`, `spsolve_triangular`, option parsing, error mapping |
| S6 | `scipy/sparse/linalg/_matfuncs.py`, `_expm_multiply.py`, `_onenormest.py` | `inv`, `expm`, `expm_multiply`, `onenormest` |
| S7 | Demmel, Eisenstat, Gilbert, Li, Liu, *A supernodal approach to sparse partial pivoting*, UT-CS-95-304 (1995) = SIMAX 20 (1999) 720-755, pp. 1-6 | Fig. 1 (left-looking column LU), §1 (Gilbert-Peierls DFS gives time proportional to flops; Eisenstat-Liu pruning), §2.1 (T1-T5 supernodes; SuperLU uses T2), §2.2 (storage) |
| S8 | Higham, Tisseur, *A block algorithm for matrix 1-norm estimation...*, SIMAX 21 (2000), MIMS EPrint 2006.145, pp. 1188-1190 | Algorithms 2.2 and 2.4 |
| S9 | Al-Mohy, Higham, *Computing the action of the matrix exponential...*, SISC 33 (2011), pp. 493-495 | (3.3)-(3.14), Table 3.1, Code Fragment 3.1 |
| S10 | Al-Mohy, Higham, *A new scaling and squaring algorithm for the matrix exponential*, SIMAX 31 (2009), pp. 981-982 | (5.3)-(5.9), the `ell` correction |
| S11 | HiGHS v1.15.1 `LICENSE.txt` (MIT), `highs/util/HFactor.h`, `HFactorConst.h`, `HFactor.cpp` (structure) | INVERT, FTRAN/BTRAN, FT/PF/MPF/APF updates, thresholds |
| S12 | M. Saunders, Stanford CME 338 Notes 8 *Basis updates* (2019), pp. 55-57 | PF, Bartels-Golub-Reid and Forrest-Tomlin updates |
| S13 | MathWorks `lu` reference page | sparse syntaxes, `thresh` defaults, `'vector'` |
| S14 | MathWorks `mldivide` reference page (text only; the sparse flowchart is an image the tool could not read) | warnings on `rcond` |
| S15 | The Go specification, "Floating-point operators" | fusion and explicit conversion |

Not obtained: Gilbert & Peierls 1988 (SISSC 9(5); the Cornell TR 86-783 PDF
returned HTTP 405) - used only through S7's description and SuperLU's code;
Forrest & Tomlin 1972 (Math. Prog. 2, paywalled) - used through S12 and S11;
Davis 2006 *Direct Methods for Sparse Linear Systems* (not accessible);
SuperLU user guide; SciPy `_dsolve/tests` (to be read in task 5); MATLAB's
sparse `mldivide` flowchart (image). LGPL/GPL code (UMFPACK, KLU, CSparse,
CXSparse, CHOLMOD) was not read and must not be read by any task below.

## 2. Licensing

- SuperLU (S1, S2): BSD-3, may be read and ported. The ported files keep the
  LBNL copyright notice and the per-file notice (S4, e.g. `dpivotL.c` lines
  20-31: notices retained, "a notice that the code was modified"); `NOTICE`
  gains a SuperLU paragraph like the QDLDL one. `mc64ad.c` is HSL code, absent
  from SciPy, never read: `RowPerm=LargeDiag_MC64` is not offered (SciPy's
  `rowperm_cvt` accepts only `NOROWPERM` and `MY_PERMR`).
- `colamd.c` and `mmd.c` (Liu's GENMMD) inside SuperLU are under the same BSD
  header (S2) and are ported as is, because they decide `perm_c`.
- HiGHS (S11): MIT, (c) 2026 HiGHS. `HFactor` may be read and ported; the MIT
  notice goes into `NOTICE` and the ported files.
- Netlib reference BLAS (loop order of `dtrsv`, `dgemv`, `dtrsm`, `dgemm`) is
  used as the order specification of the dense kernels; it is under LAPACK's
  modified BSD licence. OpenBLAS kernels are not ported.
- Davis's book, UMFPACK, KLU, CSparse: nothing taken. A Gilbert-Peierls engine
  written from the book alone was considered and rejected (section 3.3).

## 3. Algorithm decisions

### 3.1 General sparse LU = SuperLU 7.x port (`splu`, `spsolve`, `spilu`)

Port SuperLU's sequential left-looking supernodal factorization exactly as
SciPy runs it (S4):

- Column ordering `get_perm_c` (NATURAL, MMD_ATA, MMD_AT_PLUS_A via GENMMD,
  COLAMD via SuperLU's `colamd.c`), then `sp_preorder`: column etree of
  `AcᵀAc` (or of `Ac+Acᵀ` when `SymmetricMode`), postorder, and `perm_c`
  replaced by `post∘perm_c` (sp_preorder.c 162-192). The existing
  `order.COLAMD` is an independent implementation (README: LU fill within
  1.20x of SuperLU's COLAMD), so it cannot reproduce `perm_c`; a bit-exact port
  of SuperLU's ordering is task 1.
- `dgstrf`: relaxed supernodes at the leaves of the etree (`relax_snode`, or
  `heap_relax_snode` when `SymmetricMode`), otherwise panels of `panel_size`
  columns; per panel `panel_dfs` + `panel_bmod`, then per column
  `column_dfs` (Gilbert-Peierls DFS with Eisenstat-Liu pruning, S7 §1),
  `column_bmod`, `copy_to_ucol`, `pivotL`, `pruneL` (dgstrf.c 298-409).
  Supernodes are T2 (S7 §2.1). Defaults from `sp_ienv`: panel 20, relax 10,
  max supernode 200, fill ratio 30 (sp_ienv.c 76-82).
- Pivoting (`dpivotL.c` 50-58, 117-201): `thresh = u·max|a_ij|`; reuse the
  user row if `SamePattern_SameRowPerm` and it passes, else the diagonal of
  `Pc A Pcᵀ` if `|a_jj| >= thresh` and nonzero, else the largest. `u =
  DiagPivotThresh`, default 1.0 for `splu` and 0.1 for `spilu` (util.c 58, 74).
  `pivmax == 0` returns `jcol+1`, factorization continues, the first such
  column is reported; the `SCIPY_FIX` branch picks a pivot row so `perm_r`
  stays a permutation (dpivotL.c 136-155). Rank-deficient `perm_r` is
  completed after the loop (dgstrf.c 413-428).
- `dgstrs` with `NOTRANS` uses supernodal forward/back solves; `TRANS` uses
  `sp_dtrsv` (dgstrs.c 151-316). `dgssv` on CSR input factors `Aᵀ` and solves
  with `TRANS` (dgssv.c 183-189, 247): `spsolve` on CSR is not the same
  arithmetic as on CSC, and the port keeps that.
- ILU (`dgsitrf`, `ilu_*`): drop rules BASIC|AREA, drop_tol 1e-4, fill_factor
  10, ILU_FillTol 1e-2, MILU variants, zero pivots replaced (util.c 74-82;
  dgsitrf.c 228-608). SciPy forces `SymmetricMode=True` when
  `permc_spec="NATURAL"` (linsolve.py 441, 579).
- SciPy does no equilibration in `splu`/`spilu`/`spsolve` (it calls
  `get_perm_c`, `sp_preorder`, `gstrf`/`gsitrf` directly, _superluobject.c
  742-781; `dgssv` likewise): `dgsequ`/`dlaqgs` are ported only for the MATLAB
  delta and `IterRefine` (`dgsrfs`) only if a consumer asks.

Why SuperLU and not a separate Gilbert-Peierls code: SuperLU *is* a
Gilbert-Peierls column method (S7 §1, Fig. 1) with supernodes added. On the
small, very sparse matrices of KKT and simplex work its supernodes are mostly
of width 1-3, where `column_bmod`/`panel_bmod` take scalar unrolled paths
(dcolumn_bmod.c 150-213) and the overhead over plain GP is the panel DFS - to
be measured, not assumed (section 6). On large matrices with fill (3-D PDE,
circuit) supernodes reach tens of columns and the dense kernels dominate,
which is where a GP column code loses (S7 §2: sup-col updates replace sparse
axpys). One engine, one oracle, one set of bugs.

### 3.2 Simplex basis factorization = HiGHS `HFactor` port (`basis`)

A SuperLU factor cannot be updated: L is stored as dense supernode blocks and
U as columns tied to them. A simplex needs `B' = B + (a_q − B e_p) e_pᵀ`
hundreds of times between refactorizations (S11 HFactor.h 74, S12 §1.1).
Decision: port `HFactor` v1.15.1 (MIT):

- INVERT `build()`: logical (unit) columns, then repeated row/column singleton
  passes (`buildSimple`), then a Markowitz kernel (`buildKernel`) with search
  limit 8 (HFactor.cpp 919), `pivot_threshold` default 0.1 in [8e-4, 0.5],
  `pivot_tolerance` 1e-10 in [0, 1] (HFactorConst.h 25-34); rank deficiency
  is reported and replaced by logicals (`buildHandleRankDeficiency`).
- FTRAN/BTRAN with hyper-sparse switching (thresholds HFactorConst.h 39-51).
- Updates: Forrest-Tomlin by default (`kUpdateMethodFt`), PF, MPF, APF; PF/MPF
  raise a reinversion hint when `u_total_x > u_merit_x` (HFactor.cpp
  1475-1478, 2440, 2472).

Bartels-Golub-Reid (S12 §4) is more stable but needs row-wise U storage with
garbage collection; no permissively licensed implementation was opened. FT is
BGR without row interchanges (S12 §5), is what HiGHS ships, and the simplex
that consumes it will itself be a HiGHS port, so the pair stays coherent.

KKT systems of conic/interior-point methods are quasi-definite: they keep
using `ldl` (QDLDL port) with `order.AMD`; this design does not change that.

### 3.3 Rejected

- Gilbert-Peierls LU written from Davis's book: book not obtained; and a
  second general engine would have no bit-exact oracle (KLU/CSparse may not be
  read).
- Multifrontal (UMFPACK-class): LGPL/GPL references only.
- `mc64` row permutation: HSL licence.

## 4. Exactness decision

SciPy's SuperLU calls vendor BLAS `dtrsv`/`dgemv` inside `column_bmod`,
`panel_bmod` (segments longer than 3) and `snode_bmod`, and `dtrsm`/`dgemm` in
`dgstrs` for supernodes wider than 1 (S3, S4 call sites). OpenBLAS kernels are
architecture-specific, may use FMA and blocked summation, so the wheel's
values are not reproducible bit-for-bit in general. Note also that the vendor
path computes `y ← y − A x` inside `dgemv` while the non-vendor path forms
`t = A x` then `y − t` (dsnode_bmod.c 113-127): the two SuperLU builds differ
from each other. Decision, by tier:

1. **Integers, bit-exact against the SciPy 1.18.1 wheel**: `perm_c` (ordering
   is integer and deterministic), etree, relaxed supernodes. `perm_r`, the L/U
   patterns, `nnz` and the supernode partition are bit-exact whenever no pivot
   test is within a margin of 1e-12 relative of its threshold; the Go code
   records the smallest margin, the gate reports rather than fails below it.
2. **Values, bit-exact against a C oracle**: SciPy's own SuperLU sources (S1:
   commit + both patches, `-DUSE_VENDOR_BLAS=1`), linked against netlib
   reference BLAS, compiled `-O2 -ffp-contract=off` (clang on darwin/arm64
   contracts by default). The Go port follows the `USE_VENDOR_BLAS` path and
   implements the four BLAS routines in reference loop order. Go may fuse
   `x*y+z` (S15): every product feeding a sum is written `float64(x*y) + z`,
   and a test on arm64 checks a known fused/unfused pair. The C oracle is
   built by a driver from a user-supplied checkout, never vendored (same rule
   as `ldl/testdata/qdldl_bench.c`).
3. **Values against the wheel, by tolerance**: `‖P A Q − L U‖₁ / ‖A‖₁ <= 8 n u ρ`
   (ρ = SuperLU pivot growth), solution backward error
   `η = ‖b − A x‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) <= max(4 η_scipy, n u)`.
4. `spsolve_triangular`: every column is a size-1 supernode and `dgstrs` takes
   the scalar paths (dgstrs.c 170-179, 248-253), so no BLAS is reached: bit-exact
   against the wheel, including SciPy's quirks - columns pre-scaled by
   `1/diag` (linsolve.py 739-747), the back-solve divides by the scaled
   diagonal that may differ from 1 by an ulp, result rescaled (790-792).
5. `basis` (HFactor): plain loops, no BLAS: bit-exact against HiGHS v1.15.1
   built `-ffp-contract=off` for sequences build → ftran/btran → update.
6. `onenormest`: Algorithm 2.4 (S8) with X(:,1)=ones and `rand{−1,1}` columns
   drawn from `np.random` legacy global state (`_onenormest.py` 213-214, 280-288);
   `t >= n` computes the exact norm (84-105). The Go API takes the starting
   columns/random source; equality with SciPy is bit-exact for `t >= n`, for
   nonnegative matrices (converges on iteration 2, S8) and when the test
   injects SciPy's draws; otherwise `est <= ‖A‖₁` and the value within the
   estimator's documented behaviour.
7. `expm`: SciPy uses the exact 1-norm for `n < 200` (`_matfuncs.py` 632), so
   degree and scaling `s` are deterministic; tier 3 tolerance on the result
   (relative 1-norm error <= 1e3 n u ‖e^A‖ bound), equality of `(m, s)` exact
   for n < 200. `expm_multiply`: `(m*, s)` from Code Fragment 3.1 (S9) equal
   when `‖A‖₁` is computed exactly (sparse input, `_exact_1_norm`), values by
   tolerance.

Permutation conventions are those of SciPy's `SuperLU` object: `Pr A Pc = L U`
with `Pr = csc((ones, (perm_r, arange)))`, `Pc = csc((ones, (arange, perm_c)))`;
L returned unit lower with explicit unit diagonal, U upper (the CSC export in
_superluobject.c 577-683 is ported for `L()`/`U()`).

## 5. API (Go)

Package `lu` (float64 first; complex128 is task 17):

```go
type ColPerm int // Natural, MMDATA, MMDATPlusA, COLAMD, MyPermC
type Fact int    // DoFact, SamePattern, SamePatternSameRowPerm
type Trans int   // NoTrans, Trans

type Options struct {
	ColPerm         ColPerm
	PermC           []int   // MyPermC
	DiagPivotThresh float64 // 0 means the SciPy default (1.0 splu, 0.1 spilu)
	PanelSize       int     // 0 means 20
	Relax           int     // 0 means 10
	SymmetricMode   bool    // forced when ColPerm == Natural, as SciPy
	Limits          Limits
}
type ILUOptions struct {
	Options
	DropTol, FillFactor, FillTol float64
	DropRule                     DropRule // Basic|Area default
	MILU                         MILU
}
type Limits struct {
	MaxFactorNNZ int64 // nnz(L)+nnz(U); default and ceiling 2^31-1 (SciPy: int indices)
	MaxBytes     int64 // whole factor + workspace; default 0 = MaxFactorNNZ-derived
}

type Factor struct{ /* unexported */ }
func Splu(a *sparse.CSC[float64], o *Options) (*Factor, error)
func SpluWith(w *Workspace, a *sparse.CSC[float64], o *Options) (*Factor, error)
func Spilu(a *sparse.CSC[float64], o *ILUOptions) (*Factor, error)
func (f *Factor) Solve(b []float64, t Trans) error          // in place, gstrs
func (f *Factor) SolveMany(b []float64, nrhs int, t Trans) error // column-major
func (f *Factor) Refactor(a *sparse.CSC[float64]) error     // SamePattern(_SameRowPerm)
func (f *Factor) L() *sparse.CSC[float64]
func (f *Factor) U() *sparse.CSC[float64]
func (f *Factor) PermR() []int
func (f *Factor) PermC() []int
func (f *Factor) NNZ() int
func (f *Factor) RCond() (float64, error)   // dgscon + dlangs, 1-norm
func (f *Factor) PivotGrowth() float64      // dpivotgrowth

func Spsolve(a sparse.Compressed, b []float64, o *Options) ([]float64, error)
func Factorized(a *sparse.CSC[float64]) (func(dst, b []float64) error, error)
func SpsolveTriangular(a sparse.Compressed, b []float64, lower, unitDiagonal bool) error
func Inv(a *sparse.CSC[float64]) (*sparse.CSC[float64], error)
func Mldivide(a sparse.Compressed, b []float64) ([]float64, Info, error) // MATLAB delta
```

`sparse.Compressed` stands for "CSR or CSC" (the existing types; the exact
spelling is decided in task 6). Errors: `*SingularError{Column}` wrapping
`ErrSingular` (SciPy: `RuntimeError("Factor is exactly singular")` in `splu`;
`MatrixRankWarning` + NaN-filled `x` in `spsolve`, which Go returns as the NaN
vector together with the error), `ErrNotSquare`, `ErrTooLarge` (limits),
structure errors from `validate`. `Info` carries `RCond` and a warning kind
(`CloseToSingular` when `0 < rcond < eps`, `Singular` when 0, S14).

Package `basis`: `New(m int, a *sparse.CSC[float64], basic []int, o Options)`,
`Build() (rankDeficiency int, err error)`, `Ftran(v *Vector, density float64)`,
`Btran(...)`, `Update(aq, ep *Vector, row int) (reinvert bool)`,
`Vector{Index []int32; Value []float64; Count int}` (HVector),
`Options{PivotThreshold, PivotTolerance float64; Update UpdateMethod}`.

Package `matfun`: `Onenormest(a sparse.TransposeOperator, o *OnenormOptions)
(est float64, v, w []float64, err error)` with `T` (default 2), `ItMax`
(default 5), `X0`/`Rand`; `Expm(a *sparse.CSC[float64]) (*sparse.CSC[float64],
error)`; `ExpmMultiply(a, b, o)` with `Start/Stop/Num/Endpoint`, `Traceless`.

MATLAB delta: `lu` with up to 3 outputs and `thresh` scalar default 1.0 maps
to `Splu` with `DiagPivotThresh=thresh`; 4/5 outputs (`[0.1 0.001]`, `Q`, `D`,
`'vector'`, S13) map to `Splu` + COLAMD + optional row scaling, documented as a
different pivot strategy (MATLAB's own is not reproducible; no oracle).
`mldivide` sparse path: triangular / permuted triangular → `SpsolveTriangular`,
otherwise `Splu` + `RCond` warning; the Cholesky branch waits for a sparse
Cholesky (out of scope).

## 6. Performance plan

Reference: SciPy 1.18.1 `splu`/`spsolve`/`spilu` (SuperLU is sequential;
`OPENBLAS_NUM_THREADS=1`), HiGHS v1.15.1 `HFactor`, single core, interleaved
rounds, not above 1-min load 8, through the existing `cmd/perfbench` and
`internal/scipy` session. Workloads: `gallery.Laplacian2D/3D`,
`ConvectionDiffusion2D`, KKT matrices already used by `ldl`, the corpus of
section 7. Phases timed separately: ordering, symbolic+numeric factor,
`Refactor`, solve (1 and 16 RHS); allocations counted exactly.

1. Exact port first (it is the oracle); measure; open a defect for every
   matrix slower than SciPy.
2. Dense kernels: reference-order `dgemv N`, `dtrsv L/U`, `dgemm NN`,
   `dtrsm` are column-axpy loops; lanes over rows keep each row's addition
   order, so a SIMD version (go-simd, no FMA) keeps tier-2 parity. The kernels
   go into the lowest shared library (go-simd/floats or the fleet's BLAS
   package, whichever already owns level-2/3 kernels), one implementation.
3. Zero-allocation `Refactor` and `SpluWith(Workspace)` like `order.Workspace`
   and `ldl.Permuted.Reset`.
4. `basis`: build/ftran/btran/update per iteration against HiGHS on recorded
   basis sequences (recorded from HiGHS runs on netlib LPs by the harness).

## 7. Validation plan

- Oracles: SciPy live session (perm_c, perm_r, L, U, x, spilu factors,
  `inv`, `expm`, `expm_multiply`, `onenormest` with `np.random.seed`); the C
  SuperLU oracle (tier 2); HiGHS C++ oracle driven through its C API (tier 5).
  Fixtures committed as JSON like `internal/conformance/testdata`.
- SciPy's `_dsolve/tests/test_linsolve.py` cases ported (singular, rank
  deficient, NATURAL, transposed solves, SamePattern).
- Matrices: the existing NIST Harwell-Boeing corpus (`west0067`, `west0479`,
  `sherman1/4`, `orsirr_1`, `orsreg_1`, `pores_3`...), plus SuiteSparse Matrix
  Collection matrices as data - downloaded into the corpus cache outside any
  work tree, SHA-256 pinned, never committed (corpus.go policy): unsymmetric
  LU `HB/gre_1107`, `HB/lns_3937`, `HB/west2021`, `HB/sherman5`, `HB/mcfe`,
  `Mallya/lhr10`, `Goodwin/goodwin`, `Bai/af23560`, `Simon/raefsky3` (large
  supernodes); simplex bases `HB/gemat11`, `HB/gemat12` and `LPnetlib/*`
  constraint matrices; KKT `GHS_indef/*` for the `ldl` comparison. Names to be
  confirmed against the collection index in the task that adds them.
- Properties (all inputs): `P A Q = L U` residual, `perm_r`/`perm_c`
  permutations, `Refactor` equals a fresh `Splu`, `Solve(Trans)` against
  `Aᵀ`, `Inv(A)·A ≈ I`, `basis` after k updates equals a fresh `Build` (tol).

## 8. Security and resource bounds

- Input: `validate` (sorted, in-range, no duplicates after `sum_duplicates`,
  `indptr` monotone, `len(indices) == indptr[n]`) before any allocation sized
  from the input; non-square refused; NaN/Inf never chosen as pivot (`fabs(NaN)
  > pivmax` is false) and reported as singular, as SuperLU.
- Memory: SuperLU allocates `30·nnz(A)` per array and grows by 1.5 with
  `Reduce` on failure (dmemory.c 216, 503-574). The port computes every size in
  `int64`, checks against `Limits` before `make`, and returns `ErrTooLarge`;
  index arrays are `int32` (as SciPy: `int_t` is `int`, indices cast to
  `np.intc`, linsolve.py 286-287), offsets `int`, products like
  `nsupr·nsupc` checked for overflow on 32-bit GOARCH.
- Supernode guard: SciPy aborts when `nsupr < nsupc` (dsnode_bmod.c 107-111);
  the port returns an error instead of panicking.
- Fuzz (corpus committed): `FuzzSplu` (bytes → small CSC, malformed
  structure, explicit zeros, NaN/Inf, empty rows/columns), `FuzzSolveTrans`,
  `FuzzSpsolveTriangular`, `FuzzBasisUpdate` (random column replacement
  sequences vs rebuild), `FuzzOnenormest`.
- Bomb tests assert refusal under `GOMEMLIMIT` and `ulimit -v`: an arrowhead
  with NATURAL ordering (dense fill, nnz(L+U) ~ n²) above `MaxFactorNNZ`; a
  header claiming `n = 2^31−1` with three entries; `MaxFactorNNZ` overflow of
  `int32`.
- `govulncheck` already in CI; no network, no files read by the library.

## 9. File layout

```text
order/permc.go            get_perm_c, GENMMD, SuperLU COLAMD (exact ports)
lu/etree.go               sp_coletree, sp_symetree, TreePostorder, sp_preorder
lu/relax.go               relax_snode, heap_relax_snode
lu/dfs.go                 column_dfs, panel_dfs, snode_dfs, pruneL
lu/bmod.go                column_bmod, panel_bmod, snode_bmod, copy_to_ucol
lu/pivot.go               pivotL
lu/memory.go              LUMemInit/Xpand with Limits
lu/gstrf.go lu/gstrs.go   driver, solves, sp_dtrsv
lu/ilu.go                 gsitrf, ilu_* (drop_row, pivotL, copy_to_ucol, ...)
lu/con.go                 gscon, langs, lacon2, pivotgrowth, gsequ, laqgs
lu/api.go                 Splu, Spilu, Spsolve, Factorized, SpsolveTriangular, Inv, Mldivide
lu/fuzz_test.go lu/testdata/
basis/factor.go build.go kernel.go tran.go update.go vector.go
matfun/onenormest.go expm.go expm_multiply.go
internal/scipy/           new ops; internal/conformance: LU/ILU/basis gates
cmd/sluoracle/            C-oracle driver (user-supplied SuperLU + netlib BLAS)
NOTICE                    SuperLU (LBNL BSD-3) and HiGHS (MIT) paragraphs
```

## 10. Ordered tasks (one PR each)

1. `order`: exact port of SuperLU `get_perm_c` (COLAMD, GENMMD on AᵀA and
   A+Aᵀ, NATURAL); gate `perm_c` bit-equal to SciPy on the corpus.
2. Dense kernels in reference-BLAS order without fusion (`dtrsv`, `dgemv`,
   `dtrsm`, `dgemm`), in the lowest shared library; fusion test on arm64.
3. `lu`: column etree, postorder, `sp_preorder`, relaxed supernodes.
4. `lu`: `dgstrf` + `dgstrs` port, `Splu`, `Factor` (Solve, L, U, PermR,
   PermC), `Limits`, `SingularError`; NOTICE.
5. C-oracle driver and SciPy gates: tiers 1-3, SciPy `test_linsolve.py` cases.
6. `lu`: `Spsolve` (CSR→transposed path), `Factorized`, `SpsolveTriangular`
   (tier 4), `Inv`.
7. `lu`: `Refactor` (SamePattern, SamePatternSameRowPerm) and `Workspace`,
   zero allocation once sized.
8. `lu`: `RCond`, `PivotGrowth`, equilibration; MATLAB `lu`/`Mldivide` delta.
9. `lu`: `Spilu` (dgsitrf, ILU drop rules, MILU); compare with `iter.NewILUT`.
10. `lu`: fuzz targets and bomb tests (hardening pass over 4-9).
11. Performance: SIMD kernels, perfbench rows vs SciPy, defects filed.
12. `basis`: HFactor INVERT (singletons, Markowitz kernel, rank deficiency)
    and FTRAN/BTRAN, bit-exact vs HiGHS.
13. `basis`: FT update (default), PF, MPF, APF, hyper-sparse solves, update
    fuzz; perf vs HiGHS.
14. `matfun`: `Onenormest` (Algorithm 2.4).
15. `matfun`: `Expm` (sparse Padé 3-13 with the S10 `ell` correction, uses
    `lu.Spsolve`).
16. `matfun`: `ExpmMultiply` (Algorithms 3.2 and 5.2 of S9).
17. `lu`: complex128 (SuperLU `z*` routines) - only if a consumer asks.

Each PR updates README "Not provided yet" and `docs/coverage.md` rows 71-72
and the related rows; the docs site and the org landing get their own PRs
after each release.
