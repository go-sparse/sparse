# sparse

Sparse matrices, fill-reducing orderings and iterative solvers in pure Go
(`CGO_ENABLED=0`, every architecture the fleet's CI tests). Module
`github.com/go-sparse/sparse`, BSD-3-Clause.

```go
a := gallery.Laplacian2D(100, 100)           // *sparse.CSR[float64]
b := make([]float64, a.Rows)
ic, _ := iter.NewIC0(a)                      // incomplete Cholesky
res, _ := iter.CG(a, b, &iter.Options{RTol: 1e-8, M: ic})
fmt.Println(res.Info, res.Iterations)        // 0 = converged, as in SciPy
```

## What is provided

| package | provides | SciPy | MATLAB |
|---|---|---|---|
| `sparse` | `COO`, `CSR`, `CSC`, `DOK` over `float64` and `complex128`; conversions (duplicates summed, indices sorted); O(1) `Transpose`, `T`, `H`; `MulVec`/`MulVecT` (SpMV), `MulDense` (SpMM), `Mul` (SpGEMM), `Add`/`Sub`/`Axpby`; `Scale`, `ScaleRows`, `ScaleCols`; `Diagonal(k)`, `Tril`, `Triu`, `Permute`, `Prune`, norms; `Eye`, `Diags`; `LinearOperator` | `scipy.sparse.{coo,csr,csc,dok}_array`, `diags`, `eye`, `LinearOperator` | `sparse`, `spdiags`, `speye`, `nnz`, `tril`, `triu` |
| `nd` | `*ndarray.Array` ([go-ndarray](https://github.com/go-ndarray/ndarray) v0.12.0) in and out: vectors, dense matrices, SpMV and SpMM | – | – |
| `mtx` | Matrix Market read/write: coordinate and array, real/integer/pattern/complex, general/symmetric/skew-symmetric/hermitian, gzip input | `scipy.io.mmread`, `mmwrite` | `mmread.m` |
| `order` | `RCM`, `AMD`, `COLAMD`; `Bandwidth`, `Profile`, `CholeskyNNZ`, `QRFillNNZ` | `reverse_cuthill_mckee` | `symrcm`, `amd`, `symamd`, `colamd` |
| `iter` | `CG`, `GMRES(m)`, `BiCGSTAB`, `MINRES`, `LSQR`, `LSMR`; preconditioners `Jacobi`, `SSOR`, `IC0`, `ILU0`, `ILUT` | `cg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `lsmr` | `pcg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `ichol`, `ilu` |
| `gallery` | Laplacians (1-D/2-D/3-D), convection-diffusion, seeded random sparse matrices | – | `gallery` |

Solvers follow `scipy.sparse.linalg`: the stopping rule is
`‖b − A x‖ ≤ max(rtol·‖b‖, atol)`, `rtol` defaults to 1e-5, `maxiter` to
10n (5n for MINRES), `info` is 0 on success, the iteration limit when it was
reached, and negative on breakdown; GMRES uses SciPy's left preconditioning
and inner-tolerance control. The solvers take any `sparse.LinearOperator`.

Not provided yet: sparse direct factorisations (LU, Cholesky, QR), nested
dissection, eigensolvers, algebraic multigrid, Rutherford-Boeing I/O.

## Orderings: provenance

`AMD` and `COLAMD` are written from the papers (Amestoy, Davis and Duff,
SIAM J. Matrix Anal. Appl. 17(4), 1996; Davis, Gilbert, Larimore and Ng, ACM
TOMS 30(3), 2004) on one quotient-graph engine. SuiteSparse AMD and COLAMD
(BSD-3-Clause) are the reference algorithms; CHOLMOD, UMFPACK and CSparse
(GPL/LGPL) were not consulted. Tie-breaking differs, so permutations differ;
quality is compared by fill instead.

## Verification

`docs/verification.md` compares every solver, LSQR/LSMR and the orderings
with SciPy 1.18.1 / NumPy 2.5.4 (and SuiteSparse AMD through cvxopt as a
binary oracle). A Go harness (`cmd/genfixtures`) drives SciPy over a pipe,
using only the minimal SciPy call per request, and records the answers in
`internal/conformance/testdata/oracle.json`; the regression tests re-run the
Go side from that file, so they need neither Python nor network.
`cmd/conformance -corpus` also compares live on small Harwell-Boeing
matrices from NIST Matrix Market, downloaded into a cache outside the
repository (their licence is unstated) and checked against pinned SHA-256 sums.

Results on the committed suite: CG, GMRES, MINRES, LSQR and LSMR iteration
counts equal SciPy's; BiCGSTAB agrees within a few iterations. AMD Cholesky
fill is within 1% of SuiteSparse AMD; COLAMD LU fill is within 4% of
SuperLU's COLAMD; RCM bandwidth equals SciPy's.

## Performance

`go run ./cmd/perfbench` measures SpMV, CG and GMRES(30) on 2-D Laplacians of
1e5 and 1e6 nonzeros against SciPy on one core, interleaved, one at a time,
and refuses to measure while the host's load average is above 8. The table
is `docs/perf.md`. **Status: not yet measured** — the development host's load
average was 20 to 110 throughout, so every cell is empty rather than wrong;
re-run the command on an idle machine and commit the table.

The kernels (SpMV, SpMM, dot, axpy, norms) are scalar pure Go for now. SIMD
and generated-assembly versions belong in shared code (go-asmgen / go-ndarray)
rather than here, one implementation per kernel; they are not done yet.

## Untrusted input

`mtx.Read` is meant to be safe on hostile files. `mtx.ReadLimited` takes
`Limits` (defaults: dimensions 2^27, 2^28 entries, 64 KiB lines, 16 GiB read
after decompression, 4096 comment lines); declared sizes are checked before
anything is allocated from them, the stored-value count is computed in `int64`
(so it cannot overflow a 32-bit `int`), memory grows only as entries are
actually read, and a gzip bomb is cut off at the byte limit. `NewCSR`/`NewCSC`
validate hand-built arrays, `COO.Validate` does the same for a hand-built COO,
and `FromDense` checks `rows*cols` without overflow. `mtx.FuzzRead` has a
committed corpus under `mtx/testdata/fuzz` and its seeds run in every `go test`.

## Quality gates

`gofmt -l`, `go vet`, `go test -race` with 100% statement coverage over all
packages (`-coverpkg=./...`), 386 and the other architectures in
`.github/workflows/ci.yml`, `govulncheck`.

## Tooling rule

Generators, harnesses and comparisons are Go programs. The only Python is the
oracle's own SciPy call (`internal/scipy/script.go`).
