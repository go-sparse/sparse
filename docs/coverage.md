# Coverage against SciPy, MATLAB and SuiteSparse

Inventory of what `go-sparse/sparse` (main, v0.3.0) implements against
`scipy.sparse`, `scipy.sparse.linalg` and `scipy.sparse.csgraph` v1.18.1, the
MATLAB sparse-matrix functions and the capabilities of SuiteSparse. Status:
**done** (same semantics and options that matter), **partial** (what is missing
is named), **missing**. "Dep" is what a missing item needs beyond this package.

Sources. SciPy: `scipy/sparse/__init__.py`, `scipy/sparse/linalg/__init__.py`
and `scipy/sparse/csgraph/__init__.py` at tag v1.18.1 (the rows below follow
their autosummary lists; the per-class method rows follow the documented
`sparray` API from knowledge, not from the source). MATLAB: the MathWorks
function lists could not be read by the tool used (client-rendered pages), so
the MATLAB column is from knowledge of the "Sparse Matrices" and "Graph and
Network Algorithms" function families and must be re-checked against the
MathWorks reference before a slice is started. SuiteSparse: names of packages
and capabilities only; no LGPL/GPL source (CHOLMOD, UMFPACK, SPQR, CXSparse,
KLU) was read, and the same rule holds for the slices below: they are written
from the papers or from BSD/Apache sources. Go side: README "Not provided yet",
`go doc -all`, and the sources on main.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 | 19 | 21 | 65 | 105 |
| MATLAB | 19 | 23 | 56 | 98 |

A row counts under SciPy when its SciPy cell is filled and under MATLAB when its
MATLAB cell is filled. SuiteSparse is listed in the last section, uncounted.

## Formats and classes (`scipy.sparse`)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `coo_array` | `sparse(i,j,v,m,n)` | `COO[T]` (float64, complex128) | partial | N-D COO (SciPy 1.15+ has 1-D and N-D), other dtypes (float32, ints, bool); no `sum_duplicates` in place, `has_canonical_format` flag |
| 2 | `csr_array` | `sparse` | `CSR[T]` | partial | 1-D CSR, `int64` indices (dimensions limited to `MaxDim` = 2^31-1), unsorted or non-canonical indices (only canonical form is accepted), other dtypes |
| 3 | `csc_array` | `sparse` | `CSC[T]` | partial | same as `csr_array` |
| 4 | `dok_array` | | `DOK[T]` | partial | slicing, `update`, `pop`, arithmetic (only `At`, `Set`, `Len`, `ToCSR`, `ToCOO`) |
| 5 | `lil_array` | | | missing | none (row-list builder) |
| 6 | `dia_array` | `spdiags` storage | | missing | none (`Diags` builds CSR but no DIA type or DIA matvec) |
| 7 | `bsr_array` | | | missing | none (block CSR, block SpMV/SpMM) |
| 8 | `sparray` base, `spmatrix` | | `Scalar`, `LinearOperator` | partial | no common interface across formats |
| 9 | `issparse`, `isspmatrix` | `issparse` | | missing | none (type checks are Go's) |
| 10 | `SparseEfficiencyWarning`, `SparseWarning` | | | missing | not applicable (errors are returned) |

## Constructors and combination (`scipy.sparse`)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 11 | `diags_array`, `diags` | `spdiags` (build) | `Diags` | done | |
| 12 | `eye_array`, `eye` | `speye` | `Eye` (square only) | partial | rectangular `m x n`, offset `k` |
| 13 | `identity` | `speye` | `Eye` | done | |
| 14 | `random_array`, `random` | `sprand`, `sprandn` | `gallery.RandomSparse` (seeded, fixed nonzeros per row) | partial | density parameter, `data_rvs`, `sprandsym` forms; random state is not NumPy's |
| 15 | `block_array`, `bmat` | `[A B; C D]`, `blkdiag` | | missing | none |
| 16 | `block_diag` | `blkdiag` | | missing | none |
| 17 | `kron` | `kron` | | missing | none (SpGEMM-like loop) |
| 18 | `kronsum` | | | missing | `kron`, `Eye` |
| 19 | `hstack`, `vstack` | `horzcat`, `vertcat`, `cat` | | missing | none |
| 20 | `tril`, `triu` | `tril`, `triu` | `CSR.Tril(k)`, `CSR.Triu(k)` | done | |
| 21 | `matrix_transpose`, `swapaxes`, `expand_dims`, `permute_dims` | `permute` | | missing | N-D arrays (see row 1) |
| 22 | `find` | `find` | `COO` via `ToCOO` | partial | no `find` returning three vectors (trivial) |
| 23 | `get_index_dtype`, `safely_cast_index_arrays` | | | missing | not applicable |
| 24 | `save_npz`, `load_npz` | `save`, `load` | | missing | an npz/zip reader (stdlib `archive/zip`) and NumPy `.npy` codec (go-ndarray may have one) |
| 25 | `scipy.io.mmread`, `mmwrite` | `mmread.m` | `mtx.Read`, `ReadLimited`, `Write` | done | |
| 26 | `scipy.io.hb_read`, `hb_write` | `hb_to_msm` | | missing | Harwell-Boeing reader/writer |
| 27 | `scipy.io.savemat` sparse | `save -v7` | | missing | MAT-file codec |
| 28 | | `spconvert` | | missing | none |
| 29 | | `spones`, `spfun` | | missing | none (pattern, map over values) |
| 30 | | `nnz`, `nonzeros`, `nzmax`, `spalloc` | `NNZ`, `Data` | done | |
| 31 | | `gallery` (`wathen`, `poisson`, `tridiag`, ...) , `delsq`, `numgrid` | `gallery.Laplacian1D/2D/3D`, `ConvectionDiffusion2D` | partial | `numgrid`/`delsq` on irregular regions, Wathen, other gallery matrices |
| 32 | | `spy`, `gplot` | | missing | plotting (a go-plot dependency; out of scope here) |

## Methods of a sparse array (`sparray` API)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 33 | `@`, `dot` (matvec, matmat, dense) | `*` | `MulVec`, `MulVecT`, `MulDense`, `Mul` | done | |
| 34 | `+`, `-` between sparse | `+`, `-` | `Add`, `Sub`, `Axpby` | done | sparse plus dense, broadcast |
| 35 | scalar `*`, `/` | `*`, `/` | `Scale` | partial | `/` and non-mutating forms |
| 36 | `multiply` (elementwise, with broadcast) | `.*` | | missing | none |
| 37 | `maximum`, `minimum` | `max`, `min` (two args) | | missing | none |
| 38 | `power`, `**`, `sqrt`, `abs`, `sign`, `rint`, `floor`, `ceil`, `trunc`, `expm1`, `log1p`, `sin`, `tan`, `arcsin`, ... | `abs`, `sqrt`, `sign`, ... | | missing | none (value maps) |
| 39 | comparison `==`, `<`, ... | `==`, `<` | | missing | none |
| 40 | `T`, `transpose`, `H` | `'`, `.'` | `Transpose`, `T`, `H` | done | |
| 41 | `conj`, `real`, `imag` | `conj`, `real`, `imag` | `H` only | partial | `Conj`, `Real`, `Imag` |
| 42 | `diagonal(k)` | `diag` | `Diagonal(k)` | done | |
| 43 | `setdiag` | | | missing | none |
| 44 | `trace` | `trace` | | missing | none |
| 45 | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `nanmax`, `nanmin` (with `axis`) | `sum`, `mean`, `max`, `min` | `Norm1`, `NormInf`, `NormFro` only | partial | axis reductions |
| 46 | `count_nonzero`, `nnz` | `nnz` | `NNZ` | partial | `count_nonzero` (excludes explicit zeros) |
| 47 | `eliminate_zeros` | | `Prune(0)` | done | |
| 48 | `sum_duplicates`, `sort_indices`, `has_sorted_indices`, `has_canonical_format` | | conversions sum and sort | partial | in-place forms and flags |
| 49 | `astype`, `copy` | `double`, `single` | `Clone` | partial | `astype` (float32 and integer dtypes) |
| 50 | `reshape`, `resize` | `reshape` | | missing | none |
| 51 | `nonzero` | `find` | | missing | none |
| 52 | `tocoo`, `tocsr`, `tocsc`, `todense`, `toarray` | `full`, `sparse` | `ToCOO`, `ToCSR`, `ToCSC`, `Dense`, `FromDense` | done | |
| 53 | `todia`, `tobsr`, `tolil`, `todok` | | `DOK.ToCSR` | missing | `dia_array`, `bsr_array`, `lil_array` (rows 5 to 7) |
| 54 | indexing, slicing, fancy indexing, boolean masks, assignment | `A(i,j)`, `A(i,j)=v` | `At` | partial | row/column slicing, fancy indexing, assignment; the largest missing API |
| 55 | `permute` via indexing `A[p][:, q]` | `A(p,q)` | `Permute` | done | |
| 56 | `getrow`, `getcol`, `getnnz`, `getformat` | | | missing | none (slicing) |
| 57 | `Dims()`, `shape`, `ndim` | `size` | `Dims` | done | |
| 58 | | `issymmetric`, `ishermitian` | `IsSymmetric(tol)` | partial | Hermitian form, `isbanded`, `istril`, `istriu` |
| 59 | `bandwidth` helpers (`spbandwidth`) | `bandwidth` | `order.Bandwidth` | done | |
| 60 | `is_sptriangular` | `istril`, `istriu` | | missing | none |

## `scipy.sparse.linalg`

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 61 | `LinearOperator` | function handles | `LinearOperator`, `TransposeOperator`, `FuncOperator` | partial | real only; no `rmatvec` for complex, `matmat`, composition (`+`, `*`, `**`, adjoint), dtype |
| 62 | `aslinearoperator` | | `*CSR[float64]`, `*CSC[float64]` satisfy it | partial | wrapper for dense arrays (`nd`), complex |
| 63 | `inv` | `inv` | | missing | sparse LU (row 79) |
| 64 | `expm`, `matrix_power` | `expm`, `mpower` | | missing | `matrix_power`: `Mul` only; `expm`: Pade + sparse solve (sparse LU) |
| 65 | `expm_multiply` | | | missing | none (Al-Mohy and Higham 2011; matvec only) |
| 66 | `funm_multiply_krylov` | | | missing | Arnoldi (GMRES basis exists) plus a dense `f(H)` (dense `funm`/eig) |
| 67 | `norm` | `norm`, `normest` | `Norm1`, `NormInf`, `NormFro` | partial | `ord=2` (needs `svds`), `-1`, `-inf`, `nuclear`, `axis` |
| 68 | `onenormest` | `normest1`, `condest` | | missing | none (Higham-Tisseur, matvec only; `condest` also needs LU) |
| 69 | `spsolve` | `\` | | missing | sparse LU |
| 70 | `spsolve_triangular` | `\` on triangular | | missing | none (forward/back substitution on CSR) |
| 71 | `factorized`, `splu`, `SuperLU` | `lu`, `decomposition` | `ldl` (quasi-definite LDLt only) | missing | sparse LU with partial pivoting, COLAMD column ordering (have `order.COLAMD`), supernodes |
| 72 | `spilu` | `ilu` | `iter.NewILU0`, `NewILUT` | partial | `SuperLU`-style `drop_tol`/`fill_factor` options, `solve` object; ILU(k) |
| 73 | `use_solver`, `MatrixRankWarning` | `spparms` | | missing | not applicable |
| 74 | `cg` | `pcg` | `iter.CG` | done | `callback`, `x0` forms to confirm; complex |
| 75 | `bicg` | `bicg` | | missing | none (needs `MulVecT`) |
| 76 | `bicgstab` | `bicgstab`, `bicgstabl` | `iter.BiCGSTAB` | done | `bicgstabl` (BiCGSTAB(l)) is not in SciPy |
| 77 | `cgs` | `cgs` | | missing | none |
| 78 | `gmres` | `gmres` | `iter.GMRES` | done | `restart`/callback forms equal; complex |
| 79 | `lgmres` | | | missing | none (GMRES with augmentation vectors) |
| 80 | `minres` | `minres`, `symmlq` | `iter.MINRES` | partial | `symmlq` (not in SciPy); `shift`; `show` |
| 81 | `qmr` | `qmr` | | missing | none (needs `MulVecT`, optional two preconditioners) |
| 82 | `gcrotmk` | | | missing | none |
| 83 | `tfqmr` | `tfqmr` | | missing | none |
| 84 | `lsqr` | `lsqr` | `iter.LSQR` | done | |
| 85 | `lsmr` | | `iter.LSMR` | done | |
| 86 | `eigs` | `eigs` | | missing | ARPACK-equivalent implicitly restarted Arnoldi (Lehoucq-Sorensen-Yang) written from the book; dense Hessenberg eig and Schur reordering (dense linalg), shift-invert needs sparse LU |
| 87 | `eigsh` | `eigs` (symmetric) | | missing | implicitly restarted Lanczos (thick restart is simpler); dense symmetric tridiagonal eig; shift-invert needs `ldl` or sparse LU |
| 88 | `lobpcg` | | | missing | dense generalised symmetric eig, Cholesky |
| 89 | `svds` (arpack, lobpcg, propack) | `svds` | | missing | `eigsh` on `A^T A` / augmented matrix, or Golub-Kahan-Lanczos (PROPACK-style) with reorthogonalisation |
| 90 | `LaplacianNd` | `delsq` | `gallery.Laplacian1D/2D/3D` | partial | N-D, boundary-condition choices, eigenpairs in closed form, Neumann/periodic |
| 91 | `ArpackNoConvergence`, `ArpackError` | | | missing | with `eigs` |
| 92 | `splu` for complex | | | missing | sparse LU, complex |
| 93 | | `chol`, `ichol` | `iter.NewIC0` | partial | sparse Cholesky (supernodal or up-looking, written from the papers); `ichol` threshold variant (`ict`) and shifts |
| 94 | | `qr` (sparse, `spqr`) | | missing | sparse QR (Householder multifrontal or Givens); `order.COLAMD` and `QRFillNNZ` exist for the symbolic side |
| 95 | | `ldl` (sparse) | `ldl` package (quasi-definite, no pivoting) | partial | Bunch-Kaufman style 2x2 pivoting (MA57-like), general symmetric indefinite |
| 96 | | `decomposition` object | | missing | with sparse LU/Cholesky |
| 97 | | `sprank`, `dmperm` | | missing | maximum matching (row 123) and Dulmage-Mendelsohn decomposition |
| 98 | | `etree`, `symbfact` | `ldl.Analyze` (etree and counts inside) | partial | public `Etree`, `SymbFact` on a general pattern; `treelayout`, `treeplot`, `etreeplot` are plotting |
| 99 | | `colperm`, `symrcm`, `amd`, `symamd`, `colamd` | `order.RCM`, `AMD`, `COLAMD` | partial | `colperm` (count-based, trivial), `symamd` (symmetric ordering from the pattern; `AMD` covers the use), `ccolamd`, `csymamd` (constrained variants) |
| 100 | | `pcg`, `gmres` with function-handle preconditioners | `iter.Options.M` | done | |

## `scipy.sparse.csgraph`

No part of csgraph is provided. All of it needs only this package's CSR, plus
a binary heap and a union-find (own, in Go) unless noted.

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 101 | `connected_components` (weak, strong) | `conncomp` | | missing | none (DFS, Tarjan or Pearce) |
| 102 | `laplacian` (normed, `use_out_degree`, `symmetrized`, returns operator) | `laplacian(G)` | | missing | none; the `LaplacianNd` operator form needs a `LinearOperator` |
| 103 | `shortest_path` | `shortestpath`, `distances` | | missing | the methods below |
| 104 | `dijkstra` (`indices`, `limit`, `min_only`, predecessors) | `shortestpath('positive')` | | missing | none |
| 105 | `floyd_warshall` | `distances` | | missing | none |
| 106 | `bellman_ford` | `shortestpath('mixed')` | | missing | none |
| 107 | `johnson` | `distances` | | missing | `bellman_ford`, `dijkstra` |
| 108 | `yen` (K shortest paths) | | | missing | `dijkstra` |
| 109 | `breadth_first_order`, `breadth_first_tree` | `bfsearch` | | missing | none |
| 110 | `depth_first_order`, `depth_first_tree` | `dfsearch` | | missing | none |
| 111 | `minimum_spanning_tree` | `minspantree` | | missing | none (Kruskal with union-find) |
| 112 | `reverse_cuthill_mckee` | `symrcm` | `order.RCM` | done | |
| 113 | `maximum_flow` (Edmonds-Karp, Dinic), `MaximumFlowResult` | `maxflow` | | missing | none |
| 114 | `maximum_bipartite_matching` | `matching`, `dmperm` | | missing | none (Hopcroft-Karp) |
| 115 | `min_weight_full_bipartite_matching` | `matchpairs` | | missing | none (Jonker-Volgenant, LAPJVsp) |
| 116 | `structural_rank` | `sprank` | | missing | `maximum_bipartite_matching` |
| 117 | `NegativeCycleError` | | | missing | with `bellman_ford` |
| 118 | `construct_dist_matrix`, `reconstruct_path` | `shortestpath` (paths) | | missing | `dijkstra` predecessors |
| 119 | `csgraph_from_dense`, `csgraph_to_dense`, `csgraph_masked_from_dense`, `csgraph_from_masked`, `csgraph_to_masked` | `graph`, `digraph`, `adjacency` | | missing | masked arrays have no Go equivalent: dense with an explicit "no edge" value instead |
| 120 | | `toposort`, `isdag`, `transclosure`, `transreduction`, `centrality`, `biconncomp`, `condensation` | | missing | none (outside SciPy) |
| 121 | | `graph`, `digraph` classes | | missing | out of scope here (a graph library) |

## SuiteSparse capabilities (names only, uncounted)

| SuiteSparse | Capability | Go | Status | Dep |
|---|---|---|---|---|
| AMD, CAMD | approximate minimum degree | `order.AMD` | done (CAMD constrained form missing) | |
| COLAMD, CCOLAMD | column ordering | `order.COLAMD` | done (constrained form missing) | |
| BTF | block triangular form (Dulmage-Mendelsohn) | | missing | matching, strong components |
| LDL (Davis), QDLDL | simple sparse LDLt | `ldl` (QDLDL port) | done | |
| CHOLMOD | supernodal Cholesky/LDLt, updates and downdates | | missing | dense BLAS-like kernels |
| UMFPACK, KLU | sparse LU (multifrontal; circuit matrices) | | missing | sparse LU slice |
| SPQR | multifrontal sparse QR | | missing | dense QR kernels |
| CXSparse | sparse basics, complex | partial overlap | partial | |
| METIS / nested dissection | partitioning ordering | | missing | graph partitioning (METIS is Apache-2.0 since 5.x; a port is possible) |
| SPEX | exact integer-arithmetic LU | | missing | big-integer arithmetic |
| ParU | parallel multifrontal LU | | missing | sparse LU slice |
| RBio | Rutherford-Boeing I/O | | missing | Harwell-Boeing codec (row 26) |
| GraphBLAS | semiring sparse matrix algebra | | missing | generic semiring SpGEMM |
| Mongoose | graph partitioning | | missing | |

## Proposed slices

Ordered by value per cost; "dep" names what has to exist first.

1. **Containers and algebra** (rows 5 to 7, 15 to 19, 36 to 39, 41, 43 to 46, 49 to 51, 56, 60): `LIL`, `DIA`, `BSR`; `Kron`, `HStack`, `VStack`, `BlockDiag`, `BMat`; elementwise `Multiply`, `Maximum`, comparisons; reductions with `axis`; `Trace`, `SetDiag`. Dep: none.
2. **Indexing** (row 54): row/column slices, fancy indexing, assignment. Dep: slice 1 (`LIL`).
3. **csgraph** (rows 101 to 119): components, shortest paths, traversals, MST, flow, matching. Dep: none; `Laplacian` returning an operator needs row 61.
4. **More Krylov methods** (rows 61, 75, 77, 79, 81 to 83): `BiCG`, `CGS`, `QMR`, `TFQMR`, `LGMRES`, `GCROT(m,k)`, a composable `LinearOperator`, complex scalars. Dep: none.
5. **Triangular solves and sparse LU** (rows 63, 64, 69 to 71, 92, 96): `SpsolveTriangular`, then left-looking Gilbert-Peierls LU with partial pivoting over `order.COLAMD`, then `Spsolve`, `Splu`, `Inv`, `Expm`. Dep: slice 5a has none; `Expm` needs LU.
6. **Eigen and singular values** (rows 86 to 89, 91): thick-restart Lanczos `Eigsh`, implicitly restarted Arnoldi `Eigs`, `Lobpcg`, `Svds`. Dep: dense symmetric/nonsymmetric eig and SVD from the dense linear algebra library; shift-invert needs `ldl` (symmetric) or slice 5 (general).
7. **Cholesky, QR, matching-based structure** (rows 93 to 98): sparse Cholesky, sparse QR, `Sprank`, `Dmperm`, `Etree`. Dep: dense kernels; `Dmperm` needs csgraph matching (slice 3).
8. **Matrix functions and estimators** (rows 65, 66, 68): `ExpmMultiply`, `FunmMultiplyKrylov`, `OneNormEst`. Dep: none beyond slice 4 for the Arnoldi basis.
9. **I/O** (rows 24, 26, 27): npz, Harwell-Boeing, MAT. Dep: go-ndarray for `.npy`.
10. **Other dtypes and N-D** (rows 1 to 3, 21): `float32`, integer, bool; `int64` indices; N-D COO. Dep: a design decision on generics (`Scalar` constraint) and the 32-bit-index policy.
