# Public functions without a timed reference row

`refbench` requires every public function to have a case in `bench/` or to be
named here with its reason (docs/performance-campaign.md in go-scikit/roadmap,
section 2.1). The rows that exist are in [perf-table.md](perf-table.md).

## Accessors, constructors of a container, adapters (no work to time)

- `COO.Dims`: returns two fields.
- `COO.NNZ`: returns a length.
- `COO.Append`: appends one triplet; the cost is the amortised `append`, and the container has no reference that is not Python-interpreter overhead.
- `COO.Validate`: one pass over the triplets; scipy's `check_format` on COO is covered by the `CSR.Validate` row's reference and has no separate public function.
- `COO.Transpose`: swaps the row and column slices (O(1) aliasing copy).
- `CSC.Dims`: returns two fields.
- `CSC.NNZ`: returns a length.
- `CSR.Dims`: returns two fields.
- `CSR.NNZ`: returns a length.
- `DOK.Dims`: returns two fields.
- `DOK.Len`: returns a map length.
- `NewCOO`: allocates an empty container.
- `NewCSC`: wraps and validates caller-owned arrays (the validation is the `CSR.Validate` row).
- `NewCSR`: wraps and validates caller-owned arrays (the validation is the `CSR.Validate` row).
- `NewDOK`: allocates an empty map.
- `FuncOperator.Dims`: returns two fields.
- `FuncOperator.MulVec`: one indirect call to the caller's function.
- `FuncOperator.MulVecT`: one indirect call to the caller's function.
- `Shifted`: builds a closure; the product it returns is the wrapped operator's.
- `nd.FromCSR`: converts to a go-ndarray array; go-ndarray is not in the pinned reference set and the conversion has no scipy counterpart.
- `nd.FromVector`: ditto, a copy of a vector.
- `nd.MulMat`: adapter around `CSR.MulDense`, which has a row.
- `nd.MulVec`: adapter around `CSR.MulVec`, which has a row.
- `nd.ToCSR`: ditto, the reverse conversion.
- `nd.Vector`: ditto, a copy of a vector.
- `mtx.Matrix.RealCSR`: converts the parsed triplets, the work `mtx.Read` is measured with.
- `mtx.Matrix.ComplexCSR`: ditto.
- `mtx.ReadFile`: `mtx.Read` plus a file open; the file system is not the library's.
- `mtx.ReadLimited`: `mtx.Read` with explicit limits (`mtx.Read` is the row).

## Container operations whose reference is Python's own dispatch

- `CSR.At`: one entry lookup (a binary search in a row); scipy's `A[i, j]` is dominated by the Python indexing machinery, so a ratio would measure the interpreter.
- `CSC.At`: ditto.
- `DOK.At`: a map lookup; scipy's `dok_matrix[i, j]` is the same Python machinery.
- `DOK.Set`: a map store, ditto.
- `DOK.ToCOO`: collects a map; scipy's `dok_matrix.tocoo` converts a Python dict.
- `DOK.ToCSR`: `DOK.ToCOO` then `COO.ToCSR` (which has a row).
- `CSC.Clone`: the `CSR.Clone` row measures the same copy (a CSC clone is a CSR clone of its transpose).
- `CSC.Dense`: `CSC.ToCSR` (row) then `CSR.Dense` (row).
- `CSC.Diagonal`: `CSR.Diagonal` (row) on the transposed view.
- `CSC.Scale`: `CSR.Scale` (row) on the transposed view.
- `CSC.ToCOO`: `CSR.ToCOO` (row) on the transposed view.
- `CSC.Transpose`: the O(1) view, as the `CSR.Transpose` row.
- `CSR.H`: `CSR.T` (row) with a conjugation that is the identity on real data; the complex path has no scipy row in this campaign (SciPy's complex sparse kernels are the same C++ templates).

## No reference in the pinned set

SciPy has no AMD, COLAMD, symbolic fill count, incomplete Cholesky, ILU(0),
SSOR or Jacobi preconditioner (its `spilu` is SuperLU's ILUT with a different
drop rule and a column ordering, a different algorithm), and no QDLDL. The
SuiteSparse and OSQP packages are not in `refs.txt`. These functions are
guarded by allocation tests that run in CI (`order/alloc_test.go`,
`order/workspace_test.go`, `ldl/reset_test.go`, `ldl/ldl_test.go`) and by the
conformance suite; a timing reference is open work tracked in issue #10 for the
ordering step (the reference would be SuiteSparse AMD, BSD-3, called by cgo
without reading its source).

- `order.AMD`: approximate minimum degree; allocations asserted by `order/alloc_test.go`.
- `order.AMDWith`: ditto, zero allocations on a reused workspace (`order/workspace_test.go`).
- `order.COLAMD`: column approximate minimum degree, ditto.
- `order.COLAMDWith`: ditto with a workspace.
- `order.Bandwidth`: one pass over the pattern; no scipy sparse equivalent (`scipy.linalg.bandwidth` is dense).
- `order.Profile`: one pass over the pattern.
- `order.CholeskyNNZ`: symbolic count from the elimination tree; allocations asserted by `order/alloc_test.go`.
- `order.QRFillNNZ`: `sparse.Mul` (row) then `order.CholeskyNNZ`.
- `order.Inverse`: one allocation and a loop over a permutation.
- `order.IsPermutation`: one allocation and a loop.
- `gallery.ConvectionDiffusion2D`: no scipy.sparse constructor; the structure is the `gallery.Laplacian2D` row's.
- `gallery.RandomSparse`: seeded by a PCG generator; scipy's `sparse.random` draws a different matrix (the timed work differs), so no honest ratio.
- `gallery.Scramble`: a seeded permutation then `CSR.Permute` (row).
- `iter.NewJacobi`: extracts the diagonal (`CSR.Diagonal` row).
- `iter.Jacobi.Solve`: one elementwise product.
- `iter.NewSSOR`: factors nothing; copies the splitting.
- `iter.SSOR.Solve`: two triangular sweeps; scipy has no SSOR preconditioner.
- `iter.NewIC0`: incomplete Cholesky with no fill; scipy has none.
- `iter.IC0.L`: returns the factor.
- `iter.IC0.Solve`: two triangular solves on the factor.
- `iter.NewILU0`: ILU(0); scipy's `spilu` is SuperLU's ILUT.
- `iter.NewILUT`: ILUT with a Go drop rule; scipy's `spilu` uses SuperLU's with a column permutation.
- `iter.ILU.L`: returns the factor.
- `iter.ILU.U`: returns the factor.
- `iter.ILU.Solve`: two triangular solves on the factors.
- `ldl.Factorize`: `ldl.Analyze`, `Symbolic.NewFactor` and `Factor.Refactor` in a row; the first and last have rows, the middle one is an allocation.
- `ldl.Symbolic.NewFactor`: allocates L and the workspace once (QDLDL's caller mallocs the same arrays).
- `ldl.Symbolic.N`: returns a field.
- `ldl.Symbolic.NNZL`: returns a field.
- `ldl.Symbolic.Parent`: returns a slice.
- `ldl.Symbolic.ColCounts`: returns a slice.
- `ldl.Factor.D`: returns a slice.
- `ldl.Factor.NNZL`: returns a length.
- `ldl.Factor.NumPositive`: returns a field.
- `ldl.Factor.Symbolic`: returns a field.
- `ldl.NewPermuted`: `Permuted.Reset` on a new value; QDLDL has no permutation, so there is no C counterpart for the pattern permutation (the numeric phase `Permuted.Refactor` and `Permuted.Solve` have rows against QDLDL wrapped in the gather and scatter an OSQP-style caller writes). Allocation counts asserted by `ldl/reset_test.go`.
- `ldl.Permuted.Reset`: ditto.
- `ldl.Permuted.Factor`: returns a field.
- `ldl.Permuted.NNZL`: returns a length.
- `ldl.Permuted.NumPositive`: returns a field.
