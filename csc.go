package sparse

// CSC is a compressed-sparse-column matrix. Column j occupies
// Indices[Indptr[j]:Indptr[j+1]] (row indices, strictly increasing).
type CSC[T Scalar] struct {
	Rows, Cols int
	Indptr     []int
	Indices    []int32
	Data       []T
}

// NewCSC wraps the given arrays (without copying) after validation.
func NewCSC[T Scalar](rows, cols int, indptr []int, indices []int32, data []T) (*CSC[T], error) {
	a := &CSC[T]{Rows: rows, Cols: cols, Indptr: indptr, Indices: indices, Data: data}
	if err := validate(cols, rows, indptr, indices, len(data)); err != nil {
		return nil, err
	}
	return a, nil
}

// Dims returns the number of rows and columns.
func (a *CSC[T]) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ returns the number of stored entries.
func (a *CSC[T]) NNZ() int { return len(a.Data) }

// view returns Aᵀ as a CSR matrix sharing storage.
func (a *CSC[T]) view() *CSR[T] {
	return &CSR[T]{Rows: a.Cols, Cols: a.Rows, Indptr: a.Indptr, Indices: a.Indices, Data: a.Data}
}

// Clone returns a deep copy.
func (a *CSC[T]) Clone() *CSC[T] { return a.view().Clone().Transpose() }

// At returns A[i, j].
func (a *CSC[T]) At(i, j int) T { return a.view().At(j, i) }

// MulVec computes dst = A x (column-oriented scatter).
func (a *CSC[T]) MulVec(dst, x []T) { a.view().MulVecT(dst, x) }

// MulVecSymUpper computes dst = S x for the symmetric matrix S whose upper
// triangle (diagonal included) is stored in A, as OSQP stores its cost matrix
// P: entry (i, j) with i < j also acts as (j, i), so the lower half need not be
// stored. Entries stored below the diagonal are ignored. S is symmetric, not
// Hermitian: complex entries are not conjugated. A must be square and dst, x
// must not alias.
func (a *CSC[T]) MulVecSymUpper(dst, x []T) {
	if a.Rows != a.Cols || len(x) != a.Cols || len(dst) != a.Rows {
		panic(shapeErr("MulVecSymUpper", [2]int{a.Rows, a.Cols}, [2]int{len(dst), len(x)}))
	}
	clear(dst)
	ptr := a.Indptr[:len(x)+1]
	ind, d := a.Indices[:ptr[len(x)]], a.Data[:ptr[len(x)]]
	for j, xj := range x {
		var s T
		for p := ptr[j]; p < ptr[j+1]; p++ {
			i := int(ind[p])
			switch {
			case i < j:
				dst[i] += d[p] * xj
				s += d[p] * x[i]
			case i == j:
				s += d[p] * xj
			}
		}
		dst[j] += s
	}
}

// MulVecT computes dst = Aᵀ x.
func (a *CSC[T]) MulVecT(dst, x []T) { a.view().MulVec(dst, x) }

// Transpose returns Aᵀ as a CSR matrix sharing A's arrays (O(1)).
func (a *CSC[T]) Transpose() *CSR[T] { return a.view() }

// ToCSR converts to CSR (a copy).
func (a *CSC[T]) ToCSR() *CSR[T] { return a.view().T() }

// ToCOO converts to coordinate format.
func (a *CSC[T]) ToCOO() *COO[T] { return a.view().ToCOO().Transpose() }

// Dense returns the matrix as a row-major dense slice.
func (a *CSC[T]) Dense() []T { return a.ToCSR().Dense() }

// Diagonal returns the k-th diagonal.
func (a *CSC[T]) Diagonal(k int) []T { return a.view().Diagonal(-k) }

// Scale multiplies every entry by s in place.
func (a *CSC[T]) Scale(s T) { a.view().Scale(s) }
