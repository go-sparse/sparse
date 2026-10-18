package sparse

// LinearOperator is a real matrix-free linear map x -> A x, the abstraction
// the iterative solvers work on (scipy.sparse.linalg.LinearOperator).
// *CSR[float64] and *CSC[float64] implement it.
type LinearOperator interface {
	// Dims returns the number of rows and columns of A.
	Dims() (rows, cols int)
	// MulVec stores A x in dst; len(x) == cols, len(dst) == rows.
	MulVec(dst, x []float64)
}

// TransposeOperator is a LinearOperator that can also apply Aᵀ, which LSQR
// requires. *CSR[float64] and *CSC[float64] implement it.
type TransposeOperator interface {
	LinearOperator
	// MulVecT stores Aᵀ x in dst; len(x) == rows, len(dst) == cols.
	MulVecT(dst, x []float64)
}

// FuncOperator builds an operator from functions. MvT may be nil when no
// transpose product is available.
type FuncOperator struct {
	Rows, Cols int
	Mv, MvT    func(dst, x []float64)
}

// Dims implements LinearOperator.
func (f FuncOperator) Dims() (int, int) { return f.Rows, f.Cols }

// MulVec implements LinearOperator.
func (f FuncOperator) MulVec(dst, x []float64) { f.Mv(dst, x) }

// MulVecT implements TransposeOperator; it panics when MvT is nil.
func (f FuncOperator) MulVecT(dst, x []float64) { f.MvT(dst, x) }

// Shifted returns the operator A + shift*I.
func Shifted(a LinearOperator, shift float64) LinearOperator {
	r, c := a.Dims()
	return FuncOperator{Rows: r, Cols: c, Mv: func(dst, x []float64) {
		a.MulVec(dst, x)
		for i := range dst {
			dst[i] += shift * x[i]
		}
	}}
}
