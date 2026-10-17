// Package nd connects sparse matrices to go-ndarray arrays: dense vectors
// and matrices go in and out of the solvers as *ndarray.Array.
package nd

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
	"github.com/go-sparse/sparse"
)

// Vector copies a 1-D float64 array into a slice.
func Vector(a *ndarray.Array) ([]float64, error) {
	if a.Ndim() != 1 {
		return nil, fmt.Errorf("%w: need a 1-D array, got %d-D", sparse.ErrShape, a.Ndim())
	}
	out := make([]float64, a.Size())
	for i := range out {
		out[i] = a.At(i)
	}
	return out, nil
}

// FromVector copies a slice into a new 1-D array.
func FromVector(v []float64) *ndarray.Array {
	a, _ := ndarray.FromData(v, len(v)) // the shape always matches
	return a
}

// FromCSR returns the dense 2-D array of a real CSR matrix.
func FromCSR(a *sparse.CSR[float64]) (*ndarray.Array, error) {
	return ndarray.FromData(a.Dense(), a.Rows, a.Cols)
}

// ToCSR converts a 2-D float64 array to CSR, dropping zeros.
func ToCSR(a *ndarray.Array) (*sparse.CSR[float64], error) {
	if a.Ndim() != 2 {
		return nil, fmt.Errorf("%w: need a 2-D array, got %d-D", sparse.ErrShape, a.Ndim())
	}
	m, n := a.Shape()[0], a.Shape()[1]
	d := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			d[i*n+j] = a.At(i, j)
		}
	}
	return sparse.FromDense(m, n, d)
}

// MulMat computes the sparse-times-dense product A B for a 2-D array B.
func MulMat(a *sparse.CSR[float64], b *ndarray.Array) (*ndarray.Array, error) {
	if b.Ndim() != 2 || b.Shape()[0] != a.Cols {
		return nil, fmt.Errorf("%w: %dx%d times %v", sparse.ErrShape, a.Rows, a.Cols, b.Shape())
	}
	k := b.Shape()[1]
	bd := make([]float64, a.Cols*k)
	for i := 0; i < a.Cols; i++ {
		for j := 0; j < k; j++ {
			bd[i*k+j] = b.At(i, j)
		}
	}
	out := make([]float64, a.Rows*k)
	a.MulDense(out, bd, k)
	return ndarray.FromData(out, a.Rows, k)
}

// MulVec computes A x for a 1-D array x.
func MulVec(a *sparse.CSR[float64], x *ndarray.Array) (*ndarray.Array, error) {
	xv, err := Vector(x)
	if err != nil {
		return nil, err
	}
	if len(xv) != a.Cols {
		return nil, fmt.Errorf("%w: %dx%d times %d", sparse.ErrShape, a.Rows, a.Cols, len(xv))
	}
	y := make([]float64, a.Rows)
	a.MulVec(y, xv)
	return FromVector(y), nil
}
