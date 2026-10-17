// Package sparse provides sparse matrices (COO, CSR, CSC, DOK), conversions,
// products, sums and scaling in pure Go, plus the LinearOperator abstraction
// used by the iterative solvers in package iter.
//
// Element types are float64 and complex128 (the Scalar constraint). Indices
// are Go ints, so the package works unchanged on 32-bit targets.
//
// Equivalents: scipy.sparse.{coo,csr,csc,dok}_array, MATLAB sparse, find,
// nnz, spdiags, speye.
package sparse

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
)

// Scalar is the element type of a sparse matrix.
type Scalar interface{ float64 | complex128 }

// Errors returned (or wrapped) by this package.
var (
	// ErrShape reports incompatible dimensions.
	ErrShape = errors.New("sparse: incompatible shapes")
	// ErrIndex reports an index outside the matrix.
	ErrIndex = errors.New("sparse: index out of range")
	// ErrFormat reports a malformed compressed structure.
	ErrFormat = errors.New("sparse: malformed structure")
)

func abs[T Scalar](v T) float64 {
	if c, ok := any(v).(complex128); ok {
		return cmplx.Abs(c)
	}
	return math.Abs(any(v).(float64))
}

func conj[T Scalar](v T) T {
	if c, ok := any(v).(complex128); ok {
		return any(cmplx.Conj(c)).(T)
	}
	return v
}

func fromReal[T Scalar](f float64) T {
	var z T
	if _, ok := any(z).(complex128); ok {
		return any(complex(f, 0)).(T)
	}
	return any(f).(T)
}

func shapeErr(what string, a, b any) error {
	return fmt.Errorf("%w: %s %v vs %v", ErrShape, what, a, b)
}
