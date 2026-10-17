// Package iter provides Krylov-subspace solvers for sparse linear systems
// (CG, GMRES(m), BiCGSTAB, MINRES, LSQR) and preconditioners (Jacobi, SSOR,
// IC(0), ILU(0), ILUT). The solvers follow scipy.sparse.linalg: the same
// stopping rule ||b - A x|| <= max(rtol*||b||, atol), the same default
// maxiter, the same info codes, so results are directly comparable.
//
// The algorithms are the textbook ones (Hestenes-Stiefel 1952; Saad-Schultz
// 1986; van der Vorst 1992; Paige-Saunders 1975 and 1982; Meijerink-van der
// Vorst 1977; Saad 1994). The stopping logic and the GMRES inner-tolerance
// control mirror SciPy's BSD-3-Clause implementation, which is credited here.
package iter

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-sparse/sparse"
)

// Operator is the matrix-free input of the solvers.
type Operator = sparse.LinearOperator

// Preconditioner applies an approximate inverse: Solve stores M⁻¹ r in dst.
// dst and r never alias.
type Preconditioner interface {
	Solve(dst, r []float64)
}

// Info codes of Result.Info (SciPy convention: 0 success, > 0 the iteration
// limit was reached, < 0 breakdown).
const (
	Converged      = 0
	BreakdownRho   = -10 // BiCGSTAB: (r̃, r) vanished
	BreakdownOmega = -11 // BiCGSTAB: omega or (r̃, v) vanished
)

// ErrShape reports an operator that is not square or a vector of the wrong size.
var ErrShape = errors.New("iter: incompatible shapes")

// ErrBreakdown reports a zero or non-positive pivot while factorising.
var ErrBreakdown = errors.New("iter: factorisation breakdown")

// Options configures CG, GMRES, BiCGSTAB and MINRES. The zero value is
// SciPy's default call.
type Options struct {
	// RTol is the relative tolerance; 0 means the SciPy default 1e-5 and a
	// negative value means exactly 0 (use ATol alone).
	RTol float64
	// ATol is the absolute tolerance (default 0).
	ATol float64
	// MaxIter bounds the iterations: 0 means 10n (5n for MINRES). For GMRES
	// it bounds restart cycles.
	MaxIter int
	// X0 is the initial guess (default zero). It is not modified.
	X0 []float64
	// M is the preconditioner (default none). For CG and MINRES it must be
	// symmetric positive definite. GMRES preconditions on the left.
	M Preconditioner
	// Callback, if set, is called after each iteration with the current
	// iterate (after each restart cycle for GMRES). x must not be retained.
	Callback func(x []float64)
	// Restart is the GMRES restart length m (default 20).
	Restart int
	// Shift solves (A - Shift*I) x = b in MINRES.
	Shift float64
}

// Result is the outcome of a solve.
type Result struct {
	X          []float64
	Info       int // see the info constants; > 0 is the iteration limit
	Iterations int // iterations performed (GMRES: inner iterations)
}

func (o *Options) rtol() float64 {
	switch {
	case o.RTol == 0:
		return 1e-5
	case o.RTol < 0:
		return 0
	}
	return o.RTol
}

// setup validates a square system and returns x0 (copied), the tolerance
// and ||b||.
func setup(a Operator, b []float64, o *Options) (x []float64, atol, bnorm float64, err error) {
	r, c := a.Dims()
	if r != c || len(b) != r {
		return nil, 0, 0, fmt.Errorf("%w: operator %dx%d, rhs %d", ErrShape, r, c, len(b))
	}
	x = make([]float64, r)
	if o.X0 != nil {
		if len(o.X0) != r {
			return nil, 0, 0, fmt.Errorf("%w: x0 has %d entries, want %d", ErrShape, len(o.X0), r)
		}
		copy(x, o.X0)
	}
	bnorm = nrm2(b)
	return x, math.Max(o.ATol, o.rtol()*bnorm), bnorm, nil
}

func identity(dst, r []float64) { copy(dst, r) }

func (o *Options) psolve() func(dst, r []float64) {
	if o.M == nil {
		return identity
	}
	return o.M.Solve
}

func dot(x, y []float64) float64 {
	var s float64
	for i, v := range x {
		s += v * y[i]
	}
	return s
}

func nrm2(x []float64) float64 {
	// scaled to avoid overflow, as BLAS dnrm2 does
	var scale, ssq float64 = 0, 1
	for _, v := range x {
		if v != 0 {
			a := math.Abs(v)
			if scale < a {
				ssq = 1 + ssq*(scale/a)*(scale/a)
				scale = a
			} else {
				ssq += (a / scale) * (a / scale)
			}
		}
	}
	return scale * math.Sqrt(ssq)
}

func axpy(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func anyNonzero(x []float64) bool {
	for _, v := range x {
		if v != 0 {
			return true
		}
	}
	return false
}

// residual stores b - A x in r (r = b when x is zero, avoiding a product).
func residual(a Operator, r, b, x []float64) {
	if !anyNonzero(x) {
		copy(r, b)
		return
	}
	a.MulVec(r, x)
	for i := range r {
		r[i] = b[i] - r[i]
	}
}

// TrueResidual returns ||b - A x|| / ||b|| (the quantity to verify, not the
// recurrence residual); it is ||b - A x|| when b is zero.
func TrueResidual(a Operator, b, x []float64) float64 {
	r := make([]float64, len(b))
	a.MulVec(r, x)
	for i := range r {
		r[i] = b[i] - r[i]
	}
	if nb := nrm2(b); nb > 0 {
		return nrm2(r) / nb
	}
	return nrm2(r)
}
