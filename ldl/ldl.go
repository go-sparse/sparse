// Package ldl is a sparse LDLᵀ factorisation of quasi-definite (more generally,
// strongly factorisable) symmetric matrices, the factorisation OSQP uses for
// its KKT systems.
//
// The algorithms and the three-phase interface (elimination tree and column
// counts, numeric factorisation, solve) are a port of QDLDL v0.1.9,
// https://github.com/osqp/qdldl, Copyright 2018 Paul Goulart, Bartolomeo
// Stellato, Goran Banjac and the OSQP developers, licensed under the Apache
// License 2.0 (see NOTICE in the repository root). Only QDLDL was consulted;
// Davis's LDL (LGPL) was not.
//
// A matrix K is quasi-definite when K = [[A, Bᵀ], [B, −C]] with A and C
// positive definite: every symmetric permutation of K then has an LDLᵀ
// factorisation with a diagonal D (positive and negative entries, no pivoting)
// so the ordering can be chosen for sparsity alone. The factorisation fails,
// with an error, on an exact zero pivot (or a NaN or infinite one); it does
// not pivot.
//
// The matrix is given as its upper triangle (diagonal included) in CSC form,
// as OSQP stores it. The symbolic phase ([Analyze]) depends on the pattern
// only; the numeric phase ([Factor.Refactor]) can be redone with new values on
// the same pattern, without allocating, which is what an ADMM loop that
// changes ρ needs.
//
//	sym, err := ldl.Analyze(kkt)       // pattern only
//	f := sym.NewFactor()
//	err = f.Refactor(kkt)              // numeric; again after changing values
//	err = f.Solve(x)                   // x ← K⁻¹ x, in place
//
// [Permuted] wraps the same phases around a fill-reducing permutation such as
// order.AMD.
package ldl

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-sparse/sparse"
)

// Errors returned by the package, matched with errors.Is.
var (
	// ErrDimension: the matrix is not square, or a vector has the wrong length.
	ErrDimension = errors.New("ldl: dimension mismatch")
	// ErrNotUpper: an entry lies below the diagonal. Pass the upper triangle only.
	ErrNotUpper = errors.New("ldl: entry below the diagonal (the upper triangle is expected)")
	// ErrEmptyColumn: a column has no stored entry. QDLDL requires at least
	// one per column; store an explicit zero on the diagonal if needed.
	ErrEmptyColumn = errors.New("ldl: empty column")
	// ErrDuplicate: a row index appears twice in a column.
	ErrDuplicate = errors.New("ldl: duplicate entry")
	// ErrPattern: the matrix passed to Refactor does not have the sparsity
	// pattern that was analysed.
	ErrPattern = errors.New("ldl: sparsity pattern differs from the analysed one")
	// ErrPermutation: the permutation is not a permutation of 0..n-1.
	ErrPermutation = errors.New("ldl: invalid permutation")
	// ErrOverflow: a size does not fit the index types (or would not fit
	// in memory on this platform).
	ErrOverflow = errors.New("ldl: size overflow")
	// ErrZeroPivot: a pivot of D is exactly zero; K is not quasi-definite.
	ErrZeroPivot = errors.New("ldl: zero pivot")
	// ErrNonFinite: a pivot of D is NaN or infinite.
	ErrNonFinite = errors.New("ldl: non-finite pivot")
	// ErrNotFactored: Solve was called before a successful Refactor.
	ErrNotFactored = errors.New("ldl: not factored")
)

// PivotError reports the column whose pivot made the factorisation fail. It
// unwraps to ErrZeroPivot or ErrNonFinite.
type PivotError struct {
	Column int
	Err    error
}

func (e *PivotError) Error() string { return fmt.Sprintf("%v at column %d", e.Err, e.Column) }
func (e *PivotError) Unwrap() error { return e.Err }

const unknown = -1

// maxLNZ bounds nnz(L) so that 12 bytes per entry (int32 index, float64 value)
// cannot overflow an int, whatever the platform.
var maxLNZ = math.MaxInt / 16 // a variable so a test can lower it

// Symbolic is the result of the symbolic phase: the elimination tree and the
// structure of L. It depends on the sparsity pattern of the matrix only and is
// immutable, so it can back several Factors.
type Symbolic struct {
	n     int
	ap    []int   // pattern of the analysed matrix (copy)
	ai    []int32 // ditto
	etree []int32 // parent of each column, -1 for a root
	lnz   []int32 // nonzeros below the diagonal in each column of L
	lp    []int   // column pointers of L
}

// N returns the order of the matrix.
func (s *Symbolic) N() int { return s.n }

// NNZL returns the number of stored (strictly lower) entries of L, the unit
// diagonal excluded.
func (s *Symbolic) NNZL() int { return s.lp[s.n] }

// Parent returns the elimination tree: Parent()[j] is the parent of column j,
// or -1 for a root. The slice must not be modified.
func (s *Symbolic) Parent() []int32 { return s.etree }

// ColCounts returns the number of entries below the diagonal in each column of
// L. The slice must not be modified.
func (s *Symbolic) ColCounts() []int32 { return s.lnz }

// checkCSC validates the structure of the upper triangle a: square, indptr
// well formed, indices in range and on or above the diagonal, no duplicates,
// at least one entry per column. It accepts unsorted columns, as QDLDL does.
// stamp is scratch of length n.
func checkCSC(a *sparse.CSC[float64]) (stamp []int32, err error) {
	if a == nil {
		return nil, fmt.Errorf("%w: nil matrix", ErrDimension)
	}
	n := a.Cols
	if a.Rows != n {
		return nil, fmt.Errorf("%w: matrix is %d x %d, not square", ErrDimension, a.Rows, a.Cols)
	}
	if n < 0 || n > math.MaxInt32-1 {
		return nil, fmt.Errorf("%w: order %d", ErrOverflow, n)
	}
	stamp = make([]int32, n)
	return stamp, checkStructure(a, stamp)
}

func checkStructure(a *sparse.CSC[float64], stamp []int32) error {
	n := a.Cols
	if len(a.Indptr) != n+1 || a.Indptr[0] != 0 {
		return fmt.Errorf("%w: indptr must have length %d and start at 0", sparse.ErrFormat, n+1)
	}
	nnz := a.Indptr[n]
	if nnz < 0 || nnz > len(a.Indices) || nnz > len(a.Data) {
		return fmt.Errorf("%w: indptr end %d, %d indices, %d values", sparse.ErrFormat, nnz, len(a.Indices), len(a.Data))
	}
	for j := 0; j < n; j++ {
		lo, hi := a.Indptr[j], a.Indptr[j+1]
		if hi < lo || hi > nnz {
			return fmt.Errorf("%w: indptr decreases at %d", sparse.ErrFormat, j)
		}
		if hi == lo {
			return fmt.Errorf("%w %d", ErrEmptyColumn, j)
		}
		for p := lo; p < hi; p++ {
			i := a.Indices[p]
			if i < 0 {
				return fmt.Errorf("%w: index %d in column %d", sparse.ErrIndex, i, j)
			}
			if int(i) > j {
				return fmt.Errorf("%w: row %d in column %d", ErrNotUpper, i, j)
			}
			if stamp[i] == int32(j)+1 {
				return fmt.Errorf("%w: row %d in column %d", ErrDuplicate, i, j)
			}
			stamp[i] = int32(j) + 1
		}
	}
	return nil
}

// Analyze runs the symbolic phase on the upper triangle a of a symmetric
// matrix: it validates the structure, computes the elimination tree and the
// column counts of L (QDLDL_etree). Only O(n + nnz(a)) memory is allocated;
// L itself is allocated by [Symbolic.NewFactor], so a caller can refuse a
// factor that is too large by looking at NNZL first.
func Analyze(a *sparse.CSC[float64]) (*Symbolic, error) {
	work, err := checkCSC(a)
	if err != nil {
		return nil, err
	}
	n := a.Cols
	nnz := a.Indptr[n]
	s := &Symbolic{
		n:     n,
		ap:    append([]int(nil), a.Indptr...),
		ai:    append([]int32(nil), a.Indices[:nnz]...),
		etree: make([]int32, n),
		lnz:   make([]int32, n),
		lp:    make([]int, n+1),
	}
	etree, lnz := s.etree, s.lnz
	for i := range work {
		work[i] = 0
		etree[i] = unknown
	}
	for j := 0; j < n; j++ {
		work[j] = int32(j)
		for p := a.Indptr[j]; p < a.Indptr[j+1]; p++ {
			i := a.Indices[p]
			for work[i] != int32(j) {
				if etree[i] == unknown {
					etree[i] = int32(j)
				}
				lnz[i]++
				work[i] = int32(j)
				i = etree[i]
			}
		}
	}
	sum := 0
	for i, c := range lnz {
		if int(c) > maxLNZ-sum {
			return nil, fmt.Errorf("%w: nnz(L) above %d", ErrOverflow, maxLNZ)
		}
		sum += int(c)
		s.lp[i+1] = sum
	}
	return s, nil
}

// samePattern reports whether a has exactly the analysed pattern.
func (s *Symbolic) samePattern(a *sparse.CSC[float64]) error {
	if a == nil {
		return fmt.Errorf("%w: nil matrix", ErrDimension)
	}
	if a.Rows != s.n || a.Cols != s.n {
		return fmt.Errorf("%w: matrix is %d x %d, analysed %d x %d", ErrDimension, a.Rows, a.Cols, s.n, s.n)
	}
	nnz := s.ap[s.n]
	if len(a.Indptr) != s.n+1 || len(a.Indices) < nnz || len(a.Data) < nnz {
		return ErrPattern
	}
	for j, v := range s.ap {
		if a.Indptr[j] != v {
			return ErrPattern
		}
	}
	for p, v := range s.ai {
		if a.Indices[p] != v {
			return ErrPattern
		}
	}
	return nil
}

// Factor holds L, D and the workspace of the numeric phase for a Symbolic. A
// Factor is not safe for concurrent use; Solve calls on a factored Factor
// are read-only and may run concurrently.
type Factor struct {
	sym  *Symbolic
	li   []int32
	lx   []float64
	d    []float64
	dinv []float64
	npos int
	ok   bool

	// workspace of Refactor
	ymark []bool
	yidx  []int32
	elim  []int32
	next  []int
	yval  []float64
}

// NewFactor allocates the storage for L, D and the workspace. It returns nil
// only when the symbolic factor is nil.
func (s *Symbolic) NewFactor() *Factor {
	if s == nil {
		return nil
	}
	nl := s.lp[s.n]
	return &Factor{
		sym:   s,
		li:    make([]int32, nl),
		lx:    make([]float64, nl),
		d:     make([]float64, s.n),
		dinv:  make([]float64, s.n),
		ymark: make([]bool, s.n),
		yidx:  make([]int32, s.n),
		elim:  make([]int32, s.n),
		next:  make([]int, s.n),
		yval:  make([]float64, s.n),
	}
}

// Factorize is Analyze followed by NewFactor and Refactor.
func Factorize(a *sparse.CSC[float64]) (*Factor, error) {
	s, err := Analyze(a)
	if err != nil {
		return nil, err
	}
	f := s.NewFactor()
	if err := f.Refactor(a); err != nil {
		return nil, err
	}
	return f, nil
}

// Symbolic returns the symbolic factor this Factor was built from.
func (f *Factor) Symbolic() *Symbolic { return f.sym }

// D returns the diagonal of D. It is valid after a successful Refactor, and
// overwritten by the next one. The slice must not be modified.
func (f *Factor) D() []float64 { return f.d }

// NumPositive returns the number of positive entries of D (the positive
// inertia). For a KKT matrix [[P+σI, Aᵀ], [A, −diag(1/ρ)]] it is n when the
// matrix is quasi-definite; fewer means P is not positive semidefinite.
func (f *Factor) NumPositive() int { return f.npos }

// NNZL returns the number of stored entries of L.
func (f *Factor) NNZL() int { return len(f.li) }

// Refactor runs the numeric phase (QDLDL_factor) with the values of a, which
// must have the analysed pattern (checked in O(nnz)). It does not allocate.
// On a zero or non-finite pivot it returns a *PivotError and the Factor is
// not usable until a later Refactor succeeds.
func (f *Factor) Refactor(a *sparse.CSC[float64]) error {
	f.ok = false
	s := f.sym
	if err := s.samePattern(a); err != nil {
		return err
	}
	n := s.n
	if n == 0 {
		f.npos = 0
		f.ok = true
		return nil
	}
	ap, ai, ax := s.ap, s.ai, a.Data
	lp, li, lx := s.lp, f.li, f.lx
	d, dinv := f.d, f.dinv
	etree := s.etree
	ymark, yidx, elim, next, yval := f.ymark, f.yidx, f.elim, f.next, f.yval

	for i := 0; i < n; i++ {
		ymark[i] = false
		yval[i] = 0
		d[i] = 0
		next[i] = lp[i]
	}
	npos := 0

	// Column 0: its only possible entry is the diagonal.
	d[0] = ax[0]
	if err := checkPivot(d[0], 0); err != nil {
		return err
	}
	if d[0] > 0 {
		npos++
	}
	dinv[0] = 1 / d[0]

	for k := 1; k < n; k++ {
		// Row k of L is the solution of L[0:k,0:k] y = A[0:k,k]; its pattern
		// is found by walking the elimination tree from each nonzero of the
		// column, its values by the sparse triangular solve below.
		nnzY := 0
		for p := ap[k]; p < ap[k+1]; p++ {
			bidx := int(ai[p])
			if bidx == k {
				d[k] = ax[p]
				continue
			}
			yval[bidx] = ax[p]
			if !ymark[bidx] {
				ymark[bidx] = true
				elim[0] = int32(bidx)
				nnzE := 1
				nx := etree[bidx]
				for nx != unknown && int(nx) < k {
					if ymark[nx] {
						break
					}
					ymark[nx] = true
					elim[nnzE] = nx
					nnzE++
					nx = etree[nx]
				}
				for nnzE > 0 {
					nnzE--
					yidx[nnzY] = elim[nnzE]
					nnzY++
				}
			}
		}

		dk := d[k]
		for i := nnzY - 1; i >= 0; i-- {
			cidx := int(yidx[i])
			tmp := next[cidx]
			yc := yval[cidx]
			cols := li[lp[cidx]:tmp]
			vals := lx[lp[cidx]:tmp]
			for j, r := range cols {
				yval[r] -= vals[j] * yc
			}
			li[tmp] = int32(k)
			l := yc * dinv[cidx]
			lx[tmp] = l
			dk -= yc * l
			next[cidx] = tmp + 1
			yval[cidx] = 0
			ymark[cidx] = false
		}
		d[k] = dk
		if err := checkPivot(dk, k); err != nil {
			return err
		}
		if dk > 0 {
			npos++
		}
		dinv[k] = 1 / dk
	}
	f.npos = npos
	f.ok = true
	return nil
}

func checkPivot(v float64, k int) error {
	switch {
	case v == 0:
		return &PivotError{Column: k, Err: ErrZeroPivot}
	case math.IsNaN(v) || math.IsInf(v, 0):
		return &PivotError{Column: k, Err: ErrNonFinite}
	}
	return nil
}

func (f *Factor) check(x []float64) error {
	if !f.ok {
		return ErrNotFactored
	}
	if len(x) != f.sym.n {
		return fmt.Errorf("%w: vector of length %d, order %d", ErrDimension, len(x), f.sym.n)
	}
	return nil
}

// LSolve solves (L+I) y = x in place (QDLDL_Lsolve).
func (f *Factor) LSolve(x []float64) error {
	if err := f.check(x); err != nil {
		return err
	}
	lp, li, lx := f.sym.lp, f.li, f.lx
	for i, v := range x {
		lo, hi := lp[i], lp[i+1]
		cols, vals := li[lo:hi], lx[lo:hi]
		for j, r := range cols {
			x[r] -= vals[j] * v
		}
	}
	return nil
}

// LTSolve solves (L+I)ᵀ y = x in place (QDLDL_Ltsolve).
func (f *Factor) LTSolve(x []float64) error {
	if err := f.check(x); err != nil {
		return err
	}
	lp, li, lx := f.sym.lp, f.li, f.lx
	for i := len(x) - 1; i >= 0; i-- {
		lo, hi := lp[i], lp[i+1]
		cols, vals := li[lo:hi], lx[lo:hi]
		v := x[i]
		for j, r := range cols {
			v -= vals[j] * x[r]
		}
		x[i] = v
	}
	return nil
}

// DSolve solves D y = x in place.
func (f *Factor) DSolve(x []float64) error {
	if err := f.check(x); err != nil {
		return err
	}
	dinv := f.dinv[:len(x)]
	for i := range x {
		x[i] *= dinv[i]
	}
	return nil
}

// Solve solves K y = x in place, where K = L D Lᵀ is the factored matrix
// (QDLDL_solve).
func (f *Factor) Solve(x []float64) error {
	if err := f.LSolve(x); err != nil {
		return err
	}
	_ = f.DSolve(x)
	return f.LTSolve(x)
}
