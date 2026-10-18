package iter

import (
	"container/heap"
	"fmt"
	"math"
	"sort"

	"github.com/go-sparse/sparse"
)

// Jacobi is the diagonal preconditioner M = diag(A).
type Jacobi struct{ inv []float64 }

// NewJacobi builds the Jacobi preconditioner; a zero diagonal entry is an error.
func NewJacobi(a *sparse.CSR[float64]) (*Jacobi, error) {
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: matrix is not square", ErrShape)
	}
	d := a.Diagonal(0)
	for i, v := range d {
		if v == 0 {
			return nil, fmt.Errorf("%w: zero diagonal at %d", ErrBreakdown, i)
		}
		d[i] = 1 / v
	}
	return &Jacobi{inv: d}, nil
}

// Solve implements Preconditioner.
func (j *Jacobi) Solve(dst, r []float64) {
	for i, v := range r {
		dst[i] = j.inv[i] * v
	}
}

// SSOR is the symmetric successive over-relaxation preconditioner
// M = 1/(ω(2-ω)) (D+ωL) D⁻¹ (D+ωU), for the splitting A = L + D + U.
type SSOR struct {
	a     *sparse.CSR[float64]
	diag  []float64
	omega float64
}

// NewSSOR builds SSOR with relaxation 0 < omega < 2 (omega = 1 is symmetric
// Gauss-Seidel).
func NewSSOR(a *sparse.CSR[float64], omega float64) (*SSOR, error) {
	if !(omega > 0 && omega < 2) {
		return nil, fmt.Errorf("%w: omega %g outside (0,2)", ErrBreakdown, omega)
	}
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: matrix is not square", ErrShape)
	}
	d := a.Diagonal(0)
	for i, v := range d {
		if v == 0 {
			return nil, fmt.Errorf("%w: zero diagonal at %d", ErrBreakdown, i)
		}
	}
	return &SSOR{a: a, diag: d, omega: omega}, nil
}

// Solve implements Preconditioner.
func (s *SSOR) Solve(dst, r []float64) {
	a, w := s.a, s.omega
	n := a.Rows
	// (D + ωL) y = r
	for i := 0; i < n; i++ {
		t := r[i]
		for p := a.Indptr[i]; p < a.Indptr[i+1] && a.Indices[p] < i; p++ {
			t -= w * a.Data[p] * dst[a.Indices[p]]
		}
		dst[i] = t / s.diag[i]
	}
	// y := D y ; (D + ωU) z = y
	for i := 0; i < n; i++ {
		dst[i] *= s.diag[i]
	}
	for i := n - 1; i >= 0; i-- {
		t := dst[i]
		for p := a.Indptr[i+1] - 1; p >= a.Indptr[i] && a.Indices[p] > i; p-- {
			t -= w * a.Data[p] * dst[a.Indices[p]]
		}
		dst[i] = t / s.diag[i]
	}
	f := w * (2 - w)
	for i := range dst {
		dst[i] *= f
	}
}

// IC0 is the incomplete Cholesky factorisation with zero fill, M = L Lᵀ,
// L having the sparsity of the lower triangle of A (Meijerink and van der
// Vorst 1977).
type IC0 struct{ l *sparse.CSR[float64] }

// NewIC0 factorises the lower triangle of a symmetric positive definite
// matrix; a non-positive pivot returns ErrBreakdown.
func NewIC0(a *sparse.CSR[float64]) (*IC0, error) {
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: matrix is not square", ErrShape)
	}
	l := a.Tril(0)
	n := l.Rows
	pos := make([]int, n)
	for i := range pos {
		pos[i] = -1
	}
	diag := make([]int, n)
	for i := 0; i < n; i++ {
		lo, hi := l.Indptr[i], l.Indptr[i+1]
		if lo == hi || l.Indices[hi-1] != i {
			return nil, fmt.Errorf("%w: missing diagonal at %d", ErrBreakdown, i)
		}
		diag[i] = hi - 1
		for p := lo; p < hi; p++ {
			pos[l.Indices[p]] = p
		}
		for p := lo; p < hi-1; p++ {
			k := l.Indices[p]
			s := l.Data[p]
			for q := l.Indptr[k]; q < diag[k]; q++ {
				if pp := pos[l.Indices[q]]; pp >= 0 {
					s -= l.Data[pp] * l.Data[q]
				}
			}
			l.Data[p] = s / l.Data[diag[k]]
		}
		s := l.Data[diag[i]]
		for p := lo; p < hi-1; p++ {
			s -= l.Data[p] * l.Data[p]
		}
		if !(s > 0) {
			return nil, fmt.Errorf("%w: non-positive pivot %g at row %d", ErrBreakdown, s, i)
		}
		l.Data[diag[i]] = math.Sqrt(s)
		for p := lo; p < hi; p++ {
			pos[l.Indices[p]] = -1
		}
	}
	return &IC0{l: l}, nil
}

// L returns the Cholesky factor (shared, do not modify).
func (c *IC0) L() *sparse.CSR[float64] { return c.l }

// Solve implements Preconditioner: dst = (L Lᵀ)⁻¹ r.
func (c *IC0) Solve(dst, r []float64) {
	l := c.l
	n := l.Rows
	for i := 0; i < n; i++ {
		t := r[i]
		hi := l.Indptr[i+1] - 1
		for p := l.Indptr[i]; p < hi; p++ {
			t -= l.Data[p] * dst[l.Indices[p]]
		}
		dst[i] = t / l.Data[hi]
	}
	for i := n - 1; i >= 0; i-- {
		hi := l.Indptr[i+1] - 1
		dst[i] /= l.Data[hi]
		for p := l.Indptr[i]; p < hi; p++ {
			dst[l.Indices[p]] -= l.Data[p] * dst[i]
		}
	}
}

// ILU is an incomplete LU factorisation A ≈ L U, L unit lower triangular
// (strict part stored), U upper triangular with the diagonal stored first
// in each row. It is produced by NewILU0 and NewILUT.
type ILU struct {
	n      int
	lp, li []int
	lx     []float64
	up, ui []int
	ux     []float64
}

// L returns the strict lower factor (the unit diagonal is implicit).
func (f *ILU) L() *sparse.CSR[float64] {
	return &sparse.CSR[float64]{Rows: f.n, Cols: f.n, Indptr: f.lp, Indices: f.li, Data: f.lx}
}

// U returns the upper factor including its diagonal.
func (f *ILU) U() *sparse.CSR[float64] {
	return &sparse.CSR[float64]{Rows: f.n, Cols: f.n, Indptr: f.up, Indices: f.ui, Data: f.ux}
}

// Solve implements Preconditioner: dst = U⁻¹ L⁻¹ r.
func (f *ILU) Solve(dst, r []float64) {
	for i := 0; i < f.n; i++ {
		t := r[i]
		for p := f.lp[i]; p < f.lp[i+1]; p++ {
			t -= f.lx[p] * dst[f.li[p]]
		}
		dst[i] = t
	}
	for i := f.n - 1; i >= 0; i-- {
		t := dst[i]
		for p := f.up[i] + 1; p < f.up[i+1]; p++ {
			t -= f.ux[p] * dst[f.ui[p]]
		}
		dst[i] = t / f.ux[f.up[i]]
	}
}

// NewILU0 computes ILU(0): the factors have the sparsity pattern of A.
func NewILU0(a *sparse.CSR[float64]) (*ILU, error) {
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: matrix is not square", ErrShape)
	}
	n := a.Rows
	w := a.Clone()
	diag := make([]int, n)
	pos := make([]int, n)
	for i := range pos {
		pos[i] = -1
	}
	for i := 0; i < n; i++ {
		lo, hi := w.Indptr[i], w.Indptr[i+1]
		diag[i] = -1
		for p := lo; p < hi; p++ {
			pos[w.Indices[p]] = p
			if w.Indices[p] == i {
				diag[i] = p
			}
		}
		if diag[i] < 0 {
			return nil, fmt.Errorf("%w: missing diagonal at %d", ErrBreakdown, i)
		}
		for p := lo; p < diag[i]; p++ {
			k := w.Indices[p]
			w.Data[p] /= w.Data[diag[k]]
			m := w.Data[p]
			for q := diag[k] + 1; q < w.Indptr[k+1]; q++ {
				if pp := pos[w.Indices[q]]; pp >= 0 {
					w.Data[pp] -= m * w.Data[q]
				}
			}
		}
		if w.Data[diag[i]] == 0 {
			return nil, fmt.Errorf("%w: zero pivot at row %d", ErrBreakdown, i)
		}
		for p := lo; p < hi; p++ {
			pos[w.Indices[p]] = -1
		}
	}
	f := &ILU{n: n, lp: make([]int, n+1), up: make([]int, n+1)}
	for i := 0; i < n; i++ {
		for p := w.Indptr[i]; p < w.Indptr[i+1]; p++ {
			if p < diag[i] {
				f.li, f.lx = append(f.li, w.Indices[p]), append(f.lx, w.Data[p])
			} else {
				f.ui, f.ux = append(f.ui, w.Indices[p]), append(f.ux, w.Data[p])
			}
		}
		f.lp[i+1], f.up[i+1] = len(f.li), len(f.ui)
	}
	return f, nil
}

type minHeap []int

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(int)) }
func (h *minHeap) Pop() any {
	old := *h
	v := old[len(old)-1]
	*h = old[:len(old)-1]
	return v
}

// NewILUT computes the dual-threshold incomplete LU of Saad (1994): entries
// smaller than droptol times the 2-norm of their row of A are dropped, and
// at most lfil entries are kept in each row of L and in each row of U
// (the diagonal is always kept). lfil <= 0 means no cap.
func NewILUT(a *sparse.CSR[float64], droptol float64, lfil int) (*ILU, error) {
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: matrix is not square", ErrShape)
	}
	n := a.Rows
	if lfil <= 0 {
		lfil = n
	}
	f := &ILU{n: n, lp: make([]int, n+1), up: make([]int, n+1)}
	w := make([]float64, n)
	seen := make([]bool, n)
	var lower minHeap
	var upper []int
	type ent struct {
		j int
		v float64
	}
	var lrow, urow []ent
	for i := 0; i < n; i++ {
		lower, upper = lower[:0], upper[:0]
		rn := 0.0
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			j := a.Indices[p]
			w[j] = a.Data[p]
			seen[j] = true
			rn += a.Data[p] * a.Data[p]
			if j < i {
				lower = append(lower, j)
			} else {
				upper = append(upper, j)
			}
		}
		tau := droptol * math.Sqrt(rn)
		heap.Init(&lower)
		lrow = lrow[:0]
		for lower.Len() > 0 {
			k := heap.Pop(&lower).(int)
			ukk := f.ux[f.up[k]]
			w[k] /= ukk
			if math.Abs(w[k]) < tau {
				w[k] = 0
				seen[k] = false
				continue
			}
			m := w[k]
			lrow = append(lrow, ent{k, m})
			for q := f.up[k] + 1; q < f.up[k+1]; q++ {
				j := f.ui[q]
				if !seen[j] {
					seen[j] = true
					w[j] = 0
					if j < i {
						heap.Push(&lower, j)
					} else {
						upper = append(upper, j)
					}
				}
				w[j] -= m * f.ux[q]
			}
		}
		urow = urow[:0]
		var dv float64
		hasDiag := false
		for _, j := range upper {
			if j == i {
				dv, hasDiag = w[j], true
			} else if math.Abs(w[j]) >= tau {
				urow = append(urow, ent{j, w[j]})
			}
			w[j] = 0
			seen[j] = false
		}
		for _, e := range lrow {
			w[e.j] = 0
			seen[e.j] = false
		}
		if !hasDiag || dv == 0 {
			return nil, fmt.Errorf("%w: zero pivot at row %d", ErrBreakdown, i)
		}
		keepLargest := func(r []ent) []ent {
			if len(r) > lfil {
				sort.Slice(r, func(x, y int) bool { return math.Abs(r[x].v) > math.Abs(r[y].v) })
				r = r[:lfil]
			}
			sort.Slice(r, func(x, y int) bool { return r[x].j < r[y].j })
			return r
		}
		lrow, urow = keepLargest(lrow), keepLargest(urow)
		for _, e := range lrow {
			f.li, f.lx = append(f.li, e.j), append(f.lx, e.v)
		}
		f.ui, f.ux = append(f.ui, i), append(f.ux, dv)
		for _, e := range urow {
			f.ui, f.ux = append(f.ui, e.j), append(f.ux, e.v)
		}
		f.lp[i+1], f.up[i+1] = len(f.li), len(f.ui)
	}
	return f, nil
}
