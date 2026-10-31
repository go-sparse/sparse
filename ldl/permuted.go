package ldl

import (
	"fmt"

	"github.com/go-sparse/sparse"
)

// Permuted factors P K Pᵀ for a fill-reducing permutation, where K is given by
// its upper triangle. The permutation follows the package order: p[k] is the
// original index placed at position k, as returned by order.AMD, so that
// (P K Pᵀ)[i, j] = K[p[i], p[j]].
//
// The symbolic phase is done once in [NewPermuted]; [Permuted.Refactor] takes
// the original matrix with new values (same pattern) and refactors without
// allocating; [Permuted.Solve] permutes, solves and permutes back.
type Permuted struct {
	f    *Factor
	perm []int
	pa   *sparse.CSC[float64] // upper triangle of P K Pᵀ, values refreshed by Refactor
	src  []int                // pa.Data[k] = a.Data[src[k]]
	ap   []int                // pattern of the original matrix
	ai   []int32
	work []float64
}

// NewPermuted validates a (upper triangle, CSC) and perm, builds the upper
// triangle of P K Pᵀ and runs the symbolic phase on it. Call Refactor before
// Solve.
func NewPermuted(a *sparse.CSC[float64], perm []int) (*Permuted, error) {
	if _, err := checkCSC(a); err != nil {
		return nil, err
	}
	n := a.Cols
	if len(perm) != n {
		return nil, fmt.Errorf("%w: length %d, order %d", ErrPermutation, len(perm), n)
	}
	inv := make([]int, n)
	for i := range inv {
		inv[i] = -1
	}
	for k, v := range perm {
		if v < 0 || v >= n || inv[v] >= 0 {
			return nil, fmt.Errorf("%w: entry %d at position %d", ErrPermutation, v, k)
		}
		inv[v] = k
	}
	nnz := a.Indptr[n]

	// Entry (i, j) of K lands at (min, max) of (inv[i], inv[j]). Two stable
	// counting sorts (by row, then by column) give sorted columns.
	row := make([]int32, nnz)
	col := make([]int32, nnz)
	for j := 0; j < n; j++ {
		for p := a.Indptr[j]; p < a.Indptr[j+1]; p++ {
			r, c := inv[a.Indices[p]], inv[j]
			if r > c {
				r, c = c, r
			}
			row[p], col[p] = int32(r), int32(c)
		}
	}
	byRow := bucket(n, nnz, row, nil)
	byCol := bucket(n, nnz, col, byRow)
	ptr := make([]int, n+1)
	for _, c := range col {
		ptr[c+1]++
	}
	for j := 0; j < n; j++ {
		ptr[j+1] += ptr[j]
	}
	idx := make([]int32, nnz)
	for k, p := range byCol {
		idx[k] = row[p]
	}
	pa := &sparse.CSC[float64]{Rows: n, Cols: n, Indptr: ptr, Indices: idx, Data: make([]float64, nnz)}
	sym, err := Analyze(pa)
	if err != nil {
		return nil, err
	}
	return &Permuted{
		f:    sym.NewFactor(),
		perm: append([]int(nil), perm...),
		pa:   pa,
		src:  byCol,
		ap:   append([]int(nil), a.Indptr...),
		ai:   append([]int32(nil), a.Indices[:nnz]...),
		work: make([]float64, n),
	}, nil
}

// bucket returns the entry numbers 0..nnz-1 stably ordered by key (values in
// 0..n-1); with order non-nil the entries are visited in that order, so the
// result is sorted by key and, within equal keys, by the previous order.
func bucket(n, nnz int, key []int32, order []int) []int {
	cnt := make([]int, n+1)
	for _, k := range key {
		cnt[k+1]++
	}
	for i := 0; i < n; i++ {
		cnt[i+1] += cnt[i]
	}
	out := make([]int, nnz)
	for q := 0; q < nnz; q++ {
		p := q
		if order != nil {
			p = order[q]
		}
		out[cnt[key[p]]] = p
		cnt[key[p]]++
	}
	return out
}

// Refactor refreshes the permuted values from a (the original matrix, same
// pattern as at NewPermuted) and runs the numeric phase.
func (p *Permuted) Refactor(a *sparse.CSC[float64]) error {
	p.f.ok = false
	n := p.f.sym.n
	if a == nil {
		return fmt.Errorf("%w: nil matrix", ErrDimension)
	}
	if a.Rows != n || a.Cols != n {
		return fmt.Errorf("%w: matrix is %d x %d, analysed %d x %d", ErrDimension, a.Rows, a.Cols, n, n)
	}
	nnz := p.ap[n]
	if len(a.Indptr) != n+1 || len(a.Indices) < nnz || len(a.Data) < nnz {
		return ErrPattern
	}
	for j, v := range p.ap {
		if a.Indptr[j] != v {
			return ErrPattern
		}
	}
	for q, v := range p.ai {
		if a.Indices[q] != v {
			return ErrPattern
		}
	}
	for k, q := range p.src {
		p.pa.Data[k] = a.Data[q]
	}
	return p.f.Refactor(p.pa)
}

// Solve solves K y = x in place.
func (p *Permuted) Solve(x []float64) error {
	if err := p.f.check(x); err != nil {
		return err
	}
	w := p.work
	for k, v := range p.perm {
		w[k] = x[v]
	}
	_ = p.f.Solve(w) // cannot fail: factored and length checked above
	for k, v := range p.perm {
		x[v] = w[k]
	}
	return nil
}

// Factor returns the underlying factorisation of P K Pᵀ.
func (p *Permuted) Factor() *Factor { return p.f }

// NumPositive returns the number of positive entries of D (see Factor).
func (p *Permuted) NumPositive() int { return p.f.npos }

// NNZL returns the number of stored entries of L.
func (p *Permuted) NNZL() int { return p.f.NNZL() }
