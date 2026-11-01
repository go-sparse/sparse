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
//
// A Permuted owns all the memory of the symbolic phase, so [Permuted.Reset]
// can redo it for another matrix and permutation (say, the next problem of a
// sequence) without allocating once the storage is large enough.
type Permuted struct {
	f    *Factor
	perm []int
	pa   *sparse.CSC[float64] // upper triangle of P K Pᵀ, values refreshed by Refactor
	src  []int                // pa.Data[k] = a.Data[src[k]]
	ap   []int                // pattern of the original matrix
	ai   []int32
	work []float64

	// scratch of Reset
	stamp      []int32
	inv        []int
	row, col   []int32
	byRow, cnt []int
	sym        Symbolic
	valid      bool
}

// NewPermuted validates a (upper triangle, CSC) and perm, builds the upper
// triangle of P K Pᵀ and runs the symbolic phase on it. Call Refactor before
// Solve.
func NewPermuted(a *sparse.CSC[float64], perm []int) (*Permuted, error) {
	p := new(Permuted)
	if err := p.Reset(a, perm); err != nil {
		return nil, err
	}
	return p, nil
}

// Reset is [NewPermuted] on p itself: it validates a and perm and redoes the
// symbolic phase, reusing the storage of the previous one where it is large
// enough, so that a sequence of problems of similar size allocates nothing
// after the first. The Symbolic and Factor values obtained from p before the
// call are overwritten and must not be used afterwards. On an error p is
// unusable until a later Reset succeeds. Call Refactor before Solve.
func (p *Permuted) Reset(a *sparse.CSC[float64], perm []int) error {
	p.valid = false
	if p.f != nil {
		p.f.ok = false
	}
	stamp, err := checkCSC(a, p.stamp)
	p.stamp = stamp
	if err != nil {
		return err
	}
	n := a.Cols
	if len(perm) != n {
		return fmt.Errorf("%w: length %d, order %d", ErrPermutation, len(perm), n)
	}
	inv := grown(p.inv, n)
	p.inv = inv
	for i := range inv {
		inv[i] = -1
	}
	for k, v := range perm {
		if v < 0 || v >= n || inv[v] >= 0 {
			return fmt.Errorf("%w: entry %d at position %d", ErrPermutation, v, k)
		}
		inv[v] = k
	}
	nnz := a.Indptr[n]

	// Entry (i, j) of K lands at (min, max) of (inv[i], inv[j]). Two stable
	// counting sorts (by row, then by column) give sorted columns.
	row, col := grown(p.row, nnz), grown(p.col, nnz)
	p.row, p.col = row, col
	for j := 0; j < n; j++ {
		for q := a.Indptr[j]; q < a.Indptr[j+1]; q++ {
			r, c := inv[a.Indices[q]], inv[j]
			if r > c {
				r, c = c, r
			}
			row[q], col[q] = int32(r), int32(c)
		}
	}
	p.cnt = grown(p.cnt, n+1)
	p.byRow = bucket(n, nnz, row, nil, p.cnt, p.byRow)
	p.src = bucket(n, nnz, col, p.byRow, p.cnt, p.src)
	if p.pa == nil {
		p.pa = new(sparse.CSC[float64])
	}
	pa := p.pa
	pa.Rows, pa.Cols = n, n
	pa.Indptr = zeroed(pa.Indptr, n+1)
	for _, c := range col {
		pa.Indptr[c+1]++
	}
	for j := 0; j < n; j++ {
		pa.Indptr[j+1] += pa.Indptr[j]
	}
	pa.Indices, pa.Data = grown(pa.Indices, nnz), grown(pa.Data, nnz)
	for k, q := range p.src {
		pa.Indices[k] = row[q]
	}
	if err := p.sym.fill(pa, stamp, true); err != nil {
		return err
	}
	if p.f == nil {
		p.f = new(Factor)
	}
	p.f.reinit(&p.sym)
	p.perm = append(p.perm[:0], perm...)
	p.ap = append(p.ap[:0], a.Indptr...)
	p.ai = append(p.ai[:0], a.Indices[:nnz]...)
	p.work = grown(p.work, n)
	p.valid = true
	return nil
}

// bucket returns in out (grown if needed) the entry numbers 0..nnz-1 stably
// ordered by key (values in 0..n-1); with order non-nil the entries are visited
// in that order, so the result is sorted by key and, within equal keys, by the
// previous order. cnt is scratch of length n+1.
func bucket(n, nnz int, key []int32, order, cnt, out []int) []int {
	clear(cnt)
	for _, k := range key {
		cnt[k+1]++
	}
	for i := 0; i < n; i++ {
		cnt[i+1] += cnt[i]
	}
	out = grown(out, nnz)
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
	if !p.valid {
		return ErrNotFactored
	}
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
	p.f.ok = false
	return p.f.numeric(p.pa.Data) // p.pa has the analysed pattern by construction
}

// Solve solves K y = x in place.
func (p *Permuted) Solve(x []float64) error {
	if !p.valid {
		return ErrNotFactored
	}
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
