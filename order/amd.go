package order

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/go-sparse/sparse"
)

// Workspace is the reusable scratch memory of [AMDWith] and [COLAMDWith]: the
// graph of A + Aᵀ, the quotient graph and the degree lists. After one call has
// sized it, further calls on matrices of no larger order and similar fill
// allocate nothing. The zero value is ready to use. A Workspace is not safe
// for concurrent use.
type Workspace struct {
	g                 graph
	gptr, gfill, gadj []int
	dense, colCount   []int
	e                 engine
}

// hvSorter sorts the (hash, variable) pairs of supervariable detection. It is a
// sort.Interface on a pointer so that sorting allocates nothing (sort.Slice
// allocates a closure and a swapper); it runs the same pdqsort as sort.Slice,
// so the order of equal hashes is the one AMD always produced.
type hvSorter struct{ s []hashed }

func (h *hvSorter) Len() int           { return len(h.s) }
func (h *hvSorter) Less(a, b int) bool { return h.s[a].h < h.s[b].h }
func (h *hvSorter) Swap(a, b int)      { h.s[a], h.s[b] = h.s[b], h.s[a] }

// engine is the approximate-minimum-degree machinery shared by AMD and
// COLAMD, on a quotient graph of variables and elements (Amestoy, Davis and
// Duff 1996, section 2-4):
//
//   - a variable i has a list of adjacent variables and a list of adjacent
//     elements;
//   - an element e (a clique already eliminated, or for COLAMD an original
//     row of A) has the list of its variables;
//   - when the variable p of minimum approximate degree is eliminated, its
//     new element is Lp = adjV[p] ∪ ⋃ elemVars[e], e ∈ adjE[p]; the elements
//     of adjE[p] are absorbed;
//   - the external degree of each i in Lp is bounded by
//     min(n-k-|i|, d_i + |Lp\i|, |A_i\i| + |Lp\i| + Σ_e |Le\Lp|), with
//     |Le\Lp| obtained by one pass over Lp (the w(e) trick);
//   - variables with identical adjacency are merged into supervariables,
//     variables whose only neighbour is p are eliminated with it (mass
//     elimination), and elements contained in Lp are absorbed (aggressive
//     absorption).
//
// AMD starts with no elements and variable-variable edges; COLAMD starts with
// the rows of A as elements and no edges, which is the same quotient graph as
// that of AᵀA without forming it.
//
// All the lists live in one arena, pool: the variable list of i is
// pool[vst[i] : vst[i]+vln[i]] (it only shrinks, in place), the element list of
// i is pool[est[i] : est[i]+eln[i]] with room for ecp[i] entries (it shrinks in
// place and grows by one when the new element is appended; when it is full it
// moves to the end of the arena), and the variables of element e are
// pool[xst[e] : xst[e]+xln[e]]. Nothing is allocated per node.
type engine struct {
	n, nlive int
	nel      int
	pool     []int
	vst, vln []int
	est, eln []int
	ecp      []int
	xst, xln []int
	elen     []int
	alive    []bool
	nv       []int
	deg      []int

	head, next, prev []int
	mindeg           int

	mark, wmark, w []int
	stamp          int

	chainNext, chainTail []int
	order                []int

	lp []int
	hv hvSorter
}

// reset sizes and clears the engine for n variables and nelem initial
// elements.
func (e *engine) reset(n, nelem int) {
	e.n, e.nlive, e.nel, e.mindeg, e.stamp = n, 0, 0, 0, 0
	e.pool = e.pool[:0]
	e.vst, e.vln = zeroed(e.vst, n), zeroed(e.vln, n)
	e.est, e.eln, e.ecp = zeroed(e.est, n), zeroed(e.eln, n), zeroed(e.ecp, n)
	e.xst, e.xln = zeroed(e.xst, n+nelem), zeroed(e.xln, n+nelem)
	e.elen, e.alive = zeroed(e.elen, n+nelem), zeroed(e.alive, n+nelem)
	e.nv, e.deg = zeroed(e.nv, n), zeroed(e.deg, n)
	e.head, e.next, e.prev = grown(e.head, n+1), zeroed(e.next, n), zeroed(e.prev, n)
	e.mark, e.wmark, e.w = zeroed(e.mark, n+nelem), zeroed(e.wmark, n+nelem), zeroed(e.w, n+nelem)
	e.chainNext, e.chainTail = grown(e.chainNext, n), grown(e.chainTail, n)
	for i := range e.head {
		e.head[i] = -1
	}
	for i := 0; i < n; i++ {
		e.chainNext[i], e.chainTail[i] = -1, i
	}
	e.order = e.order[:0]
	e.lp = e.lp[:0]
	e.hv.s = e.hv.s[:0]
}

func (e *engine) vlist(i int) []int { return e.pool[e.vst[i] : e.vst[i]+e.vln[i]] }
func (e *engine) elist(i int) []int { return e.pool[e.est[i] : e.est[i]+e.eln[i]] }
func (e *engine) xlist(el int) []int {
	return e.pool[e.xst[el] : e.xst[el]+e.xln[el]]
}

// pushE appends element v to the element list of i.
func (e *engine) pushE(i, v int) {
	if e.eln[i] < e.ecp[i] {
		e.pool[e.est[i]+e.eln[i]] = v
		e.eln[i]++
		return
	}
	n := e.eln[i]
	c := 2 * (n + 1)
	st := len(e.pool)
	e.pool = slices.Grow(e.pool, c)
	e.pool = e.pool[:st+c]
	copy(e.pool[st:], e.pool[e.est[i]:e.est[i]+n])
	e.pool[st+n] = v
	e.est[i], e.eln[i], e.ecp[i] = st, n+1, c
}

func (e *engine) insert(v int) {
	d := e.deg[v]
	e.prev[v], e.next[v] = -1, e.head[d]
	if e.head[d] >= 0 {
		e.prev[e.head[d]] = v
	}
	e.head[d] = v
	e.mindeg = min(e.mindeg, d)
}

func (e *engine) remove(v int) {
	if e.prev[v] >= 0 {
		e.next[e.prev[v]] = e.next[v]
	} else {
		e.head[e.deg[v]] = e.next[v]
	}
	if e.next[v] >= 0 {
		e.prev[e.next[v]] = e.prev[v]
	}
}

// absorb appends the supervariable chain of src to that of dst.
func (e *engine) absorb(dst, src int) {
	e.chainNext[e.chainTail[dst]] = src
	e.chainTail[dst] = e.chainTail[src]
}

type hashed struct{ h, v int }

// run eliminates all live variables and fills e.order with the principal
// variables in pivot order.
func (e *engine) run() {
	n := e.n
	lp := e.lp[:0]
	step := 0
	for e.nel < e.nlive {
		for e.head[e.mindeg] < 0 {
			e.mindeg++
		}
		p := e.head[e.mindeg]
		e.remove(p)
		step++
		e.stamp++
		nvp := e.nv[p]
		e.mark[p] = e.stamp
		lp = lp[:0]
		for _, v := range e.vlist(p) {
			if e.nv[v] > 0 && e.mark[v] != e.stamp {
				e.mark[v] = e.stamp
				lp = append(lp, v)
			}
		}
		for _, el := range e.elist(p) {
			for _, v := range e.xlist(el) {
				if e.nv[v] > 0 && e.mark[v] != e.stamp {
					e.mark[v] = e.stamp
					lp = append(lp, v)
				}
			}
			e.alive[el], e.xln[el] = false, 0
		}
		e.vln[p], e.eln[p] = 0, 0
		e.order = append(e.order, p)
		e.nel += nvp
		e.nv[p] = 0 // p is now an element, no longer a variable

		degme := 0
		for _, v := range lp {
			e.remove(v)
			degme += e.nv[v]
		}
		// pass 1: |Le \ Lp| for every element adjacent to Lp
		for _, i := range lp {
			for _, el := range e.elist(i) {
				if !e.alive[el] {
					continue
				}
				if e.wmark[el] != step {
					e.wmark[el], e.w[el] = step, e.elen[el]
				}
				e.w[el] -= e.nv[i]
			}
		}
		// pass 2: clean lists in place, approximate degree, hash
		e.hv.s = e.hv.s[:0]
		kept := lp[:0]
		for _, i := range lp {
			eb, ne := e.est[i], 0
			sumE := 0
			hash := 0
			for k, ke := 0, e.eln[i]; k < ke; k++ {
				el := e.pool[eb+k]
				if e.alive[el] && e.w[el] > 0 {
					e.pool[eb+ne] = el
					ne++
					sumE += e.w[el]
					hash += el
				} else if e.alive[el] { // w == 0: element contained in Lp
					e.alive[el], e.xln[el] = false, 0
				}
			}
			e.eln[i] = ne
			vb, nvv := e.vst[i], 0
			sumV := 0
			for k, kv := 0, e.vln[i]; k < kv; k++ {
				v := e.pool[vb+k]
				if e.nv[v] > 0 && e.mark[v] != e.stamp {
					e.pool[vb+nvv] = v
					nvv++
					sumV += e.nv[v]
					hash += v
				}
			}
			e.vln[i] = nvv
			if ne == 0 && nvv == 0 {
				// mass elimination: i is indistinguishable from p
				e.nel += e.nv[i]
				degme -= e.nv[i]
				e.absorb(p, i)
				e.nv[i] = 0
				continue
			}
			e.pushE(i, p)
			hash += p
			d := min(e.nlive-e.nel-e.nv[i], e.deg[i]+degme-e.nv[i], sumV+degme-e.nv[i]+sumE)
			e.deg[i] = max(d, 0)
			kept = append(kept, i)
			e.hv.s = append(e.hv.s, hashed{hash % (n + 1), i})
		}
		lp = kept
		// supervariable detection among equal hashes
		hv := e.hv.s
		sort.Sort(&e.hv)
		for a := 0; a < len(hv); a++ {
			i := hv[a].v
			if e.nv[i] == 0 {
				continue
			}
			for b := a + 1; b < len(hv) && hv[b].h == hv[a].h; b++ {
				j := hv[b].v
				if e.nv[j] > 0 && e.sameAdjacency(i, j) {
					e.deg[i] = max(e.deg[i]-e.nv[j], 0)
					e.nv[i] += e.nv[j]
					e.nv[j] = 0
					e.absorb(i, j)
				}
			}
		}
		// the new element and the degree lists
		live := lp[:0]
		size := 0
		for _, i := range lp {
			if e.nv[i] > 0 {
				live = append(live, i)
				size += e.nv[i]
			}
		}
		e.xst[p], e.xln[p] = len(e.pool), len(live)
		e.pool = append(e.pool, live...)
		e.elen[p] = size
		e.alive[p] = true
		for _, i := range live {
			e.insert(i)
		}
	}
	e.lp = lp
}

// sameAdjacency reports whether variables i and j have the same element and
// variable neighbours (both lists are duplicate-free).
func (e *engine) sameAdjacency(i, j int) bool {
	if e.eln[i] != e.eln[j] || e.vln[i] != e.vln[j] {
		return false
	}
	e.stamp++
	for _, el := range e.elist(i) {
		e.mark[el] = e.stamp
	}
	for _, el := range e.elist(j) {
		if e.mark[el] != e.stamp {
			return false
		}
	}
	e.stamp++
	for _, v := range e.vlist(i) {
		e.mark[v] = e.stamp
	}
	for _, v := range e.vlist(j) {
		if e.mark[v] != e.stamp {
			return false
		}
	}
	return true
}

// permutation expands the pivot order into a full permutation of 0..n-1,
// each principal variable followed by the variables merged into it, and
// appends the variables that were set aside as dense. It is written to
// dst[:0] (grown if needed).
func (e *engine) permutation(dst, dense []int) []int {
	if cap(dst) < e.n {
		dst = make([]int, 0, e.n)
	}
	p := dst[:0]
	for _, v := range e.order {
		for ; v >= 0; v = e.chainNext[v] {
			p = append(p, v)
		}
	}
	return append(p, dense...)
}

// AMD returns an approximate minimum degree ordering of the pattern of
// A + Aᵀ for a square matrix. Rows denser than max(16, 10 sqrt(n)) are
// ordered last, as in SuiteSparse AMD. It allocates its working storage on
// every call; see [AMDWith] to reuse it.
func AMD[T sparse.Scalar](a *sparse.CSR[T]) ([]int, error) {
	return AMDWith(new(Workspace), nil, a)
}

// AMDWith is [AMD] with the working storage in w and the permutation written
// to dst[:0] (a new slice if its capacity is below n). When w has been used for
// a matrix of no smaller order and dst has capacity n, it allocates nothing.
// The result is the same as AMD's.
func AMDWith[T sparse.Scalar](w *Workspace, dst []int, a *sparse.CSR[T]) ([]int, error) {
	g, err := symGraphOf(w, a)
	if err != nil {
		return nil, err
	}
	n := g.n
	e := &w.e
	e.reset(n, 0)
	limit := int(math.Max(16, 10*math.Sqrt(float64(n))))
	dense := w.dense[:0]
	for i := 0; i < n; i++ {
		if g.degree(i) > limit {
			dense = append(dense, i)
		} else {
			e.nv[i] = 1
		}
	}
	w.dense = dense
	e.nlive = n - len(dense)
	for i := 0; i < n; i++ {
		if e.nv[i] == 0 {
			continue
		}
		e.vst[i] = len(e.pool)
		for _, v := range g.neighbours(i) {
			if e.nv[v] > 0 {
				e.pool = append(e.pool, v)
			}
		}
		e.vln[i] = len(e.pool) - e.vst[i]
		e.deg[i] = e.vln[i]
	}
	for i := 0; i < n; i++ {
		if e.nv[i] > 0 {
			e.insert(i)
		}
	}
	e.run()
	if len(dense) > 1 {
		slices.SortStableFunc(dense, func(x, y int) int { return cmp.Compare(g.degree(x), g.degree(y)) })
	}
	return e.permutation(dst, dense), nil
}

// COLAMD returns a column ordering of the m x n matrix A for sparse LU or QR
// (an approximate minimum degree ordering of AᵀA computed from A alone).
// Rows with more than max(16, 10 sqrt(n)) entries are ignored and columns
// with more than max(16, 10 sqrt(m)) entries are ordered last, the default
// thresholds of SuiteSparse COLAMD. It allocates its working storage on every
// call; see [COLAMDWith] to reuse it.
func COLAMD[T sparse.Scalar](a *sparse.CSR[T]) ([]int, error) {
	return COLAMDWith(new(Workspace), nil, a)
}

// COLAMDWith is [COLAMD] with the working storage in w and the permutation
// written to dst[:0], as [AMDWith].
func COLAMDWith[T sparse.Scalar](w *Workspace, dst []int, a *sparse.CSR[T]) ([]int, error) {
	m, n := a.Rows, a.Cols
	if m == 0 || n == 0 {
		return nil, fmt.Errorf("%w: empty %dx%d matrix", ErrNotSquare, m, n)
	}
	if len(a.Indptr) != m+1 || a.Indptr[0] != 0 || a.Indptr[m] < 0 || a.Indptr[m] > len(a.Indices) {
		return nil, fmt.Errorf("%w: indptr of length %d for %d rows", sparse.ErrFormat, len(a.Indptr), m)
	}
	for i := 0; i < m; i++ {
		lo, hi := a.Indptr[i], a.Indptr[i+1]
		if lo > hi || hi > a.Indptr[m] {
			return nil, fmt.Errorf("%w: indptr decreases at row %d", sparse.ErrFormat, i)
		}
		for _, j := range a.Indices[lo:hi] {
			if j < 0 || int(j) >= n {
				return nil, fmt.Errorf("%w: column %d in row %d", sparse.ErrIndex, j, i)
			}
		}
	}
	e := &w.e
	e.reset(n, m)
	rowLimit := int(math.Max(16, 10*math.Sqrt(float64(n))))
	colLimit := int(math.Max(16, 10*math.Sqrt(float64(m))))
	colCount := zeroed(w.colCount, n)
	w.colCount = colCount
	for _, j := range a.Indices[:a.Indptr[m]] {
		colCount[j]++
	}
	dense := w.dense[:0]
	for j := 0; j < n; j++ {
		if colCount[j] > colLimit {
			dense = append(dense, j)
		} else {
			e.nv[j] = 1
		}
	}
	w.dense = dense
	e.nlive = n - len(dense)
	// rows kept as elements: count, per column, the elements it will be in
	// (eln is used as the counter, then reset), so each element list is laid
	// out with its exact size.
	keep := func(i int) (cnt int, ok bool) {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			if e.nv[a.Indices[p]] > 0 {
				cnt++
			}
		}
		return cnt, cnt > 0 && a.Indptr[i+1]-a.Indptr[i] <= rowLimit
	}
	for i := 0; i < m; i++ {
		if _, ok := keep(i); ok {
			for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
				if j := a.Indices[p]; e.nv[j] > 0 {
					e.ecp[j]++
				}
			}
		}
	}
	for j := 0; j < n; j++ {
		e.est[j] = len(e.pool)
		e.pool = slices.Grow(e.pool, e.ecp[j])[:len(e.pool)+e.ecp[j]]
	}
	for i := 0; i < m; i++ {
		cnt, ok := keep(i)
		if !ok {
			continue
		}
		id := n + i
		e.xst[id], e.xln[id] = len(e.pool), cnt
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			if j := int(a.Indices[p]); e.nv[j] > 0 {
				e.pool = append(e.pool, j)
				e.pool[e.est[j]+e.eln[j]] = id
				e.eln[j]++
				e.deg[j] += cnt - 1
			}
		}
		e.elen[id], e.alive[id] = cnt, true
	}
	for j := 0; j < n; j++ {
		if e.nv[j] > 0 {
			e.deg[j] = min(e.deg[j], e.nlive-1)
			e.insert(j)
		}
	}
	e.run()
	if len(dense) > 1 {
		slices.SortStableFunc(dense, func(x, y int) int { return cmp.Compare(colCount[x], colCount[y]) })
	}
	return e.permutation(dst, dense), nil
}
