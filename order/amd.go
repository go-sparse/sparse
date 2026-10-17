package order

import (
	"fmt"
	"math"
	"sort"

	"github.com/go-sparse/sparse"
)

// engine is the approximate-minimum-degree machinery shared by AMD and
// COLAMD, on a quotient graph of variables and elements (Amestoy, Davis and
// Duff 1996, section 2-4):
//
//   - a variable i has a list adjV[i] of adjacent variables and a list
//     adjE[i] of adjacent elements;
//   - an element e (a clique already eliminated, or for COLAMD an original
//     row of A) has the list elemVars[e] of its variables;
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
type engine struct {
	n, nlive int
	nel      int
	adjV     [][]int
	adjE     [][]int
	elemVars [][]int
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
}

func newEngine(n, nelem int) *engine {
	e := &engine{
		n: n, adjV: make([][]int, n), adjE: make([][]int, n),
		elemVars: make([][]int, n+nelem), elen: make([]int, n+nelem), alive: make([]bool, n+nelem),
		nv: make([]int, n), deg: make([]int, n),
		head: make([]int, n+1), next: make([]int, n), prev: make([]int, n),
		mark: make([]int, n+nelem), wmark: make([]int, n+nelem), w: make([]int, n+nelem),
		chainNext: make([]int, n), chainTail: make([]int, n),
	}
	for i := range e.head {
		e.head[i] = -1
	}
	for i := 0; i < n; i++ {
		e.chainNext[i], e.chainTail[i] = -1, i
	}
	return e
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
	var lp []int
	var hv []hashed
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
		for _, v := range e.adjV[p] {
			if e.nv[v] > 0 && e.mark[v] != e.stamp {
				e.mark[v] = e.stamp
				lp = append(lp, v)
			}
		}
		for _, el := range e.adjE[p] {
			for _, v := range e.elemVars[el] {
				if e.nv[v] > 0 && e.mark[v] != e.stamp {
					e.mark[v] = e.stamp
					lp = append(lp, v)
				}
			}
			e.alive[el], e.elemVars[el] = false, nil
		}
		e.adjV[p], e.adjE[p] = nil, nil
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
			for _, el := range e.adjE[i] {
				if !e.alive[el] {
					continue
				}
				if e.wmark[el] != step {
					e.wmark[el], e.w[el] = step, e.elen[el]
				}
				e.w[el] -= e.nv[i]
			}
		}
		// pass 2: clean lists, approximate degree, hash
		hv = hv[:0]
		kept := lp[:0]
		for _, i := range lp {
			ne := e.adjE[i][:0]
			sumE := 0
			hash := 0
			for _, el := range e.adjE[i] {
				if e.alive[el] && e.w[el] > 0 {
					ne = append(ne, el)
					sumE += e.w[el]
					hash += el
				} else if e.alive[el] { // w == 0: element contained in Lp
					e.alive[el], e.elemVars[el] = false, nil
				}
			}
			nvv := e.adjV[i][:0]
			sumV := 0
			for _, v := range e.adjV[i] {
				if e.nv[v] > 0 && e.mark[v] != e.stamp {
					nvv = append(nvv, v)
					sumV += e.nv[v]
					hash += v
				}
			}
			e.adjV[i] = nvv
			if len(ne) == 0 && len(nvv) == 0 {
				// mass elimination: i is indistinguishable from p
				e.nel += e.nv[i]
				degme -= e.nv[i]
				e.absorb(p, i)
				e.nv[i] = 0
				continue
			}
			e.adjE[i] = append(ne, p)
			hash += p
			d := min(e.nlive-e.nel-e.nv[i], e.deg[i]+degme-e.nv[i], sumV+degme-e.nv[i]+sumE)
			e.deg[i] = max(d, 0)
			kept = append(kept, i)
			hv = append(hv, hashed{hash % (n + 1), i})
		}
		lp = kept
		// supervariable detection among equal hashes
		sort.Slice(hv, func(a, b int) bool { return hv[a].h < hv[b].h })
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
		e.elemVars[p] = append([]int(nil), live...)
		e.elen[p] = size
		e.alive[p] = true
		for _, i := range live {
			e.insert(i)
		}
	}
}

// sameAdjacency reports whether variables i and j have the same element and
// variable neighbours (both lists are duplicate-free).
func (e *engine) sameAdjacency(i, j int) bool {
	if len(e.adjE[i]) != len(e.adjE[j]) || len(e.adjV[i]) != len(e.adjV[j]) {
		return false
	}
	e.stamp++
	for _, el := range e.adjE[i] {
		e.mark[el] = e.stamp
	}
	for _, el := range e.adjE[j] {
		if e.mark[el] != e.stamp {
			return false
		}
	}
	e.stamp++
	for _, v := range e.adjV[i] {
		e.mark[v] = e.stamp
	}
	for _, v := range e.adjV[j] {
		if e.mark[v] != e.stamp {
			return false
		}
	}
	return true
}

// permutation expands the pivot order into a full permutation of 0..n-1,
// each principal variable followed by the variables merged into it, and
// appends the variables that were set aside as dense.
func (e *engine) permutation(dense []int) []int {
	p := make([]int, 0, e.n)
	for _, v := range e.order {
		for ; v >= 0; v = e.chainNext[v] {
			p = append(p, v)
		}
	}
	return append(p, dense...)
}

// AMD returns an approximate minimum degree ordering of the pattern of
// A + Aᵀ for a square matrix. Rows denser than max(16, 10 sqrt(n)) are
// ordered last, as in SuiteSparse AMD.
func AMD[T sparse.Scalar](a *sparse.CSR[T]) ([]int, error) {
	g, err := symGraph(a)
	if err != nil {
		return nil, err
	}
	n := g.n
	e := newEngine(n, 0)
	limit := int(math.Max(16, 10*math.Sqrt(float64(n))))
	var dense []int
	for i := 0; i < n; i++ {
		if g.degree(i) > limit {
			dense = append(dense, i)
		} else {
			e.nv[i] = 1
		}
	}
	e.nlive = n - len(dense)
	for i := 0; i < n; i++ {
		if e.nv[i] == 0 {
			continue
		}
		for _, v := range g.neighbours(i) {
			if e.nv[v] > 0 {
				e.adjV[i] = append(e.adjV[i], v)
			}
		}
		e.deg[i] = len(e.adjV[i])
	}
	for i := 0; i < n; i++ {
		if e.nv[i] > 0 {
			e.insert(i)
		}
	}
	e.run()
	sort.SliceStable(dense, func(x, y int) bool { return g.degree(dense[x]) < g.degree(dense[y]) })
	return e.permutation(dense), nil
}

// COLAMD returns a column ordering of the m x n matrix A for sparse LU or QR
// (an approximate minimum degree ordering of AᵀA computed from A alone).
// Rows with more than max(16, 10 sqrt(n)) entries are ignored and columns
// with more than max(16, 10 sqrt(m)) entries are ordered last, the default
// thresholds of SuiteSparse COLAMD.
func COLAMD[T sparse.Scalar](a *sparse.CSR[T]) ([]int, error) {
	m, n := a.Rows, a.Cols
	if m == 0 || n == 0 {
		return nil, fmt.Errorf("%w: empty %dx%d matrix", ErrNotSquare, m, n)
	}
	e := newEngine(n, m)
	rowLimit := int(math.Max(16, 10*math.Sqrt(float64(n))))
	colLimit := int(math.Max(16, 10*math.Sqrt(float64(m))))
	colCount := make([]int, n)
	for _, j := range a.Indices {
		colCount[j]++
	}
	var dense []int
	for j := 0; j < n; j++ {
		if colCount[j] > colLimit {
			dense = append(dense, j)
		} else {
			e.nv[j] = 1
		}
	}
	e.nlive = n - len(dense)
	for i := 0; i < m; i++ {
		id := n + i
		var vars []int
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			if j := a.Indices[p]; e.nv[j] > 0 {
				vars = append(vars, j)
			}
		}
		if len(vars) == 0 || a.Indptr[i+1]-a.Indptr[i] > rowLimit {
			continue
		}
		e.elemVars[id], e.elen[id], e.alive[id] = vars, len(vars), true
		for _, j := range vars {
			e.adjE[j] = append(e.adjE[j], id)
			e.deg[j] += len(vars) - 1
		}
	}
	for j := 0; j < n; j++ {
		if e.nv[j] > 0 {
			e.deg[j] = min(e.deg[j], e.nlive-1)
			e.insert(j)
		}
	}
	e.run()
	sort.SliceStable(dense, func(x, y int) bool { return colCount[dense[x]] < colCount[dense[y]] })
	return e.permutation(dense), nil
}
