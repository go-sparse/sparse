// Package order computes sparse-matrix orderings and measures their quality.
//
//   - RCM: reverse Cuthill-McKee with a George-Liu pseudo-peripheral start
//     (bandwidth and profile reduction; MATLAB symrcm, scipy
//     csgraph.reverse_cuthill_mckee).
//   - AMD: approximate minimum degree on the pattern of A+Aᵀ, written from
//     Amestoy, Davis and Duff, SIAM J. Matrix Anal. Appl. 17(4), 1996,
//     doi:10.1137/S0895479894278952 (MATLAB amd, symamd).
//   - COLAMD: approximate minimum degree on the column intersection graph of
//     A, never forming AᵀA, written from Davis, Gilbert, Larimore and Ng,
//     ACM TOMS 30(3), 2004, doi:10.1145/1024074.1024079 (MATLAB colamd).
//
// Provenance: the implementations are written from the papers. SuiteSparse
// AMD and COLAMD are BSD-3-Clause and are credited as the reference
// algorithms; CHOLMOD, UMFPACK and CSparse (GPL/LGPL) were not consulted.
// Tie-breaking differs from SuiteSparse, so permutations are not expected to
// be identical; quality is compared through Bandwidth, Profile and
// CholeskyNNZ.
//
// A permutation p lists, for each new position i, the original index p[i],
// so the reordered matrix is B[i][j] = A[p[i]][p[j]] (CSR.Permute(p, p)).
package order

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"github.com/go-sparse/sparse"
)

// ErrNotSquare is returned for orderings that need a square matrix.
var ErrNotSquare = errors.New("order: matrix is not square")

// graph is a simple undirected graph in adjacency-list (CSR) form without
// self loops.
type graph struct {
	n   int
	ptr []int
	adj []int
}

func (g *graph) neighbours(i int) []int { return g.adj[g.ptr[i]:g.ptr[i+1]] }
func (g *graph) degree(i int) int       { return g.ptr[i+1] - g.ptr[i] }

// symGraph builds the graph of the pattern of A + Aᵀ in fresh storage.
func symGraph[T sparse.Scalar](a *sparse.CSR[T]) (*graph, error) {
	return symGraphOf(new(Workspace), a)
}

// symGraphOf builds the graph of the pattern of A + Aᵀ (no self loops, sorted
// duplicate-free neighbour lists) in the storage of w, by counting: every
// off-diagonal entry (i, j) contributes j to the list of i and i to the list of
// j, then each list is sorted and compacted. The graph is valid until the next
// call on w.
func symGraphOf[T sparse.Scalar](w *Workspace, a *sparse.CSR[T]) (*graph, error) {
	if a.Rows != a.Cols {
		return nil, fmt.Errorf("%w: %dx%d", ErrNotSquare, a.Rows, a.Cols)
	}
	n := a.Rows
	if len(a.Indptr) != n+1 || a.Indptr[0] != 0 || a.Indptr[n] < 0 || a.Indptr[n] > len(a.Indices) {
		return nil, fmt.Errorf("%w: indptr of length %d for %d rows", sparse.ErrFormat, len(a.Indptr), n)
	}
	ptr := zeroed(w.gptr, n+1)
	fill := zeroed(w.gfill, n+1)
	w.gptr, w.gfill = ptr, fill
	for i := 0; i < n; i++ {
		lo, hi := a.Indptr[i], a.Indptr[i+1]
		if lo > hi || hi > a.Indptr[n] {
			return nil, fmt.Errorf("%w: indptr decreases at row %d", sparse.ErrFormat, i)
		}
		for _, j := range a.Indices[lo:hi] {
			if j < 0 || int(j) >= n {
				return nil, fmt.Errorf("%w: column %d in row %d", sparse.ErrIndex, j, i)
			}
			if int(j) != i {
				ptr[i+1]++
				ptr[j+1]++
			}
		}
	}
	for i := 0; i < n; i++ {
		ptr[i+1] += ptr[i]
	}
	copy(fill, ptr)
	adj := grown(w.gadj, ptr[n])
	w.gadj = adj
	for i := 0; i < n; i++ {
		for _, j := range a.Indices[a.Indptr[i]:a.Indptr[i+1]] {
			if int(j) != i {
				adj[fill[i]] = int(j)
				fill[i]++
				adj[fill[j]] = i
				fill[j]++
			}
		}
	}
	// fill[i] is now the end of row i; compact each sorted row in place.
	out, lo := 0, 0
	for i := 0; i < n; i++ {
		hi := fill[i]
		row := adj[lo:hi]
		slices.Sort(row)
		ptr[i] = out
		for k, v := range row {
			if k == 0 || v != row[k-1] {
				adj[out] = v
				out++
			}
		}
		lo = hi
	}
	ptr[n] = out
	w.g = graph{n: n, ptr: ptr, adj: adj[:out]}
	return &w.g, nil
}

// grown returns s with length n, reusing its storage when it is large enough
// (the contents are unspecified).
func grown[E any](s []E, n int) []E {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]E, n)
}

// zeroed is grown with every element cleared.
func zeroed[E any](s []E, n int) []E {
	s = grown(s, n)
	clear(s)
	return s
}

// isPermutation reports whether p is a permutation of 0..len(p)-1.
func isPermutation(p []int) bool {
	seen := make([]bool, len(p))
	for _, v := range p {
		if v < 0 || v >= len(p) || seen[v] {
			return false
		}
		seen[v] = true
	}
	return true
}

// Inverse returns the inverse permutation.
func Inverse(p []int) []int {
	q := make([]int, len(p))
	for i, v := range p {
		q[v] = i
	}
	return q
}

// IsPermutation reports whether p is a permutation of 0..len(p)-1.
func IsPermutation(p []int) bool { return isPermutation(p) }

// RCM returns the reverse Cuthill-McKee ordering of the pattern of A + Aᵀ.
// Each connected component is started from a pseudo-peripheral vertex.
func RCM[T sparse.Scalar](a *sparse.CSR[T]) ([]int, error) {
	g, err := symGraph(a)
	if err != nil {
		return nil, err
	}
	n := g.n
	visited := make([]bool, n)
	b := &bfs{level: make([]int, n), seen: make([]int, n)}
	order := make([]int, 0, n)
	// vertices by increasing degree, to pick component starts
	byDeg := make([]int, n)
	for i := range byDeg {
		byDeg[i] = i
	}
	sort.SliceStable(byDeg, func(x, y int) bool { return g.degree(byDeg[x]) < g.degree(byDeg[y]) })
	for _, s := range byDeg {
		if visited[s] {
			continue
		}
		start := pseudoPeripheral(g, s, b)
		first := len(order)
		order = append(order, start)
		visited[start] = true
		for h := first; h < len(order); h++ {
			v := order[h]
			from := len(order)
			for _, w := range g.neighbours(v) {
				if !visited[w] {
					visited[w] = true
					order = append(order, w)
				}
			}
			nb := order[from:]
			sort.SliceStable(nb, func(x, y int) bool { return g.degree(nb[x]) < g.degree(nb[y]) })
		}
	}
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	return order, nil
}

// bfs holds the scratch state of repeated breadth-first searches.
type bfs struct {
	level, seen []int
	stamp       int
	queue       []int
}

// run searches from s and returns the eccentricity of s; b.queue holds the
// vertices of the component in BFS order and b.level their distances.
func (b *bfs) run(g *graph, s int) int {
	b.stamp++
	b.queue = append(b.queue[:0], s)
	b.level[s], b.seen[s] = 0, b.stamp
	ecc := 0
	for h := 0; h < len(b.queue); h++ {
		v := b.queue[h]
		for _, w := range g.neighbours(v) {
			if b.seen[w] != b.stamp {
				b.seen[w] = b.stamp
				b.level[w] = b.level[v] + 1
				ecc = b.level[w]
				b.queue = append(b.queue, w)
			}
		}
	}
	return ecc
}

// pseudoPeripheral is the George-Liu algorithm: repeat BFS from the
// minimum-degree vertex of the last level until the eccentricity stops growing.
func pseudoPeripheral(g *graph, s int, b *bfs) int {
	ecc := b.run(g, s)
	for {
		best := -1
		for _, v := range b.queue {
			if b.level[v] == ecc && (best < 0 || g.degree(v) < g.degree(best)) {
				best = v
			}
		}
		if ecc2 := b.run(g, best); ecc2 <= ecc {
			return s
		} else {
			s, ecc = best, ecc2
		}
	}
}
