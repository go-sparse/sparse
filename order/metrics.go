package order

import (
	"fmt"

	"github.com/go-sparse/sparse"
)

// Bandwidth returns max |i-j| over the stored entries of A.
func Bandwidth[T sparse.Scalar](a *sparse.CSR[T]) int {
	bw := 0
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			d := a.Indices[p] - i
			if d < 0 {
				d = -d
			}
			bw = max(bw, d)
		}
	}
	return bw
}

// Profile returns the (lower) envelope size: the sum over rows of
// i - (smallest column index in row i), for entries on or below the diagonal.
func Profile[T sparse.Scalar](a *sparse.CSR[T]) int {
	prof := 0
	for i := 0; i < a.Rows; i++ {
		if a.Indptr[i] < a.Indptr[i+1] {
			if first := a.Indices[a.Indptr[i]]; first < i {
				prof += i - first
			}
		}
	}
	return prof
}

// CholeskyNNZ returns the number of nonzeros (diagonal included) of the
// Cholesky factor L of P A Pᵀ for the pattern of A + Aᵀ, computed
// symbolically from the elimination tree (Liu 1990). perm nil is the
// natural order.
func CholeskyNNZ[T sparse.Scalar](a *sparse.CSR[T], perm []int) (int, error) {
	g, err := symGraph(a)
	if err != nil {
		return 0, err
	}
	return choleskyNNZ(g, perm)
}

// QRFillNNZ returns the nonzeros of the Cholesky factor of the pattern of
// AᵀA under the column permutation perm: the size of R in the QR
// factorisation (and a bound on the fill of LU) for a COLAMD-type ordering.
func QRFillNNZ[T sparse.Scalar](a *sparse.CSR[T], perm []int) (int, error) {
	ones := &sparse.CSR[float64]{Rows: a.Rows, Cols: a.Cols, Indptr: a.Indptr, Indices: a.Indices,
		Data: make([]float64, len(a.Indices))}
	for k := range ones.Data {
		ones.Data[k] = 1
	}
	ata, _ := sparse.Mul(ones.T(), ones) // Aᵀ (n x m) times A (m x n) always conforms
	return CholeskyNNZ(ata, perm)
}

func choleskyNNZ(g *graph, perm []int) (int, error) {
	n := g.n
	if perm == nil {
		perm = make([]int, n)
		for i := range perm {
			perm[i] = i
		}
	}
	if len(perm) != n || !isPermutation(perm) {
		return 0, fmt.Errorf("order: not a permutation of 0..%d", n-1)
	}
	pos := Inverse(perm)
	parent := make([]int, n)
	anc := make([]int, n)
	mark := make([]int, n)
	for i := range mark {
		mark[i] = -1
	}
	total := n
	for i := 0; i < n; i++ {
		parent[i], anc[i] = -1, -1
		for _, w := range g.neighbours(perm[i]) {
			k := pos[w]
			if k >= i {
				continue
			}
			// elimination tree with path compression
			for anc[k] != -1 && anc[k] != i {
				next := anc[k]
				anc[k] = i
				k = next
			}
			if anc[k] == -1 {
				anc[k], parent[k] = i, i
			}
		}
	}
	// row counts: the nonzeros of row i of L are the nodes on the etree
	// paths from each lower neighbour of i up to (excluding) i
	for i := 0; i < n; i++ {
		mark[i] = i
		for _, w := range g.neighbours(perm[i]) {
			for k := pos[w]; k < i && mark[k] != i; k = parent[k] {
				mark[k] = i
				total++
			}
		}
	}
	return total, nil
}
