package ldl

import (
	"testing"

	"github.com/go-sparse/sparse"
)

// TestFactorL: the strict lower factor of [[4,1],[1,3]] is the single entry
// L[1][0] = 1/4, and D = (4, 3 - 1/4).
func TestFactorL(t *testing.T) {
	a := &sparse.CSC[float64]{Rows: 2, Cols: 2, Indptr: []int{0, 1, 3}, Indices: []int32{0, 0, 1}, Data: []float64{4, 1, 3}}
	f, err := Factorize(a)
	if err != nil {
		t.Fatal(err)
	}
	l := f.L()
	if l.Rows != 2 || l.Cols != 2 || len(l.Data) != 1 || l.Indptr[0] != 0 || l.Indptr[1] != 1 || l.Indptr[2] != 1 {
		t.Fatalf("L = %+v", l)
	}
	if l.Indices[0] != 1 || l.Data[0] != 0.25 {
		t.Fatalf("L entry (%d) = %v, want (1) 0.25", l.Indices[0], l.Data[0])
	}
	if d := f.D(); d[0] != 4 || d[1] != 2.75 {
		t.Fatalf("D = %v", d)
	}
}
