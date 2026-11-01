package order

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/go-sparse/sparse"
)

// randomPattern is an m x n pattern with the given density, a full diagonal
// when square, and dense rows (and columns, when square).
func randomPattern(r *rand.Rand, m, n int, dens float64, dense int) *sparse.CSR[float64] {
	c := sparse.NewCOO[float64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if r.Float64() < dens || i == j {
				_ = c.Append(i, j, 1)
			}
		}
	}
	for k := 0; k < dense; k++ {
		i := r.IntN(m)
		for j := 0; j < n; j++ {
			_ = c.Append(i, j, 1)
			if m == n {
				_ = c.Append(j, i, 1)
			}
		}
	}
	return c.ToCSR()
}

// A Workspace reused across matrices of every size gives what fresh storage
// gives, for AMD and COLAMD, with and without dense rows.
func TestWorkspaceReuseMatchesFresh(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	w := new(Workspace)
	var dst []int
	for it := 0; it < 300; it++ {
		n := 1 + r.IntN(120)
		if it%40 == 0 {
			n = 200 + r.IntN(300)
		}
		dens := []float64{0.01, 0.05, 0.2}[r.IntN(3)] * 10 / float64(n+10)
		a := randomPattern(r, n, n, dens, r.IntN(3))
		want, err := AMD(a)
		if err != nil {
			t.Fatal(err)
		}
		dst, err = AMDWith(w, dst, a)
		if err != nil || !slices.Equal(want, dst) || !IsPermutation(dst) {
			t.Fatalf("AMDWith differs from AMD (n=%d, err %v)", n, err)
		}
		m := 1 + r.IntN(120)
		b := randomPattern(r, m, n, 2*dens, r.IntN(2))
		want, err = COLAMD(b)
		if err != nil {
			t.Fatal(err)
		}
		got, err := COLAMDWith(w, nil, b)
		if err != nil || !slices.Equal(want, got) || !IsPermutation(got) {
			t.Fatalf("COLAMDWith differs from COLAMD (%dx%d, err %v)", m, n, err)
		}
	}
}

func TestWorkspaceDoesNotAllocate(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	a := randomPattern(r, 300, 300, 0.01, 1)
	b := randomPattern(r, 200, 300, 0.02, 0)
	w := new(Workspace)
	dst, _ := AMDWith(w, nil, a)
	dstC, _ := COLAMDWith(w, nil, b)
	if n := testing.AllocsPerRun(20, func() { dst, _ = AMDWith(w, dst, a) }); n != 0 {
		t.Errorf("AMDWith allocates %v on reuse", n)
	}
	if n := testing.AllocsPerRun(20, func() { dstC, _ = COLAMDWith(w, dstC, b) }); n != 0 {
		t.Errorf("COLAMDWith allocates %v on reuse", n)
	}
}

func TestBadStructure(t *testing.T) {
	mk := func(rows, cols int, ptr []int, idx []int32) *sparse.CSR[float64] {
		return &sparse.CSR[float64]{Rows: rows, Cols: cols, Indptr: ptr, Indices: idx, Data: make([]float64, len(idx))}
	}
	bad := map[string]struct {
		a    *sparse.CSR[float64]
		want error
	}{
		"short indptr":  {mk(3, 3, []int{0, 1}, []int32{0}), sparse.ErrFormat},
		"nonzero start": {mk(2, 2, []int{1, 1, 1}, []int32{0}), sparse.ErrFormat},
		"past indices":  {mk(2, 2, []int{0, 1, 5}, []int32{0}), sparse.ErrFormat},
		"decreasing":    {mk(3, 3, []int{0, 2, 1, 3}, []int32{0, 1, 2}), sparse.ErrFormat},
		"row past end":  {mk(3, 3, []int{0, 5, 3, 3}, []int32{0, 1, 2}), sparse.ErrFormat},
		"column high":   {mk(2, 2, []int{0, 1, 2}, []int32{0, 7}), sparse.ErrIndex},
		"column low":    {mk(2, 2, []int{0, 1, 2}, []int32{0, -1}), sparse.ErrIndex},
	}
	for name, c := range bad {
		if _, err := AMD(c.a); !errors.Is(err, c.want) {
			t.Errorf("AMD %s: %v, want %v", name, err, c.want)
		}
		if _, err := RCM(c.a); !errors.Is(err, c.want) {
			t.Errorf("RCM %s: %v, want %v", name, err, c.want)
		}
		if _, err := COLAMD(c.a); !errors.Is(err, c.want) {
			t.Errorf("COLAMD %s: %v, want %v", name, err, c.want)
		}
	}
}
