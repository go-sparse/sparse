package ldl

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/order"
)

// Reset on a used Permuted gives exactly what NewPermuted gives, for matrices
// of every size in any order, and keeps the earlier factor's answers correct.
func TestPermutedResetMatchesNew(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	reused := new(Permuted)
	var w order.Workspace
	var perm []int
	for trial := 0; trial < 60; trial++ {
		n, m := 2+rng.IntN(40), 2+rng.IntN(40)
		q := randomKKT(rng, n, m, 0.1, trial%2 == 0)
		var err error
		perm, err = order.AMDWith(&w, perm, fullPattern(t, q))
		must(t, err)
		fresh, err := NewPermuted(q.k, perm)
		must(t, err)
		must(t, reused.Reset(q.k, perm))
		must(t, fresh.Refactor(q.k))
		must(t, reused.Refactor(q.k))
		if fresh.NNZL() != reused.NNZL() || !slices.Equal(fresh.f.lx, reused.f.lx) ||
			!slices.Equal(fresh.f.li, reused.f.li) || !slices.Equal(fresh.f.d, reused.f.d) {
			t.Fatalf("trial %d: Reset and NewPermuted factor differently", trial)
		}
		b := make([]float64, n+m)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		x, y := slices.Clone(b), slices.Clone(b)
		must(t, fresh.Solve(x))
		must(t, reused.Solve(y))
		if !slices.Equal(x, y) {
			t.Fatalf("trial %d: solutions differ", trial)
		}
	}
}

func TestPermutedResetDoesNotAllocate(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))
	q := randomKKT(rng, 40, 40, 0.1, true)
	perm, _ := order.AMD(fullPattern(t, q))
	p, err := NewPermuted(q.k, perm)
	must(t, err)
	if a := testing.AllocsPerRun(20, func() { _ = p.Reset(q.k, perm); _ = p.Refactor(q.k) }); a != 0 {
		t.Fatalf("Reset+Refactor allocates %v", a)
	}
}

// A failed Reset leaves the Permuted unusable, not half-updated.
func TestPermutedResetFailure(t *testing.T) {
	a := upper(2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, -1})
	p, err := NewPermuted(a, []int{1, 0})
	must(t, err)
	must(t, p.Refactor(a))
	x := []float64{1, 2}
	must(t, p.Solve(x))

	if err := p.Reset(a, []int{0, 0}); !errors.Is(err, ErrPermutation) {
		t.Fatalf("got %v", err)
	}
	if err := p.Refactor(a); !errors.Is(err, ErrNotFactored) {
		t.Fatalf("Refactor after failed Reset: %v", err)
	}
	if err := p.Solve(x); !errors.Is(err, ErrNotFactored) {
		t.Fatalf("Solve after failed Reset: %v", err)
	}

	old := maxLNZ
	maxLNZ = 0
	err = p.Reset(a, []int{1, 0})
	maxLNZ = old
	if !errors.Is(err, ErrOverflow) {
		t.Fatalf("got %v", err)
	}
	if err := p.Refactor(a); !errors.Is(err, ErrNotFactored) {
		t.Fatalf("Refactor after overflow: %v", err)
	}

	must(t, p.Reset(a, []int{1, 0}))
	must(t, p.Refactor(a))
	must(t, p.Solve(x))

	var zero Permuted
	if err := zero.Reset(&sparse.CSC[float64]{Rows: 1, Cols: 2}, nil); !errors.Is(err, ErrDimension) {
		t.Fatalf("got %v", err)
	}
}
