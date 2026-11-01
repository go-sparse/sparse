package ldl

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/go-sparse/sparse/order"
)

func bitsEqual(a, b []float64) bool {
	return slices.EqualFunc(a, b, func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) })
}

// The fused solves give, bit for bit, what the separate passes give: Factor.Solve
// against LSolve, DSolve, LTSolve, and Permuted.Solve against an explicit
// gather, those three passes and a scatter.
func TestFusedSolvesAreBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	for trial := 0; trial < 40; trial++ {
		n, m := 2+rng.IntN(40), 2+rng.IntN(40)
		q := randomKKT(rng, n, m, 0.1, trial%2 == 0)
		N := n + m
		perm, err := order.AMD(fullPattern(t, q))
		must(t, err)
		p, err := NewPermuted(q.k, perm)
		must(t, err)
		must(t, p.Refactor(q.k))
		plain, err := Factorize(q.k)
		must(t, err)

		b := make([]float64, N)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		want := slices.Clone(b)
		must(t, plain.LSolve(want))
		must(t, plain.DSolve(want))
		must(t, plain.LTSolve(want))
		got := slices.Clone(b)
		must(t, plain.Solve(got))
		if !bitsEqual(want, got) {
			t.Fatalf("trial %d: Factor.Solve differs from LSolve+DSolve+LTSolve", trial)
		}

		w := make([]float64, N)
		for k, v := range perm {
			w[k] = b[v]
		}
		f := p.Factor()
		must(t, f.LSolve(w))
		must(t, f.DSolve(w))
		must(t, f.LTSolve(w))
		want = make([]float64, N)
		for k, v := range perm {
			want[v] = w[k]
		}
		got = slices.Clone(b)
		must(t, p.Solve(got))
		if !bitsEqual(want, got) {
			t.Fatalf("trial %d: Permuted.Solve differs from the separate passes", trial)
		}
	}
}
