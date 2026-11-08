//go:build !race

// The race detector's instrumentation allocates: the counts here are exact
// only without it (CI runs them in the plain test run of each lane).

package sparse

import "testing"

// triDiag is tridiag(-1, 2, -1) of order n, built by Diags.
func triDiag(t testing.TB, n int) *CSR[float64] {
	t.Helper()
	lo, di := make([]float64, n-1), make([]float64, n)
	for i := range lo {
		lo[i] = -1
	}
	for i := range di {
		di[i] = 2
	}
	a, err := Diags([][]float64{lo, di, lo}, []int{-1, 0, 1}, n, n)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// TestAllocsDoNotGrowWithSize asserts that the allocations of the conversions
// and constructors are the same for a matrix of order 101 and of order 1001
// (slack: allowed extra allocations, for an output whose size is not known
// before it is computed): no allocation per row, per entry or per doubling of
// an output. Counts are exact under any machine load. Before: Permute
// allocated 3 per row, and Tril, Triu, FromDense and Diags grew their outputs
// by doubling.
func TestAllocsDoNotGrowWithSize(t *testing.T) {
	perm := func(n int) []int {
		p := make([]int, n)
		for i := range p {
			p[i] = (i*7 + 3) % n // n is coprime to 7 below
		}
		return p
	}
	cases := []struct {
		name       string
		max, slack float64
		run        func(a *CSR[float64], p []int)
	}{
		// A slack of 2 absorbs the runtime's own allocations that the counter
		// sees on a large run (an allocation per row would be hundreds).
		{"Permute", 6, 2, func(a *CSR[float64], p []int) { _, _ = a.Permute(p, p) }},
		{"Tril", 4, 2, func(a *CSR[float64], _ []int) { _ = a.Tril(0) }},
		{"Triu", 4, 2, func(a *CSR[float64], _ []int) { _ = a.Triu(0) }},
		// the size of a product is only known by computing it: the capacity
		// hint is the larger operand, growth beyond it costs a step or two.
		{"Mul", 12, 4, func(a *CSR[float64], _ []int) { _, _ = Mul(a, a) }},
		{"Diags", 12, 2, func(a *CSR[float64], _ []int) { _ = triDiag(t, a.Rows) }},
	}
	for _, c := range cases {
		a1, a2 := triDiag(t, 101), triDiag(t, 1003)
		p1, p2 := perm(101), perm(1003)
		small := testing.AllocsPerRun(5, func() { c.run(a1, p1) })
		big := testing.AllocsPerRun(5, func() { c.run(a2, p2) })
		t.Logf("%s: %v allocations (order 101), %v (order 1001)", c.name, small, big)
		if big > small+c.slack || big > c.max+c.slack {
			t.Errorf("%s: %v allocations at order 1001, %v at order 101 (want at most %v more, and at most %v)", c.name, big, small, c.slack, c.max+c.slack)
		}
	}
	// COO conversions: three arrays (before: seven, with an n-sized counting
	// array and a permutation), whatever the size; Prune and NormFro none.
	for _, n := range []int{101, 1003} {
		a := triDiag(t, n)
		coo := a.ToCOO()
		if got := testing.AllocsPerRun(5, func() { _ = coo.ToCSC() }); got > 4 {
			t.Errorf("COO.ToCSC of order %d: %v allocations, want at most 4", n, got)
		}
		if got := testing.AllocsPerRun(5, func() { _ = coo.ToCSR() }); got > 4 {
			t.Errorf("COO.ToCSR of order %d: %v allocations, want at most 4", n, got)
		}
		if got := testing.AllocsPerRun(5, func() { a.Prune(0) }); got != 0 {
			t.Errorf("Prune of order %d: %v allocations, want 0", n, got)
		}
		if got := testing.AllocsPerRun(5, func() { _ = a.NormFro() }); got != 0 {
			t.Errorf("NormFro of order %d: %v allocations, want 0", n, got)
		}
	}
	for _, n := range []int{30, 60} {
		a := triDiag(t, n)
		d := a.Dense()
		if got := testing.AllocsPerRun(5, func() { _, _ = FromDense(n, n, d) }); got > 5 {
			t.Errorf("FromDense of order %d: %v allocations, want at most 5", n, got)
		}
	}
}
