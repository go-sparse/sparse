//go:build !race

// The race detector's instrumentation allocates: the counts here are exact
// only without it (CI runs them in the plain test run of each lane).

package order

import (
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
)

// TestAllocsDoNotGrowWithOrder asserts that the allocations of every ordering
// call are the same for a grid of 64 and of 1024 unknowns: no allocation per
// vertex, per row or per doubling of an arena. Allocation counts are exact
// under any machine load; each bound is the measured count, so a regression
// fails.
func TestAllocsDoNotGrowWithOrder(t *testing.T) {
	cases := []struct {
		name string
		max  float64
		run  func(a *sparse.CSR[float64])
	}{
		// before: AMD 50, 58, 66 and COLAMD 49, 57, 62 at grids of 8, 16, 32;
		// RCM 88, 290, 1077 (a swapper per vertex).
		{"AMD", 28, func(a *sparse.CSR[float64]) { _, _ = AMD(a) }},
		{"COLAMD", 26, func(a *sparse.CSR[float64]) { _, _ = COLAMD(a) }},
		{"RCM", 9, func(a *sparse.CSR[float64]) { _, _ = RCM(a) }},
		{"CholeskyNNZ", 9, func(a *sparse.CSR[float64]) { _, _ = CholeskyNNZ(a, nil) }},
	}
	a8, a32 := gallery.Laplacian2D(8, 8), gallery.Laplacian2D(32, 32)
	for _, c := range cases {
		small := testing.AllocsPerRun(5, func() { c.run(a8) })
		big := testing.AllocsPerRun(5, func() { c.run(a32) })
		t.Logf("%s: %v allocations (order 64), %v (order 1024)", c.name, small, big)
		if big != small || big > c.max {
			t.Errorf("%s: %v allocations at order 1024, %v at order 64 (want equal and at most %v)", c.name, big, small, c.max)
		}
	}
}
