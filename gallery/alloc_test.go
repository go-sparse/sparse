//go:build !race

// The race detector's instrumentation allocates: the counts here are exact
// only without it (CI runs them in the plain test run of each lane).

package gallery

import "testing"

// TestAllocsDoNotGrowWithGrid asserts that building a Laplacian allocates the
// same number of times on a 10 x 10 and a 40 x 40 grid (before: one closure per
// grid point, and the triplets grown by doubling: 10186 allocations on 100 x 100).
func TestAllocsDoNotGrowWithGrid(t *testing.T) {
	for name, f := range map[string]func(g int){
		"Laplacian1D":  func(g int) { _ = Laplacian1D(g * g) },
		"Laplacian2D":  func(g int) { _ = Laplacian2D(g, g) },
		"Laplacian3D":  func(g int) { _ = Laplacian3D(g/4, g/4, g/4) },
		"ConvDiff2D":   func(g int) { _ = ConvectionDiffusion2D(g, g, 1) },
		"RandomSparse": func(g int) { _ = RandomSparse(g*g, 4, true, 1) },
	} {
		small := testing.AllocsPerRun(5, func() { f(12) })
		big := testing.AllocsPerRun(5, func() { f(40) })
		t.Logf("%s: %v allocations (small), %v (large)", name, small, big)
		if name == "RandomSparse" {
			continue // grown by append (one COO per call); not a per-point cost
		}
		// slack 2: the runtime's own allocations that the counter sees on a
		// large run (an allocation per grid point would be thousands)
		if big > small+2 || big > 20 {
			t.Errorf("%s: %v allocations large, %v small (want at most 2 more, and at most 20)", name, big, small)
		}
	}
}
