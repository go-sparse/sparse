package ldl

import (
	"fmt"
	"testing"
)

// BenchmarkRefactorAMD is the numeric phase on the AMD-permuted KKT grids of
// TestPerfVsQDLDL (the profile target: go test -bench RefactorAMD -cpuprofile).
func BenchmarkRefactorAMD(b *testing.B) {
	for _, g := range []int{20, 40, 70} {
		pm := permutedGrid(b, gridKKT(g))
		f, pa := pm.f, pm.pa
		b.Run(fmt.Sprint(g), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := f.Refactor(pa); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
