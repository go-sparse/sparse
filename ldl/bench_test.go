package ldl

import (
	"fmt"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/order"
)

// fullOf is the pattern of K + Kᵀ for the upper triangle a, the input of AMD.
func fullOf(tb testing.TB, a *sparse.CSC[float64]) *sparse.CSR[float64] {
	tb.Helper()
	full := sparse.NewCOO[float64](a.Cols, a.Cols)
	for j := 0; j < a.Cols; j++ {
		for p := a.Indptr[j]; p < a.Indptr[j+1]; p++ {
			i := int(a.Indices[p])
			must(tb, full.Append(i, j, 1))
			if i != j {
				must(tb, full.Append(j, i, 1))
			}
		}
	}
	return full.ToCSR()
}

// BenchmarkSetupAMD is the symbolic side of an ADMM setup on the KKT grids:
// the AMD ordering followed by NewPermuted ("fresh": new storage on every
// iteration) against the same on a reused order.Workspace and Permuted.Reset
// ("reused").
func BenchmarkSetupAMD(b *testing.B) {
	for _, g := range []int{6, 20, 40} {
		a := gridKKT(g)
		full := fullOf(b, a)
		b.Run(fmt.Sprintf("fresh/%d", g), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				perm, err := order.AMD(full)
				must(b, err)
				_, err = NewPermuted(a, perm)
				must(b, err)
			}
		})
		b.Run(fmt.Sprintf("reused/%d", g), func(b *testing.B) {
			var w order.Workspace
			p := new(Permuted)
			var perm []int
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var err error
				perm, err = order.AMDWith(&w, perm, full)
				must(b, err)
				must(b, p.Reset(a, perm))
			}
		})
	}
}

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
