package ldl

import (
	"fmt"
	"testing"
)

// BenchmarkPermutedSolve is Permuted.Solve (gather, L, D, Lᵀ, scatter) on the
// AMD-ordered KKT grids; BenchmarkFactorSolve is Factor.Solve on the same
// permuted matrices. Each iteration restores the right-hand side first.
func BenchmarkPermutedSolve(b *testing.B) {
	for _, g := range []int{6, 20, 40, 70} {
		pm := permutedGrid(b, gridKKT(g))
		rhs := make([]float64, pm.f.sym.n)
		for i := range rhs {
			rhs[i] = 1 + float64(i%7)
		}
		x := make([]float64, len(rhs))
		b.Run(fmt.Sprint(g), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(x, rhs)
				if err := pm.Solve(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkFactorSolve(b *testing.B) {
	for _, g := range []int{6, 20, 40, 70} {
		pm := permutedGrid(b, gridKKT(g))
		rhs := make([]float64, pm.f.sym.n)
		for i := range rhs {
			rhs[i] = 1 + float64(i%7)
		}
		x := make([]float64, len(rhs))
		b.Run(fmt.Sprint(g), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				copy(x, rhs)
				if err := pm.f.Solve(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
