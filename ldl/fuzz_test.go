package ldl

import (
	"math"
	"testing"

	"github.com/go-sparse/sparse"
)

// decode turns fuzz bytes into a (possibly malformed) CSC matrix, a
// permutation candidate and a right-hand side. n is at most 16, so any L fits
// in a few KiB; indptr, indices and values are otherwise unconstrained
// (negative or too large rows, duplicates, unsorted columns, NaN, Inf, a
// corrupted indptr).
func decode(data []byte) (a *sparse.CSC[float64], perm []int, b []float64) {
	next := func() byte {
		if len(data) == 0 {
			return 0
		}
		v := data[0]
		data = data[1:]
		return v
	}
	n := int(next() % 17)
	flags := next()
	ap := make([]int, n+1)
	var ai []int32
	var ax []float64
	for j := 0; j < n; j++ {
		c := int(next() % 5)
		for k := 0; k < c; k++ {
			r := int32(next()) - 8
			if flags&1 == 0 {
				r = int32(next() % byte(j+1)) // mostly valid
			}
			ai = append(ai, r)
			switch v := next(); v {
			case 255:
				ax = append(ax, math.NaN())
			case 254:
				ax = append(ax, math.Inf(1))
			default:
				ax = append(ax, float64(int8(v))/8)
			}
		}
		ap[j+1] = len(ai)
	}
	if flags&2 != 0 && n > 0 { // corrupt the indptr
		ap[int(next())%(n+1)] += int(int8(next()))
	}
	cols := n
	if flags&4 != 0 {
		cols = n + 1 // not square
	}
	a = &sparse.CSC[float64]{Rows: n, Cols: cols, Indptr: ap, Indices: ai, Data: ax}
	perm = make([]int, n)
	for i := range perm {
		perm[i] = n - 1 - i
	}
	if flags&8 != 0 && n > 1 { // possibly not a permutation
		perm[int(next())%n] = int(int8(next()))
	}
	b = make([]float64, n)
	for i := range b {
		b[i] = float64(int8(next())) / 16
	}
	return
}

// FuzzFactor drives every entry point with arbitrary input: nothing may
// panic, and an accepted matrix must have a finite nonzero D.
func FuzzFactor(f *testing.F) {
	f.Add([]byte{3, 0, 1, 0, 8, 2, 0, 8, 1, 1, 8, 1, 2, 8})
	f.Add([]byte{4, 1, 2, 0, 8, 255, 1, 3, 0, 1, 9, 2})
	f.Add([]byte{5, 2, 1, 0, 16, 1, 0, 16, 1, 0, 16, 1, 0, 16, 1, 0, 16, 9, 200})
	f.Add([]byte{2, 8, 1, 0, 8, 2, 0, 8, 1, 1, 8, 1, 2, 3})
	f.Add([]byte{6, 4, 1, 0, 8})
	f.Add([]byte{2, 0, 1, 0, 8, 2, 0, 8, 1, 1, 254})
	f.Fuzz(func(t *testing.T, data []byte) {
		a, perm, b := decode(data)
		s, err := Analyze(a)
		if err == nil {
			fac := s.NewFactor()
			if err := fac.Refactor(a); err == nil {
				for _, d := range fac.D() {
					if d == 0 || math.IsNaN(d) || math.IsInf(d, 0) {
						t.Fatalf("accepted pivot %v", d)
					}
				}
				if fac.NumPositive() < 0 || fac.NumPositive() > s.N() {
					t.Fatal("inertia out of range")
				}
				x := append([]float64(nil), b...)
				if err := fac.Solve(x); err != nil {
					t.Fatal(err)
				}
			}
			_ = fac.Refactor(a) // twice: the workspace must be reusable after a failure
		}
		if p, err := NewPermuted(a, perm); err == nil {
			if err := p.Refactor(a); err == nil {
				x := append([]float64(nil), b...)
				if err := p.Solve(x); err != nil {
					t.Fatal(err)
				}
			}
		}
		_, _ = Factorize(a)
	})
}
