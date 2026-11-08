package sparse

import (
	"math/rand/v2"
	"testing"
)

// compressRef is compress as it was before the single-scatter rewrite: two
// stable counting sorts. The new one must return the same structure and the
// same bits (duplicates summed in input order).
func compressRef[T Scalar](nmajor, nminor int, maj, min []int, v []T) (ptr []int, ind []int32, data []T) {
	nnz := len(v)
	cnt := make([]int, nminor+1)
	for _, j := range min {
		cnt[j+1]++
	}
	for j := 0; j < nminor; j++ {
		cnt[j+1] += cnt[j]
	}
	order := make([]int, nnz)
	for k, j := range min {
		order[cnt[j]] = k
		cnt[j]++
	}
	ptr = make([]int, nmajor+1)
	for _, i := range maj {
		ptr[i+1]++
	}
	for i := 0; i < nmajor; i++ {
		ptr[i+1] += ptr[i]
	}
	pos := append([]int(nil), ptr[:nmajor]...)
	ind = make([]int32, nnz)
	data = make([]T, nnz)
	for _, k := range order {
		i := maj[k]
		ind[pos[i]] = int32(min[k])
		data[pos[i]] = v[k]
		pos[i]++
	}
	w, start := 0, 0
	for i := 0; i < nmajor; i++ {
		end := ptr[i+1]
		first := w
		for p := start; p < end; p++ {
			if w > first && ind[w-1] == ind[p] {
				data[w-1] += data[p]
				continue
			}
			ind[w], data[w] = ind[p], data[p]
			w++
		}
		start = end
		ptr[i+1] = w
	}
	return ptr, ind[:w:w], data[:w:w]
}

func TestCompressMatchesTwoPassSort(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	// shapes: sparse, dense rows (above the insertion-sort limit), empty
	// rows, many duplicates, sorted and shuffled input
	for _, c := range []struct{ m, n, nnz int }{{1, 1, 0}, {5, 7, 30}, {40, 3, 400}, {3, 200, 500}, {50, 50, 2000}, {200, 90, 100}} {
		maj, min := make([]int, c.nnz), make([]int, c.nnz)
		v := make([]float64, c.nnz)
		for k := range v {
			maj[k], min[k], v[k] = r.IntN(c.m), r.IntN(c.n), r.NormFloat64()*1e3
		}
		for _, sorted := range []bool{false, true} {
			if sorted {
				// the order of a COO made from a canonical matrix
				idx := make([]int, c.nnz)
				for k := range idx {
					idx[k] = k
				}
				sortByMajorMinor(idx, maj, min)
				m2, n2, v2 := make([]int, c.nnz), make([]int, c.nnz), make([]float64, c.nnz)
				for k, p := range idx {
					m2[k], n2[k], v2[k] = maj[p], min[p], v[p]
				}
				maj, min, v = m2, n2, v2
			}
			gp, gi, gd := compress(c.m, c.n, maj, min, v)
			wp, wi, wd := compressRef(c.m, c.n, maj, min, v)
			if len(gp) != len(wp) || len(gi) != len(wi) {
				t.Fatalf("%+v: sizes differ", c)
			}
			for i := range wp {
				if gp[i] != wp[i] {
					t.Fatalf("%+v: ptr[%d] = %d, want %d", c, i, gp[i], wp[i])
				}
			}
			for k := range wi {
				if gi[k] != wi[k] || gd[k] != wd[k] {
					t.Fatalf("%+v: entry %d is (%d, %v), want (%d, %v)", c, k, gi[k], gd[k], wi[k], wd[k])
				}
			}
		}
	}
}

func sortByMajorMinor(idx, maj, min []int) {
	for a := 1; a < len(idx); a++ {
		for b := a; b > 0 && (maj[idx[b-1]] > maj[idx[b]] || (maj[idx[b-1]] == maj[idx[b]] && min[idx[b-1]] > min[idx[b]])); b-- {
			idx[b-1], idx[b] = idx[b], idx[b-1]
		}
	}
}

func TestCompressRefusesMinorOutOfRange(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("no panic for a column index outside the matrix")
		}
	}()
	c := NewCOO[float64](2, 2)
	c.I, c.J, c.V = []int{0}, []int{2}, []float64{1}
	c.ToCSR()
}

func TestPruneRealAndComplex(t *testing.T) {
	r, _ := NewCSR(2, 3, []int{0, 2, 4}, []int32{0, 2, 1, 2}, []float64{0, 3, 0.5, 0})
	r.Prune(0.5) // removes |v| <= 0.5
	if r.NNZ() != 1 || r.At(0, 2) != 3 || r.Indptr[2] != 1 {
		t.Fatalf("real prune: %+v", r)
	}
	c, _ := NewCSR(1, 3, []int{0, 3}, []int32{0, 1, 2}, []complex128{0, 3 + 4i, 1e-9i})
	c.Prune(1e-6)
	if c.NNZ() != 1 || c.At(0, 1) != 3+4i {
		t.Fatalf("complex prune: %+v", c)
	}
}

func TestNormFroMatchesSerialSum(t *testing.T) {
	d := make([]float64, 1003)
	s := 0.0
	for i := range d {
		d[i] = float64(i%17) - 8.25
		s += d[i] * d[i]
	}
	ind := make([]int32, len(d))
	for i := range ind {
		ind[i] = int32(i)
	}
	a, _ := NewCSR(1, len(d), []int{0, len(d)}, ind, d)
	want := sqrtRef(s)
	if got := a.NormFro(); got < want*(1-1e-14) || got > want*(1+1e-14) {
		t.Fatalf("NormFro = %v, serial sum gives %v", got, want)
	}
	z, _ := NewCSR(1, 2, []int{0, 2}, []int32{0, 1}, []complex128{3 + 4i, 12i})
	if got := z.NormFro(); got != 13 {
		t.Fatalf("complex NormFro = %v, want 13", got)
	}
}

func sqrtRef(x float64) float64 {
	// Newton from a good start: no math import for one test
	g := x
	if g > 1 {
		g = x / 2
	}
	for i := 0; i < 100; i++ {
		g = (g + x/g) / 2
	}
	return g
}
