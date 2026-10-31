package sparse

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// symUpperCase builds a random symmetric S, returns its upper triangle in CSC
// (columns in increasing row order) and the dense full matrix.
func symUpperCase[T Scalar](r *rand.Rand, n int) (*CSC[T], []T) {
	full := make([]T, n*n)
	up := NewCOO[T](n, n)
	for j := 0; j < n; j++ {
		for i := 0; i <= j; i++ {
			if r.Float64() < 0.4 {
				v := rnd[T](r)
				full[i*n+j], full[j*n+i] = v, v
				_ = up.Append(i, j, v)
			}
		}
	}
	return up.ToCSC(), full
}

func testMulVecSymUpper[T Scalar](t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{0, 1, 2, 5, 17, 40} {
		a, full := symUpperCase[T](r, n)
		x := make([]T, n)
		for i := range x {
			x[i] = rnd[T](r)
		}
		got := make([]T, n)
		for i := range got {
			got[i] = rnd[T](r) // must be overwritten
		}
		a.MulVecSymUpper(got, x)
		for i := 0; i < n; i++ {
			var want T
			for j := 0; j < n; j++ {
				want += full[i*n+j] * x[j]
			}
			if abs(got[i]-want) > 1e-12*(1+abs(want)) {
				t.Fatalf("n=%d row %d: %v want %v", n, i, got[i], want)
			}
		}
	}
}

func TestMulVecSymUpper(t *testing.T) {
	t.Run("float64", testMulVecSymUpper[float64])
	t.Run("complex128", testMulVecSymUpper[complex128])
}

func TestMulVecSymUpperComplexSymmetricNotHermitian(t *testing.T) {
	// S = [[1, i], [i, 2]] (symmetric): S·(1, 1) = (1+i, i+2)
	a, err := NewCSC(2, 2, []int{0, 1, 3}, []int32{0, 0, 1}, []complex128{1, 1i, 2})
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]complex128, 2)
	a.MulVecSymUpper(dst, []complex128{1, 1})
	if cmplx.Abs(dst[0]-(1+1i)) > 1e-15 || cmplx.Abs(dst[1]-(2+1i)) > 1e-15 {
		t.Fatalf("got %v", dst)
	}
}

func TestMulVecSymUpperIgnoresLowerAndPanics(t *testing.T) {
	// the stored (1, 0) entry is below the diagonal and ignored
	a := &CSC[float64]{Rows: 2, Cols: 2, Indptr: []int{0, 2, 3}, Indices: []int32{0, 1, 1}, Data: []float64{1, 9, 2}}
	dst := make([]float64, 2)
	a.MulVecSymUpper(dst, []float64{1, 1})
	if dst[0] != 1 || dst[1] != 2 {
		t.Fatalf("got %v", dst)
	}
	for name, f := range map[string]func(){
		"non-square": func() {
			(&CSC[float64]{Rows: 2, Cols: 3, Indptr: []int{0, 0, 0, 0}}).MulVecSymUpper(make([]float64, 2), make([]float64, 3))
		},
		"x length":   func() { a.MulVecSymUpper(dst, make([]float64, 3)) },
		"dst length": func() { a.MulVecSymUpper(make([]float64, 3), make([]float64, 2)) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", name)
				}
			}()
			f()
		}()
	}
}
