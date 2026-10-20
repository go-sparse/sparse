package sparse

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

func rnd[T Scalar](r *rand.Rand) T {
	var z T
	if _, ok := any(z).(complex128); ok {
		return any(complex(r.NormFloat64(), r.NormFloat64())).(T)
	}
	return any(r.NormFloat64()).(T)
}

// randCOO builds a random matrix with duplicates, plus its dense value.
func randCOO[T Scalar](r *rand.Rand, m, n, nnz int) (*COO[T], []T) {
	c := NewCOO[T](m, n)
	d := make([]T, m*n)
	for k := 0; k < nnz; k++ {
		i, j := r.IntN(m), r.IntN(n)
		v := rnd[T](r)
		if err := c.Append(i, j, v); err != nil {
			panic(err)
		}
		d[i*n+j] += v
	}
	return c, d
}

func near[T Scalar](a, b []T, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func denseMul[T Scalar](a []T, m, k int, b []T, n int) []T {
	c := make([]T, m*n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			for j := 0; j < n; j++ {
				c[i*n+j] += a[i*k+p] * b[p*n+j]
			}
		}
	}
	return c
}

func transposeDense[T Scalar](a []T, m, n int) []T {
	t := make([]T, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			t[j*m+i] = a[i*n+j]
		}
	}
	return t
}

func TestAll(t *testing.T) {
	t.Run("float64", testAll[float64])
	t.Run("complex128", testAll[complex128])
}

func testAll[T Scalar](t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	const m, n = 13, 9
	coo, dense := randCOO[T](r, m, n, 60)
	if rr, cc := coo.Dims(); rr != m || cc != n || coo.NNZ() != 60 {
		t.Fatal("coo dims")
	}
	csr := coo.ToCSR()
	csc := coo.ToCSC()
	if err := csr.Validate(); err != nil {
		t.Fatal(err)
	}
	if rr, cc := csr.Dims(); rr != m || cc != n {
		t.Fatal("csr dims")
	}
	if !near(csr.Dense(), dense, 1e-12) || !near(csc.Dense(), dense, 1e-12) {
		t.Fatal("dense mismatch")
	}
	if rr, cc := csc.Dims(); rr != m || cc != n || csc.NNZ() != csr.NNZ() {
		t.Fatal("csc dims")
	}
	if !near(csr.ToCSC().Dense(), dense, 1e-12) || !near(csc.ToCSR().Dense(), dense, 1e-12) {
		t.Fatal("conversion")
	}
	if !near(csr.ToCOO().ToCSR().Dense(), dense, 1e-12) || !near(csc.ToCOO().ToCSR().Dense(), dense, 1e-12) {
		t.Fatal("to coo")
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if abs(csr.At(i, j)-dense[i*n+j]) > 1e-12 || abs(csc.At(i, j)-dense[i*n+j]) > 1e-12 {
				t.Fatal("At")
			}
		}
	}
	// transposes
	dt := transposeDense(dense, m, n)
	if !near(csr.T().Dense(), dt, 1e-12) || !near(csr.Transpose().Dense(), dt, 1e-12) ||
		!near(csc.Transpose().Dense(), dt, 1e-12) || !near(coo.Transpose().ToCSR().Dense(), dt, 1e-12) {
		t.Fatal("transpose")
	}
	h := csr.H().Dense()
	for i := range dt {
		if abs(h[i]-conj(dt[i])) > 1e-12 {
			t.Fatal("H")
		}
	}
	// clone independence
	cl := csr.Clone()
	cl.Data[0] += fromReal[T](5)
	if near(cl.Data, csr.Data, 1e-12) || !near(csc.Clone().Dense(), dense, 1e-12) {
		t.Fatal("clone")
	}
	// products
	x := make([]T, n)
	y := make([]T, m)
	for i := range x {
		x[i] = rnd[T](r)
	}
	for i := range y {
		y[i] = rnd[T](r)
	}
	want := denseMul(dense, m, n, x, 1)
	got := make([]T, m)
	csr.MulVec(got, x)
	if !near(got, want, 1e-12) {
		t.Fatal("csr matvec")
	}
	csc.MulVec(got, x)
	if !near(got, want, 1e-12) {
		t.Fatal("csc matvec")
	}
	wantT := denseMul(dt, n, m, y, 1)
	gotT := make([]T, n)
	csr.MulVecT(gotT, y)
	if !near(gotT, wantT, 1e-12) {
		t.Fatal("csr matvecT")
	}
	csc.MulVecT(gotT, y)
	if !near(gotT, wantT, 1e-12) {
		t.Fatal("csc matvecT")
	}
	const k = 4
	bm := make([]T, n*k)
	for i := range bm {
		bm[i] = rnd[T](r)
	}
	cm := make([]T, m*k)
	csr.MulDense(cm, bm, k)
	if !near(cm, denseMul(dense, m, n, bm, k), 1e-12) {
		t.Fatal("spmm")
	}
	// sparse + sparse, sparse * sparse
	coo2, dense2 := randCOO[T](r, m, n, 50)
	b := coo2.ToCSR()
	sum, err := Add(csr, b)
	if err != nil {
		t.Fatal(err)
	}
	diff, _ := Sub(csr, b)
	ws, wd := make([]T, m*n), make([]T, m*n)
	for i := range ws {
		ws[i] = dense[i] + dense2[i]
		wd[i] = dense[i] - dense2[i]
	}
	if !near(sum.Dense(), ws, 1e-12) || !near(diff.Dense(), wd, 1e-12) {
		t.Fatal("add/sub")
	}
	if err := sum.Validate(); err != nil {
		t.Fatal(err)
	}
	_, dense3 := randCOO[T](r, n, 7, 40)
	c3, _ := FromDense(n, 7, dense3)
	prod, err := Mul(csr, c3)
	if err != nil || !near(prod.Dense(), denseMul(dense, m, n, dense3, 7), 1e-11) {
		t.Fatal("spgemm")
	}
	if prod.Validate() != nil {
		t.Fatal("spgemm invalid")
	}
	// scaling
	sc := csr.Clone()
	dr := make([]T, m)
	dc := make([]T, n)
	for i := range dr {
		dr[i] = rnd[T](r)
	}
	for i := range dc {
		dc[i] = rnd[T](r)
	}
	if sc.ScaleRows(dr) != nil || sc.ScaleCols(dc) != nil {
		t.Fatal("scale err")
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if abs(sc.At(i, j)-dr[i]*dense[i*n+j]*dc[j]) > 1e-11 {
				t.Fatal("scale")
			}
		}
	}
	two := fromReal[T](2)
	sc = csr.Clone()
	sc.Scale(two)
	csc2 := csc.Clone()
	csc2.Scale(two)
	if abs(sc.At(0, 0)-two*csr.At(0, 0)) > 1e-12 || abs(csc2.At(0, 0)-two*csr.At(0, 0)) > 1e-12 {
		t.Fatal("Scale")
	}
	// diagonals
	for off := -m; off <= n; off++ {
		dg := csr.Diagonal(off)
		dgc := csc.Diagonal(off)
		for s := range dg {
			i, j := s, s+off
			if off < 0 {
				i, j = s-off, s
			}
			if dg[s] != csr.At(i, j) || dgc[s] != csr.At(i, j) {
				t.Fatal("diagonal", off)
			}
		}
	}
	// triangles
	for _, k := range []int{-1, 0, 2} {
		lo, up := csr.Tril(k), csr.Triu(k)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				wl, wu := dense[i*n+j], dense[i*n+j]
				if j-i > k {
					wl = 0
				}
				if j-i < k {
					wu = 0
				}
				if lo.At(i, j) != wl || up.At(i, j) != wu {
					t.Fatal("tri")
				}
			}
		}
	}
	// permute
	p := r.Perm(m)
	q := r.Perm(n)
	pm, err := csr.Permute(p, q)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if pm.At(i, j) != dense[p[i]*n+q[j]] {
				t.Fatal("permute")
			}
		}
	}
	id, _ := csr.Permute(nil, nil)
	if !near(id.Dense(), dense, 0) {
		t.Fatal("permute nil")
	}
	// norms
	var n1, ni, nf float64
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < m; i++ {
			s += abs(dense[i*n+j])
		}
		n1 = math.Max(n1, s)
	}
	for i := 0; i < m; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += abs(dense[i*n+j])
			nf += abs(dense[i*n+j]) * abs(dense[i*n+j])
		}
		ni = math.Max(ni, s)
	}
	if math.Abs(csr.Norm1()-n1) > 1e-11 || math.Abs(csr.NormInf()-ni) > 1e-11 || math.Abs(csr.NormFro()-math.Sqrt(nf)) > 1e-11 {
		t.Fatal("norms")
	}
	// prune
	pr := csr.Clone()
	pr.Data[0] = 0
	nn := pr.NNZ()
	pr.Prune(0)
	if pr.NNZ() != nn-1 || pr.Validate() != nil {
		t.Fatal("prune")
	}
	// symmetric
	sq, _ := FromDense(3, 3, []T{1, 2, 0, 2, 1, 0, 0, 0, 1})
	if !sq.IsSymmetric(0) {
		t.Fatal("symmetric")
	}
	sq.Data[1] += fromReal[T](1)
	if sq.IsSymmetric(0.5) || csr.IsSymmetric(0) {
		t.Fatal("asymmetric")
	}
	// diags / eye
	dm, err := Diags([][]T{{1, 2, 3}, {4, 5}, {6, 7}}, []int{0, 1, -1}, 3, 3)
	if err != nil || !near(dm.Dense(), []T{1, 4, 0, 6, 2, 5, 0, 7, 3}, 0) {
		t.Fatal("diags", dm.Dense())
	}
	if _, err := Diags([][]T{{1}}, []int{0, 1}, 3, 3); !errors.Is(err, ErrShape) {
		t.Fatal("diags len")
	}
	if _, err := Diags([][]T{{1, 2, 3, 4}}, []int{0}, 3, 3); !errors.Is(err, ErrIndex) {
		t.Fatal("diags range")
	}
	if !near(Eye[T](3).Dense(), []T{1, 0, 0, 0, 1, 0, 0, 0, 1}, 0) {
		t.Fatal("eye")
	}
	// DOK
	dk := NewDOK[T](3, 3)
	if rr, cc := dk.Dims(); rr != 3 || cc != 3 {
		t.Fatal("dok dims")
	}
	must(t, dk.Set(0, 1, 2), dk.Set(2, 2, 3), dk.Set(1, 1, 4), dk.Set(1, 1, 0))
	if dk.Len() != 2 || dk.At(0, 1) != 2 || dk.At(1, 1) != 0 {
		t.Fatal("dok")
	}
	if !near(dk.ToCSR().Dense(), []T{0, 2, 0, 0, 0, 0, 0, 0, 3}, 0) {
		t.Fatal("dok csr")
	}
	if !errors.Is(dk.Set(3, 0, 1), ErrIndex) || !errors.Is(coo.Append(-1, 0, 1), ErrIndex) {
		t.Fatal("index errors")
	}
}

func must(t *testing.T, errs ...error) {
	t.Helper()
	for _, e := range errs {
		if e != nil {
			t.Fatal(e)
		}
	}
}

func TestErrors(t *testing.T) {
	a, _ := FromDense(2, 3, []float64{1, 0, 2, 0, 3, 0})
	b, _ := FromDense(2, 2, []float64{1, 0, 0, 1})
	if _, err := Add(a, b); !errors.Is(err, ErrShape) {
		t.Fatal("add shape")
	}
	if _, err := Mul(a, b); !errors.Is(err, ErrShape) {
		t.Fatal("mul shape")
	}
	if _, err := FromDense(2, 2, []float64{1}); !errors.Is(err, ErrShape) {
		t.Fatal("fromdense")
	}
	if a.ScaleRows([]float64{1}) == nil || a.ScaleCols([]float64{1}) == nil {
		t.Fatal("scale shape")
	}
	if _, err := a.Permute([]int{0}, nil); !errors.Is(err, ErrShape) {
		t.Fatal("permute shape")
	}
	if _, err := a.Permute(nil, []int{0}); !errors.Is(err, ErrShape) {
		t.Fatal("permute shape")
	}
	for name, f := range map[string]func(){
		"MulVec":   func() { a.MulVec(make([]float64, 2), make([]float64, 2)) },
		"MulVecT":  func() { a.MulVecT(make([]float64, 3), make([]float64, 3)) },
		"MulDense": func() { a.MulDense(make([]float64, 2), make([]float64, 2), 1) },
	} {
		func() {
			defer func() {
				e, ok := recover().(error)
				if !ok || !errors.Is(e, ErrShape) {
					t.Fatal(name, "did not panic with ErrShape")
				}
			}()
			f()
		}()
	}
	if len(a.Diagonal(5)) != 0 || len(a.Diagonal(-5)) != 0 {
		t.Fatal("empty diagonal")
	}
	if a.IsSymmetric(0) {
		t.Fatal("rectangular is not symmetric")
	}
}

func TestValidate(t *testing.T) {
	good := func() ([]int, []int32, []float64) { return []int{0, 1, 2}, []int32{0, 1}, []float64{1, 2} }
	p, i, d := good()
	if _, err := NewCSR(2, 2, p, i, d); err != nil {
		t.Fatal(err)
	}
	if _, err := NewCSC(2, 2, p, i, d); err != nil {
		t.Fatal(err)
	}
	// MaxDim+1 does not fit a 32-bit int: computed at run time it wraps to
	// a negative dimension there, refused all the same.
	tooLarge := int64(MaxDim)
	tooLarge++
	bad := []struct {
		name       string
		rows, cols int
		p          []int
		i          []int32
		d          []float64
		want       error
	}{
		{"negative", -1, 2, []int{0}, nil, nil, ErrFormat},
		{"too large", int(tooLarge), 2, []int{0}, nil, nil, ErrFormat}, // refused before indptr is looked at
		{"ptr len", 3, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{1, 2}, ErrFormat},
		{"ptr start", 2, 2, []int{1, 1, 2}, []int32{0, 1}, []float64{1, 2}, ErrFormat},
		{"nnz", 2, 2, []int{0, 1, 3}, []int32{0, 1}, []float64{1, 2}, ErrFormat},
		{"decreasing", 2, 2, []int{0, 2, 1}, []int32{0}, []float64{1}, ErrFormat},
		{"range", 2, 2, []int{0, 1, 2}, []int32{0, 2}, []float64{1, 2}, ErrIndex},
		{"negidx", 2, 2, []int{0, 1, 2}, []int32{-1, 1}, []float64{1, 2}, ErrIndex},
		{"order", 2, 2, []int{0, 2, 2}, []int32{1, 0}, []float64{1, 2}, ErrFormat},
	}
	for _, c := range bad {
		if _, err := NewCSR(c.rows, c.cols, c.p, c.i, c.d); !errors.Is(err, c.want) {
			t.Errorf("%s: got %v", c.name, err)
		}
	}
	if _, err := NewCSC(2, 2, []int{0, 1}, i, d); err == nil {
		t.Fatal("csc validate")
	}
}

func TestOperators(t *testing.T) {
	a, _ := FromDense(2, 2, []float64{2, 1, 1, 3})
	var _ LinearOperator = a
	var _ TransposeOperator = a
	var _ TransposeOperator = a.ToCSC()
	f := FuncOperator{Rows: 2, Cols: 2, Mv: a.MulVec, MvT: a.MulVecT}
	if r, c := f.Dims(); r != 2 || c != 2 {
		t.Fatal("dims")
	}
	y := make([]float64, 2)
	f.MulVec(y, []float64{1, 1})
	if y[0] != 3 || y[1] != 4 {
		t.Fatal(y)
	}
	f.MulVecT(y, []float64{1, 0})
	if y[0] != 2 || y[1] != 1 {
		t.Fatal(y)
	}
	s := Shifted(a, 10)
	s.MulVec(y, []float64{1, 1})
	if y[0] != 13 || y[1] != 14 {
		t.Fatal(y)
	}
	if conj(complex(1, 2)) != complex(1, -2) || cmplx.Abs(complex(3, 4)) != 5 {
		t.Fatal("conj")
	}
}
