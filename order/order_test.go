package order

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
)

// bruteFill counts the nonzeros of the Cholesky factor of P A Pᵀ by dense
// symbolic elimination: an independent oracle for CholeskyNNZ.
func bruteFill(a *sparse.CSR[float64], perm []int) int {
	n := a.Rows
	if perm == nil {
		perm = make([]int, n)
		for i := range perm {
			perm[i] = i
		}
	}
	m := make([][]bool, n)
	for i := range m {
		m[i] = make([]bool, n)
	}
	for i := 0; i < n; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			j := a.Indices[p]
			m[Inverse(perm)[i]][Inverse(perm)[j]] = true
			m[Inverse(perm)[j]][Inverse(perm)[i]] = true
		}
	}
	count := 0
	for k := 0; k < n; k++ {
		count++
		var rows []int
		for i := k + 1; i < n; i++ {
			if m[i][k] {
				rows = append(rows, i)
				count++
			}
		}
		for _, i := range rows {
			for _, j := range rows {
				m[i][j] = true
			}
		}
	}
	return count
}

func TestCholeskyNNZAgainstBruteForce(t *testing.T) {
	for seed := uint64(0); seed < 20; seed++ {
		a := gallery.RandomSparse(40, 2, seed%2 == 0, seed)
		perm := rand.New(rand.NewPCG(seed, 5)).Perm(40)
		for _, p := range [][]int{nil, perm} {
			got, err := CholeskyNNZ(a, p)
			if err != nil || got != bruteFill(a, p) {
				t.Fatalf("seed %d: %d vs %d (%v)", seed, got, bruteFill(a, p), err)
			}
		}
	}
	// the factor of the 1-D Laplacian is bidiagonal
	if n, _ := CholeskyNNZ(gallery.Laplacian1D(10), nil); n != 19 {
		t.Fatal(n)
	}
}

func TestMetricErrors(t *testing.T) {
	a := gallery.Laplacian1D(4)
	if _, err := CholeskyNNZ(a, []int{0, 1}); err == nil {
		t.Fatal("short perm")
	}
	if _, err := CholeskyNNZ(a, []int{0, 0, 1, 2}); err == nil {
		t.Fatal("non perm")
	}
	rect, _ := sparse.FromDense(2, 3, []float64{1, 0, 1, 0, 1, 0})
	if _, err := CholeskyNNZ(rect, nil); !errors.Is(err, ErrNotSquare) {
		t.Fatal("rect")
	}
	if _, err := RCM(rect); !errors.Is(err, ErrNotSquare) {
		t.Fatal("rcm rect")
	}
	if _, err := AMD(rect); !errors.Is(err, ErrNotSquare) {
		t.Fatal("amd rect")
	}
	empty := &sparse.CSR[float64]{Indptr: []int{0}}
	if _, err := COLAMD(empty); err == nil {
		t.Fatal("colamd empty")
	}
	if _, err := QRFillNNZ(rect, []int{0}); err == nil { // wrong length
		t.Fatal("qrfill perm")
	}
	if IsPermutation([]int{0, 2}) || !IsPermutation([]int{1, 0}) || IsPermutation([]int{-1}) {
		t.Fatal("IsPermutation")
	}
	// A is 3x2 so AᵀA is fine, but the Mul inside needs conforming shapes: always true
	tall, _ := sparse.FromDense(3, 2, []float64{1, 0, 1, 1, 0, 1})
	if n, err := QRFillNNZ(tall, nil); err != nil || n != 3 {
		t.Fatal(n, err)
	}
}

func TestBandwidthProfile(t *testing.T) {
	a, _ := sparse.FromDense(4, 4, []float64{
		1, 0, 0, 1,
		0, 1, 0, 0,
		0, 1, 1, 0,
		1, 0, 0, 1})
	if Bandwidth(a) != 3 {
		t.Fatal(Bandwidth(a))
	}
	if Profile(a) != 0+0+1+3 {
		t.Fatal(Profile(a))
	}
}

func apply(a *sparse.CSR[float64], p []int) *sparse.CSR[float64] {
	b, err := a.Permute(p, p)
	if err != nil {
		panic(err)
	}
	return b
}

func TestRCM(t *testing.T) {
	grid := gallery.Laplacian2D(25, 25)
	sc, _ := gallery.Scramble(grid, 1)
	p, err := RCM(sc)
	if err != nil || !IsPermutation(p) {
		t.Fatal(err)
	}
	if bw, bw0 := Bandwidth(apply(sc, p)), Bandwidth(sc); bw > 60 || bw*5 > bw0 {
		t.Fatalf("bandwidth %d from %d", bw, bw0)
	}
	if pr, pr0 := Profile(apply(sc, p)), Profile(sc); pr*5 > pr0 {
		t.Fatalf("profile %d from %d", pr, pr0)
	}
	// disconnected: two blocks plus isolated vertices
	c := sparse.NewCOO[float64](8, 8)
	for _, e := range [][2]int{{0, 1}, {1, 2}, {5, 6}} {
		_ = c.Append(e[0], e[1], 1)
	}
	for i := 0; i < 8; i++ {
		_ = c.Append(i, i, 1)
	}
	p, _ = RCM(c.ToCSR())
	if !IsPermutation(p) || len(p) != 8 {
		t.Fatal("components", p)
	}
	// the minimum-degree vertex (0) is a leaf hanging off the middle of a
	// path, so the first start is not peripheral and the search must move
	lc := sparse.NewCOO[float64](10, 10)
	_ = lc.Append(0, 5, 1)
	for i := 1; i < 9; i++ {
		_ = lc.Append(i, i+1, 1)
	}
	lp, _ := RCM(lc.ToCSR())
	if Bandwidth(apply(lc.ToCSR(), lp)) > 2 {
		t.Fatal("leaf on a path", lp)
	}
	// a path ordered badly gets bandwidth 1
	path := gallery.Laplacian1D(30)
	sp, _ := gallery.Scramble(path, 2)
	p, _ = RCM(sp)
	if Bandwidth(apply(sp, p)) != 1 {
		t.Fatal("path bandwidth", Bandwidth(apply(sp, p)))
	}
}

func TestAMDQuality(t *testing.T) {
	grid := gallery.Laplacian2D(30, 30)
	sc, _ := gallery.Scramble(grid, 3)
	p, err := AMD(sc)
	if err != nil || !IsPermutation(p) {
		t.Fatal(err)
	}
	amd, _ := CholeskyNNZ(sc, p)
	nat, _ := CholeskyNNZ(grid, nil)
	scr, _ := CholeskyNNZ(sc, nil)
	rcmP, _ := RCM(sc)
	rcm, _ := CholeskyNNZ(sc, rcmP)
	t.Logf("2-D 30x30: scrambled %d natural %d rcm %d amd %d", scr, nat, rcm, amd)
	if amd >= nat || amd >= rcm || amd*3 >= scr {
		t.Fatalf("AMD fill %d is not better than natural %d, rcm %d, scrambled %d", amd, nat, rcm, scr)
	}
	// 3-D grid
	g3 := gallery.Laplacian3D(8, 8, 8)
	p3, _ := AMD(g3)
	f3, _ := CholeskyNNZ(g3, p3)
	n3, _ := CholeskyNNZ(g3, nil)
	if !IsPermutation(p3) || f3 >= n3 {
		t.Fatalf("3-D fill %d vs natural %d", f3, n3)
	}
	// tridiagonal: no fill at all
	tri := gallery.Laplacian1D(50)
	ts, _ := gallery.Scramble(tri, 4)
	tp, _ := AMD(ts)
	if f, _ := CholeskyNNZ(ts, tp); f != 99 {
		t.Fatalf("path fill %d, want 99", f)
	}
}

// arrow builds an arrow matrix with a dense first row and column.
func arrow(n int) *sparse.CSR[float64] {
	c := sparse.NewCOO[float64](n, n)
	for i := 0; i < n; i++ {
		_ = c.Append(i, i, 4)
		if i > 0 {
			_ = c.Append(0, i, 1)
			_ = c.Append(i, 0, 1)
		}
	}
	return c.ToCSR()
}

func TestAMDDenseAndSupervariables(t *testing.T) {
	a := arrow(300)
	p, err := AMD(a)
	if err != nil || !IsPermutation(p) || p[len(p)-1] != 0 {
		t.Fatalf("dense row must be ordered last: %v", err)
	}
	if f, _ := CholeskyNNZ(a, p); f != 599 {
		t.Fatalf("arrow fill %d, want 599", f)
	}
	// two dense rows: both go last, sparser first
	c2 := sparse.NewCOO[float64](300, 300)
	for i := 0; i < 300; i++ {
		_ = c2.Append(i, i, 4)
		for _, d := range []int{0, 1} {
			if i > 1 && (d == 0 || i%4 != 3) {
				_ = c2.Append(d, i, 1)
				_ = c2.Append(i, d, 1)
			}
		}
	}
	p2, _ := AMD(c2.ToCSR())
	if !IsPermutation(p2) || p2[299] != 0 || p2[298] != 1 {
		t.Fatal("two dense", p2[298:])
	}
	// complete bipartite blocks: many indistinguishable vertices
	c := sparse.NewCOO[float64](60, 60)
	for i := 0; i < 60; i++ {
		_ = c.Append(i, i, 1)
		for j := 0; j < 60; j++ {
			if i/10 != j/10 && (i/10+j/10)%2 == 1 {
				_ = c.Append(i, j, 1)
			}
		}
	}
	bp, _ := AMD(c.ToCSR())
	if !IsPermutation(bp) {
		t.Fatal("blocks")
	}
	// random matrices of several densities
	for seed := uint64(0); seed < 30; seed++ {
		r := gallery.RandomSparse(120, int(1+seed%6), seed%3 == 0, seed)
		rp, _ := AMD(r)
		if !IsPermutation(rp) {
			t.Fatal("random", seed)
		}
		cp, _ := COLAMD(r)
		if !IsPermutation(cp) {
			t.Fatal("colamd random", seed)
		}
	}
}

func TestCOLAMD(t *testing.T) {
	// nonsymmetric square: fill of AᵀA under COLAMD beats the scrambled order
	a := gallery.ConvectionDiffusion2D(25, 25, 1)
	sc, _ := gallery.Scramble(a, 9)
	cp, err := COLAMD(sc)
	if err != nil || !IsPermutation(cp) {
		t.Fatal(err)
	}
	col, _ := QRFillNNZ(sc, cp)
	nat, _ := QRFillNNZ(sc, nil)
	if col*3 >= nat {
		t.Fatalf("COLAMD fill %d vs natural %d", col, nat)
	}
	// rectangular, with a dense row and a dense column
	m, n := 400, 150
	r := rand.New(rand.NewPCG(1, 1))
	c := sparse.NewCOO[float64](m, n)
	for i := 0; i < m; i++ {
		for k := 0; k < 3; k++ {
			_ = c.Append(i, r.IntN(n), 1)
		}
	}
	for j := 0; j < n; j++ {
		_ = c.Append(0, j, 1) // dense row
	}
	for i := 0; i < m; i++ {
		_ = c.Append(i, 7, 1) // dense columns
		if i%2 == 0 {
			_ = c.Append(i, 9, 1)
		}
	}
	rect := c.ToCSR()
	rp, err := COLAMD(rect)
	if err != nil || !IsPermutation(rp) || len(rp) != n || rp[n-1] != 7 {
		t.Fatalf("dense column must be last: %v %v", err, rp[n-1])
	}
	// empty rows and empty columns are tolerated
	z := sparse.NewCOO[float64](5, 5)
	_ = z.Append(1, 1, 1)
	_ = z.Append(2, 3, 1)
	zp, _ := COLAMD(z.ToCSR())
	if !IsPermutation(zp) {
		t.Fatal("empty rows/cols")
	}
}
