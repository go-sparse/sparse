package gallery

import "testing"

func TestLaplacians(t *testing.T) {
	a := Laplacian1D(4)
	want := []float64{2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2}
	for i, v := range a.Dense() {
		if v != want[i] {
			t.Fatal(a.Dense())
		}
	}
	b := Laplacian2D(3, 3)
	if b.Rows != 9 || b.NNZ() != 9+2*12 || b.At(4, 4) != 4 || b.At(4, 1) != -1 || !b.IsSymmetric(0) {
		t.Fatal("2d", b.NNZ())
	}
	c := Laplacian3D(3, 3, 3)
	if c.Rows != 27 || c.At(13, 13) != 6 || c.NNZ() != 27+2*54 || !c.IsSymmetric(0) {
		t.Fatal("3d", c.NNZ())
	}
	if Laplacian3D(1, 1, 1).At(0, 0) != 2 {
		t.Fatal("1x1x1")
	}
}

func TestConvectionDiffusion(t *testing.T) {
	a := ConvectionDiffusion2D(3, 3, 0.5)
	if a.IsSymmetric(0.1) || a.At(4, 4) != 5 || a.At(4, 3) != -1.5 || a.At(4, 5) != -1 {
		t.Fatal("cd")
	}
}

func TestRandomSparse(t *testing.T) {
	a := RandomSparse(50, 4, false, 1)
	b := RandomSparse(50, 4, false, 1)
	s := RandomSparse(50, 4, true, 1)
	if a.NNZ() != b.NNZ() || !s.IsSymmetric(0) || a.IsSymmetric(0) {
		t.Fatal("symmetry or determinism")
	}
	for i := 0; i < 50; i++ {
		off := 0.0
		for j := 0; j < 50; j++ {
			if j != i {
				off += abs(s.At(i, j))
			}
		}
		if s.At(i, i) <= off {
			t.Fatal("not dominant")
		}
	}
}

func TestScramble(t *testing.T) {
	a := Laplacian2D(4, 4)
	b, p := Scramble(a, 3)
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if b.At(i, j) != a.At(p[i], p[j]) {
				t.Fatal("scramble")
			}
		}
	}
}
