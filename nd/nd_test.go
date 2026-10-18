package nd

import (
	"errors"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-sparse/sparse"
)

func TestInterop(t *testing.T) {
	d, _ := ndarray.FromData([]float64{2, 0, 1, 0, 3, 0}, 2, 3)
	a, err := ToCSR(d)
	if err != nil || a.NNZ() != 3 {
		t.Fatal(err, a)
	}
	back, err := FromCSR(a)
	if err != nil || back.At(0, 2) != 1 || back.At(1, 1) != 3 || back.At(1, 0) != 0 {
		t.Fatal("round trip")
	}
	x := FromVector([]float64{1, 2, 3})
	v, err := Vector(x)
	if err != nil || len(v) != 3 || v[2] != 3 {
		t.Fatal(v, err)
	}
	y, err := MulVec(a, x)
	if err != nil {
		t.Fatal(err)
	}
	if y.At(0) != 5 || y.At(1) != 6 {
		t.Fatal(y)
	}
	b, _ := ndarray.FromData([]float64{1, 0, 0, 1, 1, 1}, 3, 2)
	c, err := MulMat(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := d.MatMul(b)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if c.At(i, j) != want.At(i, j) {
				t.Fatal("MulMat", i, j)
			}
		}
	}
}

func TestErrors(t *testing.T) {
	a, _ := ndarray.Zeros(2, 2)
	v := FromVector([]float64{1, 2})
	if _, err := Vector(a); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
	if _, err := ToCSR(v); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
	m := sparse.Eye[float64](3)
	if _, err := MulMat(m, a); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
	if _, err := MulMat(m, v); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
	if _, err := MulVec(m, a); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
	if _, err := MulVec(m, v); !errors.Is(err, sparse.ErrShape) {
		t.Fatal(err)
	}
}
