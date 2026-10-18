package sparse

import (
	"errors"
	"math"
	"testing"
)

func TestCOOValidate(t *testing.T) {
	c := NewCOO[float64](2, 2)
	_ = c.Append(1, 1, 1)
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := map[string]*COO[float64]{
		"negative":  {Rows: -1, Cols: 2},
		"lengths":   {Rows: 2, Cols: 2, I: []int{0}, J: []int{0}, V: nil},
		"lengths J": {Rows: 2, Cols: 2, I: []int{0}, J: nil, V: []float64{1}},
		"row":       {Rows: 2, Cols: 2, I: []int{2}, J: []int{0}, V: []float64{1}},
		"col":       {Rows: 2, Cols: 2, I: []int{0}, J: []int{-1}, V: []float64{1}},
	}
	for name, b := range bad {
		if err := b.Validate(); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if !errors.Is(bad["row"].Validate(), ErrIndex) || !errors.Is(bad["negative"].Validate(), ErrFormat) {
		t.Error("error kinds")
	}
}

func TestFromDenseHostileSizes(t *testing.T) {
	// 1<<32 x 1<<32 overflows int64; 1<<17 x 1<<16 overflows int32
	for _, rc := range [][2]int{{-1, 2}, {2, -1}, {math.MaxInt, 2}, {1 << 31, 1 << 31}} {
		if _, err := FromDense(rc[0], rc[1], []float64{1}); !errors.Is(err, ErrShape) {
			t.Errorf("%v accepted", rc)
		}
	}
	if a, err := FromDense(0, 5, []float64{}); err != nil || a.NNZ() != 0 {
		t.Error("empty matrix")
	}
}
