package gallery

import (
	"testing"

	"github.com/go-sparse/sparse"
)

// stencilRef builds the matrix as grid did before it assembled CSR directly:
// triplets in emission order, converted (sorted, duplicates summed) by ToCSR.
func stencilRef(nx, ny, nz int, offs [][3]int, vals []float64) *sparse.CSR[float64] {
	n := nx * ny * nz
	c := sparse.NewCOO[float64](n, n)
	for z := 0; z < nz; z++ {
		for y := 0; y < ny; y++ {
			for x := 0; x < nx; x++ {
				for t, o := range offs {
					X, Y, Z := x+o[0], y+o[1], z+o[2]
					if X < 0 || X >= nx || Y < 0 || Y >= ny || Z < 0 || Z >= nz {
						continue
					}
					_ = c.Append((z*ny+y)*nx+x, (Z*ny+Y)*nx+X, vals[t])
				}
			}
		}
	}
	return c.ToCSR()
}

func same(t *testing.T, name string, got, want *sparse.CSR[float64]) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols || len(got.Indptr) != len(want.Indptr) || len(got.Data) != len(want.Data) {
		t.Fatalf("%s: shape or size differs", name)
	}
	for i, v := range want.Indptr {
		if got.Indptr[i] != v {
			t.Fatalf("%s: indptr[%d]", name, i)
		}
	}
	for k := range want.Data {
		if got.Indices[k] != want.Indices[k] || got.Data[k] != want.Data[k] {
			t.Fatalf("%s: entry %d", name, k)
		}
	}
}

// TestDirectCSRMatchesTripletAssembly: the matrices are those of the triplet
// assembly, bit for bit, on grids from 1 to 3 dimensions including degenerate
// ones (the empty grid too).
func TestDirectCSRMatchesTripletAssembly(t *testing.T) {
	for _, g := range [][3]int{{0, 3, 1}, {1, 1, 1}, {2, 1, 1}, {7, 1, 1}, {1, 5, 1}, {5, 4, 1}, {3, 4, 5}, {2, 2, 2}} {
		nx, ny, nz := g[0], g[1], g[2]
		dims := 0
		var offs [][3]int
		var vals []float64
		for _, d := range []int{nx, ny, nz} {
			if d > 1 {
				dims++
			}
		}
		offs, vals = append(offs, [3]int{}), append(vals, float64(2*max(dims, 1)))
		for axis, d := range []int{nx, ny, nz} {
			if d > 1 {
				lo, hi := [3]int{}, [3]int{}
				lo[axis], hi[axis] = -1, 1
				offs, vals = append(offs, lo, hi), append(vals, -1, -1)
			}
		}
		same(t, "Laplacian3D", Laplacian3D(nx, ny, nz), stencilRef(nx, ny, nz, offs, vals))
	}
	// the nonsymmetric stencil emits its entries out of column order
	offs := [][3]int{{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}}
	vals := []float64{4 + 2*0.7, -1 - 0.7, -1, -1 - 0.7, -1}
	same(t, "ConvectionDiffusion2D", ConvectionDiffusion2D(6, 5, 0.7), stencilRef(6, 5, 1, offs, vals))
}
