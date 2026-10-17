// Package gallery generates deterministic test matrices: finite-difference
// Laplacians, a convection-diffusion operator, and seeded random sparse
// matrices. They are the inputs of the verification and benchmark harnesses.
package gallery

import (
	"math/rand/v2"

	"github.com/go-sparse/sparse"
)

// stencil assembles a matrix on an nx x ny x nz grid (lexicographic order)
// from a per-point function that appends (neighbour offset, value) pairs.
func grid(nx, ny, nz int, point func(emit func(dx, dy, dz int, v float64))) *sparse.CSR[float64] {
	n := nx * ny * nz
	c := sparse.NewCOO[float64](n, n)
	for z := 0; z < nz; z++ {
		for y := 0; y < ny; y++ {
			for x := 0; x < nx; x++ {
				row := (z*ny+y)*nx + x
				point(func(dx, dy, dz int, v float64) {
					X, Y, Z := x+dx, y+dy, z+dz
					if X < 0 || X >= nx || Y < 0 || Y >= ny || Z < 0 || Z >= nz {
						return
					}
					_ = c.Append(row, (Z*ny+Y)*nx+X, v)
				})
			}
		}
	}
	return c.ToCSR()
}

// Laplacian1D is the n x n tridiagonal matrix tridiag(-1, 2, -1).
func Laplacian1D(n int) *sparse.CSR[float64] { return Laplacian3D(n, 1, 1) }

// Laplacian2D is the 5-point Laplacian on an nx x ny grid (Dirichlet).
func Laplacian2D(nx, ny int) *sparse.CSR[float64] { return Laplacian3D(nx, ny, 1) }

// Laplacian3D is the 7-point Laplacian on an nx x ny x nz grid (Dirichlet).
func Laplacian3D(nx, ny, nz int) *sparse.CSR[float64] {
	dims := 0
	for _, d := range []int{nx, ny, nz} {
		if d > 1 {
			dims++
		}
	}
	diag := float64(2 * max(dims, 1))
	return grid(nx, ny, nz, func(emit func(int, int, int, float64)) {
		emit(0, 0, 0, diag)
		if nx > 1 {
			emit(-1, 0, 0, -1)
			emit(1, 0, 0, -1)
		}
		if ny > 1 {
			emit(0, -1, 0, -1)
			emit(0, 1, 0, -1)
		}
		if nz > 1 {
			emit(0, 0, -1, -1)
			emit(0, 0, 1, -1)
		}
	})
}

// ConvectionDiffusion2D is the nonsymmetric upwind 5-point operator
// -Δu + peclet*(u_x + u_y) on an nx x ny grid with unit spacing.
func ConvectionDiffusion2D(nx, ny int, peclet float64) *sparse.CSR[float64] {
	return grid(nx, ny, 1, func(emit func(int, int, int, float64)) {
		emit(0, 0, 0, 4+2*peclet)
		emit(-1, 0, 0, -1-peclet)
		emit(1, 0, 0, -1)
		emit(0, -1, 0, -1-peclet)
		emit(0, 1, 0, -1)
	})
}

// RandomSparse is an n x n matrix with about perRow random off-diagonal
// entries per row (uniform in [-1, 1]) and a diagonal making it strictly
// diagonally dominant; sym mirrors the pattern and values (symmetric
// positive definite then).
func RandomSparse(n, perRow int, sym bool, seed uint64) *sparse.CSR[float64] {
	r := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	c := sparse.NewCOO[float64](n, n)
	rowsum := make([]float64, n)
	for i := 0; i < n; i++ {
		for k := 0; k < perRow; k++ {
			j := r.IntN(n)
			if j == i {
				continue
			}
			v := 2*r.Float64() - 1
			_ = c.Append(i, j, v)
			rowsum[i] += abs(v)
			if sym {
				_ = c.Append(j, i, v)
				rowsum[j] += abs(v)
			}
		}
	}
	for i := 0; i < n; i++ {
		_ = c.Append(i, i, rowsum[i]+1+r.Float64())
	}
	return c.ToCSR()
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Scramble returns A with rows and columns symmetrically permuted by a
// seeded random permutation, and the permutation (B[i,j] = A[p[i],p[j]]).
func Scramble(a *sparse.CSR[float64], seed uint64) (*sparse.CSR[float64], []int) {
	r := rand.New(rand.NewPCG(seed, 7))
	p := r.Perm(a.Rows)
	b, _ := a.Permute(p, p) // p matches a's order, so Permute cannot fail
	return b, p
}
