package bench

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-scikit/refbench"
	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/ldl"
)

// The contract of a solve is a stated backward-error bound, not equality with
// another float64 code (bit identity is a porting step).
//
// LDLᵀ is the LU factorisation with U = D Lᵀ. Higham, Accuracy and Stability
// of Numerical Algorithms, 2nd ed., Theorem 9.4 (LU solve: two triangular
// solves, Theorem 8.5, after the factorisation, Theorem 9.3) gives, for the
// computed solution x̂ of A x = b,
//
//	(A + ΔA) x̂ = b,   |ΔA| ≤ γ_{3n} |L̂| |D̂| |L̂ᵀ|,   γ_k = k u / (1 − k u),
//
// u = 2⁻⁵³, so the residual r = b − A x̂ = ΔA x̂ satisfies, componentwise,
//
//	|r| ≤ γ_{3n} |L̂| |D̂| |L̂ᵀ| |x̂|.                                   (1)
//
// The forward error follows from x̂ − x = A⁻¹ r:
//
//	‖x̂ − x‖∞ ≤ ‖A⁻¹‖∞ γ_{3n} ‖ |L̂||D̂||L̂ᵀ||x̂| ‖∞.                    (2)
//
// n is the order of A (sharper counts, the row length of L, are not needed).
// The residual and the exact solution x are computed in 300-bit arithmetic
// (mixed-precision refinement with the float64 solve, Higham ch. 12, converges
// to working precision of the residual because κ(A) u < 1); ‖A⁻¹‖∞ is formed
// from the N columns of A⁻¹ (A is symmetric). QDLDL's output is held to the
// same bound, using this factorisation's |L||D||Lᵀ|: QDLDL runs the same
// algorithm, so its factors differ from ours by rounding only, and the
// slack γ_{3n} is orders of magnitude above that.

const bigPrec = 300

// bigResidual returns b − A x in big arithmetic; A is symmetric, a holds its
// upper triangle.
func bigResidual(a *sparse.CSC[float64], b []float64, x []*big.Float) []*big.Float {
	r := make([]*big.Float, len(b))
	for i, v := range b {
		r[i] = new(big.Float).SetPrec(bigPrec).SetFloat64(v)
	}
	t := new(big.Float).SetPrec(bigPrec)
	for j := 0; j < a.Cols; j++ {
		for p := a.Indptr[j]; p < a.Indptr[j+1]; p++ {
			i := int(a.Indices[p])
			v := new(big.Float).SetPrec(bigPrec).SetFloat64(a.Data[p])
			r[i].Sub(r[i], t.Mul(v, x[j]))
			if i != j {
				r[j].Sub(r[j], t.Mul(v, x[i]))
			}
		}
	}
	return r
}

// oracleSolve is A⁻¹ b to well beyond float64, by refinement with the float64
// solve.
func oracleSolve(t *testing.T, a *sparse.CSC[float64], b []float64, solve func([]float64) error) []*big.Float {
	t.Helper()
	x := make([]*big.Float, len(b))
	for i := range x {
		x[i] = new(big.Float).SetPrec(bigPrec)
	}
	var r []*big.Float
	for it := 0; it < 12; it++ {
		r = bigResidual(a, b, x)
		d := make([]float64, len(b))
		for i, v := range r {
			d[i], _ = v.Float64()
		}
		if err := solve(d); err != nil {
			t.Fatal(err)
		}
		for i := range x {
			x[i].Add(x[i], new(big.Float).SetPrec(bigPrec).SetFloat64(d[i]))
		}
	}
	// The oracle is only an oracle if its own residual is far below u‖b‖.
	r = bigResidual(a, b, x)
	var rmax, bmax float64
	for i := range r {
		v, _ := r[i].Float64()
		rmax, bmax = math.Max(rmax, math.Abs(v)), math.Max(bmax, math.Abs(b[i]))
	}
	if rmax > 1e-30*bmax {
		t.Fatalf("oracle did not converge: ‖r‖∞ = %g, ‖b‖∞ = %g", rmax, bmax)
	}
	return x
}

// invNormInf is ‖A⁻¹‖∞ from the columns of A⁻¹ (A symmetric: rows = columns).
func invNormInf(t *testing.T, n int, solve func([]float64) error) float64 {
	t.Helper()
	rows := make([]float64, n)
	e := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := range e {
			e[i] = 0
		}
		e[j] = 1
		if err := solve(e); err != nil {
			t.Fatal(err)
		}
		for i, v := range e {
			rows[i] += math.Abs(v)
		}
	}
	m := 0.0
	for _, v := range rows {
		m = math.Max(m, v)
	}
	return m
}

// growth is |L||D||Lᵀ||x| for the unit lower factor with strict part l.
func growth(l *sparse.CSC[float64], d, x []float64) []float64 {
	n := len(x)
	u := make([]float64, n) // |Lᵀ||x|: row i of Lᵀ is column i of L
	for i := 0; i < n; i++ {
		s := math.Abs(x[i])
		for p := l.Indptr[i]; p < l.Indptr[i+1]; p++ {
			s += math.Abs(l.Data[p]) * math.Abs(x[l.Indices[p]])
		}
		u[i] = s * math.Abs(d[i])
	}
	w := append([]float64(nil), u...) // |L| (|D||Lᵀ||x|)
	for j := 0; j < n; j++ {
		for p := l.Indptr[j]; p < l.Indptr[j+1]; p++ {
			w[l.Indices[p]] += math.Abs(l.Data[p]) * u[j]
		}
	}
	return w
}

// checkBound asserts (1) and (2) for the computed solution xh of a x = b.
func checkBound(t *testing.T, what string, a *sparse.CSC[float64], f *ldl.Factor, b, xh []float64, xstar []*big.Float, ainv float64) {
	t.Helper()
	n := len(b)
	const u = 0x1p-53
	gam := 3 * float64(n) * u / (1 - 3*float64(n)*u)
	w := growth(f.L(), f.D(), xh)
	bx := make([]*big.Float, n)
	for i := range bx {
		bx[i] = new(big.Float).SetPrec(bigPrec).SetFloat64(xh[i])
	}
	r := bigResidual(a, b, bx)
	wmax, rmax, slack := 0.0, 0.0, 0.0
	for i := range r {
		ri, _ := r[i].Float64()
		rmax = math.Max(rmax, math.Abs(ri))
		wmax = math.Max(wmax, w[i])
		if bound := gam * w[i]; math.Abs(ri) > bound {
			t.Errorf("%s: |r_%d| = %g exceeds γ_3n|L||D||Lᵀ||x̂| = %g", what, i, math.Abs(ri), bound)
		} else if bound > 0 {
			slack = math.Max(slack, math.Abs(ri)/bound)
		}
	}
	var ferr float64
	for i := range xh {
		d := new(big.Float).SetPrec(bigPrec).Sub(bx[i], xstar[i])
		v, _ := d.Float64()
		ferr = math.Max(ferr, math.Abs(v))
	}
	if fb := ainv * gam * wmax; ferr > fb {
		t.Errorf("%s: ‖x̂−x‖∞ = %g exceeds ‖A⁻¹‖∞γ_3n‖|L||D||Lᵀ||x̂|‖∞ = %g", what, ferr, fb)
	}
	t.Logf("%s: n=%d ‖x̂−x‖∞=%.3g ‖r‖∞=%.3g bound(2)=%.3g, worst |r|/bound(1)=%.3g",
		what, n, ferr, rmax, ainv*gam*wmax, slack)
}

// TestParityOf: the full solves are validated against the bound, every other
// direct-path row bit for bit.
func TestParityOf(t *testing.T) {
	for kind, want := range map[string]string{
		"solve": refbench.ParityTolerance, "psolve": refbench.ParityTolerance,
		"etree": refbench.ParityExact, "factor": refbench.ParityExact, "lsolve": refbench.ParityExact,
		"ltsolve": refbench.ParityExact, "dsolve": refbench.ParityExact, "pfactor": refbench.ParityExact,
	} {
		if got := parityOf(kind); got != want {
			t.Errorf("parityOf(%q) = %q, want %q", kind, got, want)
		}
	}
}

// TestParityWithQDLDL holds both this package and QDLDL to the backward- and
// forward-error bounds, measured against an extended-precision solution.
func TestParityWithQDLDL(t *testing.T) {
	for _, g := range []int{12, 20} {
		m, err := newKKT(g)
		if err != nil {
			t.Fatal(err)
		}
		N := m.k.Cols
		plain, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		permd, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, m.perm, m.pmap, m.k.Data)
		if err != nil {
			t.Fatal(err)
		}
		f, err := ldl.Factorize(m.pa)
		if err != nil {
			t.Fatal(err)
		}
		if f.NNZL() != plain.NNZL() {
			t.Fatalf("g=%d: nnz(L) %d, QDLDL %d", g, f.NNZL(), plain.NNZL())
		}
		// The system every path solves is pa x = b (pa = upper triangle of
		// P K Pᵀ); the permuted solvers answer K y = b, and y[perm[k]] = x[k]
		// for the permuted ordering, so compare in pa's ordering.
		xstar := oracleSolve(t, m.pa, m.b, f.Solve)
		ainv := invNormInf(t, N, f.Solve)

		x := append([]float64(nil), m.b...)
		if err := f.Solve(x); err != nil {
			t.Fatal(err)
		}
		checkBound(t, "Factor.Solve", m.pa, f, m.b, x, xstar, ainv)
		checkBound(t, "QDLDL Solve", m.pa, f, m.b, plain.Solve(N), xstar, ainv)

		pf, err := ldl.NewPermuted(m.k, m.perm)
		if err != nil {
			t.Fatal(err)
		}
		if err := pf.Refactor(m.k); err != nil {
			t.Fatal(err)
		}
		y := append([]float64(nil), m.b...)
		if err := pf.Solve(y); err != nil {
			t.Fatal(err)
		}
		yp, qp := make([]float64, N), make([]float64, N)
		qs := permd.PSolve(N)
		for k, v := range m.perm {
			yp[k], qp[k] = y[v], qs[v]
		}
		// b in the permuted ordering is what pa x = b' means; vec's b is
		// the right-hand side of K y = b, so permute it too.
		bp := make([]float64, N)
		for k, v := range m.perm {
			bp[k] = m.b[v]
		}
		xp := oracleSolve(t, m.pa, bp, f.Solve)
		checkBound(t, "Permuted.Solve", m.pa, f, bp, yp, xp, ainv)
		checkBound(t, "QDLDL PSolve", m.pa, f, bp, qp, xp, ainv)
	}
}
