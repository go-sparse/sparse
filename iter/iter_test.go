package iter

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
)

func rhs(n int, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 1))
	b := make([]float64, n)
	for i := range b {
		b[i] = r.NormFloat64()
	}
	return b
}

type solver func(Operator, []float64, *Options) (Result, error)

var solvers = map[string]solver{"cg": CG, "bicgstab": BiCGSTAB, "gmres": GMRES, "minres": MINRES}

func TestSPDAllSolvers(t *testing.T) {
	a := gallery.Laplacian2D(12, 12)
	b := rhs(a.Rows, 1)
	jac, _ := NewJacobi(a)
	ssor, _ := NewSSOR(a, 1.2)
	ic, err := NewIC0(a)
	if err != nil {
		t.Fatal(err)
	}
	ilu, _ := NewILU0(a)
	ilut, _ := NewILUT(a, 1e-3, 20)
	pcs := map[string]Preconditioner{"none": nil, "jacobi": jac, "ssor": ssor, "ic0": ic, "ilu0": ilu, "ilut": ilut}
	for sn, s := range solvers {
		for pn, pc := range pcs {
			calls := 0
			res, err := s(a, b, &Options{RTol: 1e-10, M: pc, Restart: 30, Callback: func([]float64) { calls++ }})
			if err != nil {
				t.Fatal(err)
			}
			if res.Info != 0 {
				t.Errorf("%s/%s: info %d after %d", sn, pn, res.Info, res.Iterations)
			}
			if r := TrueResidual(a, b, res.X); r > 1e-8 {
				t.Errorf("%s/%s: residual %g", sn, pn, r)
			}
			if calls == 0 {
				t.Errorf("%s/%s: callback never called", sn, pn)
			}
		}
	}
}

func TestPreconditionersHelp(t *testing.T) {
	a := gallery.Laplacian2D(20, 20)
	b := rhs(a.Rows, 2)
	plain, _ := CG(a, b, &Options{RTol: 1e-8})
	ic, _ := NewIC0(a)
	pre, _ := CG(a, b, &Options{RTol: 1e-8, M: ic})
	if pre.Iterations >= plain.Iterations {
		t.Fatalf("IC0 did not reduce iterations: %d vs %d", pre.Iterations, plain.Iterations)
	}
}

func TestNonsymmetric(t *testing.T) {
	a := gallery.ConvectionDiffusion2D(15, 15, 2)
	b := rhs(a.Rows, 3)
	ilu, _ := NewILU0(a)
	ilut, _ := NewILUT(a, 1e-2, 10)
	for name, pc := range map[string]Preconditioner{"none": nil, "ilu0": ilu, "ilut": ilut} {
		for sn, s := range map[string]solver{"bicgstab": BiCGSTAB, "gmres": GMRES} {
			res, err := s(a, b, &Options{RTol: 1e-9, M: pc, Restart: 40})
			if err != nil || res.Info != 0 || TrueResidual(a, b, res.X) > 1e-7 {
				t.Errorf("%s/%s: %v info=%d res=%g", sn, name, err, res.Info, TrueResidual(a, b, res.X))
			}
		}
	}
}

func TestILU0ExactOnTridiagonal(t *testing.T) {
	a := gallery.Laplacian1D(50)
	b := rhs(50, 4)
	ilu, _ := NewILU0(a)
	x := make([]float64, 50)
	ilu.Solve(x, b)
	if TrueResidual(a, b, x) > 1e-12 {
		t.Fatal("ILU0 of a tridiagonal matrix must be exact")
	}
	ic, _ := NewIC0(a)
	ic.Solve(x, b)
	if TrueResidual(a, b, x) > 1e-12 {
		t.Fatal("IC0 of a tridiagonal matrix must be exact")
	}
	lt, _ := NewILUT(a, 0, 0)
	lt.Solve(x, b)
	if TrueResidual(a, b, x) > 1e-12 {
		t.Fatal("ILUT without dropping of a tridiagonal matrix must be exact")
	}
	if ilu.L().NNZ() != 49 || ilu.U().NNZ() != 99 || ic.L().NNZ() != 99 {
		t.Fatal("factor shapes")
	}
}

func TestILUTExactWhenNoDropping(t *testing.T) {
	a := gallery.ConvectionDiffusion2D(6, 6, 1)
	b := rhs(36, 5)
	f, err := NewILUT(a, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, 36)
	f.Solve(x, b)
	if TrueResidual(a, b, x) > 1e-12 {
		t.Fatal("complete LU must solve exactly")
	}
	// lfil caps the fill
	g, _ := NewILUT(a, 0, 1)
	if g.L().NNZ() > 36 || g.U().NNZ() > 36*2 {
		t.Fatal("lfil not honoured")
	}
}

func TestMINRESIndefinite(t *testing.T) {
	a := gallery.Laplacian1D(60)
	b := rhs(60, 6)
	res, err := MINRES(a, b, &Options{RTol: 1e-10, Shift: 0.5, MaxIter: 1000})
	if err != nil || res.Info != 0 {
		t.Fatal(err, res.Info)
	}
	shifted := sparse.Shifted(a, -0.5)
	if TrueResidual(shifted, b, res.X) > 1e-8 {
		t.Fatal("residual")
	}
	// initial guess
	res2, _ := MINRES(a, b, &Options{RTol: 1e-10, X0: res.X, Shift: 0.5})
	if res2.Iterations > 2 {
		t.Fatalf("restart from the solution took %d iterations", res2.Iterations)
	}
}

func TestZeroRHSAndX0(t *testing.T) {
	a := gallery.Laplacian1D(5)
	z := make([]float64, 5)
	for name, s := range solvers {
		res, err := s(a, z, nil)
		if err != nil || res.Info != 0 || nrm2(res.X) != 0 {
			t.Errorf("%s zero rhs", name)
		}
	}
	// x0 already the solution: converged without iterating
	b := rhs(5, 7)
	sol, _ := CG(a, b, &Options{RTol: -1, ATol: 1e-12, MaxIter: 100})
	for name, s := range map[string]solver{"cg": CG, "bicgstab": BiCGSTAB, "gmres": GMRES} {
		res, _ := s(a, b, &Options{RTol: -1, ATol: 1e-9, X0: sol.X})
		if res.Info != 0 || res.Iterations != 0 {
			t.Errorf("%s from the solution: %+v", name, res)
		}
	}
	x0 := []float64{1, 1, 1, 1, 1}
	res, _ := CG(a, b, &Options{X0: x0, RTol: 1e-10})
	if x0[0] != 1 || TrueResidual(a, b, res.X) > 1e-8 {
		t.Fatal("x0 must be copied and used")
	}
}

func TestMaxIter(t *testing.T) {
	a := gallery.Laplacian2D(10, 10)
	b := rhs(100, 8)
	for name, s := range map[string]solver{"cg": CG, "bicgstab": BiCGSTAB, "gmres": GMRES, "minres": MINRES} {
		res, _ := s(a, b, &Options{MaxIter: 2, RTol: 1e-14})
		if res.Info != 2 {
			t.Errorf("%s: info %d, want 2", name, res.Info)
		}
	}
}

func TestShapeErrors(t *testing.T) {
	rect, _ := sparse.FromDense(2, 3, []float64{1, 0, 0, 0, 1, 0})
	sq := sparse.Eye[float64](2)
	for name, s := range solvers {
		if _, err := s(rect, []float64{1, 1}, nil); !errors.Is(err, ErrShape) {
			t.Errorf("%s rect", name)
		}
		if _, err := s(sq, []float64{1}, nil); !errors.Is(err, ErrShape) {
			t.Errorf("%s rhs", name)
		}
		if _, err := s(sq, []float64{1, 1}, &Options{X0: []float64{1}}); !errors.Is(err, ErrShape) {
			t.Errorf("%s x0", name)
		}
	}
	for _, f := range []func() error{
		func() error { _, err := NewJacobi(rect); return err },
		func() error { _, err := NewSSOR(rect, 1); return err },
		func() error { _, err := NewIC0(rect); return err },
		func() error { _, err := NewILU0(rect); return err },
		func() error { _, err := NewILUT(rect, 0, 0); return err },
	} {
		if err := f(); !errors.Is(err, ErrShape) {
			t.Errorf("rect precond: %v", err)
		}
	}
}

func TestBreakdowns(t *testing.T) {
	zeroDiag, _ := sparse.FromDense(2, 2, []float64{0, 1, 1, 0})
	if _, err := NewJacobi(zeroDiag); !errors.Is(err, ErrBreakdown) {
		t.Fatal("jacobi")
	}
	if _, err := NewSSOR(zeroDiag, 1); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ssor diag")
	}
	if _, err := NewSSOR(sparse.Eye[float64](2), 2); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ssor omega")
	}
	if _, err := NewIC0(zeroDiag); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ic0 missing diagonal")
	}
	if _, err := NewILU0(zeroDiag); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ilu0 missing diagonal")
	}
	if _, err := NewILUT(zeroDiag, 0, 0); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ilut zero pivot")
	}
	indef, _ := sparse.FromDense(2, 2, []float64{1, 2, 2, 1})
	if _, err := NewIC0(indef); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ic0 indefinite")
	}
	empty, _ := sparse.FromDense(2, 2, []float64{1, 0, 0, 0})
	empty.Indptr, empty.Indices, empty.Data = []int{0, 1, 1}, []int32{0}, []float64{1}
	if _, err := NewIC0(empty); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ic0 empty row")
	}
	sing, _ := sparse.FromDense(2, 2, []float64{1, 1, 1, 1})
	if _, err := NewILU0(sing); !errors.Is(err, ErrBreakdown) {
		t.Fatal("ilu0 zero pivot")
	}
	// indefinite preconditioner and matrix for MINRES
	neg := negPrec{}
	if _, err := MINRES(sparse.Eye[float64](2), []float64{1, 1}, &Options{M: neg}); !errors.Is(err, ErrBreakdown) {
		t.Fatal("minres indefinite M")
	}
	calls := 0
	flip := &flipPrec{n: &calls}
	if _, err := MINRES(gallery.Laplacian1D(3), []float64{1, 2, 3}, &Options{M: flip}); !errors.Is(err, ErrBreakdown) {
		t.Fatal("minres beta<0")
	}
}

type negPrec struct{}

func (negPrec) Solve(dst, r []float64) {
	for i := range r {
		dst[i] = -r[i]
	}
}

// flipPrec is positive on its first call and negative afterwards.
type flipPrec struct{ n *int }

func (f *flipPrec) Solve(dst, r []float64) {
	*f.n++
	s := 1.0
	if *f.n > 1 {
		s = -1
	}
	for i := range r {
		dst[i] = s * r[i]
	}
}

func TestBiCGSTABBreakdowns(t *testing.T) {
	skew, _ := sparse.FromDense(2, 2, []float64{0, 1, -1, 0})
	res, _ := BiCGSTAB(skew, []float64{1, 0}, nil)
	if res.Info != BreakdownOmega {
		t.Fatalf("rv == 0: info %d", res.Info)
	}
	// scripted operator: second product orthogonal to s gives omega = 0 and
	// then (r~, r) = 0
	c0 := 0
	rop := sparse.FuncOperator{Rows: 2, Cols: 2, Mv: func(dst, x []float64) {
		c0++
		if c0 == 1 {
			dst[0], dst[1] = 2, 0
		} else {
			dst[0], dst[1] = 1, 1
		}
	}}
	res, _ = BiCGSTAB(rop, []float64{1, 1}, &Options{RTol: -1, MaxIter: 5})
	if res.Info != BreakdownRho {
		t.Fatalf("rho: info %d", res.Info)
	}
	// omega breakdown needs an operator whose second product is nearly
	// orthogonal to s: a scripted operator.
	call := 0
	op := sparse.FuncOperator{Rows: 2, Cols: 2, Mv: func(dst, x []float64) {
		call++
		switch call {
		case 1:
			dst[0], dst[1] = 2, 0
		case 2:
			dst[0], dst[1] = 1e20, 1e20+16384
		default:
			dst[0], dst[1] = x[0], x[1]
		}
	}}
	res, _ = BiCGSTAB(op, []float64{1, 1}, &Options{RTol: -1, MaxIter: 5})
	if res.Info != BreakdownOmega {
		t.Fatalf("omega: info %d", res.Info)
	}
}

func TestGMRESPaths(t *testing.T) {
	// identity: lucky breakdown
	res, _ := GMRES(sparse.Eye[float64](5), []float64{1, 2, 3, 4, 5}, &Options{})
	if res.Info != 0 || res.Iterations != 1 {
		t.Fatalf("%+v", res)
	}
	// singular zero matrix: h[col][col] == 0
	zero, _ := sparse.FromDense(2, 2, []float64{0, 0, 0, 0})
	zero.Indptr = []int{0, 0, 0}
	zero.Indices, zero.Data = nil, nil
	res, _ = GMRES(zero, []float64{1, 1}, &Options{MaxIter: 3})
	if res.Info == 0 {
		t.Fatal("must not converge")
	}
	// restart shorter than the Krylov dimension needed, many cycles
	a := gallery.Laplacian1D(40)
	b := rhs(40, 9)
	res, _ = GMRES(a, b, &Options{Restart: 3, RTol: 1e-8, MaxIter: 2000})
	if res.Info != 0 || TrueResidual(a, b, res.X) > 1e-6 {
		t.Fatalf("restarted: %+v", res)
	}
	// restart larger than n is clipped
	res, _ = GMRES(a, b, &Options{Restart: 500, RTol: 1e-10})
	if res.Info != 0 {
		t.Fatal("clipped restart")
	}
	// breakdown that does not satisfy the true tolerance
	res, _ = GMRES(a, b, &Options{RTol: -1, ATol: 1e-30, Restart: 40, MaxIter: 4})
	_ = res
}

func TestMINRESPaths(t *testing.T) {
	// A = I: beta/beta1 <= 10 eps at the first step (istop = -1)
	res, err := MINRES(sparse.Eye[float64](4), []float64{1, 2, 3, 4}, nil)
	if err != nil || res.Info != 0 || TrueResidual(sparse.Eye[float64](4), []float64{1, 2, 3, 4}, res.X) > 1e-12 {
		t.Fatalf("%+v %v", res, err)
	}
	// zero rhs with a nonzero guess: the answer is zero
	z, _ := MINRES(sparse.Eye[float64](2), []float64{0, 0}, &Options{X0: []float64{1, 1}})
	if nrm2(z.X) != 0 {
		t.Fatal("zero rhs")
	}
	// singular, inconsistent: least-squares test
	sing, _ := sparse.Diags([][]float64{{1, 0}}, []int{0}, 2, 2)
	sres, _ := MINRES(sing, []float64{1, 1}, nil)
	if sres.Info != 0 || math.IsNaN(sres.X[0]) {
		t.Fatalf("%+v", sres)
	}
	// extreme ill-conditioning exercises the condition and eps tests
	ill, _ := sparse.Diags([][]float64{{1, 1e-17}}, []int{0}, 2, 2)
	res, _ = MINRES(ill, []float64{1, 1}, &Options{RTol: -1, MaxIter: 50})
	_ = res
	// tolerance below machine precision: the 1 + test <= 1 tests fire
	a := gallery.Laplacian1D(20)
	res, _ = MINRES(a, rhs(20, 10), &Options{RTol: -1, MaxIter: 500})
	if res.Info != 0 && res.Info != 500 {
		t.Fatal("info")
	}
}

func lsqrRef(a *sparse.CSR[float64], b []float64) []float64 {
	// normal equations by CG on AᵀA
	n := a.Cols
	op := sparse.FuncOperator{Rows: n, Cols: n, Mv: func(dst, x []float64) {
		t := make([]float64, a.Rows)
		a.MulVec(t, x)
		a.MulVecT(dst, t)
	}}
	atb := make([]float64, n)
	a.MulVecT(atb, b)
	r, _ := CG(op, atb, &Options{RTol: 1e-14, MaxIter: 10 * n})
	return r.X
}

func TestLSQR(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	c := sparse.NewCOO[float64](40, 15)
	for i := 0; i < 40; i++ {
		for k := 0; k < 4; k++ {
			_ = c.Append(i, r.IntN(15), r.NormFloat64())
		}
	}
	for j := 0; j < 15; j++ {
		_ = c.Append(j, j, 3)
	}
	a := c.ToCSR()
	b := rhs(40, 13)
	want := lsqrRef(a, b)
	res, err := LSQR(a, b, &LSQROptions{ATol: 1e-12, BTol: 1e-12, CalcVar: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(res.X[i]-want[i]) > 1e-7 {
			t.Fatalf("x[%d] = %g, want %g (istop %d)", i, res.X[i], want[i], res.Istop)
		}
	}
	if res.Istop == 0 || res.Itn == 0 || len(res.Var) != 15 || res.ACond <= 0 || res.XNorm <= 0 {
		t.Fatalf("%+v", res)
	}
	tmp := make([]float64, 40)
	a.MulVec(tmp, res.X)
	axpy(-1, b, tmp)
	if math.Abs(nrm2(tmp)-res.R1Norm) > 1e-8 {
		t.Fatal("r1norm")
	}
	// damping: min ||b-Ax||² + d²||x||² via the augmented normal equations
	dres, _ := LSQR(a, b, &LSQROptions{Damp: 0.5, ATol: 1e-12, BTol: 1e-12})
	if nrm2(dres.X) >= nrm2(res.X) {
		t.Fatal("damping must shrink the solution")
	}
	// warm start
	w, _ := LSQR(a, b, &LSQROptions{X0: res.X, ATol: 1e-12, BTol: 1e-12})
	if w.Itn > 3 {
		t.Fatalf("warm start took %d iterations", w.Itn)
	}
	// iteration limit and condition limit and tiny tolerances
	l, _ := LSQR(a, b, &LSQROptions{IterLim: 1})
	if l.Istop != 7 {
		t.Fatalf("istop %d", l.Istop)
	}
	cl, _ := LSQR(a, b, &LSQROptions{ConLim: 1.0001, ATol: -1, BTol: -1})
	if cl.Istop != 3 {
		t.Fatalf("conlim istop %d", cl.Istop)
	}
	tiny, _ := LSQR(a, b, &LSQROptions{ATol: -1, BTol: -1, ConLim: -1, IterLim: 500})
	if tiny.Istop == 0 || tiny.Istop == 7 {
		t.Fatalf("tiny tolerances istop %d", tiny.Istop)
	}
}

func TestLSQREdgeCases(t *testing.T) {
	one, _ := sparse.FromDense(1, 1, []float64{2})
	res, _ := LSQR(one, []float64{3}, nil)
	if math.Abs(res.X[0]-1.5) > 1e-12 {
		t.Fatalf("%v", res.X)
	}
	zero := &sparse.CSR[float64]{Rows: 2, Cols: 2, Indptr: []int{0, 0, 0}}
	res, _ = LSQR(zero, []float64{1, 1}, nil)
	if res.Itn != 0 || res.X[0] != 0 {
		t.Fatal("zero matrix")
	}
	res, _ = LSQR(one, []float64{0}, nil)
	if res.Itn != 0 {
		t.Fatal("zero rhs")
	}
	if _, err := LSQR(one, []float64{1, 2}, nil); !errors.Is(err, ErrShape) {
		t.Fatal("shape")
	}
	if _, err := LSQR(one, []float64{1}, &LSQROptions{X0: []float64{1, 2}}); !errors.Is(err, ErrShape) {
		t.Fatal("x0 shape")
	}
	// ill-conditioned: condition-number stops
	ill, _ := sparse.Diags([][]float64{{1, 1e-9, 1e-18}}, []int{0}, 3, 3)
	r1, _ := LSQR(ill, []float64{1, 1, 1}, &LSQROptions{ATol: -1, BTol: -1, ConLim: -1, IterLim: 100})
	if r1.Istop == 0 {
		t.Fatal("must stop")
	}
	// symOrtho branches
	for _, ab := range [][2]float64{{3, 0}, {0, 4}, {1, 5}, {5, 1}, {-3, 0}, {0, -2}} {
		c, s, r := symOrtho(ab[0], ab[1])
		if math.Abs(c*ab[0]+s*ab[1]-r) > 1e-12 || math.Abs(-s*ab[0]+c*ab[1]) > 1e-12 {
			t.Fatalf("symOrtho %v", ab)
		}
	}
}

func TestHelpers(t *testing.T) {
	if got := nrm2([]float64{3, 4}); got != 5 {
		t.Fatal(got)
	}
	if nrm2([]float64{1e200, 1e200}) < 1e200 || nrm2([]float64{0, 0}) != 0 || nrm2([]float64{1, 1e5}) < 1e5 {
		t.Fatal("scaled norm")
	}
	if TrueResidual(sparse.Eye[float64](2), []float64{0, 0}, []float64{3, 4}) != 5 {
		t.Fatal("residual with zero rhs")
	}
	o := &Options{RTol: -1}
	if o.rtol() != 0 || (&Options{RTol: 0.5}).rtol() != 0.5 || (&Options{}).rtol() != 1e-5 {
		t.Fatal("rtol")
	}
	if def(0, 7) != 7 || def(-1, 7) != 0 || def(2, 7) != 2 {
		t.Fatal("def")
	}
	c, s, r := givens(3, 0)
	if c != 1 || s != 0 || r != 3 {
		t.Fatal("givens")
	}
	jac, _ := NewJacobi(sparse.Eye[float64](3))
	out := make([]float64, 3)
	jac.Solve(out, []float64{1, 2, 3})
	if out[2] != 3 {
		t.Fatal("jacobi")
	}
}

// On the sparsity pattern of A, the incomplete factors reproduce A exactly:
// that is the defining property of IC(0) and ILU(0).
func TestIncompleteFactorsMatchOnPattern(t *testing.T) {
	a := gallery.RandomSparse(60, 5, true, 21)
	ic, err := NewIC0(a)
	if err != nil {
		t.Fatal(err)
	}
	llt, _ := sparse.Mul(ic.L(), ic.L().T())
	ilu, err := NewILU0(a)
	if err != nil {
		t.Fatal(err)
	}
	// unit lower factor = strict L + I
	l1, _ := sparse.Add(ilu.L(), sparse.Eye[float64](60))
	lu, _ := sparse.Mul(l1, ilu.U())
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			j := int(a.Indices[p])
			if math.Abs(lu.At(i, j)-a.At(i, j)) > 1e-12 {
				t.Fatalf("ILU0 differs from A at (%d,%d)", i, j)
			}
			if j <= i && math.Abs(llt.At(i, j)-a.At(i, j)) > 1e-12 {
				t.Fatalf("IC0 differs from A at (%d,%d)", i, j)
			}
		}
	}
}

func TestLSMR(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	c := sparse.NewCOO[float64](40, 15)
	for i := 0; i < 40; i++ {
		for k := 0; k < 4; k++ {
			_ = c.Append(i, r.IntN(15), r.NormFloat64())
		}
	}
	for j := 0; j < 15; j++ {
		_ = c.Append(j, j, 3)
	}
	a := c.ToCSR()
	b := rhs(40, 13)
	want := lsqrRef(a, b)
	res, err := LSMR(a, b, &LSMROptions{ATol: 1e-12, BTol: 1e-12, MaxIter: 100})
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(res.X[i]-want[i]) > 1e-7 {
			t.Fatalf("x[%d] = %g, want %g (istop %d)", i, res.X[i], want[i], res.Istop)
		}
	}
	tmp := make([]float64, 40)
	a.MulVec(tmp, res.X)
	axpy(-1, b, tmp)
	if math.Abs(nrm2(tmp)-res.NormR) > 1e-8 || res.NormA <= 0 || res.CondA < 1 || res.NormX <= 0 || res.NormAR < 0 {
		t.Fatalf("%+v", res)
	}
	d, _ := LSMR(a, b, &LSMROptions{Damp: 0.5, ATol: 1e-12, BTol: 1e-12, MaxIter: 100})
	if nrm2(d.X) >= nrm2(res.X) {
		t.Fatal("damping must shrink the solution")
	}
	w, _ := LSMR(a, b, &LSMROptions{X0: res.X, ATol: 1e-12, BTol: 1e-12, MaxIter: 100})
	if w.Itn > 3 {
		t.Fatalf("warm start took %d iterations", w.Itn)
	}
	def, _ := LSMR(a, b, nil)
	if def.Istop == 0 || def.Itn > 15 {
		t.Fatalf("default call: %+v", def)
	}
	if l, _ := LSMR(a, b, &LSMROptions{MaxIter: 1}); l.Istop != 7 {
		t.Fatalf("istop %d", l.Istop)
	}
	if cl, _ := LSMR(a, b, &LSMROptions{ConLim: 1.0001, ATol: -1, BTol: -1, MaxIter: 100}); cl.Istop != 3 {
		t.Fatalf("conlim istop %d", cl.Istop)
	}
	if tiny, _ := LSMR(a, b, &LSMROptions{ATol: -1, BTol: -1, ConLim: -1, MaxIter: 500}); tiny.Istop == 0 || tiny.Istop == 7 {
		t.Fatalf("tiny tolerances istop %d", tiny.Istop)
	}
}

func TestLSMREdgeCases(t *testing.T) {
	one, _ := sparse.FromDense(1, 1, []float64{2})
	res, _ := LSMR(one, []float64{3}, nil)
	if math.Abs(res.X[0]-1.5) > 1e-12 {
		t.Fatalf("%v", res.X)
	}
	zero := &sparse.CSR[float64]{Rows: 2, Cols: 2, Indptr: []int{0, 0, 0}}
	if r, _ := LSMR(zero, []float64{1, 1}, nil); r.Itn != 0 {
		t.Fatal("zero matrix")
	}
	if r, _ := LSMR(one, []float64{0}, nil); r.Itn != 0 {
		t.Fatal("zero rhs")
	}
	if _, err := LSMR(one, []float64{1, 2}, nil); !errors.Is(err, ErrShape) {
		t.Fatal("shape")
	}
	if _, err := LSMR(one, []float64{1}, &LSMROptions{X0: []float64{1, 2}}); !errors.Is(err, ErrShape) {
		t.Fatal("x0 shape")
	}
	// beta = 0 inside the loop and a rank-deficient operator
	ill, _ := sparse.Diags([][]float64{{1, 1e-9, 1e-18}}, []int{0}, 3, 3)
	if r, _ := LSMR(ill, []float64{1, 1, 1}, &LSMROptions{ATol: -1, BTol: -1, ConLim: -1, MaxIter: 100}); r.Istop == 0 {
		t.Fatal("must stop")
	}
	// a scripted operator whose bidiagonalisation ends (u vanishes) at once
	id := sparse.Eye[float64](3)
	if r, _ := LSMR(id, []float64{1, 2, 3}, &LSMROptions{MaxIter: 10}); r.Itn == 0 || nrm2(r.X) == 0 {
		t.Fatal("identity")
	}
}
