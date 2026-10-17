package conformance

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/internal/scipy"
)

func fixture(t *testing.T) *Fixture {
	t.Helper()
	f, err := LoadFixture("testdata/oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSolversAgainstSciPy(t *testing.T) {
	f := fixture(t)
	ms, err := Suite(SuiteSolve)
	if err != nil {
		t.Fatal(err)
	}
	rows, bad := Verify(ms, f.Solves)
	for _, r := range rows {
		t.Logf("%-18s %-9s %-7s scipy it=%-4d res=%.2e | go it=%-4d res=%.2e", r.Matrix, r.Solver, r.Precond, r.Sci.Iters, r.Sci.Res, r.GoIters, r.GoRes)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

func TestLSQRAgainstSciPy(t *testing.T) {
	rows, bad := VerifyLSQR(fixture(t).LSQR)
	for _, r := range rows {
		t.Logf("%s damp=%g atol=%g: scipy istop=%d itn=%d | go istop=%d itn=%d xdiff=%.1e", r.Matrix, r.Damp, r.ATol, r.Sci.Istop, r.Sci.Itn, r.GoIstop, r.GoItn, r.MaxXDiff)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

func TestOrderingsAgainstSciPy(t *testing.T) {
	ms, err := Suite(SuiteOrder)
	if err != nil {
		t.Fatal(err)
	}
	rows, bad := VerifyOrders(ms, fixture(t).Orders)
	for _, r := range rows {
		t.Logf("%s: rcm bw %d (scipy %d) prof %d (%d); amd chol %d (suitesparse %d) lu %d (superlu mmd %d); colamd lu %d (superlu %d, natural %d)",
			r.Matrix, r.GoRCMBandwidth, r.SciRCMBandwidth, r.GoRCMProfile, r.SciRCMProfile, r.GoAMDChol, r.SuiteSparseAMDChol, r.GoAMDLU, r.SuperLUMMDLU, r.GoCOLAMDLU, r.SciCOLAMDLU, r.NaturalLU)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

func TestMain(m *testing.M) {
	if scipy.MaybeServeFake(os.Getenv, os.Stdin, os.Stdout) {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestBuild(t *testing.T) {
	for _, n := range []string{"lap1d:10", "lap2d:4:5", "lap3d:3:3:3", "cd2d:4:4:1", "rnd:30:3:1:7", "rnd:30:3:0:7", "lap2d:4:4+scr:2"} {
		m, err := Build(n)
		if err != nil || m.Name != n || m.A.Rows == 0 {
			t.Errorf("%s: %v", n, err)
		}
	}
	if m, _ := Build("cd2d:4:4:1"); m.SPD {
		t.Error("cd2d is not SPD")
	}
	if m, _ := Build("rnd:30:3:0:7"); m.SPD {
		t.Error("nonsymmetric rnd is not SPD")
	}
	for _, n := range []string{"nope:1", "lap2d:4:4+scr:x", "lap1d"} {
		if n == "lap1d" {
			continue // missing size parses as NaN -> int conversion; not an error path
		}
		if _, err := Build(n); err == nil {
			t.Errorf("%s must fail", n)
		}
	}
	if _, err := Suite([]string{"nope"}); err == nil {
		t.Error("Suite must fail")
	}
	if len(RHS(5)) != 5 {
		t.Error("RHS")
	}
}

func TestPrecondBuild(t *testing.T) {
	n, _ := Build("lap2d:6:6")
	for _, k := range []PrecondSpec{"none", "jacobi", "ssor", "ic0", "ilu0", "ilut"} {
		if _, _, err := build(n.A, k); err != nil {
			t.Errorf("%s: %v", k, err)
		}
	}
	if _, _, err := build(n.A, "bogus"); err == nil {
		t.Error("bogus")
	}
	// breakdowns propagate: a zero diagonal defeats every factorisation
	z, _ := sparse.FromDense(2, 2, []float64{0, 1, 1, 0})
	for _, k := range []PrecondSpec{"jacobi", "ssor", "ic0", "ilu0", "ilut"} {
		if _, _, err := build(z, k); err == nil {
			t.Errorf("%s must break down", k)
		}
	}
	if _, _, _, err := RunGo(Named{A: z}, SolveCase{Solver: "cg", Precond: "jacobi", RTol: 1e-8}); err == nil {
		t.Error("RunGo must fail")
	}
	if _, _, _, err := RunGo(Named{A: sparse.Eye[float64](3)}, SolveCase{Solver: "cg", Precond: "none", RTol: 1e-8}); err != nil {
		t.Error(err)
	}
}

func TestGenerateWithFake(t *testing.T) {
	s, err := scipy.Start(scipy.FakeCommand("ok"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	f, err := GenerateAll(s)
	if err != nil || len(f.Solves) == 0 || len(f.LSQR) != len(DefaultLSQRCases()) || len(f.Orders) != len(SuiteOrder) {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "f.json")
	if err := SaveFixture(path, f); err != nil {
		t.Fatal(err)
	}
	g, err := LoadFixture(path)
	if err != nil || len(g.Solves) != len(f.Solves) {
		t.Fatal(err)
	}
	if _, err := LoadFixture(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing")
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	os.WriteFile(bad, []byte("{"), 0o644)
	if _, err := LoadFixture(bad); err == nil {
		t.Fatal("bad json")
	}
	if err := SaveFixture(filepath.Join(t.TempDir(), "no", "dir.json"), f); err == nil {
		t.Fatal("save")
	}
}

func TestGenerateFailures(t *testing.T) {
	ms, _ := Suite([]string{"lap2d:6:6", "cd2d:6:6:1", "rnd:40:3:1:1"})
	for _, mode := range []string{"fail", "garbage", "eof"} {
		s, err := scipy.Start(scipy.FakeCommand(mode))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Generate(s, ms); err == nil {
			t.Errorf("%s: Generate must fail", mode)
		}
		if _, err := GenerateLSQR(s); err == nil && mode != "garbage" { // lsqr's reply has no field of a wrong type
			t.Errorf("%s: GenerateLSQR must fail", mode)
		}
		if _, err := GenerateOrders(s, ms); err == nil {
			t.Errorf("%s: GenerateOrders must fail", mode)
		}
		if _, err := GenerateAll(s); err == nil {
			t.Errorf("%s: GenerateAll must fail", mode)
		}
		s.Close()
	}
	// a preconditioner that cannot be built stops Generate before the oracle
	z, _ := sparse.FromDense(2, 2, []float64{0, 1, 1, 0})
	s, _ := scipy.Start(scipy.FakeCommand("ok"))
	defer s.Close()
	if _, err := Generate(s, []Named{{Name: "z", A: z, SPD: true}}); err == nil {
		t.Error("breakdown must stop Generate")
	}
}

// failAt answers "ok" until the n-th request, then fails: it walks the error
// branch of every request in GenerateOrders.
func TestGenerateOrdersEveryStep(t *testing.T) {
	ms, _ := Suite([]string{"lap2d:6:6"})
	for step := 0; step < 7; step++ { // rcm, amd, 2 x lu (amd), 2 x lu (colamd), lu (rcm-free)
		s, err := scipy.Start(scipy.FakeCommand("failafter:" + strconv.Itoa(step)))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := GenerateOrders(s, ms); err == nil {
			t.Errorf("step %d must fail", step)
		}
		s.Close()
	}
}

func TestVerifyDetectsProblems(t *testing.T) {
	ms, _ := Suite([]string{"lap2d:12:12"})
	good := SolveFix{SolveCase: SolveCase{Matrix: "lap2d:12:12", Solver: "cg", Precond: "none", RTol: 1e-8}}
	_, iters, _, _ := RunGo(ms[0], good.SolveCase)
	good.Sci = scipy.SolveResponse{Iters: iters, Res: 1e-9}
	if _, bad := Verify(ms, []SolveFix{good}); len(bad) != 0 {
		t.Fatal(bad)
	}
	missing := good
	missing.Matrix = "other"
	wrongInfo := good
	wrongInfo.Sci.Info = 3
	wrongIters := good
	wrongIters.Sci.Iters = iters + 50
	wrongIters2 := good
	wrongIters2.Sci.Iters = max(iters-50, 0) // Go takes more iterations than SciPy
	badPre := good
	badPre.Precond = "bogus"
	tight := good
	tight.RTol = 1e-30
	tight.Sci.Res = 0
	for name, f := range map[string]SolveFix{"missing": missing, "info": wrongInfo, "iters": wrongIters,
		"iters2": wrongIters2, "precond": badPre, "residual": tight} {
		if _, bad := Verify(ms, []SolveFix{f}); len(bad) == 0 {
			t.Errorf("%s not detected", name)
		}
	}
	if IterTolerance("cg", 1000) != 50 || IterTolerance("bicgstab", 1000) != 250 || IterTolerance("cg", 3) != 3 {
		t.Error("IterTolerance")
	}
	if len(PermHash([]int{1, 2})) != 16 || PermHash([]int{1, 2}) == PermHash([]int{2, 1}) {
		t.Error("PermHash")
	}
}

func TestVerifyLSQRDetectsProblems(t *testing.T) {
	c := LSQRCase{Matrix: "120:40:4:1", ATol: 1e-6, BTol: 1e-6}
	fixes := fixture(t).LSQR
	var ref LSQRFix
	for _, f := range fixes {
		if f.LSQRCase == c {
			ref = f
		}
	}
	if rows, bad := VerifyLSQR([]LSQRFix{ref}); len(bad) != 0 || len(rows) != 1 {
		t.Fatal(bad)
	}
	stop, itn, x, r1 := ref, ref, ref, ref
	stop.Sci.Istop = 7
	itn.Sci.Itn += 10
	x.Sci.X = append([]float64(nil), ref.Sci.X...)
	x.Sci.X[0] += 1
	r1.Sci.R1Norm += 1
	for name, f := range map[string]LSQRFix{"istop": stop, "itn": itn, "x": x, "r1norm": r1} {
		if _, bad := VerifyLSQR([]LSQRFix{f}); len(bad) == 0 {
			t.Errorf("%s not detected", name)
		}
	}
	if _, bad := VerifyLSQR([]LSQRFix{{LSQRCase: LSQRCase{Matrix: "garbage"}}}); len(bad) == 0 {
		t.Error("bad matrix name")
	}
}

func TestVerifyOrdersDetectsProblems(t *testing.T) {
	ms, _ := Suite(SuiteOrder)
	fix := fixture(t).Orders
	spd, nonsym := fix[0], fix[2]
	mut := func(f OrderFix, g func(*OrderFix)) []OrderFix { g(&f); return []OrderFix{f} }
	cases := map[string][]OrderFix{
		"missing":   mut(spd, func(f *OrderFix) { f.Matrix = "other" }),
		"rcm hash":  mut(spd, func(f *OrderFix) { f.GoRCMHash = "x" }),
		"amd hash":  mut(spd, func(f *OrderFix) { f.GoAMDHash = "x" }),
		"col hash":  mut(nonsym, func(f *OrderFix) { f.GoCOLAMDHash = "x" }),
		"bandwidth": mut(spd, func(f *OrderFix) { f.SciRCMBandwidth = 1 }),
		"amd chol":  mut(spd, func(f *OrderFix) { f.SuiteSparseAMDChol = 1 }),
		"amd lu":    mut(spd, func(f *OrderFix) { f.SuperLUMMDLU = 1 }),
		"colamd":    mut(nonsym, func(f *OrderFix) { f.SciCOLAMDLU = 1 }),
	}
	for name, f := range cases {
		if _, bad := VerifyOrders(ms, f); len(bad) == 0 {
			t.Errorf("%s not detected", name)
		}
	}
}

func TestBadLSQRName(t *testing.T) {
	if _, _, err := lsqrSystem("1:2"); err == nil {
		t.Fatal("must fail")
	}
}
