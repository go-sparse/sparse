package scipy

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/go-sparse/sparse"
)

func TestMain(m *testing.M) {
	if MaybeServeFake(os.Getenv, os.Stdin, os.Stdout) {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestDefaultCommand(t *testing.T) {
	c := DefaultCommand()
	joined := strings.Join(c.Args, " ")
	for _, want := range []string{"uv run", "scipy==" + SciPyVersion, "numpy==" + NumPyVersion, "python -u -c"} {
		if !strings.Contains(joined, want) {
			t.Errorf("command %q lacks %q", joined[:min(len(joined), 120)], want)
		}
	}
	if !strings.Contains(c.Args[len(c.Args)-1], "reverse_cuthill_mckee") {
		t.Error("script not passed")
	}
	found := false
	for _, e := range c.Env {
		found = found || e == "OPENBLAS_NUM_THREADS=1"
	}
	if !found {
		t.Error("single-thread environment missing")
	}
}

func fake(t *testing.T, mode string) *Session {
	t.Helper()
	s, err := Start(FakeCommand(mode))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestTypedCalls(t *testing.T) {
	s := fake(t, "ok")
	a := FromCSR(sparse.Eye[float64](3))
	if sv, nv, err := s.Versions(); err != nil || sv != SciPyVersion || nv != NumPyVersion {
		t.Fatal(sv, nv, err)
	}
	r, err := s.Solve(SolveRequest{Solver: "cg", A: a, B: []float64{1, 1, 1}, RTol: 1e-8, M: &Precond{Kind: "jacobi"}})
	if err != nil || r.Iters != 1 {
		t.Fatal(r, err)
	}
	l, err := s.LSQR(LSQRRequest{A: a, B: []float64{1, 1, 1}})
	if err != nil || len(l.X) != 3 {
		t.Fatal(l, err)
	}
	if p, err := s.RCM(a); err != nil || len(p) != 3 {
		t.Fatal(p, err)
	}
	if p, err := s.AMD(a); err != nil || len(p) != 3 {
		t.Fatal(p, err)
	}
	if lu, err := s.LU(a, []int{0, 1, 2}, "NATURAL", true); err != nil || lu.NNZ != 10 {
		t.Fatal(lu, err)
	}
	if err := s.Load("m", a, []float64{1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if sec, err := s.Time("m", "spmv", 10, 0); err != nil || sec != 0.5 {
		t.Fatal(sec, err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestFailures(t *testing.T) {
	a := FromCSR(sparse.Eye[float64](2))
	// the oracle reports an error
	s := fake(t, "fail")
	if _, err := s.RCM(a); err == nil || !strings.Contains(err.Error(), "fake failure") {
		t.Fatal(err)
	}
	s.Close()
	// the oracle answers with the wrong shape
	s = fake(t, "garbage")
	if _, err := s.Solve(SolveRequest{Solver: "cg", A: a}); err == nil {
		t.Fatal("garbage must not decode")
	}
	s.Close()
	// the oracle dies
	s = fake(t, "eof")
	if _, err := s.RCM(a); err == nil {
		t.Fatal("eof")
	}
	// writing to a dead process / unmarshalable request
	s.Close()
	if err := s.Call(map[string]any{"op": "x"}, &struct{}{}); err == nil {
		t.Fatal("write after close")
	}
	s = fake(t, "ok")
	if err := s.Call(map[string]any{"op": make(chan int)}, &struct{}{}); err == nil {
		t.Fatal("marshal")
	}
	s.Close()
	// a command that cannot start
	if _, err := Start(func() *exec.Cmd { return exec.Command("/nonexistent/oracle") }); err == nil {
		t.Fatal("start")
	}
	// a command that already has stdin/stdout attached
	if _, err := Start(func() *exec.Cmd {
		c := exec.Command("true")
		c.Stdin = strings.NewReader("")
		return c
	}); err == nil {
		t.Fatal("stdin pipe")
	}
	if _, err := Start(func() *exec.Cmd {
		c := exec.Command("true")
		c.Stdout = &bytes.Buffer{}
		return c
	}); err == nil {
		t.Fatal("stdout pipe")
	}
}

func TestMaybeServeFakeOff(t *testing.T) {
	if MaybeServeFake(func(string) string { return "" }, nil, nil) {
		t.Fatal("must not serve without the variable")
	}
	var out bytes.Buffer
	if !MaybeServeFake(func(string) string { return "ok" }, strings.NewReader(`{"op":"nothing"}`+"\n"), &out) || strings.TrimSpace(out.String()) != "{}" {
		t.Fatal("unknown op", out.String())
	}
}
