package scipy

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// FakeEnv selects the fake-oracle mode of a re-executed test binary:
// "ok" answers every request with plausible canned numbers, "fail" answers
// with an error, "garbage" answers with the wrong JSON shape and "eof" exits
// at once. The fake lets the harness code be tested at 100% coverage without
// Python; it is not used by the real tools.
const FakeEnv = "GO_SPARSE_FAKE_ORACLE"

// FakeCommand returns a command that re-executes the running test binary as
// a fake oracle in the given mode; the binary's TestMain must call
// MaybeServeFake.
func FakeCommand(mode string) func() *exec.Cmd {
	return func() *exec.Cmd {
		c := exec.Command(os.Args[0])
		c.Env = append(os.Environ(), FakeEnv+"="+mode)
		return c
	}
}

// MaybeServeFake serves the fake oracle on in/out and returns true when
// getenv(FakeEnv) selects a mode; otherwise it returns false at once.
func MaybeServeFake(getenv func(string) string, in io.Reader, out io.Writer) bool {
	mode := getenv(FakeEnv)
	if mode == "" {
		return false
	}
	if mode == "eof" {
		return true
	}
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 1<<30)
	// "failafter:N" answers N requests and then fails every later one
	after := -1
	if n, ok := strings.CutPrefix(mode, "failafter:"); ok {
		after, _ = strconv.Atoi(n)
	}
	for served := 0; sc.Scan(); served++ {
		var req struct {
			Op string `json:"op"`
			A  Matrix `json:"A"`
			B  []float64
		}
		_ = json.Unmarshal(sc.Bytes(), &req)
		var resp any
		switch {
		case mode == "fail" || (after >= 0 && served >= after):
			resp = map[string]string{"error": "fake failure"}
		case (mode == "badrcm" && req.Op == "rcm") || (mode == "badamd" && req.Op == "amd"):
			resp = map[string]any{"perm": []int{0}} // not a permutation of the matrix
		case mode == "garbage":
			resp = map[string]string{"info": "not a number", "nnz": "x", "perm": "x", "seconds": "x", "scipy": "1"}
		default:
			resp = fakeAnswer(req.Op, req.A, req.B)
		}
		b, _ := json.Marshal(resp)
		out.Write(append(b, '\n'))
	}
	return true
}

func fakeAnswer(op string, a Matrix, b []float64) any {
	switch op {
	case "version":
		return map[string]string{"scipy": SciPyVersion, "numpy": NumPyVersion}
	case "solve":
		return SolveResponse{Iters: 1, Res: 1e-9}
	case "lsqr":
		return LSQRResponse{X: make([]float64, a.N), Itn: 1, Istop: 1}
	case "lsmr":
		return LSMRResponse{X: make([]float64, a.N), Itn: 1, Istop: 1}
	case "rcm", "amd":
		p := make([]int, a.M)
		for i := range p {
			p[i] = i
		}
		return map[string]any{"perm": p}
	case "lu":
		return LUResponse{NNZ: 10}
	case "time":
		return map[string]float64{"seconds": 0.5}
	}
	return map[string]any{}
}
