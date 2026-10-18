// Package scipy drives SciPy as a verification oracle: one Python process
// per Session, spoken to in JSON lines over a pipe. Only the minimal SciPy
// call per request is made on the Python side (see script.go); all test
// generation, comparison and reporting is Go.
//
// The default command is
//
//	uv run --with scipy==1.18.1 --with numpy==2.5.4 --with cvxopt==1.3.3 python -u -c <script>
//
// cvxopt is used only for its SuiteSparse AMD binary (BSD-3-Clause), as an
// oracle for the ordering quality; no SuiteSparse source is involved.
package scipy

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/go-sparse/sparse"
)

// Pinned oracle versions.
const (
	SciPyVersion  = "1.18.1"
	NumPyVersion  = "2.5.4"
	CvxoptVersion = "1.3.3"
)

// Matrix is the wire form of a CSR matrix.
type Matrix struct {
	M       int       `json:"m"`
	N       int       `json:"n"`
	Indptr  []int     `json:"indptr"`
	Indices []int     `json:"indices"`
	Data    []float64 `json:"data"`
}

// FromCSR converts a real CSR matrix to its wire form (no copy).
func FromCSR(a *sparse.CSR[float64]) Matrix {
	m := Matrix{M: a.Rows, N: a.Cols, Indptr: a.Indptr, Indices: a.Indices, Data: a.Data}
	if m.Indices == nil { // a nil slice would be JSON null
		m.Indices = []int{}
	}
	if m.Data == nil {
		m.Data = []float64{}
	}
	return m
}

// Session is a running oracle process.
type Session struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

// DefaultCommand returns the uv command that runs the oracle with
// single-threaded numerical libraries.
func DefaultCommand() *exec.Cmd {
	c := exec.Command("uv", "run",
		"--with", "scipy=="+SciPyVersion, "--with", "numpy=="+NumPyVersion, "--with", "cvxopt=="+CvxoptVersion,
		"python", "-u", "-c", script)
	c.Env = append(os.Environ(), "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
		"VECLIB_MAXIMUM_THREADS=1", "MKL_NUM_THREADS=1")
	return c
}

// Start launches the process built by command (normally DefaultCommand).
func Start(command func() *exec.Cmd) (*Session, error) {
	c := command()
	in, err := c.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := c.StdoutPipe()
	if err != nil {
		return nil, err
	}
	c.Stderr = os.Stderr
	if err := c.Start(); err != nil {
		return nil, err
	}
	return &Session{cmd: c, in: in, out: bufio.NewReaderSize(out, 1<<20)}, nil
}

// Close ends the session.
func (s *Session) Close() error {
	s.in.Close()
	return s.cmd.Wait()
}

// Call sends one request (a struct or map with an "op" field) and decodes
// the response into resp. An {"error": ...} reply becomes a Go error.
func (s *Session) Call(req, resp any) error {
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	if _, err := s.in.Write(append(b, '\n')); err != nil {
		return err
	}
	line, err := s.out.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("scipy oracle: %w", err)
	}
	var e struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(line, &e) == nil && e.Error != "" {
		return errors.New("scipy oracle: " + e.Error)
	}
	return json.Unmarshal(line, resp)
}

// Precond selects the preconditioner of a Solve request.
type Precond struct {
	Kind  string  `json:"kind"`
	Omega float64 `json:"omega,omitempty"`
	L     *Matrix `json:"L,omitempty"`
	U     *Matrix `json:"U,omitempty"`
}

// SolveRequest asks for scipy.sparse.linalg.{cg,bicgstab,gmres,minres}.
type SolveRequest struct {
	Op      string    `json:"op"`
	Solver  string    `json:"solver"`
	A       Matrix    `json:"A"`
	B       []float64 `json:"b"`
	M       *Precond  `json:"M,omitempty"`
	RTol    float64   `json:"rtol"`
	ATol    float64   `json:"atol"`
	MaxIter *int      `json:"maxiter"`
	Restart int       `json:"restart,omitempty"`
}

// SolveResponse is what SciPy reported: info, callback count, and the true
// relative residual it computed itself.
type SolveResponse struct {
	Info  int     `json:"info"`
	Iters int     `json:"iters"`
	Res   float64 `json:"res"`
}

// Solve runs one solve.
func (s *Session) Solve(r SolveRequest) (SolveResponse, error) {
	r.Op = "solve"
	var out SolveResponse
	return out, s.Call(r, &out)
}

// LSQRRequest asks for scipy.sparse.linalg.lsqr.
type LSQRRequest struct {
	Op      string    `json:"op"`
	A       Matrix    `json:"A"`
	B       []float64 `json:"b"`
	Damp    float64   `json:"damp"`
	ATol    float64   `json:"atol"`
	BTol    float64   `json:"btol"`
	ConLim  float64   `json:"conlim"`
	IterLim int       `json:"iter_lim"`
}

// LSQRResponse is the tuple lsqr returns.
type LSQRResponse struct {
	X      []float64 `json:"x"`
	Istop  int       `json:"istop"`
	Itn    int       `json:"itn"`
	R1Norm float64   `json:"r1norm"`
	R2Norm float64   `json:"r2norm"`
	ANorm  float64   `json:"anorm"`
	ACond  float64   `json:"acond"`
	ARNorm float64   `json:"arnorm"`
	XNorm  float64   `json:"xnorm"`
}

// LSQR runs scipy lsqr.
func (s *Session) LSQR(r LSQRRequest) (LSQRResponse, error) {
	r.Op = "lsqr"
	var out LSQRResponse
	return out, s.Call(r, &out)
}

type permResponse struct {
	Perm []int `json:"perm"`
}

// RCM returns scipy's reverse Cuthill-McKee permutation.
func (s *Session) RCM(a Matrix) ([]int, error) {
	var out permResponse
	err := s.Call(map[string]any{"op": "rcm", "A": a}, &out)
	return out.Perm, err
}

// AMD returns SuiteSparse AMD's permutation (through cvxopt) of the pattern
// of the symmetric matrix a.
func (s *Session) AMD(a Matrix) ([]int, error) {
	var out permResponse
	err := s.Call(map[string]any{"op": "amd", "A": a}, &out)
	return out.Perm, err
}

// LUResponse is the factor size of SuperLU.
type LUResponse struct {
	NNZ   int   `json:"nnz"`
	PermC []int `json:"perm_c"`
}

// LU factorises with scipy splu and returns nnz(L)+nnz(U). With perm nil the
// column ordering is spec ("COLAMD", "NATURAL", "MMD_AT_PLUS_A", "MMD_ATA");
// otherwise A[:, perm] is factorised with spec "NATURAL" (or A[perm][:, perm]
// in SuperLU's symmetric mode without row pivoting when symmetric is true).
func (s *Session) LU(a Matrix, perm []int, spec string, symmetric bool) (LUResponse, error) {
	var out LUResponse
	err := s.Call(map[string]any{"op": "lu", "A": a, "perm": perm, "spec": spec, "symmetric": symmetric}, &out)
	return out, err
}

// Versions reports the SciPy and NumPy versions in use.
func (s *Session) Versions() (scipy, numpy string, err error) {
	var raw struct {
		SciPy string `json:"scipy"`
		NumPy string `json:"numpy"`
	}
	err = s.Call(map[string]any{"op": "version"}, &raw)
	return raw.SciPy, raw.NumPy, err
}

// Load stores a system in the oracle for Time.
func (s *Session) Load(name string, a Matrix, b []float64) error {
	var out struct{}
	return s.Call(map[string]any{"op": "load", "name": name, "A": a, "b": b}, &out)
}

// Time runs a timed loop in the oracle and returns the seconds it took:
// what is "spmv" (reps products), "cg" (reps iterations, no stopping) or
// "gmres" (reps restart cycles of length restart, no stopping).
func (s *Session) Time(name, what string, reps, restart int) (float64, error) {
	var out struct {
		Seconds float64 `json:"seconds"`
	}
	err := s.Call(map[string]any{"op": "time", "name": name, "what": what, "reps": reps, "restart": restart}, &out)
	return out.Seconds, err
}
