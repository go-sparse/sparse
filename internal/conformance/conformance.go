// Package conformance compares this module with SciPy. A Generator runs the
// cases in both implementations through a live oracle session and records
// SciPy's answers as a Fixture (JSON, committed under testdata/); Verify
// re-runs the Go side from the fixture alone, so the regression tests need
// neither Python nor network.
package conformance

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
	"github.com/go-sparse/sparse/internal/scipy"
	"github.com/go-sparse/sparse/iter"
	"github.com/go-sparse/sparse/order"
)

// Named is a matrix and the name it is reported under.
type Named struct {
	Name string
	A    *sparse.CSR[float64]
	SPD  bool
}

// Build constructs a gallery matrix from its name:
// lap1d:n, lap2d:nx:ny, lap3d:nx:ny:nz, cd2d:nx:ny:peclet, rnd:n:perRow:sym:seed,
// each optionally followed by "+scr:seed" (symmetric random permutation).
func Build(name string) (Named, error) {
	base, scr, _ := strings.Cut(name, "+scr:")
	f := append(strings.Split(base, ":"), "0", "0", "0", "0", "0")
	num := func(i int) float64 { // missing or malformed numbers read as 0
		v, _ := strconv.ParseFloat(f[i], 64)
		return v
	}
	var a *sparse.CSR[float64]
	spd := true
	switch f[0] {
	case "lap1d":
		a = gallery.Laplacian1D(int(num(1)))
	case "lap2d":
		a = gallery.Laplacian2D(int(num(1)), int(num(2)))
	case "lap3d":
		a = gallery.Laplacian3D(int(num(1)), int(num(2)), int(num(3)))
	case "cd2d":
		a = gallery.ConvectionDiffusion2D(int(num(1)), int(num(2)), num(3))
		spd = false
	case "rnd":
		sym := num(3) != 0
		a = gallery.RandomSparse(int(num(1)), int(num(2)), sym, uint64(num(4)))
		spd = sym
	default:
		return Named{}, fmt.Errorf("conformance: unknown matrix %q", name)
	}
	if scr != "" {
		seed, err := strconv.ParseUint(scr, 10, 64)
		if err != nil {
			return Named{}, fmt.Errorf("conformance: bad scramble seed in %q", name)
		}
		a, _ = gallery.Scramble(a, seed)
	}
	return Named{Name: name, A: a, SPD: spd}, nil
}

// RHS is the deterministic right-hand side used for matrix a.
func RHS(n int) []float64 {
	r := rand.New(rand.NewPCG(uint64(n), 99))
	b := make([]float64, n)
	for i := range b {
		b[i] = r.NormFloat64()
	}
	return b
}

// PrecondSpec names a preconditioner: none, jacobi, ssor, ic0, ilu0, ilut.
type PrecondSpec string

// build returns the Go preconditioner and its description for SciPy.
func build(a *sparse.CSR[float64], k PrecondSpec) (iter.Preconditioner, *scipy.Precond, error) {
	switch k {
	case "none":
		return nil, nil, nil
	case "jacobi":
		p, err := iter.NewJacobi(a)
		return p, &scipy.Precond{Kind: "jacobi"}, err
	case "ssor":
		p, err := iter.NewSSOR(a, 1.2)
		return p, &scipy.Precond{Kind: "ssor", Omega: 1.2}, err
	case "ic0":
		p, err := iter.NewIC0(a)
		if err != nil {
			return nil, nil, err
		}
		l := scipy.FromCSR(p.L())
		return p, &scipy.Precond{Kind: "ic", L: &l}, nil
	case "ilu0", "ilut":
		var p *iter.ILU
		var err error
		if k == "ilu0" {
			p, err = iter.NewILU0(a)
		} else {
			p, err = iter.NewILUT(a, 1e-3, 20)
		}
		if err != nil {
			return nil, nil, err
		}
		l, u := scipy.FromCSR(p.L()), scipy.FromCSR(p.U())
		return p, &scipy.Precond{Kind: "ilu", L: &l, U: &u}, nil
	}
	return nil, nil, fmt.Errorf("conformance: unknown preconditioner %q", k)
}

var goSolvers = map[string]func(iter.Operator, []float64, *iter.Options) (iter.Result, error){
	"cg": iter.CG, "bicgstab": iter.BiCGSTAB, "gmres": iter.GMRES, "minres": iter.MINRES,
}

// SolveCase is one solver run on one matrix.
type SolveCase struct {
	Matrix  string  `json:"matrix"`
	Solver  string  `json:"solver"`
	Precond string  `json:"precond"`
	RTol    float64 `json:"rtol"`
	Restart int     `json:"restart,omitempty"`
	MaxIter int     `json:"maxiter,omitempty"` // 0: the solver default
}

// iterPtr is SciPy's maxiter argument: null for the default.
func iterPtr(n int) *int {
	if n == 0 {
		return nil
	}
	return &n
}

// SolveFix is a SolveCase with SciPy's answer.
type SolveFix struct {
	SolveCase
	Sci scipy.SolveResponse `json:"scipy"`
}

// SolveRow compares the two implementations on one case.
type SolveRow struct {
	SolveFix
	GoInfo, GoIters int
	GoRes           float64
}

// RunGo executes the Go side of a case.
func RunGo(n Named, c SolveCase) (info, iters int, res float64, err error) {
	m, _, err := build(n.A, PrecondSpec(c.Precond))
	if err != nil {
		return 0, 0, 0, err
	}
	b := RHS(n.A.Rows)
	r, _ := goSolvers[c.Solver](n.A, b, &iter.Options{RTol: c.RTol, Restart: c.Restart, MaxIter: c.MaxIter, M: m}) // square system, matching rhs
	return r.Info, r.Iterations, iter.TrueResidual(n.A, b, r.X), nil
}

// DefaultSolveCases is the committed suite: every solver on matrices of the
// right kind with each applicable preconditioner.
func DefaultSolveCases(matrices []Named) []struct {
	Name string
	Case SolveCase
} {
	var out []struct {
		Name string
		Case SolveCase
	}
	add := func(m, s, p string, restart int) {
		out = append(out, struct {
			Name string
			Case SolveCase
		}{m, SolveCase{Matrix: m, Solver: s, Precond: p, RTol: 1e-8, Restart: restart}})
	}
	for _, m := range matrices {
		if m.SPD {
			for _, p := range []string{"none", "jacobi", "ssor", "ic0"} {
				add(m.Name, "cg", p, 0)
			}
			add(m.Name, "minres", "none", 0)
		}
		for _, p := range []string{"none", "jacobi", "ilu0", "ilut"} {
			add(m.Name, "bicgstab", p, 0)
			add(m.Name, "gmres", p, 30)
		}
	}
	return out
}

// Generate runs the cases against the oracle.
func Generate(s *scipy.Session, matrices []Named) ([]SolveFix, error) {
	byName := map[string]Named{}
	for _, m := range matrices {
		byName[m.Name] = m
	}
	var out []SolveFix
	for _, c := range DefaultSolveCases(matrices) {
		n := byName[c.Name]
		_, sp, err := build(n.A, PrecondSpec(c.Case.Precond))
		if err != nil {
			return nil, fmt.Errorf("%s/%s/%s: %w", c.Name, c.Case.Solver, c.Case.Precond, err)
		}
		resp, err := s.Solve(scipy.SolveRequest{Solver: c.Case.Solver, A: scipy.FromCSR(n.A), B: RHS(n.A.Rows),
			M: sp, RTol: c.Case.RTol, Restart: c.Case.Restart, MaxIter: iterPtr(c.Case.MaxIter)})
		if err != nil {
			return nil, fmt.Errorf("%s/%s/%s: %w", c.Name, c.Case.Solver, c.Case.Precond, err)
		}
		out = append(out, SolveFix{SolveCase: c.Case, Sci: resp})
	}
	return out, nil
}

// IterTolerance is the allowed difference in iteration counts between Go and
// SciPy for the same case: floating-point summation order differs, which
// moves a Krylov method by an iteration or two (more for BiCGSTAB, whose
// convergence is erratic).
func IterTolerance(solver string, sciIters int) int {
	frac := 0.05
	if solver == "bicgstab" {
		frac = 0.25
	}
	return max(3, int(math.Ceil(frac*float64(sciIters))))
}

// Verify runs the Go side of every fixture and returns the comparison rows
// and the list of violations.
func Verify(matrices []Named, fixes []SolveFix) ([]SolveRow, []string) {
	byName := map[string]Named{}
	for _, m := range matrices {
		byName[m.Name] = m
	}
	var rows []SolveRow
	var bad []string
	for _, f := range fixes {
		n, ok := byName[f.Matrix]
		label := fmt.Sprintf("%s/%s/%s", f.Matrix, f.Solver, f.Precond)
		if !ok {
			bad = append(bad, label+": matrix missing")
			continue
		}
		info, iters, res, err := RunGo(n, f.SolveCase)
		if err != nil {
			bad = append(bad, label+": "+err.Error())
			continue
		}
		rows = append(rows, SolveRow{SolveFix: f, GoInfo: info, GoIters: iters, GoRes: res})
		if info != f.Sci.Info {
			bad = append(bad, fmt.Sprintf("%s: info %d, SciPy %d", label, info, f.Sci.Info))
		}
		// MINRES stops on ||r||/(||A|| ||x||), so its true residual may exceed
		// rtol; the criterion is then "no worse than SciPy's".
		if res > math.Max(1.5*f.RTol, 1.5*f.Sci.Res) {
			bad = append(bad, fmt.Sprintf("%s: residual %.3g exceeds both 1.5 rtol and 1.5 x SciPy's %.3g", label, res, f.Sci.Res))
		}
		if d := iters - f.Sci.Iters; d > IterTolerance(f.Solver, f.Sci.Iters) || -d > IterTolerance(f.Solver, f.Sci.Iters) {
			bad = append(bad, fmt.Sprintf("%s: %d iterations, SciPy %d", label, iters, f.Sci.Iters))
		}
	}
	return rows, bad
}

// PermHash is a short checksum of a permutation, used to detect that an
// ordering changed since the fixture was generated.
func PermHash(p []int) string {
	h := sha256.New()
	for _, v := range p {
		fmt.Fprintf(h, "%d,", v)
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// OrderFix records the ordering comparison for one matrix.
type OrderFix struct {
	Matrix string `json:"matrix"`
	N      int    `json:"n"`
	NNZ    int    `json:"nnz"`
	// RCM: bandwidth and profile after SciPy's reverse_cuthill_mckee.
	SciRCMBandwidth int `json:"sci_rcm_bandwidth"`
	SciRCMProfile   int `json:"sci_rcm_profile"`
	// AMD (symmetric matrices): Cholesky nnz under SuiteSparse AMD's
	// permutation and nnz(L+U) of SuperLU's minimum degree on A+Aᵀ.
	SuiteSparseAMDChol int    `json:"suitesparse_amd_chol,omitempty"`
	SuperLUMMDLU       int    `json:"superlu_mmd_lu,omitempty"`
	GoAMDHash          string `json:"go_amd_hash,omitempty"`
	GoAMDLU            int    `json:"go_amd_lu,omitempty"`
	// COLAMD: nnz(L+U) of SuperLU's COLAMD and of SuperLU on our ordering.
	SciCOLAMDLU  int    `json:"sci_colamd_lu"`
	NaturalLU    int    `json:"natural_lu"`
	GoCOLAMDHash string `json:"go_colamd_hash"`
	GoCOLAMDLU   int    `json:"go_colamd_lu"`
	GoRCMHash    string `json:"go_rcm_hash"`
}

// GenerateOrders runs the ordering comparison on the oracle.
func GenerateOrders(s *scipy.Session, matrices []Named) ([]OrderFix, error) {
	var out []OrderFix
	for _, n := range matrices {
		w := scipy.FromCSR(n.A)
		f := OrderFix{Matrix: n.Name, N: n.A.Rows, NNZ: n.A.NNZ()}
		p, err := s.RCM(w)
		if err != nil {
			return nil, err
		}
		b, err := n.A.Permute(p, p)
		if err != nil {
			return nil, err
		}
		f.SciRCMBandwidth, f.SciRCMProfile = order.Bandwidth(b), order.Profile(b)
		if n.SPD {
			ap, err := s.AMD(w)
			if err != nil {
				return nil, err
			}
			if f.SuiteSparseAMDChol, err = order.CholeskyNNZ(n.A, ap); err != nil {
				return nil, err
			}
			mmd, err := s.LU(w, nil, "MMD_AT_PLUS_A", true)
			if err != nil {
				return nil, err
			}
			f.SuperLUMMDLU = mmd.NNZ
			gp, _ := order.AMD(n.A)
			f.GoAMDHash = PermHash(gp)
			lu, err := s.LU(w, gp, "NATURAL", true)
			if err != nil {
				return nil, err
			}
			f.GoAMDLU = lu.NNZ
		}
		cl, err := s.LU(w, nil, "COLAMD", false)
		if err != nil {
			return nil, err
		}
		nat, err := s.LU(w, nil, "NATURAL", false)
		if err != nil {
			return nil, err
		}
		f.SciCOLAMDLU, f.NaturalLU = cl.NNZ, nat.NNZ
		gc, _ := order.COLAMD(n.A)
		f.GoCOLAMDHash = PermHash(gc)
		lu, err := s.LU(w, gc, "NATURAL", false)
		if err != nil {
			return nil, err
		}
		f.GoCOLAMDLU = lu.NNZ
		gr, _ := order.RCM(n.A)
		f.GoRCMHash = PermHash(gr)
		out = append(out, f)
	}
	return out, nil
}

// OrderRow is the Go-side measurement next to the recorded oracle values.
type OrderRow struct {
	OrderFix
	GoRCMBandwidth, GoRCMProfile int
	GoAMDChol                    int
}

// OrderBounds are the quality ratios Verify enforces (Go over reference).
var OrderBounds = struct{ RCMBandwidth, RCMProfile, AMDChol, AMDLU, COLAMDLU float64 }{
	RCMBandwidth: 1.25, RCMProfile: 1.25, AMDChol: 1.15, AMDLU: 1.15, COLAMDLU: 1.2,
}

// VerifyOrders re-runs the Go orderings and checks them against the fixture.
func VerifyOrders(matrices []Named, fixes []OrderFix) ([]OrderRow, []string) {
	byName := map[string]Named{}
	for _, m := range matrices {
		byName[m.Name] = m
	}
	var rows []OrderRow
	var bad []string
	for _, f := range fixes {
		n, ok := byName[f.Matrix]
		if !ok {
			bad = append(bad, f.Matrix+": matrix missing")
			continue
		}
		row := OrderRow{OrderFix: f}
		rp, _ := order.RCM(n.A)
		if PermHash(rp) != f.GoRCMHash {
			bad = append(bad, f.Matrix+": RCM changed since the fixture was generated; regenerate")
		}
		b, _ := n.A.Permute(rp, rp)
		row.GoRCMBandwidth, row.GoRCMProfile = order.Bandwidth(b), order.Profile(b)
		check := func(what string, got, ref int, bound float64) {
			if float64(got) > bound*float64(ref) {
				bad = append(bad, fmt.Sprintf("%s: %s %d exceeds %.2g x reference %d", f.Matrix, what, got, bound, ref))
			}
		}
		check("RCM bandwidth", row.GoRCMBandwidth, f.SciRCMBandwidth, OrderBounds.RCMBandwidth)
		check("RCM profile", row.GoRCMProfile, f.SciRCMProfile, OrderBounds.RCMProfile)
		if n.SPD {
			ap, _ := order.AMD(n.A)
			if PermHash(ap) != f.GoAMDHash {
				bad = append(bad, f.Matrix+": AMD changed since the fixture was generated; regenerate")
			}
			row.GoAMDChol, _ = order.CholeskyNNZ(n.A, ap)
			check("AMD Cholesky nnz vs SuiteSparse AMD", row.GoAMDChol, f.SuiteSparseAMDChol, OrderBounds.AMDChol)
			check("AMD LU nnz vs SuperLU MMD", f.GoAMDLU, f.SuperLUMMDLU, OrderBounds.AMDLU)
		}
		cp, _ := order.COLAMD(n.A)
		if PermHash(cp) != f.GoCOLAMDHash {
			bad = append(bad, f.Matrix+": COLAMD changed since the fixture was generated; regenerate")
		}
		check("COLAMD LU nnz vs SuperLU COLAMD", f.GoCOLAMDLU, f.SciCOLAMDLU, OrderBounds.COLAMDLU)
		rows = append(rows, row)
	}
	return rows, bad
}
