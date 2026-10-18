package conformance

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/internal/scipy"
	"github.com/go-sparse/sparse/iter"
)

// LSQRMatrix builds a full-column-rank m x n least-squares matrix with about
// perRow random entries per row plus a strengthened diagonal block.
func LSQRMatrix(m, n, perRow int, seed uint64) *sparse.CSR[float64] {
	r := rand.New(rand.NewPCG(seed, 31))
	c := sparse.NewCOO[float64](m, n)
	for i := 0; i < m; i++ {
		for k := 0; k < perRow; k++ {
			_ = c.Append(i, r.IntN(n), r.NormFloat64())
		}
	}
	for j := 0; j < n; j++ {
		_ = c.Append(j, j, 3)
	}
	return c.ToCSR()
}

// LSQRCase is one lsqr run.
type LSQRCase struct {
	Matrix string  `json:"matrix"` // "m:n:perRow:seed"
	Damp   float64 `json:"damp"`
	ATol   float64 `json:"atol"`
	BTol   float64 `json:"btol"`
}

// LSQRFix is an LSQRCase with SciPy's answer.
type LSQRFix struct {
	LSQRCase
	Sci scipy.LSQRResponse `json:"scipy"`
}

// DefaultLSQRCases is the committed least-squares suite.
func DefaultLSQRCases() []LSQRCase {
	var out []LSQRCase
	for _, m := range []string{"120:40:4:1", "400:100:5:2", "60:60:3:3"} {
		out = append(out,
			LSQRCase{m, 0, 1e-6, 1e-6},
			LSQRCase{m, 0, 1e-12, 1e-12},
			LSQRCase{m, 0.5, 1e-8, 1e-8})
	}
	return out
}

func lsqrSystem(name string) (*sparse.CSR[float64], []float64, error) {
	var m, n, k int
	var seed uint64
	if _, err := fmt.Sscanf(name, "%d:%d:%d:%d", &m, &n, &k, &seed); err != nil {
		return nil, nil, fmt.Errorf("conformance: bad lsqr matrix %q", name)
	}
	return LSQRMatrix(m, n, k, seed), RHS(m), nil
}

// GenerateLSQR runs the cases against the oracle.
func GenerateLSQR(s *scipy.Session) ([]LSQRFix, error) {
	var out []LSQRFix
	for _, c := range DefaultLSQRCases() {
		a, b, _ := lsqrSystem(c.Matrix) // default case names are well-formed
		r, err := s.LSQR(scipy.LSQRRequest{A: scipy.FromCSR(a), B: b, Damp: c.Damp, ATol: c.ATol, BTol: c.BTol,
			ConLim: 1e8, IterLim: 2 * a.Cols})
		if err != nil {
			return nil, err
		}
		out = append(out, LSQRFix{c, r})
	}
	return out, nil
}

// LSQRRow compares the two implementations on one case.
type LSQRRow struct {
	LSQRFix
	GoIstop, GoItn int
	GoR1Norm       float64
	MaxXDiff       float64 // max |x_go - x_scipy| / max(1, |x_scipy|)
}

// VerifyLSQR runs the Go side and checks it against the fixtures: same
// stopping reason, iteration count within 3, solution and residual norm
// equal to 1e-6 relative.
func VerifyLSQR(fixes []LSQRFix) ([]LSQRRow, []string) {
	var rows []LSQRRow
	var bad []string
	for _, f := range fixes {
		a, b, err := lsqrSystem(f.Matrix)
		if err != nil {
			bad = append(bad, err.Error())
			continue
		}
		r, _ := iter.LSQR(a, b, &iter.LSQROptions{Damp: f.Damp, ATol: f.ATol, BTol: f.BTol}) // shapes match by construction
		row := LSQRRow{LSQRFix: f, GoIstop: r.Istop, GoItn: r.Itn, GoR1Norm: r.R1Norm}
		for i, v := range f.Sci.X {
			row.MaxXDiff = math.Max(row.MaxXDiff, math.Abs(r.X[i]-v)/math.Max(1, math.Abs(v)))
		}
		rows = append(rows, row)
		label := fmt.Sprintf("lsqr %s damp=%g atol=%g", f.Matrix, f.Damp, f.ATol)
		if r.Istop != f.Sci.Istop {
			bad = append(bad, fmt.Sprintf("%s: istop %d, SciPy %d", label, r.Istop, f.Sci.Istop))
		}
		if d := r.Itn - f.Sci.Itn; d > 3 || d < -3 {
			bad = append(bad, fmt.Sprintf("%s: %d iterations, SciPy %d", label, r.Itn, f.Sci.Itn))
		}
		if row.MaxXDiff > 1e-6 {
			bad = append(bad, fmt.Sprintf("%s: solution differs by %.3g", label, row.MaxXDiff))
		}
		if math.Abs(r.R1Norm-f.Sci.R1Norm) > 1e-6*math.Max(1, f.Sci.R1Norm) {
			bad = append(bad, fmt.Sprintf("%s: r1norm %.10g, SciPy %.10g", label, r.R1Norm, f.Sci.R1Norm))
		}
	}
	return rows, bad
}
