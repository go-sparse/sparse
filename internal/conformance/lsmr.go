package conformance

import (
	"fmt"
	"math"

	"github.com/go-sparse/sparse/internal/scipy"
	"github.com/go-sparse/sparse/iter"
)

// LSMRFix is an LSQRCase (the matrices are shared) with SciPy's lsmr answer.
type LSMRFix struct {
	LSQRCase
	Sci scipy.LSMRResponse `json:"scipy"`
}

// GenerateLSMR runs the least-squares cases through scipy lsmr with maxiter 5n.
func GenerateLSMR(s *scipy.Session) ([]LSMRFix, error) {
	var out []LSMRFix
	for _, c := range DefaultLSQRCases() {
		a, b, _ := lsqrSystem(c.Matrix) // default case names are well-formed
		r, err := s.LSMR(scipy.LSQRRequest{A: scipy.FromCSR(a), B: b, Damp: c.Damp, ATol: c.ATol, BTol: c.BTol,
			ConLim: 1e8, IterLim: 5 * a.Cols})
		if err != nil {
			return nil, err
		}
		out = append(out, LSMRFix{c, r})
	}
	return out, nil
}

// LSMRRow compares the two implementations on one case.
type LSMRRow struct {
	LSMRFix
	GoIstop, GoItn int
	GoNormR        float64
	MaxXDiff       float64
}

// VerifyLSMR checks stopping reason, iteration count (within 3), solution
// and residual norm (1e-6 relative) against the fixtures.
func VerifyLSMR(fixes []LSMRFix) ([]LSMRRow, []string) {
	var rows []LSMRRow
	var bad []string
	for _, f := range fixes {
		a, b, err := lsqrSystem(f.Matrix)
		if err != nil {
			bad = append(bad, err.Error())
			continue
		}
		r, _ := iter.LSMR(a, b, &iter.LSMROptions{Damp: f.Damp, ATol: f.ATol, BTol: f.BTol, MaxIter: 5 * a.Cols})
		row := LSMRRow{LSMRFix: f, GoIstop: r.Istop, GoItn: r.Itn, GoNormR: r.NormR}
		for i, v := range f.Sci.X {
			row.MaxXDiff = math.Max(row.MaxXDiff, math.Abs(r.X[i]-v)/math.Max(1, math.Abs(v)))
		}
		rows = append(rows, row)
		label := fmt.Sprintf("lsmr %s damp=%g atol=%g", f.Matrix, f.Damp, f.ATol)
		if r.Istop != f.Sci.Istop {
			bad = append(bad, fmt.Sprintf("%s: istop %d, SciPy %d", label, r.Istop, f.Sci.Istop))
		}
		if d := r.Itn - f.Sci.Itn; d > 3 || d < -3 {
			bad = append(bad, fmt.Sprintf("%s: %d iterations, SciPy %d", label, r.Itn, f.Sci.Itn))
		}
		if row.MaxXDiff > 1e-6 {
			bad = append(bad, fmt.Sprintf("%s: solution differs by %.3g", label, row.MaxXDiff))
		}
		if math.Abs(r.NormR-f.Sci.NormR) > 1e-6*math.Max(1, f.Sci.NormR) {
			bad = append(bad, fmt.Sprintf("%s: normr %.10g, SciPy %.10g", label, r.NormR, f.Sci.NormR))
		}
	}
	return rows, bad
}
