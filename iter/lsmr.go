package iter

import (
	"fmt"
	"math"

	"github.com/go-sparse/sparse"
)

// LSMROptions configures LSMR; the zero value is SciPy's default call
// (atol = btol = 1e-6, conlim = 1e8, maxiter = min(m, n)).
type LSMROptions struct {
	Damp    float64
	ATol    float64 // 0 means 1e-6; negative means 0
	BTol    float64 // 0 means 1e-6; negative means 0
	ConLim  float64 // 0 means 1e8; negative disables the test
	MaxIter int     // 0 means min(m, n)
	X0      []float64
}

// LSMRResult mirrors the tuple returned by scipy.sparse.linalg.lsmr.
type LSMRResult struct {
	X      []float64
	Istop  int
	Itn    int
	NormR  float64 // ||b - A x||
	NormAR float64 // ||Aᵀ(b - A x)||
	NormA  float64
	CondA  float64
	NormX  float64
}

// LSMR solves min ||b - A x|| (Fong and Saunders 2011, doi:10.1137/10079687X)
// by Golub-Kahan bidiagonalisation, minimising ||Aᵀ r|| at each step, with
// the stopping rules of scipy.sparse.linalg.lsmr.
func LSMR(a sparse.TransposeOperator, b []float64, o *LSMROptions) (LSMRResult, error) {
	if o == nil {
		o = &LSMROptions{}
	}
	m, n := a.Dims()
	if len(b) != m || (o.X0 != nil && len(o.X0) != n) {
		return LSMRResult{}, fmt.Errorf("%w: operator %dx%d, rhs %d", ErrShape, m, n, len(b))
	}
	atol, btol := def(o.ATol, 1e-6), def(o.BTol, 1e-6)
	conlim := o.ConLim
	if conlim == 0 {
		conlim = 1e8
	}
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = min(m, n)
	}
	res := LSMRResult{X: make([]float64, n)}
	x := res.X
	u := append([]float64(nil), b...)
	beta := nrm2(b)
	if o.X0 != nil {
		copy(x, o.X0)
		tmp := make([]float64, m)
		a.MulVec(tmp, x)
		axpy(-1, tmp, u)
		beta = nrm2(u)
	}
	v := make([]float64, n)
	var alpha float64
	if beta > 0 {
		scale(u, 1/beta)
		a.MulVecT(v, u)
		alpha = nrm2(v)
	}
	if alpha > 0 {
		scale(v, 1/alpha)
	}
	ctol := 0.0
	if conlim > 0 {
		ctol = 1 / conlim
	}
	normb := nrm2(b)
	zetabar := alpha * beta
	alphabar := alpha
	rho, rhobar, cbar, sbar := 1.0, 1.0, 1.0, 0.0
	h := append([]float64(nil), v...)
	hbar := make([]float64, n)
	betadd, betad, rhodold, tautildeold, thetatilde, zeta, d := beta, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
	var thetabar float64
	normA2 := alpha * alpha
	maxrbar, minrbar := 0.0, 1e100
	res.NormR = beta
	res.NormAR = alpha * beta
	if res.NormAR == 0 {
		return res, nil
	}
	tmpm := make([]float64, m)
	tmpn := make([]float64, n)
	for res.Itn < maxiter {
		res.Itn++
		a.MulVec(tmpm, v)
		for i := range u {
			u[i] = tmpm[i] - alpha*u[i]
		}
		beta = nrm2(u)
		if beta > 0 {
			scale(u, 1/beta)
			a.MulVecT(tmpn, u)
			for i := range v {
				v[i] = tmpn[i] - beta*v[i]
			}
			alpha = nrm2(v)
			if alpha > 0 {
				scale(v, 1/alpha)
			}
		}
		chat, shat, alphahat := symOrtho(alphabar, o.Damp)
		rhoold := rho
		c, s, rhoNew := symOrtho(alphahat, beta)
		rho = rhoNew
		thetanew := s * alpha
		alphabar = c * alpha
		rhobarold, zetaold := rhobar, zeta
		thetabar = sbar * rho
		rhotemp := cbar * rho
		cbar, sbar, rhobar = symOrtho(cbar*rho, thetanew)
		zeta = cbar * zetabar
		zetabar = -sbar * zetabar
		k1 := thetabar * rho / (rhoold * rhobarold)
		k2 := zeta / (rho * rhobar)
		k3 := thetanew / rho
		for i := range h {
			hbar[i] = h[i] - k1*hbar[i]
			x[i] += k2 * hbar[i]
			h[i] = v[i] - k3*h[i]
		}
		betaacute := chat * betadd
		betacheck := -shat * betadd
		betahat := c * betaacute
		betadd = -s * betaacute
		thetatildeold := thetatilde
		ctildeold, stildeold, rhotildeold := symOrtho(rhodold, thetabar)
		thetatilde = stildeold * rhobar
		rhodold = ctildeold * rhobar
		betad = -stildeold*betad + ctildeold*betahat
		tautildeold = (zetaold - thetatildeold*tautildeold) / rhotildeold
		taud := (zeta - thetatilde*tautildeold) / rhodold
		d += betacheck * betacheck
		res.NormR = math.Sqrt(d + (betad-taud)*(betad-taud) + betadd*betadd)
		normA2 += beta * beta
		res.NormA = math.Sqrt(normA2)
		normA2 += alpha * alpha
		maxrbar = math.Max(maxrbar, rhobarold)
		if res.Itn > 1 {
			minrbar = math.Min(minrbar, rhobarold)
		}
		res.CondA = math.Max(maxrbar, rhotemp) / math.Min(minrbar, rhotemp)
		res.NormAR = math.Abs(zetabar)
		res.NormX = nrm2(x)
		test1 := res.NormR / normb
		test2 := math.Inf(1)
		if p := res.NormA * res.NormR; p != 0 {
			test2 = res.NormAR / p
		}
		test3 := 1 / res.CondA
		t1 := test1 / (1 + res.NormA*res.NormX/normb)
		rtol := btol + atol*res.NormA*res.NormX/normb
		for _, g := range [...]struct {
			stop int
			hit  bool
		}{
			{7, res.Itn >= maxiter}, {6, 1+test3 <= 1}, {5, 1+test2 <= 1}, {4, 1+t1 <= 1},
			{3, test3 <= ctol}, {2, test2 <= atol}, {1, test1 <= rtol},
		} {
			if g.hit {
				res.Istop = g.stop
			}
		}
		if res.Istop != 0 {
			break
		}
	}
	return res, nil
}
