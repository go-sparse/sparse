package iter

import (
	"fmt"
	"math"

	"github.com/go-sparse/sparse"
)

// LSQROptions configures LSQR; the zero value is SciPy's default call
// (atol = btol = 1e-6, conlim = 1e8, iter_lim = 2n).
type LSQROptions struct {
	Damp    float64 // solves min ||b-Ax||² + damp²||x||²
	ATol    float64 // 0 means 1e-6; negative means 0
	BTol    float64 // 0 means 1e-6; negative means 0
	ConLim  float64 // 0 means 1e8; negative disables the test
	IterLim int     // 0 means 2n
	X0      []float64
	// CalcVar requests the diagonal estimate of (AᵀA + damp²I)⁻¹.
	CalcVar bool
}

// LSQRResult mirrors the tuple returned by scipy.sparse.linalg.lsqr.
type LSQRResult struct {
	X      []float64
	Istop  int // 0..7 as documented by SciPy
	Itn    int
	R1Norm float64 // ||b - A x||
	R2Norm float64 // sqrt(||b - A x||² + damp²||x||²)
	ANorm  float64
	ACond  float64
	ARNorm float64
	XNorm  float64
	Var    []float64
}

func def(v, d float64) float64 {
	switch {
	case v == 0:
		return d
	case v < 0:
		return 0
	}
	return v
}

func symOrtho(a, b float64) (c, s, r float64) {
	switch {
	case b == 0:
		return math.Copysign(1, a), 0, math.Abs(a)
	case a == 0:
		return 0, math.Copysign(1, b), math.Abs(b)
	case math.Abs(b) > math.Abs(a):
		tau := a / b
		s = math.Copysign(1, b) / math.Sqrt(1+tau*tau)
		return s * tau, s, b / s
	}
	tau := b / a
	c = math.Copysign(1, a) / math.Sqrt(1+tau*tau)
	return c, c * tau, a / c
}

// LSQR solves the least-squares problem min ||b - A x|| (Paige and Saunders
// 1982) for a rectangular operator that can apply Aᵀ, following
// scipy.sparse.linalg.lsqr.
func LSQR(a sparse.TransposeOperator, b []float64, o *LSQROptions) (LSQRResult, error) {
	if o == nil {
		o = &LSQROptions{}
	}
	m, n := a.Dims()
	if len(b) != m || (o.X0 != nil && len(o.X0) != n) {
		return LSQRResult{}, fmt.Errorf("%w: operator %dx%d, rhs %d", ErrShape, m, n, len(b))
	}
	atol, btol := def(o.ATol, 1e-6), def(o.BTol, 1e-6)
	conlim := o.ConLim
	if conlim == 0 {
		conlim = 1e8
	}
	iterLim := o.IterLim
	if iterLim == 0 {
		iterLim = 2 * n
	}
	const eps = 2.220446049250313e-16
	res := LSQRResult{X: make([]float64, n)}
	if o.CalcVar {
		res.Var = make([]float64, n)
	}
	ctol := 0.0
	if conlim > 0 {
		ctol = 1 / conlim
	}
	dampsq := o.Damp * o.Damp
	var ddnorm, res2, xxnorm, z float64
	cs2, sn2 := -1.0, 0.0
	u := append([]float64(nil), b...)
	bnorm := nrm2(b)
	x := res.X
	beta := bnorm
	if o.X0 != nil {
		copy(x, o.X0)
		tmp := make([]float64, m)
		a.MulVec(tmp, x)
		axpy(-1, tmp, u)
		beta = nrm2(u)
	}
	v := make([]float64, n)
	var alfa float64
	if beta > 0 {
		scale(u, 1/beta)
		a.MulVecT(v, u)
		alfa = nrm2(v)
	} else {
		copy(v, x)
	}
	if alfa > 0 {
		scale(v, 1/alfa)
	}
	w := append([]float64(nil), v...)
	rhobar, phibar := alfa, beta
	res.R1Norm, res.R2Norm = beta, beta
	res.ARNorm = alfa * beta
	if res.ARNorm == 0 {
		return res, nil
	}
	tmpm := make([]float64, m)
	tmpn := make([]float64, n)
	dk := make([]float64, n)
	for res.Itn < iterLim {
		res.Itn++
		a.MulVec(tmpm, v)
		for i := range u {
			u[i] = tmpm[i] - alfa*u[i]
		}
		beta = nrm2(u)
		if beta > 0 {
			scale(u, 1/beta)
			res.ANorm = math.Sqrt(res.ANorm*res.ANorm + alfa*alfa + beta*beta + dampsq)
			a.MulVecT(tmpn, u)
			for i := range v {
				v[i] = tmpn[i] - beta*v[i]
			}
			alfa = nrm2(v)
			if alfa > 0 {
				scale(v, 1/alfa)
			}
		}
		rhobar1, psi := rhobar, 0.0
		if o.Damp > 0 {
			rhobar1 = math.Sqrt(rhobar*rhobar + dampsq)
			cs1, sn1 := rhobar/rhobar1, o.Damp/rhobar1
			psi = sn1 * phibar
			phibar = cs1 * phibar
		}
		cs, sn, rho := symOrtho(rhobar1, beta)
		theta := sn * alfa
		rhobar = -cs * alfa
		phi := cs * phibar
		phibar = sn * phibar
		tau := sn * phi
		t1, t2 := phi/rho, -theta/rho
		for i := range w {
			dk[i] = w[i] / rho
			x[i] += t1 * w[i]
			w[i] = v[i] + t2*w[i]
		}
		nd := nrm2(dk)
		ddnorm += nd * nd
		if o.CalcVar {
			for i, d := range dk {
				res.Var[i] += d * d
			}
		}
		delta := sn2 * rho
		gambar := -cs2 * rho
		rhs := phi - delta*z
		zbar := rhs / gambar
		res.XNorm = math.Sqrt(xxnorm + zbar*zbar)
		gamma := math.Sqrt(gambar*gambar + theta*theta)
		cs2 = gambar / gamma
		sn2 = theta / gamma
		z = rhs / gamma
		xxnorm += z * z

		res.ACond = res.ANorm * math.Sqrt(ddnorm)
		res1 := phibar * phibar
		res2 += psi * psi
		rnorm := math.Sqrt(res1 + res2)
		res.ARNorm = alfa * math.Abs(tau)
		r1sq := rnorm*rnorm - dampsq*xxnorm
		res.R1Norm = math.Sqrt(math.Max(r1sq, 0)) // negative only by roundoff
		res.R2Norm = rnorm
		test1 := rnorm / bnorm
		test2 := res.ARNorm / (res.ANorm*rnorm + eps)
		test3 := 1 / (res.ACond + eps)
		t1 = test1 / (1 + res.ANorm*res.XNorm/bnorm)
		rtol := btol + atol*res.ANorm*res.XNorm/bnorm
		// Later tests override earlier ones, as in SciPy: the first four
		// guard against extremely small tolerances, the last three are the
		// user's.
		for _, g := range [...]struct {
			stop int
			hit  bool
		}{
			{7, res.Itn >= iterLim}, {6, 1+test3 <= 1}, {5, 1+test2 <= 1}, {4, 1+t1 <= 1},
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

func scale(x []float64, s float64) {
	for i := range x {
		x[i] *= s
	}
}
