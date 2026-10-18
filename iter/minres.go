package iter

import "math"

// MINRES solves A x = b for symmetric, possibly indefinite A (Paige and
// Saunders 1975), with the stopping tests of scipy.sparse.linalg.minres:
// relative residual ||r||/(||A|| ||x||) against RTol (default 1e-5), and the
// least-squares test. Info is 0 on success and the iteration limit when it
// was reached; o.MaxIter defaults to 5n. The preconditioner must be
// symmetric positive definite.
func MINRES(a Operator, b []float64, o *Options) (Result, error) {
	if o == nil {
		o = &Options{}
	}
	x, _, bnorm, err := setup(a, b, o)
	if err != nil {
		return Result{}, err
	}
	n := len(b)
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 5 * n
	}
	rtol := o.rtol()
	const eps = 2.220446049250313e-16
	psolve := o.psolve()
	r1 := make([]float64, n)
	if o.X0 == nil {
		copy(r1, b)
	} else {
		a.MulVec(r1, x)
		for i := range r1 {
			r1[i] = b[i] - r1[i] + o.Shift*x[i]
		}
	}
	y := make([]float64, n)
	psolve(y, r1)
	beta1 := dot(r1, y)
	if beta1 < 0 {
		return Result{}, ErrBreakdown // indefinite preconditioner
	}
	if beta1 == 0 {
		return Result{X: x}, nil
	}
	if bnorm == 0 {
		return Result{X: make([]float64, n)}, nil
	}
	beta1 = math.Sqrt(beta1)

	var oldb, dbar, epsln, tnorm2, gmax, rhs2 float64
	beta := beta1
	phibar := beta1
	rhs1 := beta1
	gmin := math.MaxFloat64
	cs, sn := -1.0, 0.0
	w := make([]float64, n)
	w1 := make([]float64, n)
	w2 := make([]float64, n)
	v := make([]float64, n)
	r2 := append([]float64(nil), r1...)
	info := 0
	istop := 0
	itn := 0
	for itn < maxiter {
		itn++
		s := 1 / beta
		for i := range v {
			v[i] = s * y[i]
		}
		a.MulVec(y, v)
		if o.Shift != 0 {
			axpy(-o.Shift, v, y)
		}
		if itn >= 2 {
			axpy(-beta/oldb, r1, y)
		}
		alfa := dot(v, y)
		axpy(-alfa/beta, r2, y)
		r1, r2 = r2, r1
		copy(r2, y)
		psolve(y, r2)
		oldb = beta
		beta = dot(r2, y)
		if beta < 0 {
			return Result{}, ErrBreakdown
		}
		beta = math.Sqrt(beta)
		tnorm2 += alfa*alfa + oldb*oldb + beta*beta
		if itn == 1 && beta/beta1 <= 10*eps {
			istop = -1
		}
		oldeps := epsln
		delta := cs*dbar + sn*alfa
		gbar := sn*dbar - cs*alfa
		epsln = sn * beta
		dbar = -cs * beta
		root := math.Hypot(gbar, dbar)
		gamma := math.Max(math.Hypot(gbar, beta), eps)
		cs = gbar / gamma
		sn = beta / gamma
		phi := cs * phibar
		phibar = sn * phibar
		denom := 1 / gamma
		w1, w2, w = w2, w, w1
		for i := range w {
			w[i] = (v[i] - oldeps*w1[i] - delta*w2[i]) * denom
		}
		axpy(phi, w, x)
		gmax = math.Max(gmax, gamma)
		gmin = math.Min(gmin, gamma)
		z := rhs1 / gamma
		rhs1 = rhs2 - delta*z
		rhs2 = -epsln * z

		anorm := math.Sqrt(tnorm2)
		ynorm := nrm2(x)
		epsx := anorm * ynorm * eps
		rnorm := phibar
		test1, test2 := math.Inf(1), math.Inf(1)
		if ynorm != 0 && anorm != 0 {
			test1 = rnorm / (anorm * ynorm)
		}
		if anorm != 0 {
			test2 = root / anorm
		}
		acond := gmax / gmin
		if istop == 0 {
			// later tests override earlier ones, as in SciPy
			for _, g := range [...]struct {
				stop int
				hit  bool
			}{
				{2, 1+test2 <= 1}, {1, 1+test1 <= 1}, {6, itn >= maxiter}, {4, acond >= 0.1/eps},
				{3, epsx >= beta1}, {2, test2 <= rtol}, {1, test1 <= rtol},
			} {
				if g.hit {
					istop = g.stop
				}
			}
		}
		if o.Callback != nil {
			o.Callback(x)
		}
		if istop != 0 {
			break
		}
	}
	if istop == 6 {
		info = maxiter
	}
	return Result{X: x, Info: info, Iterations: itn}, nil
}
