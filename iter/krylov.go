package iter

import "math"

// CG solves A x = b for symmetric positive definite A (scipy cg).
func CG(a Operator, b []float64, o *Options) (Result, error) {
	if o == nil {
		o = &Options{}
	}
	x, atol, bnorm, err := setup(a, b, o)
	if err != nil {
		return Result{}, err
	}
	if bnorm == 0 {
		return Result{X: make([]float64, len(b))}, nil
	}
	n := len(b)
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 10 * n
	}
	psolve := o.psolve()
	r := make([]float64, n)
	p := make([]float64, n)
	q := make([]float64, n)
	// Without a preconditioner z is r itself (scipy's identity psolve
	// returns its argument): no copy per iteration.
	z := r
	if o.M != nil {
		z = make([]float64, n)
	}
	residual(a, r, b, x)
	var rhoPrev float64
	rr := dot(r, r) // r·r, kept up to date by the fused update below
	for it := 0; it < maxiter; it++ {
		if sqrtNorm(rr, r) < atol {
			return Result{X: x, Iterations: it}, nil
		}
		rho := rr // z is r
		if o.M != nil {
			psolve(z, r)
			rho = dot(r, z)
		}
		if it > 0 {
			xpby(z, rho/rhoPrev, p)
		} else {
			copy(p, z)
		}
		a.MulVec(q, p)
		alpha := rho / dot(p, q)
		axpy(alpha, p, x)
		rr = axpyDot(-alpha, q, r, r) // r -= alpha q, and the next r·r in the same pass
		rhoPrev = rho
		if o.Callback != nil {
			o.Callback(x)
		}
	}
	return Result{X: x, Info: maxiter, Iterations: maxiter}, nil
}

// BiCGSTAB solves A x = b for general square A (scipy bicgstab).
func BiCGSTAB(a Operator, b []float64, o *Options) (Result, error) {
	if o == nil {
		o = &Options{}
	}
	x, atol, bnorm, err := setup(a, b, o)
	if err != nil {
		return Result{}, err
	}
	if bnorm == 0 {
		return Result{X: make([]float64, len(b))}, nil
	}
	n := len(b)
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 10 * n
	}
	psolve := o.psolve()
	const eps = 2.220446049250313e-16
	rhotol := eps * eps
	omegatol := rhotol
	r := make([]float64, n)
	residual(a, r, b, x)
	rt := append([]float64(nil), r...)
	p := make([]float64, n)
	v := make([]float64, n)
	s := make([]float64, n)
	t := make([]float64, n)
	ph := make([]float64, n)
	sh := make([]float64, n)
	var rhoPrev, omega, alpha float64
	for it := 0; it < maxiter; it++ {
		if nrm2(r) < atol {
			return Result{X: x, Iterations: it}, nil
		}
		rho := dot(rt, r)
		if math.Abs(rho) < rhotol {
			return Result{X: x, Info: BreakdownRho, Iterations: it}, nil
		}
		if it > 0 {
			if math.Abs(omega) < omegatol {
				return Result{X: x, Info: BreakdownOmega, Iterations: it}, nil
			}
			beta := (rho / rhoPrev) * (alpha / omega)
			for i := range p {
				p[i] = r[i] + beta*(p[i]-omega*v[i])
			}
		} else {
			copy(p, r)
		}
		psolve(ph, p)
		a.MulVec(v, ph)
		rv := dot(rt, v)
		if rv == 0 {
			return Result{X: x, Info: BreakdownOmega, Iterations: it}, nil
		}
		alpha = rho / rv
		axpy(-alpha, v, r)
		copy(s, r)
		if nrm2(s) < atol {
			axpy(alpha, ph, x)
			return Result{X: x, Iterations: it}, nil
		}
		psolve(sh, s)
		a.MulVec(t, sh)
		omega = dot(t, s) / dot(t, t)
		axpy(alpha, ph, x)
		axpy(omega, sh, x)
		axpy(-omega, t, r)
		rhoPrev = rho
		if o.Callback != nil {
			o.Callback(x)
		}
	}
	return Result{X: x, Info: maxiter, Iterations: maxiter}, nil
}

// GMRES solves A x = b with restarted GMRES(m), left-preconditioned, using
// modified Gram-Schmidt and Givens rotations, with the inner-iteration
// tolerance control of scipy.sparse.linalg.gmres (gh-8400): the inner loop
// stops on the preconditioned residual and the outer loop checks the true
// residual.
func GMRES(a Operator, b []float64, o *Options) (Result, error) {
	if o == nil {
		o = &Options{}
	}
	x, atol, bnorm, err := setup(a, b, o)
	if err != nil {
		return Result{}, err
	}
	if bnorm == 0 {
		return Result{X: make([]float64, len(b))}, nil
	}
	n := len(b)
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 10 * n
	}
	restart := o.Restart
	if restart == 0 {
		restart = 20
	}
	restart = min(restart, n)
	const eps = 2.220446049250313e-16
	psolve := o.psolve()
	tmp := make([]float64, n)
	psolve(tmp, b)
	mbnorm := nrm2(tmp)
	ptolMax := 1.0
	ptol := mbnorm * math.Min(ptolMax, atol/bnorm)
	presid := 0.0

	v := make([][]float64, restart+1)
	for i := range v {
		v[i] = make([]float64, n)
	}
	h := make([][]float64, restart) // h[col][row], row <= col+1
	for i := range h {
		h[i] = make([]float64, restart+1)
	}
	cs := make([]float64, restart)
	sn := make([]float64, restart)
	s := make([]float64, restart+1)
	y := make([]float64, restart)
	r := make([]float64, n)
	w := make([]float64, n)
	av := make([]float64, n)
	inner := 0
	var rnorm float64
	for it := 0; it < maxiter; it++ {
		if it == 0 {
			residual(a, r, b, x)
			if nrm2(r) < atol {
				return Result{X: x}, nil
			}
		}
		psolve(v[0], r)
		beta := nrm2(v[0])
		for i := range v[0] {
			v[0][i] *= 1 / beta
		}
		clear(s)
		s[0] = beta
		breakdown := false
		col := 0
		for col = 0; col < restart; col++ {
			if o.M == nil {
				a.MulVec(w, v[col]) // identity preconditioner: no copy
			} else {
				a.MulVec(av, v[col])
				psolve(w, av)
			}
			h0 := nrm2(w)
			// Modified Gram-Schmidt, each update fused with the next
			// coefficient (and the last one with the norm): one pass per
			// basis vector instead of two, the same operations in the same
			// order.
			t := dot(v[0], w)
			var ww float64
			for k := 0; k <= col; k++ {
				h[col][k] = t
				if k < col {
					t = axpyDot(-t, v[k], w, v[k+1])
				} else {
					ww = axpyDot(-t, v[k], w, w)
				}
			}
			h1 := sqrtNorm(ww, w)
			h[col][col+1] = h1
			copy(v[col+1], w)
			if h1 <= eps*h0 {
				h[col][col+1] = 0
				breakdown = true
			} else {
				for i := range v[col+1] {
					v[col+1][i] *= 1 / h1
				}
			}
			for k := 0; k < col; k++ {
				n0, n1 := h[col][k], h[col][k+1]
				h[col][k], h[col][k+1] = cs[k]*n0+sn[k]*n1, -sn[k]*n0+cs[k]*n1
			}
			c, sg, mag := givens(h[col][col], h[col][col+1])
			cs[col], sn[col] = c, sg
			h[col][col], h[col][col+1] = mag, 0
			t = -sg * s[col]
			s[col], s[col+1] = c*s[col], t
			presid = math.Abs(t)
			inner++
			if presid <= ptol || breakdown {
				break
			}
		}
		if col == restart { // loop ran to completion: col indexes the last column
			col = restart - 1
		}
		if h[col][col] == 0 {
			s[col] = 0
		}
		copy(y[:col+1], s[:col+1])
		for k := col; k > 0; k-- {
			if y[k] != 0 {
				y[k] /= h[k][k]
				axpy(-y[k], h[k][:k], y[:k])
			}
		}
		if y[0] != 0 {
			y[0] /= h[0][0]
		}
		for k := 0; k <= col; k++ {
			axpy(y[k], v[k], x)
		}
		a.MulVec(av, x)
		for i := range r {
			r[i] = b[i] - av[i]
		}
		rnorm = nrm2(r)
		if o.Callback != nil {
			o.Callback(x)
		}
		if rnorm <= atol {
			break
		} else if breakdown {
			break
		} else if presid <= ptol {
			ptolMax = math.Max(eps, 0.25*ptolMax)
		} else {
			ptolMax = math.Min(1, 1.5*ptolMax)
		}
		ptol = presid * math.Min(ptolMax, atol/rnorm)
	}
	info := 0
	if rnorm > atol {
		info = maxiter
	}
	return Result{X: x, Info: info, Iterations: inner}, nil
}

// givens returns c, s, r with [c s; -s c] [f; g] = [r; 0].
func givens(f, g float64) (c, s, r float64) {
	if g == 0 {
		return 1, 0, f
	}
	r = math.Hypot(f, g)
	return f / r, g / r, r
}
