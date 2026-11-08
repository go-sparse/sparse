package bench

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"

	"github.com/go-scikit/refbench"
	"github.com/go-sparse/sparse/gallery"
	"github.com/go-sparse/sparse/iter"
	"github.com/go-sparse/sparse/mtx"
	"github.com/go-sparse/sparse/order"
)

// krylovCases time every Krylov solver for a fixed amount of work, 25
// iterations (GMRES: 2 restart cycles of 20) with both tolerances at zero, so
// that both sides do exactly the same number of matrix products.
func krylovCases() []refbench.Case {
	n := krGrid * krGrid
	a := gallery.Laplacian2D(krGrid, krGrid)
	b := vec(n)
	L := pyLap2D(krGrid)
	Ab := L + "," + pyVec(n)
	// The solver is bound in the setup, so the lambda does not import per call.
	call := func(fn string) (string, string) { return "A,b,f", "f(A,b," + fn + ")" }
	var out []refbench.Case
	add := func(name, fn, solver, args string, body func(int)) {
		params, c := call(args)
		out = append(out, spCase(name, fn, refSPL+" ("+solver+")", params, c, Ab+",spl."+solver, body))
	}
	opt := &iter.Options{RTol: -1, MaxIter: 25}
	add("iter.CG/lap2d-50", "iter.CG", "cg", "rtol=0.0,atol=0.0,maxiter=25",
		func(it int) { each(it, func() iter.Result { r, _ := iter.CG(a, b, opt); return r }) })
	add("iter.BiCGSTAB/lap2d-50", "iter.BiCGSTAB", "bicgstab", "rtol=0.0,atol=0.0,maxiter=25",
		func(it int) { each(it, func() iter.Result { r, _ := iter.BiCGSTAB(a, b, opt); return r }) })
	add("iter.MINRES/lap2d-50", "iter.MINRES", "minres", "rtol=0.0,maxiter=25",
		func(it int) { each(it, func() iter.Result { r, _ := iter.MINRES(a, b, opt); return r }) })
	gm := &iter.Options{RTol: -1, MaxIter: 2, Restart: 20}
	add("iter.GMRES/lap2d-50", "iter.GMRES", "gmres", "rtol=0.0,atol=0.0,restart=20,maxiter=2",
		func(it int) { each(it, func() iter.Result { r, _ := iter.GMRES(a, b, gm); return r }) })
	lq := &iter.LSQROptions{ATol: -1, BTol: -1, ConLim: -1, IterLim: 25}
	add("iter.LSQR/lap2d-50", "iter.LSQR", "lsqr", "atol=0.0,btol=0.0,conlim=0.0,iter_lim=25",
		func(it int) { each(it, func() iter.LSQRResult { r, _ := iter.LSQR(a, b, lq); return r }) })
	lm := &iter.LSMROptions{ATol: -1, BTol: -1, ConLim: -1, MaxIter: 25}
	add("iter.LSMR/lap2d-50", "iter.LSMR", "lsmr", "atol=0.0,btol=0.0,conlim=0.0,maxiter=25",
		func(it int) { each(it, func() iter.LSMRResult { r, _ := iter.LSMR(a, b, lm); return r }) })

	xs := vec(n)
	out = append(out, spCase("iter.TrueResidual/lap2d-50", "iter.TrueResidual", "numpy 2.5.4 (norm(b - A@x)/norm(b))",
		"A,b,x", "np.linalg.norm(b-A@x)/np.linalg.norm(b)", Ab+","+pyVec(n),
		func(it int) { each(it, func() float64 { return iter.TrueResidual(a, b, xs) }) }))
	return out
}

// orderCases: reverse Cuthill-McKee against scipy.sparse.csgraph. AMD, COLAMD
// and the fill counts have no reference in the pinned set (SciPy ships none):
// they are named in docs/perf-exempt.md, and their allocations are asserted
// by order/alloc_test.go.
func orderCases() ([]refbench.Case, error) {
	n := rcmGrid * rcmGrid
	p := make([]int, n)
	for i := range p {
		p[i] = i * 7919 % n
	}
	b, err := gallery.Laplacian2D(rcmGrid, rcmGrid).Permute(p, p)
	if err != nil {
		return nil, err
	}
	L := pyLap2D(rcmGrid)
	return []refbench.Case{spCase("order.RCM/lap2d-64-scrambled", "order.RCM",
		"scipy.sparse.csgraph.reverse_cuthill_mckee 1.18.1", "B", "csg.reverse_cuthill_mckee(B)",
		fmt.Sprintf("(lambda A,p: (lambda B:(B.sort_indices(),B)[1])(A[p,:][:,p].tocsr()))(%s,(np.arange(%d)*7919)%%%d)", L, n, n),
		func(it int) { each(it, func() []int { r, _ := order.RCM(b); return r }) })}, nil
}

// mtxCases: Matrix Market reading and writing against scipy.io (the
// fast_matrix_market backend) on the same bytes.
func mtxCases() ([]refbench.Case, error) {
	a := gallery.Laplacian2D(smGrid, smGrid)
	var buf bytes.Buffer
	if err := mtx.Write(&buf, a, mtx.Options{}); err != nil {
		return nil, err
	}
	data := buf.Bytes()
	L := pyLap2D(smGrid)
	return []refbench.Case{
		spCase("mtx.Read/lap2d-30", "mtx.Read", "scipy.io.mmread 1.18.1 (same bytes)", "data", "sio.mmread(__import__('io').BytesIO(data))",
			"bytes.fromhex('"+hex.EncodeToString(data)+"')",
			func(it int) {
				each(it, func() *mtx.Matrix { m, _ := mtx.Read(bytes.NewReader(data)); return m })
			}),
		spCase("mtx.Write/lap2d-30", "mtx.Write", "scipy.io.mmwrite 1.18.1", "A", "sio.mmwrite(__import__('io').BytesIO(),A,symmetry='general')", L,
			func(it int) {
				for i := 0; i < it; i++ {
					_ = mtx.Write(io.Discard, a, mtx.Options{})
				}
			}),
	}, nil
}

// allCases is the registry.
func allCases() ([]refbench.Case, error) {
	out := sparseCases()
	out = append(out, krylovCases()...)
	for _, f := range []func() ([]refbench.Case, error){ldlCases, orderCases, mtxCases} {
		cs, err := f()
		if err != nil {
			return nil, err
		}
		out = append(out, cs...)
	}
	return out, nil
}
