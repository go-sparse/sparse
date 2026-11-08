package bench

import (
	"context"
	"fmt"
	"os/exec"
	"sync"
	"time"

	"github.com/go-scikit/refbench"
	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
)

// ---------------------------------------------------------------------------
// The Python reference: one worker, started on the first timing call, so the
// CI mode (allocation counts, no reference) never needs Python or the network.

var py struct {
	once sync.Once
	w    *refbench.Worker
	err  error
}

func pyWorker() (*refbench.Worker, error) {
	py.once.Do(func() {
		uv, err := exec.LookPath("uv")
		if err != nil {
			py.err = fmt.Errorf("the scipy references need uv: %w", err)
			return
		}
		py.w, py.err = refbench.StartPython(context.Background(), uv, "refs.txt")
	})
	return py.w, py.err
}

// pyTimer times iters calls of the Python lambda call on the arguments of setup.
func pyTimer(setup, call string) refbench.Timer {
	return func(iters int) (time.Duration, error) {
		w, err := pyWorker()
		if err != nil {
			return 0, err
		}
		// n is a size the setups do not use; the worker requires a nonzero one
		// (the request omits a zero n).
		return w.Timer(setup, call, 1)(iters)
	}
}

// pySetup wraps a tuple body: the arguments of every call start with the
// modules sp (scipy.sparse), spl (scipy.sparse.linalg), sio (scipy.io) and csg
// (scipy.sparse.csgraph), then the tuple's own elements.
func pySetup(tuple string) string {
	return "(lambda sp,spl,sio,csg: (sp,spl,sio,csg)+(" + tuple + ",))(" +
		"__import__('scipy.sparse',fromlist=['x']),__import__('scipy.sparse.linalg',fromlist=['x'])," +
		"__import__('scipy.io',fromlist=['x']),__import__('scipy.sparse.csgraph',fromlist=['x']))"
}

// pyCall is the lambda of a call: params names the tuple's own elements.
func pyCall(params, body string) string {
	if params != "" {
		params = "," + params
	}
	return "lambda sp,spl,sio,csg" + params + ": " + body
}

// pyCSR makes a canonical (sorted) CSR of a scipy matrix expression.
func pyCSR(m string) string {
	return "(lambda M:(M.sort_indices(),M)[1])((" + m + ").tocsr())"
}

// pyLap2D is gallery.Laplacian2D(g, g): kron(I, T) + kron(T, I), T = tridiag(-1, 2, -1).
func pyLap2D(g int) string {
	return pyCSR(fmt.Sprintf("(lambda T,I: sp.kron(I,T)+sp.kron(T,I))(sp.diags([-1.0,2.0,-1.0],[-1,0,1],shape=(%[1]d,%[1]d)),sp.identity(%[1]d))", g))
}

// pyTri is the tridiagonal sparse.Diags({-1, 2, -1}, {-1, 0, 1}, n, n).
func pyTri(n int) string {
	return pyCSR(fmt.Sprintf("sp.diags([-np.ones(%[1]d-1),2*np.ones(%[1]d),-np.ones(%[1]d-1)],[-1,0,1],shape=(%[1]d,%[1]d))", n))
}

// pyVec is vec's data: x[i] = 1 + i mod 7.
func pyVec(n int) string { return fmt.Sprintf("(np.arange(%d) %% 7 + 1.0)", n) }

func vec(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 1 + float64(i%7)
	}
	return x
}

// tri is pyTri's matrix.
func tri(n int) *sparse.CSR[float64] {
	lo, di, up := make([]float64, n-1), make([]float64, n), make([]float64, n-1)
	for i := range lo {
		lo[i], up[i] = -1, -1
	}
	for i := range di {
		di[i] = 2
	}
	b, err := sparse.Diags([][]float64{lo, di, up}, []int{-1, 0, 1}, n, n)
	if err != nil {
		panic(err)
	}
	return b
}

// Sizes of the inputs.
const (
	spGrid  = 100 // the 100 x 100 grid: n = 10000, nnz = 49600
	krGrid  = 50  // the Krylov systems: n = 2500
	smGrid  = 30  // the dense conversions: n = 900
	rcmGrid = 64  // the ordering: n = 4096

	refSP  = "scipy.sparse 1.18.1"
	refSPL = "scipy.sparse.linalg 1.18.1"
)

var sinkAny any

// each runs f iters times and keeps the last value alive.
func each[T any](iters int, f func() T) {
	var v T
	for i := 0; i < iters; i++ {
		v = f()
	}
	sinkAny = v
}

// spCase is a row against scipy: the Go body, the Python setup tuple and call.
func spCase(name, fn, ref, params, body, tuple string, goBody func(iters int)) refbench.Case {
	return refbench.Case{
		Name: name, Func: fn, Reference: ref,
		// Values agree with the SciPy oracle in internal/conformance (the
		// committed fixture); the data identity of the timed inputs is
		// TestDataIdentity.
		Parity: refbench.ParityTolerance,
		Go:     goBody,
		Ref:    pyTimer(pySetup(tuple), pyCall(params, body)),
	}
}

// loop runs f iters times.
func loop(f func()) func(int) {
	return func(it int) {
		for i := 0; i < it; i++ {
			f()
		}
	}
}

// sparseCases are the rows of the sparse package itself.
func sparseCases() []refbench.Case {
	var out []refbench.Case
	add := func(c ...refbench.Case) { out = append(out, c...) }

	n := spGrid * spGrid
	a, b := gallery.Laplacian2D(spGrid, spGrid), tri(n)
	ac, coo := a.ToCSC(), a.ToCOO()
	up := a.Triu(0).ToCSC()
	x, dst := vec(n), make([]float64, n)
	L, T := pyLap2D(spGrid), pyTri(n)
	AB, Ax := L+","+T, L+","+pyVec(n)

	add(
		spCase("CSR.MulVec/lap2d-100", "CSR.MulVec", refSP, "A,x", "A@x", Ax, loop(func() { a.MulVec(dst, x) })),
		spCase("CSR.MulVecT/lap2d-100", "CSR.MulVecT", refSP+" (A.T@x)", "A,x", "A.T@x", Ax, loop(func() { a.MulVecT(dst, x) })),
		spCase("CSC.MulVec/lap2d-100", "CSC.MulVec", refSP, "A,x", "A@x", L+".tocsc(),"+pyVec(n), loop(func() { ac.MulVec(dst, x) })),
		spCase("CSC.MulVecT/lap2d-100", "CSC.MulVecT", refSP+" (A.T@x)", "A,x", "A.T@x", L+".tocsc(),"+pyVec(n), loop(func() { ac.MulVecT(dst, x) })),
		spCase("CSC.MulVecSymUpper/lap2d-100", "CSC.MulVecSymUpper", refSP+" (U@x + U.T@x - diag(U)*x)", "U,x",
			"U@x+U.T@x-U.diagonal()*x", "sp.triu("+L+",format='csc'),"+pyVec(n), loop(func() { up.MulVecSymUpper(dst, x) })),
	)

	const k = 4
	bm, dm := make([]float64, n*k), make([]float64, n*k)
	for i := range bm {
		bm[i] = 1 + float64(i%7)
	}
	add(spCase("CSR.MulDense/lap2d-100x4", "CSR.MulDense", refSP, "A,X", "A@X",
		fmt.Sprintf("%s,(np.arange(%d) %% 7 + 1.0).reshape(%d,%d)", L, n*k, n, k), loop(func() { a.MulDense(dm, bm, k) })))

	csr := func(f func() (*sparse.CSR[float64], error)) func(int) {
		return func(it int) { each(it, func() *sparse.CSR[float64] { r, _ := f(); return r }) }
	}
	add(
		spCase("Add/lap2d+tri", "Add", refSP, "A,B", "A+B", AB, csr(func() (*sparse.CSR[float64], error) { return sparse.Add(a, b) })),
		spCase("Sub/lap2d-tri", "Sub", refSP, "A,B", "A-B", AB, csr(func() (*sparse.CSR[float64], error) { return sparse.Sub(a, b) })),
		spCase("Mul/lap2d*tri", "Mul", refSP, "A,B", "A@B", AB, csr(func() (*sparse.CSR[float64], error) { return sparse.Mul(a, b) })),
		spCase("Axpby/2lap2d+3tri", "Axpby", refSP+" (2.0*A + 3.0*B)", "A,B", "2.0*A+3.0*B", AB,
			csr(func() (*sparse.CSR[float64], error) { return sparse.Axpby(2, a, 3, b) })),
	)

	add(
		spCase("CSR.ToCSC/lap2d-100", "CSR.ToCSC", refSP, "A", "A.tocsc()", L, func(it int) { each(it, a.ToCSC) }),
		spCase("CSC.ToCSR/lap2d-100", "CSC.ToCSR", refSP, "A", "A.tocsr()", L+".tocsc()", func(it int) { each(it, ac.ToCSR) }),
		spCase("CSR.T/lap2d-100", "CSR.T", refSP+" (A.T.tocsr())", "A", "A.T.tocsr()", L, func(it int) { each(it, a.T) }),
		spCase("CSR.Transpose/lap2d-100", "CSR.Transpose", refSP+" (A.transpose(), a view)", "A", "A.transpose()", L, func(it int) { each(it, a.Transpose) }),
		spCase("CSR.ToCOO/lap2d-100", "CSR.ToCOO", refSP, "A", "A.tocoo()", L, func(it int) { each(it, a.ToCOO) }),
		spCase("COO.ToCSR/lap2d-100", "COO.ToCSR", refSP, "M", "M.tocsr()", L+".tocoo()", func(it int) { each(it, coo.ToCSR) }),
		spCase("COO.ToCSC/lap2d-100", "COO.ToCSC", refSP, "M", "M.tocsc()", L+".tocoo()", func(it int) { each(it, coo.ToCSC) }),
		spCase("CSR.Clone/lap2d-100", "CSR.Clone", refSP+" (A.copy())", "A", "A.copy()", L, func(it int) { each(it, a.Clone) }),
		spCase("CSR.Diagonal/lap2d-100", "CSR.Diagonal", refSP, "A", "A.diagonal(0)", L, func(it int) { each(it, func() []float64 { return a.Diagonal(0) }) }),
		spCase("CSR.Tril/lap2d-100", "CSR.Tril", refSP+" (sp.tril, csr)", "A", "sp.tril(A,k=0,format='csr')", L, func(it int) { each(it, func() *sparse.CSR[float64] { return a.Tril(0) }) }),
		spCase("CSR.Triu/lap2d-100", "CSR.Triu", refSP+" (sp.triu, csr)", "A", "sp.triu(A,k=0,format='csr')", L, func(it int) { each(it, func() *sparse.CSR[float64] { return a.Triu(0) }) }),
	)

	// Scaling (in place on a copy, by factors that keep the data finite) and norms.
	sc, d := a.Clone(), vec(n)
	add(
		spCase("CSR.Scale/lap2d-100", "CSR.Scale", refSP+" (in place: A.data *= 1.0)", "A", "A.data.__imul__(1.0)", L, loop(func() { sc.Scale(1) })),
		spCase("CSR.ScaleRows/lap2d-100", "CSR.ScaleRows", refSP+" (sp.diags(d) @ A, a new matrix)", "A,d", "sp.diags(d)@A", Ax, loop(func() { _ = sc.ScaleRows(d) })),
		spCase("CSR.ScaleCols/lap2d-100", "CSR.ScaleCols", refSP+" (A @ sp.diags(d), a new matrix)", "A,d", "A@sp.diags(d)", Ax, loop(func() { _ = sc.ScaleCols(d) })),
		spCase("CSR.Norm1/lap2d-100", "CSR.Norm1", refSPL, "A", "spl.norm(A,1)", L, func(it int) { each(it, a.Norm1) }),
		spCase("CSR.NormInf/lap2d-100", "CSR.NormInf", refSPL, "A", "spl.norm(A,np.inf)", L, func(it int) { each(it, a.NormInf) }),
		spCase("CSR.NormFro/lap2d-100", "CSR.NormFro", refSPL, "A", "spl.norm(A,'fro')", L, func(it int) { each(it, a.NormFro) }),
	)
	pr := a.Clone()
	add(
		spCase("CSR.Prune/lap2d-100", "CSR.Prune", refSP+" (A.eliminate_zeros(), in place)", "A", "A.eliminate_zeros()", L, loop(func() { pr.Prune(0) })),
		spCase("CSR.IsSymmetric/lap2d-100", "CSR.IsSymmetric", refSP+" ((A != A.T).nnz == 0)", "A", "(A!=A.T).nnz==0", L,
			func(it int) { each(it, func() bool { return a.IsSymmetric(0) }) }),
		spCase("CSR.Validate/lap2d-100", "CSR.Validate", refSP+" (A.check_format(full_check=True))", "A", "A.check_format(full_check=True)", L,
			func(it int) { each(it, a.Validate) }),
	)

	// B[i,j] = A[p[i], p[j]] with p[i] = 7919 i mod n.
	p := make([]int, n)
	for i := range p {
		p[i] = i * 7919 % n
	}
	add(spCase("CSR.Permute/lap2d-100", "CSR.Permute", refSP+" (A[p,:][:,p])", "A,p", "A[p,:][:,p]",
		fmt.Sprintf("%s,(np.arange(%d)*7919)%%%d", L, n, n), csr(func() (*sparse.CSR[float64], error) { return a.Permute(p, p) })))

	// Construction and dense conversion.
	sn := smGrid * smGrid
	sm := gallery.Laplacian2D(smGrid, smGrid)
	dense := sm.Dense()
	lo, di := make([]float64, n-1), make([]float64, n)
	for i := range lo {
		lo[i] = -1
	}
	for i := range di {
		di[i] = 2
	}
	add(
		spCase("CSR.Dense/lap2d-30", "CSR.Dense", refSP+" (A.toarray())", "A", "A.toarray()", pyLap2D(smGrid), func(it int) { each(it, sm.Dense) }),
		spCase("FromDense/lap2d-30", "FromDense", refSP+" (sp.csr_matrix(D))", "D", "sp.csr_matrix(D)", pyLap2D(smGrid)+".toarray()",
			csr(func() (*sparse.CSR[float64], error) { return sparse.FromDense(sn, sn, dense) })),
		spCase("Eye/10000", "Eye", refSP+" (sp.eye(n, format='csr'))", "n", "sp.eye(n,format='csr')", fmt.Sprint(n),
			func(it int) { each(it, func() *sparse.CSR[float64] { return sparse.Eye[float64](n) }) }),
		spCase("Diags/tri-10000", "Diags", refSP+" (sp.diags(...).tocsr())", "L,D,U", "sp.diags([L,D,U],[-1,0,1],format='csr')",
			fmt.Sprintf("-np.ones(%[1]d-1),2*np.ones(%[1]d),-np.ones(%[1]d-1)", n),
			csr(func() (*sparse.CSR[float64], error) {
				return sparse.Diags([][]float64{lo, di, lo}, []int{-1, 0, 1}, n, n)
			})),
		spCase("gallery.Laplacian1D/10000", "gallery.Laplacian1D", refSP+" (sp.diags)", "n",
			"sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')", fmt.Sprint(n),
			func(it int) { each(it, func() *sparse.CSR[float64] { return gallery.Laplacian1D(n) }) }),
		spCase("gallery.Laplacian2D/100", "gallery.Laplacian2D", refSP+" (kron(I,T) + kron(T,I))", "g",
			"(lambda T,I: (sp.kron(I,T)+sp.kron(T,I)).tocsr())(sp.diags([-1.0,2.0,-1.0],[-1,0,1],shape=(g,g)),sp.identity(g))", "100",
			func(it int) {
				each(it, func() *sparse.CSR[float64] { return gallery.Laplacian2D(spGrid, spGrid) })
			}),
		spCase("gallery.Laplacian3D/22", "gallery.Laplacian3D", refSP+" (three kron terms)", "g",
			"(lambda T,I: (sp.kron(sp.kron(I,I),T)+sp.kron(sp.kron(I,T),I)+sp.kron(sp.kron(T,I),I)).tocsr())(sp.diags([-1.0,2.0,-1.0],[-1,0,1],shape=(g,g)),sp.identity(g))", "22",
			func(it int) {
				each(it, func() *sparse.CSR[float64] { return gallery.Laplacian3D(22, 22, 22) })
			}),
	)
	return out
}
