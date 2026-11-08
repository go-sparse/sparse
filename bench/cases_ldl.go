package bench

import (
	"fmt"
	"time"

	"github.com/go-scikit/refbench"
	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/ldl"
	"github.com/go-sparse/sparse/order"
)

const refQDLDL = "QDLDL v0.1.9 (C, cgo, -O3)"

// gridKKT is a KKT matrix with n = g*g primal variables (5-point Laplacian +
// identity as P) and m = n constraints, each coupling x_i and x_(i+1): the
// matrix of an OSQP problem on a grid. Upper triangle, columns sorted.
func gridKKT(g int) *sparse.CSC[float64] {
	n := g * g
	ap := []int{0}
	var ai []int32
	var ax []float64
	for j := 0; j < n; j++ {
		if j >= g {
			ai, ax = append(ai, int32(j-g)), append(ax, -1)
		}
		if j%g != 0 {
			ai, ax = append(ai, int32(j-1)), append(ax, -1)
		}
		ai, ax = append(ai, int32(j)), append(ax, 4.1)
		ap = append(ap, len(ai))
	}
	for i := 0; i < n; i++ {
		a, b := i, (i+1)%n
		if a > b {
			a, b = b, a
		}
		ai = append(ai, int32(a), int32(b), int32(n+i))
		ax = append(ax, 1, 0.5, -10)
		ap = append(ap, len(ai))
	}
	N := 2 * n
	return &sparse.CSC[float64]{Rows: N, Cols: N, Indptr: ap, Indices: ai, Data: ax}
}

// kkt is one direct-solve problem: K (upper triangle), its AMD permutation, the
// upper triangle of P K Pᵀ and where its values come from.
type kkt struct {
	k    *sparse.CSC[float64]
	perm []int
	pa   *sparse.CSC[float64]
	pmap []int32 // pa.Data[q] = k.Data[pmap[q]]
	b    []float64
}

func newKKT(g int) (*kkt, error) {
	k := gridKKT(g)
	n := k.Cols
	// The symmetric matrix with the position of each upper entry as its value:
	// after the permutation the values say where each entry came from.
	full := sparse.NewCOO[float64](n, n)
	for j := 0; j < n; j++ {
		for p := k.Indptr[j]; p < k.Indptr[j+1]; p++ {
			i := int(k.Indices[p])
			if err := full.Append(i, j, float64(p+1)); err != nil {
				return nil, err
			}
			if i != j {
				if err := full.Append(j, i, float64(p+1)); err != nil {
					return nil, err
				}
			}
		}
	}
	csr := full.ToCSR()
	perm, err := order.AMD(csr)
	if err != nil {
		return nil, err
	}
	pm, err := csr.Permute(perm, perm)
	if err != nil {
		return nil, err
	}
	pa := pm.Triu(0).ToCSC()
	pmap := make([]int32, len(pa.Data))
	for q, v := range pa.Data {
		pmap[q] = int32(v) - 1
		pa.Data[q] = k.Data[pmap[q]]
	}
	return &kkt{k: k, perm: perm, pa: pa, pmap: pmap, b: vec(n)}, nil
}

// cTimer times a QDLDL loop.
func (r *Ref) timer(kind string) refbench.Timer {
	return func(iters int) (time.Duration, error) {
		t0 := time.Now()
		sink = r.Loop(kind, iters)
		return time.Since(t0), nil
	}
}

// ldlCases are the rows of the direct path against QDLDL.
func ldlCases() ([]refbench.Case, error) {
	var out []refbench.Case
	for _, g := range []int{12, 20, 40} {
		m, err := newKKT(g)
		if err != nil {
			return nil, err
		}
		N := m.k.Cols
		tag := fmt.Sprintf("kkt-%d", N)

		// Plain QDLDL on P K Pᵀ (the permuted pattern, factored in natural order).
		plain, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, nil, nil, nil)
		if err != nil {
			return nil, err
		}
		// The permuted path: QDLDL on the same factor, with the gather and
		// scatter an OSQP-style caller adds around it.
		permd, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, m.perm, m.pmap, m.k.Data)
		if err != nil {
			return nil, err
		}
		f, err := ldl.Factorize(m.pa)
		if err != nil {
			return nil, err
		}
		pf, err := ldl.NewPermuted(m.k, m.perm)
		if err != nil {
			return nil, err
		}
		if err := pf.Refactor(m.k); err != nil {
			return nil, err
		}
		x := make([]float64, N)
		solve := func(fn func([]float64) error) func(int) {
			return loop(func() { copy(x, m.b); _ = fn(x) })
		}
		row := func(name, fn, kind string, ref *Ref, body func(int)) refbench.Case {
			return refbench.Case{
				Name: name + "/" + tag, Func: fn, Reference: refQDLDL,
				// Solutions and factors agree to the last bit with the C code:
				// TestParityWithQDLDL.
				Parity: refbench.ParityExact,
				Go:     body, Ref: ref.timer(kind),
			}
		}
		out = append(out,
			row("ldl.Analyze", "ldl.Analyze", "etree", plain, func(it int) {
				each(it, func() *ldl.Symbolic { s, _ := ldl.Analyze(m.pa); return s })
			}),
			row("ldl.Factor.Refactor", "ldl.Factor.Refactor", "factor", plain, loop(func() { _ = f.Refactor(m.pa) })),
			row("ldl.Factor.Solve", "ldl.Factor.Solve", "solve", plain, solve(f.Solve)),
			row("ldl.Factor.LSolve", "ldl.Factor.LSolve", "lsolve", plain, solve(f.LSolve)),
			row("ldl.Factor.LTSolve", "ldl.Factor.LTSolve", "ltsolve", plain, solve(f.LTSolve)),
			row("ldl.Factor.DSolve", "ldl.Factor.DSolve", "dsolve", plain, solve(f.DSolve)),
			row("ldl.Permuted.Refactor", "ldl.Permuted.Refactor", "pfactor", permd, loop(func() { _ = pf.Refactor(m.k) })),
			row("ldl.Permuted.Solve", "ldl.Permuted.Solve", "psolve", permd, solve(pf.Solve)),
		)
	}
	return out, nil
}
