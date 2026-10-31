package ldl

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/nd"
	"github.com/go-sparse/sparse/order"
)

// upper builds a CSC value from QDLDL-style arrays (no validation, so that
// unsorted columns, as in QDLDL's own tests, are kept).
func upper(n int, ap []int, ai []int32, ax []float64) *sparse.CSC[float64] {
	return &sparse.CSC[float64]{Rows: n, Cols: n, Indptr: ap, Indices: ai, Data: ax}
}

func maxDiff(x, y []float64) float64 {
	m := 0.0
	for i := range x {
		m = math.Max(m, math.Abs(x[i]-y[i]))
	}
	return m
}

// The nine test cases of QDLDL v0.1.9 (tests/test_*.h), same data, same
// tolerance (QDLDL_TESTS_TOL = 1e-4 on the max-norm of x - xsol).
func TestQDLDLCases(t *testing.T) {
	cases := []struct {
		name string
		n    int
		ap   []int
		ai   []int32
		ax   []float64
		b    []float64
		xsol []float64 // nil: factorisation must fail
		want error
	}{
		{"basic", 10, []int{0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17},
			[]int32{0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9},
			[]float64{1.0, 0.460641, -0.121189, 0.417928, 0.177828, 0.1,
				-0.0290058, -1.0, 0.350321, -0.441092, -0.0845395, -0.316228,
				0.178663, -0.299077, 0.182452, -1.56506, -0.1},
			[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			[]float64{10.2171, 3.9416, -5.69096, 9.28661, 50.0, -6.11433, -26.3104, -27.7809, -45.8099, -3.74178}, nil},
		{"identity", 4, []int{0, 1, 2, 3, 4}, []int32{0, 1, 2, 3}, []float64{1, 1, 1, 1},
			[]float64{2, 2, 2, 2}, []float64{2, 2, 2, 2}, nil},
		{"rank_deficient", 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, 1},
			[]float64{1, 1}, nil, ErrZeroPivot},
		{"singleton", 1, []int{0, 1}, []int32{0}, []float64{0.2}, []float64{2}, []float64{10}, nil},
		{"sym_structure", 2, []int{0, 2, 4}, []int32{0, 1, 0, 1}, []float64{5, 1, 1, 5},
			[]float64{1, 1}, nil, ErrNotUpper},
		{"tril_structure", 2, []int{0, 2, 3}, []int32{0, 1, 1}, []float64{5, 1, 5},
			[]float64{1, 1}, nil, ErrNotUpper},
		{"two_by_two", 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, -1},
			[]float64{2, 4}, []float64{3, -1}, nil},
		// column 1 has no diagonal entry: the pivot is created by fill
		{"zero_on_diag", 3, []int{0, 1, 2, 5}, []int32{0, 0, 0, 1, 2}, []float64{4, 1, 2, 1, -3},
			[]float64{6, 9, 12}, []float64{17, -46, -8}, nil},
		// columns not sorted (column 2 is 2,1,0)
		{"osqp_kkt", 7, []int{0, 1, 2, 5, 6, 7, 8, 12}, []int32{0, 1, 2, 1, 0, 3, 4, 5, 5, 6, 4, 3},
			[]float64{-0.25, -0.25, 1.0, 0.513578, 0.529142, -0.25, -0.25, 1.10274, 0.15538, 1.25883, 0.13458, 0.621134},
			[]float64{-0.595598, -0.0193715, -0.576156, -0.168746, 0.61543, 0.419073, 1.31087},
			[]float64{1.13141, -1.1367, -0.591044, 1.68867, -2.24209, 0.32254, 0.407998}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f, err := Factorize(upper(c.n, c.ap, c.ai, c.ax))
			if c.xsol == nil {
				if !errors.Is(err, c.want) {
					t.Fatalf("got %v, want %v", err, c.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			x := append([]float64(nil), c.b...)
			if err := f.Solve(x); err != nil {
				t.Fatal(err)
			}
			if d := maxDiff(x, c.xsol); d >= 1e-4 {
				t.Fatalf("max |x - xsol| = %g", d)
			}
		})
	}
}

// qp is a random KKT matrix [[P+σI, Aᵀ], [A, -diag(1/ρ)]] with n primal and m
// dual rows, as its upper triangle and as a dense row-major matrix.
type qp struct {
	k     *sparse.CSC[float64]
	dense []float64
	n, m  int // n primal, m constraints; order n+m
}

func randomKKT(rng *rand.Rand, n, m int, density float64, shuffle bool) qp {
	N := n + m
	dense := make([]float64, N*N)
	set := func(i, j int, v float64) { dense[i*N+j] = v; dense[j*N+i] = v }
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if rng.Float64() < density {
				set(i, j, rng.NormFloat64()*0.3)
			}
		}
	}
	for i := 0; i < n; i++ { // diagonally dominant: P + σI is positive definite
		s := 0.1
		for j := 0; j < n; j++ {
			if j != i {
				s += math.Abs(dense[i*N+j])
			}
		}
		dense[i*N+i] = s + rng.Float64()
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if rng.Float64() < density {
				set(n+i, j, rng.NormFloat64())
			}
		}
		dense[(n+i)*N+n+i] = -(0.1 + 10*rng.Float64()) // -1/ρ
	}
	ap := []int{0}
	var ai []int32
	var ax []float64
	for j := 0; j < N; j++ {
		start := len(ai)
		for i := 0; i <= j; i++ {
			if v := dense[i*N+j]; v != 0 {
				ai = append(ai, int32(i))
				ax = append(ax, v)
			}
		}
		if shuffle {
			col, val := ai[start:], ax[start:]
			rng.Shuffle(len(col), func(a, b int) {
				col[a], col[b] = col[b], col[a]
				val[a], val[b] = val[b], val[a]
			})
		}
		ap = append(ap, len(ai))
	}
	return qp{k: upper(N, ap, ai, ax), dense: dense, n: n, m: m}
}

// denseLDL is an unpivoted dense LDLᵀ: the oracle for D.
func denseLDL(a []float64, n int) []float64 {
	w := append([]float64(nil), a...)
	d := make([]float64, n)
	for k := 0; k < n; k++ {
		d[k] = w[k*n+k]
		for i := k + 1; i < n; i++ { // Schur complement update uses the unscaled column
			for j := k + 1; j <= i; j++ {
				w[i*n+j] -= w[i*n+k] * w[j*n+k] / d[k]
			}
		}
	}
	return d
}

func denseSolve(t *testing.T, a, b []float64, n int) []float64 {
	t.Helper()
	am, err := ndarray.FromData(append([]float64(nil), a...), n, n)
	if err != nil {
		t.Fatal(err)
	}
	x, err := linalg.Solve(am, nd.FromVector(append([]float64(nil), b...)))
	if err != nil {
		t.Fatal(err)
	}
	v, err := nd.Vector(x)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// relResidual is ‖Kx−b‖∞ / (‖K‖∞‖x‖∞ + ‖b‖∞) with K = upper-stored.
func relResidual(k *sparse.CSC[float64], x, b []float64) float64 {
	r := make([]float64, len(b))
	k.MulVecSymUpper(r, x)
	var rn, xn, bn, kn float64
	for i := range r {
		rn = math.Max(rn, math.Abs(r[i]-b[i]))
		xn = math.Max(xn, math.Abs(x[i]))
		bn = math.Max(bn, math.Abs(b[i]))
	}
	for _, v := range k.Data {
		kn = math.Max(kn, math.Abs(v))
	}
	return rn / (kn*xn + bn)
}

func TestRandomQuasiDefiniteAgainstDense(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := 0; trial < 60; trial++ {
		n, m := 1+rng.IntN(25), 1+rng.IntN(25)
		q := randomKKT(rng, n, m, 0.05+0.4*rng.Float64(), trial%2 == 1)
		N := n + m
		f, err := Factorize(q.k)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		b := make([]float64, N)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		x := append([]float64(nil), b...)
		if err := f.Solve(x); err != nil {
			t.Fatal(err)
		}
		if r := relResidual(q.k, x, b); r > 1e-12 {
			t.Fatalf("trial %d: relative residual %g", trial, r)
		}
		if d := maxDiff(x, denseSolve(t, q.dense, b, N)); d > 1e-8*(1+maxAbs(x)) {
			t.Fatalf("trial %d: |x - dense| = %g", trial, d)
		}
		// D: values and sign pattern against the dense LDLᵀ
		dd := denseLDL(q.dense, N)
		for i, v := range f.D() {
			if (v > 0) != (dd[i] > 0) || math.Abs(v-dd[i]) > 1e-9*(1+math.Abs(dd[i])) {
				t.Fatalf("trial %d: D[%d] = %g, dense %g", trial, i, v, dd[i])
			}
		}
		// quasi-definite: n positive pivots, m negative ones
		if f.NumPositive() != n {
			t.Fatalf("trial %d: %d positive pivots, want %d", trial, f.NumPositive(), n)
		}
		// L, D, Lᵀ pieces compose to Solve
		y := append([]float64(nil), b...)
		must(t, f.LSolve(y))
		must(t, f.DSolve(y))
		must(t, f.LTSolve(y))
		if maxDiff(x, y) != 0 {
			t.Fatal("LSolve, DSolve, LTSolve differ from Solve")
		}
	}
}

func maxAbs(x []float64) float64 {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, math.Abs(v))
	}
	return m
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// fullPattern is the CSR pattern of K + Kᵀ for order.AMD.
func fullPattern(t *testing.T, q qp) *sparse.CSR[float64] {
	t.Helper()
	N := q.n + q.m
	coo := sparse.NewCOO[float64](N, N)
	for j := 0; j < N; j++ {
		for p := q.k.Indptr[j]; p < q.k.Indptr[j+1]; p++ {
			i := int(q.k.Indices[p])
			must(t, coo.Append(i, j, q.k.Data[p]))
			if i != j {
				must(t, coo.Append(j, i, q.k.Data[p]))
			}
		}
	}
	return coo.ToCSR()
}

func TestPermutedAMDAndRefactor(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for trial := 0; trial < 30; trial++ {
		n, m := 3+rng.IntN(30), 3+rng.IntN(30)
		q := randomKKT(rng, n, m, 0.1, trial%2 == 0)
		N := n + m
		perm, err := order.AMD(fullPattern(t, q))
		if err != nil {
			t.Fatal(err)
		}
		p, err := NewPermuted(q.k, perm)
		if err != nil {
			t.Fatal(err)
		}
		plain, err := Factorize(q.k)
		if err != nil {
			t.Fatal(err)
		}
		if p.NNZL() > plain.NNZL() {
			// AMD should not be worse than the natural order on these
			// small random matrices by more than a hair; record only.
			t.Logf("trial %d: AMD nnz(L) %d vs natural %d", trial, p.NNZL(), plain.NNZL())
		}
		// two numeric phases on the same pattern with different values
		for round := 0; round < 3; round++ {
			if round > 0 {
				for i := range q.k.Data {
					q.k.Data[i] *= 0.5 + rng.Float64()
				}
				// keep it quasi-definite: rescale the diagonal blocks' signs
				// by construction (positive scaling preserves every sign)
			}
			must(t, p.Refactor(q.k))
			b := make([]float64, N)
			for i := range b {
				b[i] = rng.NormFloat64()
			}
			x := append([]float64(nil), b...)
			must(t, p.Solve(x))
			// scaling entries independently can break diagonal dominance, so
			// compare with the natural-order factor instead of assuming accuracy
			must(t, plain.Refactor(q.k))
			y := append([]float64(nil), b...)
			must(t, plain.Solve(y))
			if d := maxDiff(x, y); d > 1e-6*(1+maxAbs(y)) {
				t.Fatalf("trial %d round %d: permuted vs natural differ by %g", trial, round, d)
			}
			if p.NumPositive() != plain.NumPositive() {
				t.Fatal("inertia differs under permutation")
			}
		}
		if p.Factor() == nil {
			t.Fatal("nil factor")
		}
	}
}

func TestRefactorDoesNotAllocate(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	q := randomKKT(rng, 40, 40, 0.1, true)
	perm, _ := order.AMD(fullPattern(t, q))
	f, err := Factorize(q.k)
	must(t, err)
	p, err := NewPermuted(q.k, perm)
	must(t, err)
	x := make([]float64, 80)
	if a := testing.AllocsPerRun(20, func() { _ = f.Refactor(q.k) }); a != 0 {
		t.Fatalf("Refactor allocates %v", a)
	}
	if a := testing.AllocsPerRun(20, func() { _ = p.Refactor(q.k); _ = p.Solve(x) }); a != 0 {
		t.Fatalf("Permuted Refactor+Solve allocates %v", a)
	}
	if a := testing.AllocsPerRun(20, func() { _ = f.Solve(x) }); a != 0 {
		t.Fatalf("Solve allocates %v", a)
	}
}

func TestSymbolicAccessors(t *testing.T) {
	// arrow matrix: every column links to the last one
	n := 5
	ap := []int{0}
	var ai []int32
	var ax []float64
	for j := 0; j < n; j++ {
		ai = append(ai, int32(j))
		if j == n-1 {
			ax = append(ax, -10)
		} else {
			ax = append(ax, 2)
		}
		if j == n-1 {
			for i := 0; i < n-1; i++ {
				ai = append(ai, int32(i))
				ax = append(ax, 1)
			}
		}
		ap = append(ap, len(ai))
	}
	s, err := Analyze(upper(n, ap, ai, ax))
	must(t, err)
	if s.N() != n || s.NNZL() != n-1 {
		t.Fatalf("N=%d NNZL=%d", s.N(), s.NNZL())
	}
	par := s.Parent()
	for j := 0; j < n-1; j++ {
		if par[j] != int32(n-1) || s.ColCounts()[j] != 1 {
			t.Fatalf("column %d: parent %d count %d", j, par[j], s.ColCounts()[j])
		}
	}
	if par[n-1] != -1 || s.ColCounts()[n-1] != 0 {
		t.Fatal("root")
	}
	f := s.NewFactor()
	if f.Symbolic() != s || f.NNZL() != n-1 {
		t.Fatal("factor accessors")
	}
	if (*Symbolic)(nil).NewFactor() != nil {
		t.Fatal("NewFactor on nil Symbolic")
	}
	// the same Symbolic backs a second factor
	g := s.NewFactor()
	must(t, f.Refactor(upper(n, ap, ai, ax)))
	must(t, g.Refactor(upper(n, ap, ai, ax)))
}

func TestEmptyMatrix(t *testing.T) {
	f, err := Factorize(upper(0, []int{0}, nil, nil))
	must(t, err)
	must(t, f.Solve(nil))
	if f.NumPositive() != 0 || f.NNZL() != 0 {
		t.Fatal("empty")
	}
	p, err := NewPermuted(upper(0, []int{0}, nil, nil), nil)
	must(t, err)
	must(t, p.Refactor(upper(0, []int{0}, nil, nil)))
	must(t, p.Solve(nil))
}

func TestAnalyzeErrors(t *testing.T) {
	cases := []struct {
		name string
		a    *sparse.CSC[float64]
		want error
	}{
		{"nil", nil, ErrDimension},
		{"not square", &sparse.CSC[float64]{Rows: 3, Cols: 2, Indptr: []int{0, 0, 0}}, ErrDimension},
		{"negative order", &sparse.CSC[float64]{Rows: -1, Cols: -1}, ErrOverflow},
		{"huge order", &sparse.CSC[float64]{Rows: math.MaxInt32, Cols: math.MaxInt32}, ErrOverflow},
		{"short indptr", upper(2, []int{0, 1}, []int32{0}, []float64{1}), sparse.ErrFormat},
		{"indptr not from 0", upper(1, []int{1, 1}, []int32{0}, []float64{1}), sparse.ErrFormat},
		{"indptr past indices", upper(1, []int{0, 2}, []int32{0}, []float64{1}), sparse.ErrFormat},
		{"negative indptr end", upper(1, []int{0, -1}, nil, nil), sparse.ErrFormat},
		{"short data", upper(1, []int{0, 1}, []int32{0}, nil), sparse.ErrFormat},
		{"indptr decreases", upper(3, []int{0, 1, 0, 3}, []int32{0, 0, 1}, []float64{1, 1, 1}), sparse.ErrFormat},
		{"indptr above end", upper(2, []int{0, 3, 2}, []int32{0, 0, 1}, []float64{1, 1, 1}), sparse.ErrFormat},
		{"empty column", upper(2, []int{0, 1, 1}, []int32{0}, []float64{1}), ErrEmptyColumn},
		{"negative index", upper(1, []int{0, 1}, []int32{-1}, []float64{1}), sparse.ErrIndex},
		{"below diagonal", upper(2, []int{0, 2, 3}, []int32{0, 1, 1}, []float64{1, 1, 1}), ErrNotUpper},
		{"index out of range", upper(1, []int{0, 1}, []int32{7}, []float64{1}), ErrNotUpper},
		{"duplicate", upper(2, []int{0, 1, 3}, []int32{0, 0, 0}, []float64{1, 1, 1}), ErrDuplicate},
	}
	for _, c := range cases {
		if _, err := Analyze(c.a); !errors.Is(err, c.want) {
			t.Errorf("%s: Analyze gave %v, want %v", c.name, err, c.want)
		}
		if _, err := Factorize(c.a); !errors.Is(err, c.want) {
			t.Errorf("%s: Factorize gave %v, want %v", c.name, err, c.want)
		}
		if _, err := NewPermuted(c.a, []int{0, 1}); !errors.Is(err, c.want) {
			t.Errorf("%s: NewPermuted gave %v, want %v", c.name, err, c.want)
		}
	}
}

func TestOverflowOfL(t *testing.T) {
	old := maxLNZ
	maxLNZ = 0
	defer func() { maxLNZ = old }()
	a := upper(2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, -1})
	if _, err := Analyze(a); !errors.Is(err, ErrOverflow) {
		t.Fatalf("got %v", err)
	}
	if _, err := NewPermuted(a, []int{1, 0}); !errors.Is(err, ErrOverflow) {
		t.Fatalf("got %v", err)
	}
}

func TestRefactorErrors(t *testing.T) {
	mk := func() *sparse.CSC[float64] {
		return upper(3, []int{0, 1, 3, 5}, []int32{0, 0, 1, 1, 2}, []float64{2, 1, 3, 1, -4})
	}
	a := mk()
	f, err := Factorize(a)
	must(t, err)
	p, err := NewPermuted(a, []int{2, 0, 1})
	must(t, err)
	must(t, p.Refactor(a))

	bad := map[string]*sparse.CSC[float64]{
		"nil":          nil,
		"dims":         upper(2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, 1}),
		"short indptr": {Rows: 3, Cols: 3, Indptr: []int{0, 1}, Indices: a.Indices, Data: a.Data},
		"short idx":    {Rows: 3, Cols: 3, Indptr: a.Indptr, Indices: a.Indices[:2], Data: a.Data},
		"short data":   {Rows: 3, Cols: 3, Indptr: a.Indptr, Indices: a.Indices, Data: a.Data[:2]},
		"indptr":       upper(3, []int{0, 2, 3, 5}, []int32{0, 0, 1, 1, 2}, a.Data),
		"indices":      upper(3, []int{0, 1, 3, 5}, []int32{0, 0, 1, 0, 2}, a.Data),
	}
	for name, m := range bad {
		wantDim := name == "nil" || name == "dims"
		for who, fn := range map[string]func(*sparse.CSC[float64]) error{"Factor": f.Refactor, "Permuted": p.Refactor} {
			err := fn(m)
			if wantDim && !errors.Is(err, ErrDimension) || !wantDim && !errors.Is(err, ErrPattern) {
				t.Errorf("%s %s: %v", who, name, err)
			}
		}
	}
	// after a failed Refactor the factor is unusable
	if err := f.Solve(make([]float64, 3)); !errors.Is(err, ErrNotFactored) {
		t.Fatalf("got %v", err)
	}
	if err := p.Solve(make([]float64, 3)); !errors.Is(err, ErrNotFactored) {
		t.Fatalf("got %v", err)
	}
	must(t, f.Refactor(a))
	if err := f.Solve(make([]float64, 2)); !errors.Is(err, ErrDimension) {
		t.Fatalf("got %v", err)
	}
	for name, fn := range map[string]func([]float64) error{"L": f.LSolve, "LT": f.LTSolve, "D": f.DSolve} {
		if err := fn(make([]float64, 2)); !errors.Is(err, ErrDimension) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	must(t, p.Refactor(a))
	if err := p.Solve(make([]float64, 4)); !errors.Is(err, ErrDimension) {
		t.Fatalf("got %v", err)
	}
}

func TestPivotFailures(t *testing.T) {
	mk := func(v ...float64) *sparse.CSC[float64] {
		return upper(2, []int{0, 1, 3}, []int32{0, 0, 1}, v)
	}
	cases := []struct {
		name string
		a    *sparse.CSC[float64]
		want error
		col  int
	}{
		{"zero first", mk(0, 1, 1), ErrZeroPivot, 0},
		{"zero by cancellation", mk(1, 1, 1), ErrZeroPivot, 1},
		{"NaN first", mk(math.NaN(), 1, 1), ErrNonFinite, 0},
		{"Inf later", mk(1, 1, math.Inf(1)), ErrNonFinite, 1},
	}
	for _, c := range cases {
		for who, run := range map[string]func() error{
			"Factorize": func() error { _, err := Factorize(c.a); return err },
			"Permuted": func() error {
				p, err := NewPermuted(c.a, []int{0, 1})
				must(t, err)
				return p.Refactor(c.a)
			},
		} {
			err := run()
			var pe *PivotError
			if !errors.Is(err, c.want) || !errors.As(err, &pe) || pe.Column != c.col || pe.Error() == "" {
				t.Errorf("%s %s: %v", c.name, who, err)
			}
		}
	}
}

func TestPermutationErrors(t *testing.T) {
	a := upper(2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, -1})
	for name, p := range map[string][]int{
		"short": {0}, "long": {0, 1, 2}, "duplicate": {0, 0}, "negative": {-1, 0}, "range": {0, 2}, "nil": nil,
	} {
		if _, err := NewPermuted(a, p); !errors.Is(err, ErrPermutation) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestPivotErrorMessage(t *testing.T) {
	e := &PivotError{Column: 3, Err: ErrZeroPivot}
	if e.Error() != "ldl: zero pivot at column 3" {
		t.Fatal(e.Error())
	}
}
