package sparse

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// CSR is a compressed-sparse-row matrix. Row i occupies
// Indices[Indptr[i]:Indptr[i+1]] / Data[...] with strictly increasing
// column indices (canonical form, as produced by every constructor here).
// Column indices are int32, as scipy.sparse stores them whenever they fit:
// the products stream one index per stored value, so their width is a
// third of the memory traffic (dimensions are limited to MaxDim).
type CSR[T Scalar] struct {
	Rows, Cols int
	Indptr     []int
	Indices    []int32
	Data       []T
}

// MaxDim is the largest number of rows or columns of a CSR or CSC matrix:
// minor indices are stored as int32.
const MaxDim = math.MaxInt32

// NewCSR wraps the given arrays (without copying) after validating that they
// form a canonical CSR matrix.
func NewCSR[T Scalar](rows, cols int, indptr []int, indices []int32, data []T) (*CSR[T], error) {
	a := &CSR[T]{Rows: rows, Cols: cols, Indptr: indptr, Indices: indices, Data: data}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	return a, nil
}

// Validate checks the canonical-form invariants.
func (a *CSR[T]) Validate() error {
	return validate(a.Rows, a.Cols, a.Indptr, a.Indices, len(a.Data))
}

func validate(nmajor, nminor int, ptr []int, ind []int32, ndata int) error {
	if nmajor < 0 || nminor < 0 {
		return fmt.Errorf("%w: negative dimension", ErrFormat)
	}
	if nmajor > MaxDim || nminor > MaxDim {
		return fmt.Errorf("%w: dimension above %d", ErrFormat, MaxDim)
	}
	if len(ptr) != nmajor+1 || ptr[0] != 0 {
		return fmt.Errorf("%w: indptr must have length %d and start at 0", ErrFormat, nmajor+1)
	}
	if ptr[nmajor] != len(ind) || len(ind) != ndata {
		return fmt.Errorf("%w: indptr end %d, %d indices, %d values", ErrFormat, ptr[nmajor], len(ind), ndata)
	}
	for i := 0; i < nmajor; i++ {
		if ptr[i+1] < ptr[i] || ptr[i+1] > len(ind) {
			return fmt.Errorf("%w: indptr decreases at %d", ErrFormat, i)
		}
		for p := ptr[i]; p < ptr[i+1]; p++ {
			if ind[p] < 0 || int(ind[p]) >= nminor {
				return fmt.Errorf("%w: index %d in slice %d", ErrIndex, ind[p], i)
			}
			if p > ptr[i] && ind[p] <= ind[p-1] {
				return fmt.Errorf("%w: indices not strictly increasing in slice %d", ErrFormat, i)
			}
		}
	}
	return nil
}

// searchIndex is sort.SearchInts for int32 indices: the first position
// whose index is >= j.
func searchIndex(ind []int32, j int) int {
	return sort.Search(len(ind), func(k int) bool { return int(ind[k]) >= j })
}

// Dims returns the number of rows and columns.
func (a *CSR[T]) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ returns the number of stored entries.
func (a *CSR[T]) NNZ() int { return len(a.Data) }

// Clone returns a deep copy.
func (a *CSR[T]) Clone() *CSR[T] {
	return &CSR[T]{Rows: a.Rows, Cols: a.Cols,
		Indptr:  append([]int(nil), a.Indptr...),
		Indices: append([]int32(nil), a.Indices...),
		Data:    append([]T(nil), a.Data...)}
}

// At returns A[i, j] (zero when not stored) by binary search.
func (a *CSR[T]) At(i, j int) T {
	lo, hi := a.Indptr[i], a.Indptr[i+1]
	p := lo + searchIndex(a.Indices[lo:hi], j)
	if p < hi && int(a.Indices[p]) == j {
		return a.Data[p]
	}
	var z T
	return z
}

// MulVec computes dst = A x.
func (a *CSR[T]) MulVec(dst, x []T) {
	if len(x) != a.Cols || len(dst) != a.Rows {
		panic(shapeErr("MulVec", [2]int{a.Rows, a.Cols}, [2]int{len(dst), len(x)}))
	}
	ptr := a.Indptr[:len(dst)+1]
	nnz := ptr[len(dst)]
	ind, d := a.Indices[:nnz], a.Data[:nnz]
	// One walk over the nonzeros, row by row, summing in storage order (as
	// scipy's csr_matvec does); no per-row re-slicing.
	k := ptr[0]
	for i := range dst {
		hi := ptr[i+1]
		var s T
		for ; k < hi; k++ {
			s += d[k] * x[ind[k]]
		}
		dst[i] = s
	}
}

// MulVecT computes dst = Aᵀ x (plain transpose, no conjugation).
func (a *CSR[T]) MulVecT(dst, x []T) {
	if len(x) != a.Rows || len(dst) != a.Cols {
		panic(shapeErr("MulVecT", [2]int{a.Cols, a.Rows}, [2]int{len(dst), len(x)}))
	}
	clear(dst)
	for i, xi := range x {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			dst[a.Indices[p]] += a.Data[p] * xi
		}
	}
}

// MulDense computes dst = A B for row-major dense B (Cols x k) and dst
// (Rows x k): the SpMM kernel (scipy A @ B for a dense B).
func (a *CSR[T]) MulDense(dst, b []T, k int) {
	if len(b) != a.Cols*k || len(dst) != a.Rows*k {
		panic(shapeErr("MulDense", [3]int{a.Rows, a.Cols, k}, [2]int{len(dst), len(b)}))
	}
	clear(dst)
	for i := 0; i < a.Rows; i++ {
		row := dst[i*k : (i+1)*k]
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			v := a.Data[p]
			j := int(a.Indices[p])
			src := b[j*k : (j+1)*k]
			for c := range row {
				row[c] += v * src[c]
			}
		}
	}
}

// Transpose returns Aᵀ as a CSC matrix sharing A's arrays (O(1)).
func (a *CSR[T]) Transpose() *CSC[T] {
	return &CSC[T]{Rows: a.Cols, Cols: a.Rows, Indptr: a.Indptr, Indices: a.Indices, Data: a.Data}
}

// ToCSC converts to CSC (a copy).
func (a *CSR[T]) ToCSC() *CSC[T] {
	p, ind, d := transposeCompressed(a.Rows, a.Cols, a.Indptr, a.Indices, a.Data)
	return &CSC[T]{Rows: a.Rows, Cols: a.Cols, Indptr: p, Indices: ind, Data: d}
}

// T returns Aᵀ as a new CSR matrix.
func (a *CSR[T]) T() *CSR[T] {
	p, ind, d := transposeCompressed(a.Rows, a.Cols, a.Indptr, a.Indices, a.Data)
	return &CSR[T]{Rows: a.Cols, Cols: a.Rows, Indptr: p, Indices: ind, Data: d}
}

// H returns the conjugate transpose Aᴴ as a new CSR matrix.
func (a *CSR[T]) H() *CSR[T] {
	t := a.T()
	for k, v := range t.Data {
		t.Data[k] = conj(v)
	}
	return t
}

// transposeCompressed swaps the major and minor dimensions of a compressed
// structure; rows come out sorted because the input is walked in order.
func transposeCompressed[T Scalar](nmajor, nminor int, ptr []int, ind []int32, d []T) ([]int, []int32, []T) {
	p2 := make([]int, nminor+1)
	for _, j := range ind {
		p2[j+1]++
	}
	for j := 0; j < nminor; j++ {
		p2[j+1] += p2[j]
	}
	pos := append([]int(nil), p2[:nminor]...)
	i2 := make([]int32, len(ind))
	d2 := make([]T, len(ind))
	for i := 0; i < nmajor; i++ {
		for p := ptr[i]; p < ptr[i+1]; p++ {
			q := pos[ind[p]]
			pos[ind[p]]++
			i2[q] = int32(i)
			d2[q] = d[p]
		}
	}
	return p2, i2, d2
}

// ToCOO converts to coordinate format.
func (a *CSR[T]) ToCOO() *COO[T] {
	c := &COO[T]{Rows: a.Rows, Cols: a.Cols,
		I: make([]int, len(a.Data)), J: make([]int, len(a.Data)),
		V: append([]T(nil), a.Data...)}
	for p, j := range a.Indices {
		c.J[p] = int(j)
	}
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			c.I[p] = i
		}
	}
	return c
}

// Dense returns the matrix as a row-major dense slice of length Rows*Cols.
func (a *CSR[T]) Dense() []T {
	out := make([]T, a.Rows*a.Cols)
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			out[i*a.Cols+int(a.Indices[p])] = a.Data[p]
		}
	}
	return out
}

// FromDense builds a CSR matrix from a row-major dense slice, dropping zeros.
func FromDense[T Scalar](rows, cols int, d []T) (*CSR[T], error) {
	// rows*cols is checked without overflowing (32-bit ints, hostile sizes)
	if rows < 0 || cols < 0 || rows > MaxDim || cols > MaxDim || (cols != 0 && rows > math.MaxInt/cols) || len(d) != rows*cols {
		return nil, shapeErr("FromDense", [2]int{rows, cols}, len(d))
	}
	a := &CSR[T]{Rows: rows, Cols: cols, Indptr: make([]int, rows+1)}
	var z T
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if v := d[i*cols+j]; v != z {
				a.Indices = append(a.Indices, int32(j))
				a.Data = append(a.Data, v)
			}
		}
		a.Indptr[i+1] = len(a.Data)
	}
	return a, nil
}

// Diagonal returns the k-th diagonal (k > 0 above the main one).
func (a *CSR[T]) Diagonal(k int) []T {
	var n int
	if k >= 0 {
		n = min(a.Rows, a.Cols-k)
	} else {
		n = min(a.Rows+k, a.Cols)
	}
	if n < 0 {
		n = 0
	}
	out := make([]T, n)
	for t := range out {
		i, j := t, t+k
		if k < 0 {
			i, j = t-k, t
		}
		out[t] = a.At(i, j)
	}
	return out
}

// Scale multiplies every entry by s in place.
func (a *CSR[T]) Scale(s T) {
	for k := range a.Data {
		a.Data[k] *= s
	}
}

// ScaleRows computes diag(d) * A in place.
func (a *CSR[T]) ScaleRows(d []T) error {
	if len(d) != a.Rows {
		return shapeErr("ScaleRows", a.Rows, len(d))
	}
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			a.Data[p] *= d[i]
		}
	}
	return nil
}

// ScaleCols computes A * diag(d) in place.
func (a *CSR[T]) ScaleCols(d []T) error {
	if len(d) != a.Cols {
		return shapeErr("ScaleCols", a.Cols, len(d))
	}
	for p, j := range a.Indices {
		a.Data[p] *= d[j]
	}
	return nil
}

// Axpby returns alpha*A + beta*B (sparse + sparse).
func Axpby[T Scalar](alpha T, a *CSR[T], beta T, b *CSR[T]) (*CSR[T], error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, shapeErr("Axpby", [2]int{a.Rows, a.Cols}, [2]int{b.Rows, b.Cols})
	}
	c := &CSR[T]{Rows: a.Rows, Cols: a.Cols, Indptr: make([]int, a.Rows+1),
		Indices: make([]int32, 0, len(a.Data)+len(b.Data)),
		Data:    make([]T, 0, len(a.Data)+len(b.Data))}
	for i := 0; i < a.Rows; i++ {
		p, pe := a.Indptr[i], a.Indptr[i+1]
		q, qe := b.Indptr[i], b.Indptr[i+1]
		for p < pe || q < qe {
			switch {
			case q == qe || (p < pe && a.Indices[p] < b.Indices[q]):
				c.Indices = append(c.Indices, a.Indices[p])
				c.Data = append(c.Data, alpha*a.Data[p])
				p++
			case p == pe || b.Indices[q] < a.Indices[p]:
				c.Indices = append(c.Indices, b.Indices[q])
				c.Data = append(c.Data, beta*b.Data[q])
				q++
			default:
				c.Indices = append(c.Indices, a.Indices[p])
				c.Data = append(c.Data, alpha*a.Data[p]+beta*b.Data[q])
				p++
				q++
			}
		}
		c.Indptr[i+1] = len(c.Data)
	}
	return c, nil
}

// Add returns A + B.
func Add[T Scalar](a, b *CSR[T]) (*CSR[T], error) {
	one := fromReal[T](1)
	return Axpby(one, a, one, b)
}

// Sub returns A - B.
func Sub[T Scalar](a, b *CSR[T]) (*CSR[T], error) {
	return Axpby(fromReal[T](1), a, fromReal[T](-1), b)
}

// Mul returns the sparse product A B (Gustavson's row-by-row algorithm).
func Mul[T Scalar](a, b *CSR[T]) (*CSR[T], error) {
	if a.Cols != b.Rows {
		return nil, shapeErr("Mul", [2]int{a.Rows, a.Cols}, [2]int{b.Rows, b.Cols})
	}
	c := &CSR[T]{Rows: a.Rows, Cols: b.Cols, Indptr: make([]int, a.Rows+1)}
	acc := make([]T, b.Cols)
	mark := make([]int, b.Cols)
	for j := range mark {
		mark[j] = -1
	}
	var cols []int32
	for i := 0; i < a.Rows; i++ {
		cols = cols[:0]
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			av := a.Data[p]
			k := a.Indices[p]
			for q := b.Indptr[k]; q < b.Indptr[k+1]; q++ {
				j := b.Indices[q]
				if mark[j] != i {
					mark[j] = i
					acc[j] = 0
					cols = append(cols, j)
				}
				acc[j] += av * b.Data[q]
			}
		}
		slices.Sort(cols)
		for _, j := range cols {
			c.Indices = append(c.Indices, j)
			c.Data = append(c.Data, acc[j])
		}
		c.Indptr[i+1] = len(c.Data)
	}
	return c, nil
}

// Prune removes stored entries with |value| <= tol (tol = 0 removes explicit
// zeros) in place.
func (a *CSR[T]) Prune(tol float64) {
	w := 0
	start := 0
	for i := 0; i < a.Rows; i++ {
		end := a.Indptr[i+1]
		for p := start; p < end; p++ {
			if abs(a.Data[p]) > tol {
				a.Indices[w], a.Data[w] = a.Indices[p], a.Data[p]
				w++
			}
		}
		start = end
		a.Indptr[i+1] = w
	}
	a.Indices, a.Data = a.Indices[:w], a.Data[:w]
}

// Tril returns the lower triangle (entries with j - i <= k).
func (a *CSR[T]) Tril(k int) *CSR[T] { return a.filter(func(i, j int) bool { return j-i <= k }) }

// Triu returns the upper triangle (entries with j - i >= k).
func (a *CSR[T]) Triu(k int) *CSR[T] { return a.filter(func(i, j int) bool { return j-i >= k }) }

func (a *CSR[T]) filter(keep func(i, j int) bool) *CSR[T] {
	c := &CSR[T]{Rows: a.Rows, Cols: a.Cols, Indptr: make([]int, a.Rows+1)}
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			if keep(i, int(a.Indices[p])) {
				c.Indices = append(c.Indices, a.Indices[p])
				c.Data = append(c.Data, a.Data[p])
			}
		}
		c.Indptr[i+1] = len(c.Data)
	}
	return c
}

// Permute returns B with B[i, j] = A[p[i], q[j]]. A nil permutation is the
// identity.
func (a *CSR[T]) Permute(p, q []int) (*CSR[T], error) {
	if (p != nil && len(p) != a.Rows) || (q != nil && len(q) != a.Cols) {
		return nil, shapeErr("Permute", [2]int{a.Rows, a.Cols}, [2]int{len(p), len(q)})
	}
	qinv := make([]int, a.Cols)
	for j := range qinv {
		qinv[j] = j
	}
	for j, v := range q {
		qinv[v] = j
	}
	c := &CSR[T]{Rows: a.Rows, Cols: a.Cols, Indptr: make([]int, a.Rows+1)}
	type ent struct {
		j int
		v T
	}
	var row []ent
	for i := 0; i < a.Rows; i++ {
		src := i
		if p != nil {
			src = p[i]
		}
		row = row[:0]
		for t := a.Indptr[src]; t < a.Indptr[src+1]; t++ {
			row = append(row, ent{qinv[a.Indices[t]], a.Data[t]})
		}
		sort.Slice(row, func(x, y int) bool { return row[x].j < row[y].j })
		for _, e := range row {
			c.Indices = append(c.Indices, int32(e.j))
			c.Data = append(c.Data, e.v)
		}
		c.Indptr[i+1] = len(c.Data)
	}
	return c, nil
}

// IsSymmetric reports whether A equals Aᵀ within |a_ij - a_ji| <= tol.
func (a *CSR[T]) IsSymmetric(tol float64) bool {
	if a.Rows != a.Cols {
		return false
	}
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			if abs(a.Data[p]-a.At(int(a.Indices[p]), i)) > tol {
				return false
			}
		}
	}
	return true
}

// Norm1 is the maximum absolute column sum.
func (a *CSR[T]) Norm1() float64 {
	s := make([]float64, a.Cols)
	for p, j := range a.Indices {
		s[j] += abs(a.Data[p])
	}
	m := 0.0
	for _, v := range s {
		m = math.Max(m, v)
	}
	return m
}

// NormInf is the maximum absolute row sum.
func (a *CSR[T]) NormInf() float64 {
	m := 0.0
	for i := 0; i < a.Rows; i++ {
		s := 0.0
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			s += abs(a.Data[p])
		}
		m = math.Max(m, s)
	}
	return m
}

// NormFro is the Frobenius norm.
func (a *CSR[T]) NormFro() float64 {
	s := 0.0
	for _, v := range a.Data {
		x := abs(v)
		s += x * x
	}
	return math.Sqrt(s)
}

// Eye returns the n x n identity.
func Eye[T Scalar](n int) *CSR[T] {
	a := &CSR[T]{Rows: n, Cols: n, Indptr: make([]int, n+1), Indices: make([]int32, n), Data: make([]T, n)}
	one := fromReal[T](1)
	for i := 0; i < n; i++ {
		a.Indptr[i+1] = i + 1
		a.Indices[i] = int32(i)
		a.Data[i] = one
	}
	return a
}

// Diags builds a rows x cols matrix from diagonals placed at the given
// offsets (scipy.sparse.diags / MATLAB spdiags). diagonals[t] has the length
// of diagonal offsets[t].
func Diags[T Scalar](diagonals [][]T, offsets []int, rows, cols int) (*CSR[T], error) {
	if len(diagonals) != len(offsets) {
		return nil, shapeErr("Diags", len(diagonals), len(offsets))
	}
	c := NewCOO[T](rows, cols)
	for t, k := range offsets {
		for s, v := range diagonals[t] {
			i, j := s, s+k
			if k < 0 {
				i, j = s-k, s
			}
			if err := c.Append(i, j, v); err != nil {
				return nil, err
			}
		}
	}
	return c.ToCSR(), nil
}
