package sparse

import "fmt"

// COO is a coordinate-format matrix: parallel slices of row index, column
// index and value. Duplicate (i, j) entries are allowed and are summed on
// conversion, as in scipy.sparse.coo_array.
type COO[T Scalar] struct {
	Rows, Cols int
	I, J       []int
	V          []T
}

// NewCOO returns an empty rows x cols COO matrix.
func NewCOO[T Scalar](rows, cols int) *COO[T] {
	return &COO[T]{Rows: rows, Cols: cols}
}

// Dims returns the number of rows and columns.
func (a *COO[T]) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ returns the number of stored entries (duplicates counted).
func (a *COO[T]) NNZ() int { return len(a.V) }

// Append adds entry (i, j) = v.
func (a *COO[T]) Append(i, j int, v T) error {
	if i < 0 || i >= a.Rows || j < 0 || j >= a.Cols {
		return fmt.Errorf("%w: (%d,%d) in %dx%d", ErrIndex, i, j, a.Rows, a.Cols)
	}
	a.I = append(a.I, i)
	a.J = append(a.J, j)
	a.V = append(a.V, v)
	return nil
}

// Validate checks a COO matrix assembled by hand (for example from
// untrusted arrays): equal slice lengths, non-negative dimensions and every
// index inside the matrix. Matrices built with Append, Transpose or mtx.Read
// are valid by construction; ToCSR and ToCSC assume validity.
func (a *COO[T]) Validate() error {
	if a.Rows < 0 || a.Cols < 0 || len(a.I) != len(a.V) || len(a.J) != len(a.V) {
		return fmt.Errorf("%w: COO dimensions or slice lengths", ErrFormat)
	}
	for k := range a.V {
		if a.I[k] < 0 || a.I[k] >= a.Rows || a.J[k] < 0 || a.J[k] >= a.Cols {
			return fmt.Errorf("%w: entry %d at (%d,%d) in %dx%d", ErrIndex, k, a.I[k], a.J[k], a.Rows, a.Cols)
		}
	}
	return nil
}

// compress turns coordinate triplets into a compressed structure with major
// dimension nmajor, column (minor) indices strictly increasing inside each
// major slice and duplicates summed. It is two stable counting sorts, O(nnz).
func compress[T Scalar](nmajor, nminor int, maj, min []int, v []T) (ptr, ind []int, data []T) {
	nnz := len(v)
	// pass 1: stable sort by minor
	cnt := make([]int, nminor+1)
	for _, j := range min {
		cnt[j+1]++
	}
	for j := 0; j < nminor; j++ {
		cnt[j+1] += cnt[j]
	}
	order := make([]int, nnz)
	for k, j := range min {
		order[cnt[j]] = k
		cnt[j]++
	}
	// pass 2: stable sort by major
	ptr = make([]int, nmajor+1)
	for _, i := range maj {
		ptr[i+1]++
	}
	for i := 0; i < nmajor; i++ {
		ptr[i+1] += ptr[i]
	}
	pos := append([]int(nil), ptr[:nmajor]...)
	ind = make([]int, nnz)
	data = make([]T, nnz)
	for _, k := range order {
		i := maj[k]
		ind[pos[i]] = min[k]
		data[pos[i]] = v[k]
		pos[i]++
	}
	// sum duplicates
	w := 0
	start := 0
	for i := 0; i < nmajor; i++ {
		end := ptr[i+1]
		first := w
		for p := start; p < end; p++ {
			if w > first && ind[w-1] == ind[p] {
				data[w-1] += data[p]
				continue
			}
			ind[w], data[w] = ind[p], data[p]
			w++
		}
		start = end
		ptr[i+1] = w
	}
	return ptr, ind[:w:w], data[:w:w]
}

// ToCSR converts to CSR, summing duplicates and sorting column indices.
func (a *COO[T]) ToCSR() *CSR[T] {
	p, ind, d := compress(a.Rows, a.Cols, a.I, a.J, a.V)
	return &CSR[T]{Rows: a.Rows, Cols: a.Cols, Indptr: p, Indices: ind, Data: d}
}

// ToCSC converts to CSC, summing duplicates and sorting row indices.
func (a *COO[T]) ToCSC() *CSC[T] {
	p, ind, d := compress(a.Cols, a.Rows, a.J, a.I, a.V)
	return &CSC[T]{Rows: a.Rows, Cols: a.Cols, Indptr: p, Indices: ind, Data: d}
}

// Transpose returns the transposed COO matrix (shares no storage).
func (a *COO[T]) Transpose() *COO[T] {
	return &COO[T]{Rows: a.Cols, Cols: a.Rows,
		I: append([]int(nil), a.J...), J: append([]int(nil), a.I...),
		V: append([]T(nil), a.V...)}
}
