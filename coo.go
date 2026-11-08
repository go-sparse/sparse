package sparse

import (
	"fmt"
	"slices"
)

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
// major slice and duplicates summed. One counting pass scatters the triplets
// by major index in their input order (as scipy's coo_tocsr does); each major
// slice is then put in minor order by a stable sort (skipped when it already
// is, which is the case for a COO made from a CSR), so duplicates are summed in
// input order. Two counting sorts, as this function was before, made two random
// scatters of every entry, and the second one strided by column for a CSC.
func compress[T Scalar](nmajor, nminor int, maj, min []int, v []T) (ptr []int, ind []int32, data []T) {
	nnz := len(v)
	ptr = make([]int, nmajor+1)
	for _, i := range maj {
		ptr[i]++
	}
	sum := 0
	for i := 0; i < nmajor; i++ {
		ptr[i], sum = sum, sum+ptr[i]
	}
	ptr[nmajor] = sum
	ind = make([]int32, nnz)
	data = make([]T, nnz)
	for k, i := range maj {
		j := min[k]
		if uint(j) >= uint(nminor) {
			panic(fmt.Sprintf("sparse: minor index %d outside 0..%d", j, nminor-1))
		}
		p := ptr[i]
		ind[p], data[p] = int32(j), v[k]
		ptr[i] = p + 1
	}
	// ptr[i] is now the end of slice i, the start of slice i+1: shift back.
	for i := nmajor; i > 0; i-- {
		ptr[i] = ptr[i-1]
	}
	ptr[0] = 0
	// order each slice, then sum duplicates
	var scratch []entry[T]
	w := 0
	start := 0
	for i := 0; i < nmajor; i++ {
		end := ptr[i+1]
		scratch = sortSlice(ind[start:end], data[start:end], scratch)
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

type entry[T Scalar] struct {
	j int32
	v T
}

// sortSlice sorts ind (with data alongside) by index, stably. A slice already
// in order is left alone; a short one is insertion-sorted in place; a long one
// goes through scratch (returned, grown when needed).
func sortSlice[T Scalar](ind []int32, data []T, scratch []entry[T]) []entry[T] {
	if slices.IsSorted(ind) {
		return scratch
	}
	if len(ind) <= 32 {
		for p := 1; p < len(ind); p++ {
			j, x := ind[p], data[p]
			q := p
			for ; q > 0 && ind[q-1] > j; q-- {
				ind[q], data[q] = ind[q-1], data[q-1]
			}
			ind[q], data[q] = j, x
		}
		return scratch
	}
	scratch = scratch[:0]
	for p, j := range ind {
		scratch = append(scratch, entry[T]{j, data[p]})
	}
	slices.SortStableFunc(scratch, func(a, b entry[T]) int { return int(a.j) - int(b.j) })
	for p, e := range scratch {
		ind[p], data[p] = e.j, e.v
	}
	return scratch
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
