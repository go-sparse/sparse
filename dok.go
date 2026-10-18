package sparse

import "fmt"

// DOK is a dictionary-of-keys matrix, convenient for incremental
// construction (scipy.sparse.dok_array).
type DOK[T Scalar] struct {
	Rows, Cols int
	m          map[[2]int]T
}

// NewDOK returns an empty rows x cols DOK matrix.
func NewDOK[T Scalar](rows, cols int) *DOK[T] {
	return &DOK[T]{Rows: rows, Cols: cols, m: map[[2]int]T{}}
}

// Dims returns the number of rows and columns.
func (d *DOK[T]) Dims() (int, int) { return d.Rows, d.Cols }

// Len returns the number of stored entries.
func (d *DOK[T]) Len() int { return len(d.m) }

// Set stores v at (i, j); storing zero deletes the entry.
func (d *DOK[T]) Set(i, j int, v T) error {
	if i < 0 || i >= d.Rows || j < 0 || j >= d.Cols {
		return fmt.Errorf("%w: (%d,%d) in %dx%d", ErrIndex, i, j, d.Rows, d.Cols)
	}
	var z T
	if v == z {
		delete(d.m, [2]int{i, j})
		return nil
	}
	d.m[[2]int{i, j}] = v
	return nil
}

// At returns the value at (i, j).
func (d *DOK[T]) At(i, j int) T { return d.m[[2]int{i, j}] }

// ToCOO returns the entries as a COO matrix (order unspecified).
func (d *DOK[T]) ToCOO() *COO[T] {
	c := NewCOO[T](d.Rows, d.Cols)
	for k, v := range d.m {
		c.I = append(c.I, k[0])
		c.J = append(c.J, k[1])
		c.V = append(c.V, v)
	}
	return c
}

// ToCSR converts to CSR.
func (d *DOK[T]) ToCSR() *CSR[T] { return d.ToCOO().ToCSR() }
