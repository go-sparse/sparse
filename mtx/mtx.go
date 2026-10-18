// Package mtx reads and writes the Matrix Market exchange format
// (https://math.nist.gov/MatrixMarket/formats.html): coordinate and array
// layouts; real, integer, pattern and complex fields; general, symmetric,
// skew-symmetric and hermitian symmetry. Gzip-compressed input is detected
// by its magic bytes. Equivalents: scipy.io.mmread / mmwrite, MATLAB's
// mmread.m.
package mtx

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"os"
	"strconv"
	"strings"

	"github.com/go-sparse/sparse"
)

// Layout is the storage layout of a Matrix Market file.
type Layout string

// Field is the type of the stored values.
type Field string

// Symmetry is the symmetry structure of the stored matrix.
type Symmetry string

// Constants naming the Matrix Market vocabulary.
const (
	Coordinate Layout = "coordinate"
	Array      Layout = "array"

	Real    Field = "real"
	Integer Field = "integer"
	Pattern Field = "pattern"
	Complex Field = "complex"

	General       Symmetry = "general"
	Symmetric     Symmetry = "symmetric"
	SkewSymmetric Symmetry = "skew-symmetric"
	Hermitian     Symmetry = "hermitian"
)

// ErrFormat reports a malformed Matrix Market file.
var ErrFormat = errors.New("mtx: malformed file")

// Header is the information of the banner and size lines.
type Header struct {
	Layout   Layout
	Field    Field
	Symmetry Symmetry
	Rows     int
	Cols     int
	Entries  int // stored entries (coordinate) or stored values (array)
	Comments []string
}

// Matrix is a file read by Read. Exactly one of Real and Complex is set:
// Complex for the complex field, Real otherwise (pattern entries are 1).
// Symmetry is expanded: the COO matrices hold the full matrix, and
// Header.Symmetry records what the file declared.
type Matrix struct {
	Header
	Real    *sparse.COO[float64]
	Complex *sparse.COO[complex128]
}

// RealCSR returns the matrix as a real CSR matrix; it fails for complex files.
func (m *Matrix) RealCSR() (*sparse.CSR[float64], error) {
	if m.Real == nil {
		return nil, fmt.Errorf("%w: file is complex", ErrFormat)
	}
	return m.Real.ToCSR(), nil
}

// ComplexCSR returns the matrix as a complex CSR matrix (real files are widened).
func (m *Matrix) ComplexCSR() *sparse.CSR[complex128] {
	if m.Complex != nil {
		return m.Complex.ToCSR()
	}
	c := sparse.NewCOO[complex128](m.Rows, m.Cols)
	c.I, c.J = m.Real.I, m.Real.J
	c.V = make([]complex128, len(m.Real.V))
	for k, v := range m.Real.V {
		c.V[k] = complex(v, 0)
	}
	return c.ToCSR()
}

// ReadFile reads a Matrix Market file (optionally gzip-compressed).
func ReadFile(path string) (*Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// Limits bound what Read accepts from an untrusted stream. A zero field
// takes its DefaultLimits value. Nothing is allocated from a declared size
// before the data that backs it has been read, but the Matrix is later
// converted with O(Rows+Cols) arrays, hence MaxDim.
type Limits struct {
	MaxDim      int   // rows and columns
	MaxEntries  int   // declared coordinate entries; rows*cols of an array file
	MaxLine     int   // bytes in one line
	MaxBytes    int64 // bytes read (after decompression)
	MaxComments int   // comment lines kept in Header.Comments
}

// DefaultLimits are generous for real matrices (2.7e8 entries) and refuse
// absurd headers.
var DefaultLimits = Limits{MaxDim: 1 << 27, MaxEntries: 1 << 28, MaxLine: 1 << 16, MaxBytes: 1 << 34, MaxComments: 1 << 12}

func (l Limits) fill() Limits {
	d := DefaultLimits
	if l.MaxDim > 0 {
		d.MaxDim = l.MaxDim
	}
	if l.MaxEntries > 0 {
		d.MaxEntries = l.MaxEntries
	}
	if l.MaxLine > 0 {
		d.MaxLine = max(l.MaxLine, 64)
	}
	if l.MaxBytes > 0 {
		d.MaxBytes = l.MaxBytes
	}
	if l.MaxComments > 0 {
		d.MaxComments = l.MaxComments
	}
	return d
}

// Read parses a Matrix Market stream with DefaultLimits.
func Read(r io.Reader) (*Matrix, error) { return ReadLimited(r, DefaultLimits) }

// ReadLimited parses a Matrix Market stream, refusing input beyond lim.
func ReadLimited(r io.Reader, lim Limits) (*Matrix, error) {
	lim = lim.fill()
	br := bufio.NewReaderSize(io.LimitReader(r, lim.MaxBytes), lim.MaxLine)
	if magic, _ := br.Peek(2); len(magic) == 2 && magic[0] == 0x1f && magic[1] == 0x8b {
		zr, err := gzip.NewReader(br)
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		// the decompressed size is bounded too (a gzip bomb)
		br = bufio.NewReaderSize(io.LimitReader(zr, lim.MaxBytes), lim.MaxLine)
	}
	p := &parser{br: br, lim: lim}
	return p.parse()
}

type parser struct {
	br   *bufio.Reader
	lim  Limits
	line int
}

func (p *parser) errf(format string, a ...any) error {
	return fmt.Errorf("%w: line %d: %s", ErrFormat, p.line, fmt.Sprintf(format, a...))
}

// eof turns a next() failure into the caller's message, keeping the
// "line too long" error as it is.
func (p *parser) eof(err error, msg string) error {
	if errors.Is(err, ErrFormat) {
		return err
	}
	return p.errf("%s", msg)
}

// next returns the next line (without terminator); io.EOF at the end.
func (p *parser) next() (string, error) {
	b, long, err := p.br.ReadLine()
	if err != nil {
		return "", err
	}
	p.line++
	if long {
		return "", p.errf("line longer than %d bytes", p.lim.MaxLine)
	}
	return string(b), nil
}

// nextData returns the next non-blank, non-comment line.
func (p *parser) nextData() ([]string, error) {
	for {
		s, err := p.next()
		if err != nil {
			return nil, p.eof(err, "unexpected end of file")
		}
		f := strings.Fields(s)
		if len(f) > 0 && !strings.HasPrefix(f[0], "%") {
			return f, nil
		}
	}
}

func (p *parser) parse() (*Matrix, error) {
	s, err := p.next()
	if err != nil {
		return nil, p.eof(err, "empty file")
	}
	f := strings.Fields(strings.ToLower(s))
	if len(f) != 5 || f[0] != "%%matrixmarket" || f[1] != "matrix" {
		return nil, p.errf("bad banner %q", s)
	}
	h := Header{Layout: Layout(f[2]), Field: Field(f[3]), Symmetry: Symmetry(f[4])}
	switch h.Layout {
	case Coordinate, Array:
	default:
		return nil, p.errf("unknown layout %q", f[2])
	}
	switch h.Field {
	case Real, Integer, Pattern, Complex:
	default:
		return nil, p.errf("unknown field %q", f[3])
	}
	switch h.Symmetry {
	case General, Symmetric, SkewSymmetric, Hermitian:
	default:
		return nil, p.errf("unknown symmetry %q", f[4])
	}
	if h.Field == Pattern && h.Layout == Array {
		return nil, p.errf("pattern field requires coordinate layout")
	}
	if h.Symmetry == Hermitian && h.Field != Complex {
		return nil, p.errf("hermitian symmetry requires the complex field")
	}
	var size []string
	for {
		s, err = p.next()
		if err != nil {
			return nil, p.eof(err, "missing size line")
		}
		t := strings.TrimSpace(s)
		if strings.HasPrefix(t, "%") {
			if len(h.Comments) >= p.lim.MaxComments {
				return nil, p.errf("more than %d comment lines", p.lim.MaxComments)
			}
			h.Comments = append(h.Comments, strings.TrimPrefix(t, "%"))
			continue
		}
		if t != "" {
			size = strings.Fields(t)
			break
		}
	}
	want := 3
	if h.Layout == Array {
		want = 2
	}
	if len(size) != want {
		return nil, p.errf("size line needs %d integers", want)
	}
	dims := make([]int, want)
	for i, w := range size {
		v, err := strconv.Atoi(w)
		if err != nil || v < 0 {
			return nil, p.errf("bad size %q", w)
		}
		dims[i] = v
	}
	h.Rows, h.Cols = dims[0], dims[1]
	if h.Rows > p.lim.MaxDim || h.Cols > p.lim.MaxDim {
		return nil, p.errf("dimensions %dx%d exceed the limit %d", h.Rows, h.Cols, p.lim.MaxDim)
	}
	if h.Symmetry != General && h.Rows != h.Cols {
		return nil, p.errf("%s matrix must be square", h.Symmetry)
	}
	// stored-value count in int64: rows and cols are below MaxDim here, but
	// their product overflows a 32-bit int
	n := int64(dims[0])
	var entries int64
	switch {
	case h.Layout == Coordinate:
		entries = int64(dims[2])
	case h.Symmetry == General:
		entries = n * int64(h.Cols)
	case h.Symmetry == SkewSymmetric:
		entries = n * (n - 1) / 2
	default:
		entries = n * (n + 1) / 2
	}
	if entries > int64(p.lim.MaxEntries) {
		return nil, p.errf("%d stored entries exceed the limit %d", entries, p.lim.MaxEntries)
	}
	h.Entries = int(entries)
	coo := sparse.NewCOO[complex128](h.Rows, h.Cols)
	if h.Layout == Coordinate {
		err = p.readCoordinate(&h, coo)
	} else {
		err = p.readArray(&h, coo)
	}
	if err != nil {
		return nil, err
	}
	m := &Matrix{Header: h}
	if h.Field == Complex {
		m.Complex = coo
		return m, nil
	}
	m.Real = sparse.NewCOO[float64](h.Rows, h.Cols)
	m.Real.I, m.Real.J = coo.I, coo.J
	m.Real.V = make([]float64, len(coo.V))
	for k, v := range coo.V {
		m.Real.V[k] = real(v)
	}
	return m, nil
}

func (p *parser) value(h *Header, f []string) (complex128, error) {
	switch h.Field {
	case Pattern:
		if len(f) != 0 {
			return 0, p.errf("unexpected value in pattern file")
		}
		return 1, nil
	case Complex:
		if len(f) != 2 {
			return 0, p.errf("complex entry needs two numbers")
		}
		re, e1 := strconv.ParseFloat(f[0], 64)
		im, e2 := strconv.ParseFloat(f[1], 64)
		if e1 != nil || e2 != nil {
			return 0, p.errf("bad number %q %q", f[0], f[1])
		}
		return complex(re, im), nil
	case Integer:
		if len(f) != 1 {
			return 0, p.errf("entry needs one integer")
		}
		v, err := strconv.ParseInt(f[0], 10, 64)
		if err != nil {
			return 0, p.errf("bad integer %q", f[0])
		}
		return complex(float64(v), 0), nil
	}
	if len(f) != 1 {
		return 0, p.errf("entry needs one number")
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return 0, p.errf("bad number %q", f[0])
	}
	return complex(v, 0), nil
}

// add stores v at (i, j) and its mirror when the file is symmetric.
func add(h *Header, c *sparse.COO[complex128], i, j int, v complex128) {
	c.I, c.J, c.V = append(c.I, i), append(c.J, j), append(c.V, v)
	if i == j || h.Symmetry == General {
		return
	}
	switch h.Symmetry {
	case SkewSymmetric:
		v = -v
	case Hermitian:
		v = cmplx.Conj(v)
	}
	c.I, c.J, c.V = append(c.I, j), append(c.J, i), append(c.V, v)
}

func (p *parser) readCoordinate(h *Header, c *sparse.COO[complex128]) error {
	for k := 0; k < h.Entries; k++ {
		f, err := p.nextData()
		if err != nil {
			return err
		}
		if len(f) < 2 {
			return p.errf("coordinate entry needs row and column")
		}
		i, e1 := strconv.Atoi(f[0])
		j, e2 := strconv.Atoi(f[1])
		if e1 != nil || e2 != nil {
			return p.errf("bad index %q %q", f[0], f[1])
		}
		if i < 1 || i > h.Rows || j < 1 || j > h.Cols {
			return p.errf("index (%d,%d) outside %dx%d", i, j, h.Rows, h.Cols)
		}
		if h.Symmetry != General && i < j {
			return p.errf("%s file must store the lower triangle", h.Symmetry)
		}
		if h.Symmetry == SkewSymmetric && i == j {
			return p.errf("skew-symmetric file stores a diagonal entry")
		}
		v, err := p.value(h, f[2:])
		if err != nil {
			return err
		}
		add(h, c, i-1, j-1, v)
	}
	return nil
}

func (p *parser) readArray(h *Header, c *sparse.COO[complex128]) error {
	for j := 0; j < h.Cols; j++ {
		i0 := 0
		switch h.Symmetry {
		case Symmetric, Hermitian:
			i0 = j
		case SkewSymmetric:
			i0 = j + 1
		}
		for i := i0; i < h.Rows; i++ {
			f, err := p.nextData()
			if err != nil {
				return err
			}
			v, err := p.value(h, f)
			if err != nil {
				return err
			}
			if v != 0 {
				add(h, c, i, j, v)
			}
		}
	}
	return nil
}

// Options controls Write. The zero value writes a general coordinate file
// whose field is real (float64) or complex (complex128).
type Options struct {
	Layout   Layout   // Coordinate (default) or Array
	Field    Field    // default: real for float64 data, complex for complex128
	Symmetry Symmetry // default General; others write only the lower triangle
	Comments []string
}

// Write stores a in Matrix Market format. With a symmetry other than
// General, the matrix must have that symmetry exactly.
func Write[T sparse.Scalar](w io.Writer, a *sparse.CSR[T], o Options) error {
	var z T
	_, isC := any(z).(complex128)
	if o.Layout == "" {
		o.Layout = Coordinate
	}
	if o.Field == "" {
		o.Field = Real
		if isC {
			o.Field = Complex
		}
	}
	if o.Symmetry == "" {
		o.Symmetry = General
	}
	if o.Field == Complex && !isC {
		return fmt.Errorf("%w: complex field needs complex data", ErrFormat)
	}
	if o.Field != Complex && isC {
		return fmt.Errorf("%w: %s field cannot hold complex data", ErrFormat, o.Field)
	}
	if o.Field == Pattern && o.Layout == Array {
		return fmt.Errorf("%w: pattern field requires coordinate layout", ErrFormat)
	}
	if o.Symmetry == Hermitian && o.Field != Complex {
		return fmt.Errorf("%w: hermitian symmetry requires the complex field", ErrFormat)
	}
	if o.Symmetry != General {
		if err := checkSymmetry(a, o.Symmetry); err != nil {
			return err
		}
	}
	// entries to store, in the order of the layout
	type entry struct {
		i, j int
		v    T
	}
	var ents []entry
	keep := func(i, j int) bool {
		switch o.Symmetry {
		case General:
			return true
		case SkewSymmetric:
			return i > j
		}
		return i >= j
	}
	if o.Layout == Coordinate {
		for i := 0; i < a.Rows; i++ {
			for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
				if keep(i, a.Indices[p]) {
					ents = append(ents, entry{i, a.Indices[p], a.Data[p]})
				}
			}
		}
	} else {
		for j := 0; j < a.Cols; j++ {
			for i := 0; i < a.Rows; i++ {
				if keep(i, j) {
					ents = append(ents, entry{i, j, a.At(i, j)})
				}
			}
		}
	}
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%%%%MatrixMarket matrix %s %s %s\n", o.Layout, o.Field, o.Symmetry)
	for _, c := range o.Comments {
		fmt.Fprintf(bw, "%%%s\n", c)
	}
	if o.Layout == Coordinate {
		fmt.Fprintf(bw, "%d %d %d\n", a.Rows, a.Cols, len(ents))
	} else {
		fmt.Fprintf(bw, "%d %d\n", a.Rows, a.Cols)
	}
	for _, e := range ents {
		if o.Layout == Coordinate {
			fmt.Fprintf(bw, "%d %d", e.i+1, e.j+1)
		}
		c := toComplex(e.v)
		switch o.Field {
		case Pattern:
		case Complex:
			fmt.Fprintf(bw, " %s %s", num(real(c)), num(imag(c)))
		case Integer:
			if real(c) != math.Trunc(real(c)) {
				return fmt.Errorf("%w: value %g is not an integer", ErrFormat, real(c))
			}
			fmt.Fprintf(bw, " %d", int64(real(c)))
		default:
			fmt.Fprintf(bw, " %s", num(real(c)))
		}
		bw.WriteByte('\n')
	}
	return bw.Flush()
}

func toComplex[T sparse.Scalar](v T) complex128 {
	if c, ok := any(v).(complex128); ok {
		return c
	}
	return complex(any(v).(float64), 0)
}

func num(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }

func checkSymmetry[T sparse.Scalar](a *sparse.CSR[T], s Symmetry) error {
	if a.Rows != a.Cols {
		return fmt.Errorf("%w: %s matrix must be square", ErrFormat, s)
	}
	for i := 0; i < a.Rows; i++ {
		for p := a.Indptr[i]; p < a.Indptr[i+1]; p++ {
			j := a.Indices[p]
			v, t := toComplex(a.Data[p]), toComplex(a.At(j, i))
			var want complex128
			switch s {
			case Symmetric:
				want = v
			case SkewSymmetric:
				want = -v
			default:
				want = cmplx.Conj(v)
			}
			if t != want {
				return fmt.Errorf("%w: matrix is not %s at (%d,%d)", ErrFormat, s, i, j)
			}
		}
	}
	return nil
}
