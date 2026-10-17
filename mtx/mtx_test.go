package mtx

import (
	"bytes"
	"compress/gzip"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-sparse/sparse"
)

func TestReadCoordinateGeneral(t *testing.T) {
	src := "%%MatrixMarket matrix coordinate real general\n% a comment\n\n3 3 3\n1 1 1.5\n\n3 2 -2\n2 3 4e0\n"
	m, err := Read(strings.NewReader(src))
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows != 3 || m.Entries != 3 || len(m.Comments) != 1 || m.Complex != nil {
		t.Fatalf("%+v", m.Header)
	}
	a, err := m.RealCSR()
	if err != nil || a.At(0, 0) != 1.5 || a.At(2, 1) != -2 || a.At(1, 2) != 4 {
		t.Fatal(err, a)
	}
	// no trailing newline
	if _, err := Read(strings.NewReader(strings.TrimSuffix(src, "\n"))); err != nil {
		t.Fatal(err)
	}
	// CRLF
	if _, err := Read(strings.NewReader(strings.ReplaceAll(src, "\n", "\r\n"))); err != nil {
		t.Fatal(err)
	}
	c := m.ComplexCSR()
	if c.At(2, 1) != -2 {
		t.Fatal("widen")
	}
	if _, err := m.RealCSR(); err != nil {
		t.Fatal(err)
	}
}

func TestSymmetries(t *testing.T) {
	cases := []struct {
		src  string
		at   [2]int
		want complex128
	}{
		{"%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 1\n2 1 5\n", [2]int{0, 1}, 5},
		{"%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 5\n", [2]int{0, 1}, -5},
		{"%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 5 2\n", [2]int{0, 1}, complex(5, -2)},
		{"%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n2 1\n", [2]int{0, 1}, 1},
		{"%%MatrixMarket matrix coordinate integer general\n2 2 1\n2 1 7\n", [2]int{1, 0}, 7},
		{"%%MatrixMarket matrix array real general\n2 2\n1\n2\n0\n4\n", [2]int{0, 0}, 1},
		{"%%MatrixMarket matrix array real symmetric\n2 2\n1\n2\n4\n", [2]int{0, 1}, 2},
		{"%%MatrixMarket matrix array real skew-symmetric\n2 2\n3\n", [2]int{0, 1}, -3},
		{"%%MatrixMarket matrix array complex hermitian\n2 2\n1 0\n2 1\n4 0\n", [2]int{0, 1}, complex(2, -1)},
		{"%%MatrixMarket matrix array complex general\n1 1\n2 3\n", [2]int{0, 0}, complex(2, 3)},
	}
	for _, c := range cases {
		m, err := Read(strings.NewReader(c.src))
		if err != nil {
			t.Fatalf("%q: %v", c.src, err)
		}
		got := m.ComplexCSR().At(c.at[0], c.at[1])
		if got != c.want {
			t.Errorf("%q: got %v want %v", c.src, got, c.want)
		}
	}
	// the array general case stores column-major: [[1 0][2 4]]
	m, _ := Read(strings.NewReader(cases[5].src))
	a, _ := m.RealCSR()
	if a.At(1, 0) != 2 || a.At(0, 1) != 0 || a.At(1, 1) != 4 {
		t.Fatal("column-major")
	}
}

func TestMalformed(t *testing.T) {
	hd := "%%MatrixMarket matrix coordinate real general\n"
	bad := []string{
		"",
		"hello\n",
		"%%MatrixMarket matrix weird real general\n1 1 0\n",
		"%%MatrixMarket matrix coordinate weird general\n1 1 0\n",
		"%%MatrixMarket matrix coordinate real weird\n1 1 0\n",
		"%%MatrixMarket matrix array pattern general\n1 1\n",
		"%%MatrixMarket matrix coordinate real hermitian\n1 1 0\n",
		hd,
		hd + "% only comments\n",
		hd + "1 1\n",
		hd + "a b c\n",
		hd + "-1 1 0\n",
		"%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n",
		hd + "2 2 2\n1 1 1\n",
		hd + "2 2 1\n1\n",
		hd + "2 2 1\nx 1 1\n",
		hd + "2 2 1\n3 1 1\n",
		hd + "2 2 1\n1 3 1\n",
		"%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 2 1\n",
		"%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n1 1 1\n",
		"%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 1 1\n",
		"%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1\n",
		"%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 a b\n",
		"%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 1 2\n",
		"%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 1.5\n",
		hd + "2 2 1\n1 1 1 2\n",
		hd + "2 2 1\n1 1 z\n",
		"%%MatrixMarket matrix array real general\n2 2\n1\n2\n",
		"%%MatrixMarket matrix array real general\n1 1\nx\n",
	}
	for _, s := range bad {
		if _, err := Read(strings.NewReader(s)); !errors.Is(err, ErrFormat) {
			t.Errorf("%q: got %v", s, err)
		}
	}
	cm, _ := Read(strings.NewReader("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 1\n"))
	if _, err := cm.RealCSR(); !errors.Is(err, ErrFormat) {
		t.Fatal("complex RealCSR")
	}
	if cm.ComplexCSR().At(0, 0) != complex(1, 1) {
		t.Fatal("ComplexCSR")
	}
}

func sample[T sparse.Scalar](vals ...T) *sparse.CSR[T] {
	c := sparse.NewCOO[T](3, 3)
	pos := [][2]int{{0, 0}, {1, 0}, {0, 1}, {2, 2}}
	for k, p := range pos {
		_ = c.Append(p[0], p[1], vals[k])
	}
	return c.ToCSR()
}

func roundTrip[T sparse.Scalar](t *testing.T, a *sparse.CSR[T], o Options) *Matrix {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, a, o); err != nil {
		t.Fatal(err)
	}
	m, err := Read(&buf)
	if err != nil {
		t.Fatal(err, buf.String())
	}
	return m
}

func TestWriteRoundTrip(t *testing.T) {
	a := sample(1.0, 2, 2, 0.1)
	for _, lay := range []Layout{"", Coordinate, Array} {
		for _, sym := range []Symmetry{"", General, Symmetric} {
			m := roundTrip(t, a, Options{Layout: lay, Symmetry: sym, Comments: []string{" hi"}})
			got, _ := m.RealCSR()
			if !equal(got.Dense(), a.Dense()) {
				t.Errorf("%s/%s: %v", lay, sym, got.Dense())
			}
		}
	}
	// integer field, pattern field
	ai := sample(1.0, 2, 2, 7)
	m := roundTrip(t, ai, Options{Field: Integer})
	if m.Field != Integer {
		t.Fatal(m.Field)
	}
	if g, _ := m.RealCSR(); !equal(g.Dense(), ai.Dense()) {
		t.Fatal("integer")
	}
	mp := roundTrip(t, ai, Options{Field: Pattern})
	if g, _ := mp.RealCSR(); g.At(2, 2) != 1 || g.NNZ() != 4 {
		t.Fatal("pattern")
	}
	// skew-symmetric and hermitian
	sk := sample(0.0, -3, 3, 0)
	sk.Prune(0)
	if g, _ := roundTrip(t, sk, Options{Symmetry: SkewSymmetric}).RealCSR(); !equal(g.Dense(), sk.Dense()) {
		t.Fatal("skew")
	}
	if g, _ := roundTrip(t, sk, Options{Layout: Array, Symmetry: SkewSymmetric}).RealCSR(); !equal(g.Dense(), sk.Dense()) {
		t.Fatal("skew array")
	}
	h := sample(2+0i, 1-1i, 1+1i, 3)
	mh := roundTrip(t, h, Options{Symmetry: Hermitian})
	if !equalC(mh.ComplexCSR().Dense(), h.Dense()) {
		t.Fatal("hermitian")
	}
	mc := roundTrip(t, h, Options{Layout: Array})
	if !equalC(mc.ComplexCSR().Dense(), h.Dense()) {
		t.Fatal("complex array")
	}
}

func equal(a, b []float64) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return len(a) == len(b)
}

func equalC(a, b []complex128) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return len(a) == len(b)
}

func TestWriteErrors(t *testing.T) {
	var buf bytes.Buffer
	a := sample(1.0, 2, 3, 4)
	c := sample(1+0i, 2, 3, 4)
	rect, _ := sparse.FromDense(2, 3, []float64{1, 0, 0, 0, 1, 0})
	bad := []struct {
		name string
		err  error
	}{
		{"complex on real", Write(&buf, a, Options{Field: Complex})},
		{"real on complex", Write(&buf, c, Options{Field: Real})},
		{"pattern array", Write(&buf, a, Options{Field: Pattern, Layout: Array})},
		{"hermitian real", Write(&buf, a, Options{Symmetry: Hermitian})},
		{"not symmetric", Write(&buf, a, Options{Symmetry: Symmetric})},
		{"not skew", Write(&buf, a, Options{Symmetry: SkewSymmetric})},
		{"not hermitian", Write(&buf, c, Options{Symmetry: Hermitian})},
		{"rectangular", Write(&buf, rect, Options{Symmetry: Symmetric})},
		{"not integer", Write(&buf, sample(0.5, 0, 0, 1), Options{Field: Integer})},
	}
	for _, b := range bad {
		if !errors.Is(b.err, ErrFormat) {
			t.Errorf("%s: %v", b.name, b.err)
		}
	}
}

func TestReadFileGzip(t *testing.T) {
	dir := t.TempDir()
	var raw bytes.Buffer
	a := sample(1.0, 2, 3, 4)
	if err := Write(&raw, a, Options{}); err != nil {
		t.Fatal(err)
	}
	plain := filepath.Join(dir, "a.mtx")
	if err := os.WriteFile(plain, raw.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	var gz bytes.Buffer
	zw := gzip.NewWriter(&gz)
	zw.Write(raw.Bytes())
	zw.Close()
	zipped := filepath.Join(dir, "a.mtx.gz")
	if err := os.WriteFile(zipped, gz.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{plain, zipped} {
		m, err := ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if g, _ := m.RealCSR(); !equal(g.Dense(), a.Dense()) {
			t.Fatal(p)
		}
	}
	if _, err := ReadFile(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing file")
	}
	// truncated gzip header
	if _, err := Read(bytes.NewReader([]byte{0x1f, 0x8b, 0x00})); err == nil {
		t.Fatal("bad gzip")
	}
}
