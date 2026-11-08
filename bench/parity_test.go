package bench

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"os/exec"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/gallery"
	"github.com/go-sparse/sparse/ldl"
)

// TestParityWithQDLDL: the Go factorisation and solves return the very bits
// QDLDL does (the rows of the direct path are ParityExact).
func TestParityWithQDLDL(t *testing.T) {
	for _, g := range []int{12, 20} {
		m, err := newKKT(g)
		if err != nil {
			t.Fatal(err)
		}
		N := m.k.Cols
		plain, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		permd, err := NewRef(m.pa.Indptr, m.pa.Indices, m.pa.Data, m.b, m.perm, m.pmap, m.k.Data)
		if err != nil {
			t.Fatal(err)
		}
		f, err := ldl.Factorize(m.pa)
		if err != nil {
			t.Fatal(err)
		}
		if f.NNZL() != plain.NNZL() {
			t.Fatalf("g=%d: nnz(L) %d, QDLDL %d", g, f.NNZL(), plain.NNZL())
		}
		x := append([]float64(nil), m.b...)
		if err := f.Solve(x); err != nil {
			t.Fatal(err)
		}
		equalBits(t, "Solve", x, plain.Solve(N))

		pf, err := ldl.NewPermuted(m.k, m.perm)
		if err != nil {
			t.Fatal(err)
		}
		if err := pf.Refactor(m.k); err != nil {
			t.Fatal(err)
		}
		y := append([]float64(nil), m.b...)
		if err := pf.Solve(y); err != nil {
			t.Fatal(err)
		}
		equalBits(t, "Permuted.Solve", y, permd.PSolve(N))
		if got := permd.Loop("pfactor", 1); got != float64(N/2) {
			t.Logf("QDLDL positive pivots %v of %d", got, N)
		}
	}
}

func equalBits(t *testing.T, what string, got, want []float64) {
	t.Helper()
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s: entry %d is %v, QDLDL has %v", what, i, got[i], want[i])
		}
	}
}

// hashArrays is the SHA-256 the Python worker computes for a tuple of numpy
// arrays: their bytes, little endian, in order.
func hashArrays(a *sparse.CSR[float64], x []float64) string {
	h := sha256.New()
	var b [8]byte
	for _, v := range a.Indptr {
		binary.LittleEndian.PutUint32(b[:4], uint32(v))
		h.Write(b[:4])
	}
	for _, v := range a.Indices {
		binary.LittleEndian.PutUint32(b[:4], uint32(v))
		h.Write(b[:4])
	}
	for _, v := range a.Data {
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
		h.Write(b[:])
	}
	for _, v := range x {
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestDataIdentity: the matrices and vectors the Python side times are bit for
// bit the ones the Go side times. It starts the reference worker (uv, and the
// network for the pinned wheels), so it runs only with RB_PYTHON=1.
func TestDataIdentity(t *testing.T) {
	if os.Getenv("RB_PYTHON") == "" {
		t.Skip("set RB_PYTHON=1 to start the Python reference worker")
	}
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	w, err := pyWorker()
	if err != nil {
		t.Fatal(err)
	}
	n := spGrid * spGrid
	for _, c := range []struct {
		name  string
		pyCSR string
		a     *sparse.CSR[float64]
		n     int
	}{
		{"lap2d-100", pyLap2D(spGrid), gallery.Laplacian2D(spGrid, spGrid), n},
		{"tri-10000", pyTri(n), tri(n), n},
		{"lap2d-50", pyLap2D(krGrid), gallery.Laplacian2D(krGrid, krGrid), krGrid * krGrid},
		{"lap2d-30", pyLap2D(smGrid), gallery.Laplacian2D(smGrid, smGrid), smGrid * smGrid},
	} {
		setup := "(lambda sp: (lambda M:(M.indptr,M.indices,M.data," + pyVec(c.n) + "))(" + c.pyCSR + "))(__import__('scipy.sparse',fromlist=['x']))"
		got, err := w.DataHash(setup, 1)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if want := hashArrays(c.a, vec(c.n)); got != want {
			t.Errorf("%s: scipy's arrays hash to %s, the Go side's to %s", c.name, got, want)
		}
	}
}
