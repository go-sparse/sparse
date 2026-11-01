package ldl

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/order"
)

// gridKKT is a KKT matrix with n = g*g primal variables (5-point Laplacian +
// identity as P) and m = n constraints, each coupling x_i and x_(i+1): banded
// enough that the natural ordering keeps the fill moderate. Upper triangle,
// columns sorted.
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
	// constraint i couples x_i and x_(i+1 mod n); its column in the upper
	// triangle holds the two (sorted) primal rows and the dual diagonal.
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

// permutedGrid applies the AMD permutation of K + Kᵀ to the KKT matrix a and
// returns the factored Permuted.
func permutedGrid(tb testing.TB, a *sparse.CSC[float64]) *Permuted {
	tb.Helper()
	full := sparse.NewCOO[float64](a.Cols, a.Cols)
	for j := 0; j < a.Cols; j++ {
		for p := a.Indptr[j]; p < a.Indptr[j+1]; p++ {
			i := int(a.Indices[p])
			if err := full.Append(i, j, 1); err != nil {
				tb.Fatal(err)
			}
			if i != j {
				if err := full.Append(j, i, 1); err != nil {
					tb.Fatal(err)
				}
			}
		}
	}
	perm, err := order.AMD(full.ToCSR())
	if err != nil {
		tb.Fatal(err)
	}
	pm, err := NewPermuted(a, perm)
	if err != nil {
		tb.Fatal(err)
	}
	if err := pm.Refactor(a); err != nil {
		tb.Fatal(err)
	}
	return pm
}

func writeMatrix(path string, a *sparse.CSC[float64]) error {
	n, nnz := a.Cols, a.Indptr[a.Cols]
	buf := make([]byte, 0, 8*(2+n+1+2*nnz))
	put := func(v uint64) { buf = binary.LittleEndian.AppendUint64(buf, v) }
	put(uint64(n))
	put(uint64(nnz))
	for _, v := range a.Indptr {
		put(uint64(v))
	}
	for _, v := range a.Indices[:nnz] {
		put(uint64(v))
	}
	for _, v := range a.Data[:nnz] {
		put(math.Float64bits(v))
	}
	return os.WriteFile(path, buf, 0o644)
}

func oneMinuteLoad(t *testing.T) float64 {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	i := strings.LastIndex(s, "load average")
	if i < 0 {
		t.Fatalf("no load average in %q", s)
	}
	f := strings.Fields(strings.NewReplacer(", ", " ", ":", " ").Replace(s[i:]))
	v, err := strconv.ParseFloat(strings.ReplaceAll(f[2], ",", "."), 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func median(x []float64) float64 {
	y := slices.Clone(x)
	slices.Sort(y)
	return y[len(y)/2]
}

// TestPerfVsQDLDL compares the numeric phase and the solve with the reference
// C QDLDL v0.1.9 on one core, interleaved (a C batch, a Go batch, ...). It is
// skipped unless QDLDL_SRC names a checkout of github.com/osqp/qdldl at v0.1.9
// and a C compiler is available; it refuses to measure while the one-minute
// load average is above 8 (set LDL_PERF_FORCE=1 to run anyway for a smoke test:
// the numbers are then meaningless). LDL_PERF_OUT names the markdown table to
// write. Run: QDLDL_SRC=... go test -run PerfVsQDLDL -v ./ldl
func TestPerfVsQDLDL(t *testing.T) {
	src := os.Getenv("QDLDL_SRC")
	if src == "" {
		t.Skip("QDLDL_SRC not set")
	}
	load := oneMinuteLoad(t)
	if load > 8 && os.Getenv("LDL_PERF_FORCE") == "" {
		t.Fatalf("load average %.1f is above 8: measurements would be meaningless; deferred", load)
	}
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "qdldl_types.h"), []byte(
		"#ifndef QDLDL_TYPES_H\n#define QDLDL_TYPES_H\n#include <limits.h>\ntypedef int QDLDL_int;\ntypedef double QDLDL_float;\n"+
			"typedef unsigned char QDLDL_bool;\n#define QDLDL_INT_MAX INT_MAX\n#endif\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "qdldl_version.h"), []byte("#define QDLDL_VERSION \"0.1.9\"\n"), 0o644))
	bin := filepath.Join(dir, "bench")
	cc := exec.Command("cc", "-O3", "-I", dir, "-I", filepath.Join(src, "include"), "-o", bin,
		filepath.Join(src, "src", "qdldl.c"), filepath.Join("testdata", "qdldl_bench.c"))
	if out, err := cc.CombinedOutput(); err != nil {
		t.Fatalf("cc: %v\n%s", err, out)
	}

	var rows strings.Builder
	rows.WriteString("| matrix (order, nnz(K), nnz(L)) | phase | QDLDL C | Go | Go / C |\n|---|---|---:|---:|---:|\n")
	// natural order on the small grids, the AMD permutation applied on both
	// sides (the C side reads the permuted upper triangle, so both factor the
	// same pattern) on the larger ones.
	for _, c := range []struct {
		g   int
		amd bool
	}{{6, false}, {12, false}, {20, false}, {20, true}, {40, true}, {70, true}} {
		g := c.g
		a := gridKKT(g)
		var f *Factor
		var err error
		if c.amd {
			pm := permutedGrid(t, a)
			a, f = pm.pa, pm.f // permuted upper triangle: what both sides factor
		} else {
			f, err = Factorize(a)
			must(t, err)
		}
		path := filepath.Join(dir, fmt.Sprintf("k%d%v.bin", g, c.amd))
		must(t, writeMatrix(path, a))
		N := a.Cols
		b := make([]float64, N)
		for i := range b {
			b[i] = 1 + float64(i%7)
		}
		x := make([]float64, N)

		// batch sizes: about 100 ms each
		repsF := max(3, int(1e8/float64(8*f.NNZL()+50*N)))
		repsS := max(3, int(1e8/float64(10*f.NNZL()+10*N)))
		var cF, cS, gF, gS []float64
		for round := 0; round < 9; round++ {
			out, err := exec.Command(bin, path, strconv.Itoa(repsF), strconv.Itoa(repsS)).Output()
			if err != nil {
				t.Fatalf("bench: %v", err)
			}
			fl := strings.Fields(string(out))
			tf, _ := strconv.ParseFloat(fl[0], 64)
			ts, _ := strconv.ParseFloat(fl[1], 64)
			cF, cS = append(cF, tf), append(cS, ts)

			must(t, f.Refactor(a))
			t0 := time.Now()
			for r := 0; r < repsF; r++ {
				_ = f.Refactor(a)
			}
			gF = append(gF, float64(time.Since(t0).Nanoseconds())/float64(repsF))
			copy(x, b)
			must(t, f.Solve(x))
			if cx, _ := strconv.ParseFloat(fl[2], 64); math.Abs(cx-x[0]) > 1e-9*(1+math.Abs(cx)) {
				t.Fatalf("C and Go solutions differ: %g vs %g", cx, x[0])
			}
			t0 = time.Now()
			for r := 0; r < repsS; r++ {
				copy(x, b)
				_ = f.Solve(x)
			}
			gS = append(gS, float64(time.Since(t0).Nanoseconds())/float64(repsS))
		}
		label := fmt.Sprintf("%d, %d, %d", N, a.Indptr[N], f.NNZL())
		if c.amd {
			label += ", AMD"
		}
		for _, r := range []struct {
			name string
			c, g []float64
		}{{"factor (numeric)", cF, gF}, {"solve", cS, gS}} {
			c, g := median(r.c), median(r.g)
			fmt.Fprintf(&rows, "| %s | %s | %.1f µs | %.1f µs | **%.2f** |\n", label, r.name, c/1e3, g/1e3, g/c)
		}
	}
	t.Logf("load average %.1f\n%s", load, rows.String())
	if out := os.Getenv("LDL_PERF_OUT"); out != "" {
		must(t, os.WriteFile(out, []byte(rows.String()), 0o644))
	}
}
