// Command perfbench compares this module with SciPy on one core: SpMV, CG
// and GMRES on 2-D Laplacians of about 1e5 and 1e6 nonzeros. Measurements are
// interleaved (Go, SciPy, Go, SciPy, ...) and one at a time, and the run
// refuses to measure while the machine's one-minute load average exceeds
// -maxload, because a shared host makes timings meaningless.
//
//	go run ./cmd/perfbench -o docs/perf.md
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-sparse/sparse/internal/conformance"
	"github.com/go-sparse/sparse/internal/scipy"
	"github.com/go-sparse/sparse/iter"
)

var exit = os.Exit

func main() {
	exit(run(os.Args[1:], os.Stdout, os.Stderr, scipy.DefaultCommand, loadAverage))
}

// uptime runs the system's uptime command (replaced in tests).
var uptime = func() ([]byte, error) { return exec.Command("uptime").Output() }

// loadAverage returns the one-minute load average from `uptime`.
func loadAverage() (float64, error) {
	out, err := uptime()
	if err != nil {
		return 0, err
	}
	return parseLoad(string(out))
}

// parseLoad extracts the first load average from uptime's output, which
// reads "... load averages: 12.34 11.00 10.00" (or "load average: 1.2, ...").
func parseLoad(s string) (float64, error) {
	i := strings.LastIndex(s, "load average")
	if i < 0 {
		return 0, fmt.Errorf("perfbench: no load average in %q", s)
	}
	f := strings.Fields(strings.NewReplacer(", ", " ", ":", " ").Replace(s[i:])) // ", " only: keeps a decimal comma (French locale)
	if len(f) < 3 {
		return 0, fmt.Errorf("perfbench: no load average in %q", s)
	}
	return strconv.ParseFloat(strings.ReplaceAll(f[2], ",", "."), 64)
}

type workload struct {
	matrix string
	what   string // spmv, cg, gmres
	reps   int
}

var workloads = []workload{
	{"lap2d:150:150", "spmv", 400}, {"lap2d:150:150", "cg", 200}, {"lap2d:150:150", "gmres", 10},
	{"lap2d:450:450", "spmv", 50}, {"lap2d:450:450", "cg", 40}, {"lap2d:450:450", "gmres", 2},
}

const restart = 30

func goTime(a conformance.Named, w workload) float64 {
	b := conformance.RHS(a.A.Rows)
	t0 := time.Now()
	switch w.what {
	case "spmv":
		y := make([]float64, len(b))
		for i := 0; i < w.reps; i++ {
			a.A.MulVec(y, b)
		}
	case "cg":
		iter.CG(a.A, b, &iter.Options{RTol: -1, MaxIter: w.reps})
	default:
		iter.GMRES(a.A, b, &iter.Options{RTol: -1, MaxIter: w.reps, Restart: restart})
	}
	return time.Since(t0).Seconds()
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func run(args []string, stdout, stderr io.Writer, command func() *exec.Cmd, load func() (float64, error)) int {
	fs := flag.NewFlagSet("perfbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("o", "docs/perf.md", "table to write")
	rounds := fs.Int("rounds", 7, "interleaved rounds per workload")
	maxLoad := fs.Float64("maxload", 8, "refuse to measure above this one-minute load average")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	fail := func(err error) int {
		fmt.Fprintln(stderr, "perfbench:", err)
		return 1
	}
	s, err := scipy.Start(command)
	if err != nil {
		return fail(err)
	}
	defer s.Close()
	sv, nv, err := s.Versions()
	if err != nil {
		return fail(err)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Performance against SciPy\n\nOne core, Go %s on %s/%s, SciPy %s, NumPy %s. Each cell is the median of %d interleaved rounds (Go, SciPy, Go, SciPy, ...), measured one at a time; a round is skipped if the one-minute load average exceeds %g. Ratio = Go time / SciPy time; below 1 means Go is faster. CG runs a fixed number of iterations and GMRES(30) a fixed number of restart cycles (no stopping test), so both do identical work.\n\n",
		runtime.Version(), runtime.GOOS, runtime.GOARCH, sv, nv, *rounds, *maxLoad)
	b.WriteString("| matrix | nnz | operation | work | Go (ms) | SciPy (ms) | ratio | rounds used |\n|---|---:|---|---|---:|---:|---:|---:|\n")
	loaded := false
	for _, w := range workloads {
		n, _ := conformance.Build(w.matrix)
		name := "bench-" + w.matrix
		if err := s.Load(name, scipy.FromCSR(n.A), conformance.RHS(n.A.Rows)); err != nil {
			return fail(err)
		}
		var g, p []float64
		for r := 0; r < *rounds; r++ {
			l, err := load()
			if err != nil {
				return fail(err)
			}
			if l > *maxLoad {
				loaded = true
				continue
			}
			g = append(g, goTime(n, w))
			sec, err := s.Time(name, w.what, w.reps, restart)
			if err != nil {
				return fail(err)
			}
			p = append(p, sec)
		}
		work := fmt.Sprintf("%d products", w.reps)
		if w.what == "cg" {
			work = fmt.Sprintf("%d iterations", w.reps)
		} else if w.what == "gmres" {
			work = fmt.Sprintf("%d cycles x %d", w.reps, restart)
		}
		if len(g) == 0 {
			fmt.Fprintf(&b, "| %s | %d | %s | %s | – | – | – | 0 |\n", w.matrix, n.A.NNZ(), w.what, work)
			continue
		}
		mg, mp := median(g), median(p)
		fmt.Fprintf(&b, "| %s | %d | %s | %s | %.2f | %.2f | %.2f | %d |\n", w.matrix, n.A.NNZ(), w.what, work, 1e3*mg, 1e3*mp, mg/mp, len(g))
	}
	if loaded {
		b.WriteString("\nSome rounds were skipped because the host was busy; cells with 0 rounds were not measured.\n")
	}
	if err := os.WriteFile(*out, []byte(b.String()), 0o644); err != nil {
		return fail(err)
	}
	fmt.Fprintf(stdout, "wrote %s\n", *out)
	return 0
}
