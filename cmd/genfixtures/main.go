// Command genfixtures runs the conformance suite against a live SciPy oracle
// and writes the fixture file the regression tests read. It is the only part
// of the module that needs Python (through uv); the oracle's own call is the
// minimal SciPy function per request, everything else is Go.
//
//	go run ./cmd/genfixtures -o internal/conformance/testdata/oracle.json
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/go-sparse/sparse/internal/conformance"
	"github.com/go-sparse/sparse/internal/scipy"
)

var exit = os.Exit

func main() { exit(run(os.Args[1:], os.Stdout, os.Stderr, scipy.DefaultCommand)) }

func run(args []string, stdout, stderr io.Writer, command func() *exec.Cmd) int {
	fs := flag.NewFlagSet("genfixtures", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("o", "internal/conformance/testdata/oracle.json", "fixture file to write")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	s, err := scipy.Start(command)
	if err != nil {
		fmt.Fprintln(stderr, "genfixtures:", err)
		return 1
	}
	defer s.Close()
	f, err := conformance.GenerateAll(s)
	if err != nil {
		fmt.Fprintln(stderr, "genfixtures:", err)
		return 1
	}
	if err := conformance.SaveFixture(*out, f); err != nil {
		fmt.Fprintln(stderr, "genfixtures:", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s: scipy %s numpy %s, %d solves, %d lsqr, %d orderings\n",
		*out, f.SciPy, f.NumPy, len(f.Solves), len(f.LSQR), len(f.Orders))
	return 0
}
