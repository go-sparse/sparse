// Command conformance writes docs/verification.md: the committed fixture
// compared with the Go implementation, and with -corpus a live comparison
// against SciPy on the NIST Harwell-Boeing matrices, downloaded into a cache
// directory outside the repository.
//
//	go run ./cmd/conformance -o docs/verification.md
//	go run ./cmd/conformance -corpus -o docs/verification.md
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

func main() {
	exit(run(os.Args[1:], os.Stdout, os.Stderr, scipy.DefaultCommand, conformance.HTTPGet, os.Getenv, os.UserCacheDir))
}

func run(args []string, stdout, stderr io.Writer, command func() *exec.Cmd, get func(string) ([]byte, error),
	getenv func(string) string, userCache func() (string, error)) int {
	fs := flag.NewFlagSet("conformance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fixture := fs.String("fixture", "internal/conformance/testdata/oracle.json", "committed oracle fixture")
	out := fs.String("o", "docs/verification.md", "report to write")
	corpus := fs.Bool("corpus", false, "also compare live on the Harwell-Boeing corpus (needs network and uv)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	fail := func(err error) int {
		fmt.Fprintln(stderr, "conformance:", err)
		return 1
	}
	f, err := conformance.LoadFixture(*fixture)
	if err != nil {
		return fail(err)
	}
	var live *conformance.LiveReport
	if *corpus {
		dir, err := conformance.CorpusDir(getenv, userCache)
		if err != nil {
			return fail(err)
		}
		paths, unpinned, err := conformance.Fetch(dir, get, conformance.Corpus, conformance.CorpusSums)
		if err != nil {
			return fail(err)
		}
		for name, sum := range unpinned {
			fmt.Fprintf(stdout, "unpinned %s %s\n", name, sum)
		}
		ms, err := conformance.LoadCorpus(paths)
		if err != nil {
			return fail(err)
		}
		s, err := scipy.Start(command)
		if err != nil {
			return fail(err)
		}
		defer s.Close()
		rep := conformance.RunLive(s, ms)
		live = &rep
	}
	if err := os.WriteFile(*out, []byte(conformance.Document(f, live)), 0o644); err != nil {
		return fail(err)
	}
	fmt.Fprintf(stdout, "wrote %s\n", *out)
	return 0
}
