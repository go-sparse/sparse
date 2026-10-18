package main

import (
	"bytes"
	"compress/gzip"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-sparse/sparse/internal/conformance"
	"github.com/go-sparse/sparse/internal/scipy"
)

func TestMain(m *testing.M) {
	if scipy.MaybeServeFake(os.Getenv, os.Stdin, os.Stdout) {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

const fixturePath = "../../internal/conformance/testdata/oracle.json"

func tiny() []byte {
	var b bytes.Buffer
	w := gzip.NewWriter(&b)
	w.Write([]byte("%%MatrixMarket matrix coordinate real symmetric\n3 3 5\n1 1 4\n2 1 -1\n2 2 4\n3 2 -1\n3 3 4\n"))
	w.Close()
	return b.Bytes()
}

func TestRunPlain(t *testing.T) {
	out := filepath.Join(t.TempDir(), "v.md")
	var so, se bytes.Buffer
	if code := run([]string{"-fixture", fixturePath, "-o", out}, &so, &se, nil, nil, nil, nil); code != 0 {
		t.Fatal(code, se.String())
	}
	b, err := os.ReadFile(out)
	if err != nil || !strings.Contains(string(b), "Verification against SciPy") {
		t.Fatal(err)
	}
}

func TestRunCorpus(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "v.md")
	getenv := func(string) string { return filepath.Join(tmp, "corpus") }
	get := func(string) ([]byte, error) { return tiny(), nil }
	var so, se bytes.Buffer
	saved := conformance.CorpusSums
	conformance.CorpusSums = map[string]string{} // the stand-in files are not the pinned ones
	defer func() { conformance.CorpusSums = saved }()
	args := []string{"-fixture", fixturePath, "-o", out, "-corpus"}
	if code := run(args, &so, &se, scipy.FakeCommand("ok"), get, getenv, nil); code != 0 {
		t.Fatal(code, se.String())
	}
	if !strings.Contains(so.String(), "unpinned") {
		t.Fatal("tiny files are not pinned, so Fetch must say so")
	}
	b, _ := os.ReadFile(out)
	if !strings.Contains(string(b), "Harwell-Boeing") {
		t.Fatal("corpus section missing")
	}
}

func TestRunErrors(t *testing.T) {
	saved := conformance.CorpusSums
	conformance.CorpusSums = map[string]string{}
	defer func() { conformance.CorpusSums = saved }()
	tmp := t.TempDir()
	good := filepath.Join(tmp, "v.md")
	getenv := func(string) string { return filepath.Join(tmp, "corpus") }
	get := func(string) ([]byte, error) { return tiny(), nil }
	var so, se bytes.Buffer
	cases := map[string]struct {
		args   []string
		get    func(string) ([]byte, error)
		getenv func(string) string
		want   int
	}{
		"flag":    {[]string{"-nope"}, get, getenv, 2},
		"fixture": {[]string{"-fixture", filepath.Join(tmp, "none")}, get, getenv, 1},
		"write":   {[]string{"-fixture", fixturePath, "-o", filepath.Join(tmp, "no", "v.md")}, get, getenv, 1},
		"fetch":   {[]string{"-fixture", fixturePath, "-o", good, "-corpus"}, func(string) ([]byte, error) { return nil, errors.New("down") }, getenv, 1},
		"load": {[]string{"-fixture", fixturePath, "-o", good, "-corpus"},
			func(string) ([]byte, error) { return []byte("not gzip, not mtx"), nil }, func(string) string { return filepath.Join(tmp, "bad") }, 1},
		"worktree": {[]string{"-fixture", fixturePath, "-o", good, "-corpus"}, get, func(string) string { return "../../.git/corpus" }, 1},
	}
	for name, c := range cases {
		se.Reset()
		if code := run(c.args, &so, &se, scipy.FakeCommand("ok"), c.get, c.getenv, nil); code != c.want {
			t.Errorf("%s: exit %d, want %d (%s)", name, code, c.want, se.String())
		}
	}
	// the oracle cannot start
	se.Reset()
	missingOracle := func() *exec.Cmd { return exec.Command("/nonexistent/oracle") }
	if code := run([]string{"-fixture", fixturePath, "-o", good, "-corpus"}, &so, &se, missingOracle, get,
		func(string) string { return filepath.Join(tmp, "corpus2") }, nil); code != 1 {
		t.Errorf("start: exit %d", code)
	}
}

func TestMainExit(t *testing.T) {
	old := exit
	defer func() { exit = old }()
	got := -1
	exit = func(c int) { got = c }
	saved := os.Args
	defer func() { os.Args = saved }()
	os.Args = []string{"conformance", "-nope"}
	main()
	if got != 2 {
		t.Fatal(got)
	}
}
