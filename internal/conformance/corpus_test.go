package conformance

import (
	"bytes"
	"compress/gzip"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/internal/scipy"
)

func gz(s string) []byte {
	var b bytes.Buffer
	w := gzip.NewWriter(&b)
	w.Write([]byte(s))
	w.Close()
	return b.Bytes()
}

const tinySym = "%%MatrixMarket matrix coordinate real symmetric\n3 3 5\n1 1 4\n2 1 -1\n2 2 4\n3 2 -1\n3 3 4\n"

func TestCorpusDir(t *testing.T) {
	tmp := t.TempDir()
	env := func(v string) func(string) string {
		return func(string) string { return v }
	}
	d, err := CorpusDir(env(filepath.Join(tmp, "c")), nil)
	if err != nil || d != filepath.Join(tmp, "c") {
		t.Fatal(d, err)
	}
	d, err = CorpusDir(env(""), func() (string, error) { return tmp, nil })
	if err != nil || d != filepath.Join(tmp, "go-sparse-corpus") {
		t.Fatal(d, err)
	}
	if _, err := CorpusDir(env(""), func() (string, error) { return "", errors.New("no cache") }); err == nil {
		t.Fatal("cache error")
	}
	// inside a git work tree: refused
	work := filepath.Join(tmp, "work")
	os.MkdirAll(filepath.Join(work, ".git"), 0o755)
	if _, err := CorpusDir(env(filepath.Join(work, "sub", "corpus")), nil); err == nil || !strings.Contains(err.Error(), "work tree") {
		t.Fatal("must refuse a work tree", err)
	}
}

func TestFetchAndLoad(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "corp")
	calls := 0
	get := func(url string) ([]byte, error) {
		calls++
		if !strings.HasPrefix(url, CorpusBase) {
			t.Errorf("url %s", url)
		}
		return gz(tinySym), nil
	}
	entries := []CorpusEntry{{"set", "tiny"}}
	paths, unpinned, err := Fetch(dir, get, entries, nil)
	if err != nil || len(paths) != 1 || len(unpinned) != 1 || calls != 1 {
		t.Fatal(paths, unpinned, err, calls)
	}
	sum := unpinned["tiny"]
	// second call reuses the file and checks the pin
	if _, un, err := Fetch(dir, get, entries, map[string]string{"tiny": sum}); err != nil || len(un) != 0 || calls != 1 {
		t.Fatal(un, err, calls)
	}
	if _, _, err := Fetch(dir, get, entries, map[string]string{"tiny": "00"}); err == nil {
		t.Fatal("pin mismatch")
	}
	ms, err := LoadCorpus(paths)
	if err != nil || len(ms) != 1 || ms[0].Name != "tiny" || !ms[0].SPD || ms[0].A.Rows != 3 {
		t.Fatal(ms, err)
	}
	// failures
	if _, _, err := Fetch(dir, func(string) ([]byte, error) { return nil, errors.New("down") }, []CorpusEntry{{"s", "other"}}, nil); err == nil {
		t.Fatal("get error")
	}
	file := filepath.Join(t.TempDir(), "f")
	os.WriteFile(file, nil, 0o644)
	if _, _, err := Fetch(filepath.Join(file, "sub"), get, entries, nil); err == nil {
		t.Fatal("mkdir error")
	}
	blocked := t.TempDir()
	os.Mkdir(filepath.Join(blocked, "x.mtx.gz"), 0o755)
	if _, _, err := Fetch(blocked, get, []CorpusEntry{{"s", "x"}}, nil); err == nil {
		t.Fatal("write error")
	}
	if _, err := LoadCorpus([]string{filepath.Join(dir, "missing.mtx.gz")}); err == nil {
		t.Fatal("missing file")
	}
	cpx := filepath.Join(dir, "cpx.mtx.gz")
	os.WriteFile(cpx, gz("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 1\n"), 0o644)
	if _, err := LoadCorpus([]string{cpx}); err == nil {
		t.Fatal("complex")
	}
}

func TestHTTPGet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/missing" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte("hello"))
	}))
	b, err := HTTPGet(srv.URL + "/ok")
	if err != nil || string(b) != "hello" {
		t.Fatal(string(b), err)
	}
	if _, err := HTTPGet(srv.URL + "/missing"); err == nil {
		t.Fatal("404")
	}
	srv.Close()
	if _, err := HTTPGet(srv.URL + "/ok"); err == nil {
		t.Fatal("closed server")
	}
}

func TestRunLiveAndDocument(t *testing.T) {
	spd, _ := Build("lap2d:6:6")
	zero, _ := sparse.FromDense(2, 2, []float64{0, 1, 1, 0})
	ms := []Named{spd, {Name: "zero", A: zero, SPD: true}}
	s, err := scipy.Start(scipy.FakeCommand("ok"))
	if err != nil {
		t.Fatal(err)
	}
	rep := RunLive(s, ms)
	s.Close()
	if len(rep.Solves) == 0 || len(rep.Orders) == 0 || len(rep.Notes) == 0 {
		t.Fatalf("%d %d %d", len(rep.Solves), len(rep.Orders), len(rep.Notes))
	}
	s, _ = scipy.Start(scipy.FakeCommand("fail"))
	rep2 := RunLive(s, ms[:1])
	s.Close()
	if len(rep2.Solves) != 0 || len(rep2.Notes) == 0 {
		t.Fatal("oracle failures must be noted, not fatal")
	}
	f := fixture(t)
	plain := Document(f, nil)
	full := Document(f, &rep)
	for _, want := range []string{"# Verification against SciPy", "## Iterative solvers", "## LSQR", "## LSMR", "## Orderings", "Violations of the enforced bounds when this file was written: 0."} {
		if !strings.Contains(plain, want) {
			t.Errorf("document lacks %q", want)
		}
	}
	if strings.Contains(plain, "Harwell-Boeing") || !strings.Contains(full, "Harwell-Boeing") || !strings.Contains(full, "skipped") {
		t.Error("corpus section")
	}
}
