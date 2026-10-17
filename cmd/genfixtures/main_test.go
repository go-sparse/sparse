package main

import (
	"bytes"
	"os"
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

func TestRun(t *testing.T) {
	out := filepath.Join(t.TempDir(), "o.json")
	var so, se bytes.Buffer
	if code := run([]string{"-o", out}, &so, &se, scipy.FakeCommand("ok")); code != 0 {
		t.Fatal(code, se.String())
	}
	if !strings.Contains(so.String(), "wrote") {
		t.Fatal(so.String())
	}
	f, err := conformance.LoadFixture(out)
	if err != nil || len(f.Solves) == 0 || f.SciPy != scipy.SciPyVersion {
		t.Fatal(err, f)
	}
}

func TestRunErrors(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "o.json")
	var so, se bytes.Buffer
	for name, c := range map[string]struct {
		args []string
		cmd  string
		want int
	}{
		"flag": {[]string{"-nope"}, "ok", 2},
		"fail": {[]string{"-o", good}, "fail", 1},
		"eof":  {[]string{"-o", good}, "eof", 1},
		"save": {[]string{"-o", filepath.Join(dir, "missing", "o.json")}, "ok", 1},
	} {
		se.Reset()
		if code := run(c.args, &so, &se, scipy.FakeCommand(c.cmd)); code != c.want {
			t.Errorf("%s: exit %d, want %d (%s)", name, code, c.want, se.String())
		}
	}
	se.Reset()
	if code := run([]string{"-o", good}, &so, &se, nil2); code != 1 || !strings.Contains(se.String(), "genfixtures:") {
		t.Errorf("start failure: %d %s", code, se.String())
	}
}

func TestMainExit(t *testing.T) {
	old := exit
	defer func() { exit = old }()
	got := -1
	exit = func(c int) { got = c }
	saved := os.Args
	defer func() { os.Args = saved }()
	os.Args = []string{"genfixtures", "-nope"}
	main()
	if got != 2 {
		t.Fatal(got)
	}
}
