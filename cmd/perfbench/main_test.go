package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-sparse/sparse/internal/scipy"
)

func TestMain(m *testing.M) {
	if scipy.MaybeServeFake(os.Getenv, os.Stdin, os.Stdout) {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestParseLoad(t *testing.T) {
	for in, want := range map[string]float64{
		"22:14  up 24 days,  9:13, 1 user, load averages: 11.37 13.18 16.65": 11.37,
		"10:00:00 up 1 day, load average: 0.52, 0.58, 0.59":                  0.52,
		"x load average: 3,5 1 2":                                            3.5,
	} {
		if got, err := parseLoad(in); err != nil || got != want {
			t.Errorf("%q: %v %v", in, got, err)
		}
	}
	for _, in := range []string{"nothing", "load average", "load averages: x y z"} {
		if _, err := parseLoad(in); err == nil {
			t.Errorf("%q must fail", in)
		}
	}
	loadAverage() // the real `uptime`; absent on some hosts, so only exercised
	old := uptime
	defer func() { uptime = old }()
	uptime = func() ([]byte, error) { return []byte("up 1 day, load averages: 2.5 2 1"), nil }
	if l, err := loadAverage(); err != nil || l != 2.5 {
		t.Errorf("uptime: %v %v", l, err)
	}
	uptime = func() ([]byte, error) { return nil, errors.New("no uptime") }
	if _, err := loadAverage(); err == nil {
		t.Error("missing uptime must fail")
	}
}

func TestMedianAndGoTime(t *testing.T) {
	if median([]float64{3, 1, 2}) != 2 {
		t.Fatal("median")
	}
}

func TestRun(t *testing.T) {
	out := filepath.Join(t.TempDir(), "perf.md")
	var so, se bytes.Buffer
	idle := func() (float64, error) { return 0.1, nil }
	if code := run([]string{"-o", out, "-rounds", "1"}, &so, &se, scipy.FakeCommand("ok"), idle); code != 0 {
		t.Fatal(code, se.String())
	}
	b, _ := os.ReadFile(out)
	text := string(b)
	for _, want := range []string{"| lap2d:150:150 |", "spmv", "cg", "gmres", "cycles x 30", "iterations", "products"} {
		if !strings.Contains(text, want) {
			t.Errorf("table lacks %q", want)
		}
	}
	if strings.Contains(text, "Some rounds were skipped") {
		t.Error("idle host must not skip")
	}
	// a busy host measures nothing and says so
	busy := func() (float64, error) { return 99, nil }
	if code := run([]string{"-o", out, "-rounds", "2"}, &so, &se, scipy.FakeCommand("ok"), busy); code != 0 {
		t.Fatal(code)
	}
	b, _ = os.ReadFile(out)
	if !strings.Contains(string(b), "| 0 |") || !strings.Contains(string(b), "Some rounds were skipped") {
		t.Error("busy table")
	}
}

func TestRunErrors(t *testing.T) {
	dir := t.TempDir()
	idle := func() (float64, error) { return 0.1, nil }
	var so, se bytes.Buffer
	cases := map[string]struct {
		args []string
		cmd  func() *exec.Cmd
		load func() (float64, error)
		want int
	}{
		"flag":  {[]string{"-nope"}, scipy.FakeCommand("ok"), idle, 2},
		"start": {[]string{"-o", filepath.Join(dir, "x")}, func() *exec.Cmd { return exec.Command("/nonexistent/oracle") }, idle, 1},
		"eof":   {[]string{"-o", filepath.Join(dir, "x")}, scipy.FakeCommand("eof"), idle, 1},
		"load":  {[]string{"-o", filepath.Join(dir, "x")}, scipy.FakeCommand("failafter:1"), idle, 1},
		"time":  {[]string{"-o", filepath.Join(dir, "x"), "-rounds", "1"}, scipy.FakeCommand("failafter:2"), idle, 1},
		"uptime": {[]string{"-o", filepath.Join(dir, "x"), "-rounds", "1"}, scipy.FakeCommand("ok"),
			func() (float64, error) { return 0, errors.New("no uptime") }, 1},
		"write": {[]string{"-o", filepath.Join(dir, "no", "x"), "-rounds", "1"}, scipy.FakeCommand("ok"), idle, 1},
	}
	for name, c := range cases {
		se.Reset()
		if code := run(c.args, &so, &se, c.cmd, c.load); code != c.want {
			t.Errorf("%s: exit %d, want %d (%s)", name, code, c.want, se.String())
		}
	}
}

func TestMainExit(t *testing.T) {
	old := exit
	defer func() { exit = old }()
	got := -1
	exit = func(c int) { got = c }
	saved := os.Args
	defer func() { os.Args = saved }()
	os.Args = []string{"perfbench", "-nope"}
	main()
	if got != 2 {
		t.Fatal(got)
	}
}
