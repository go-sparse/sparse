package mtx

import (
	"bytes"
	"compress/gzip"
	"errors"
	"strings"
	"testing"
)

func TestLimits(t *testing.T) {
	hd := "%%MatrixMarket matrix coordinate real general\n"
	cases := map[string]struct {
		src string
		lim Limits
	}{
		"rows":          {hd + "999999999 2 0\n", Limits{}},
		"cols":          {hd + "2 999999999 0\n", Limits{}},
		"entries":       {hd + "2 2 999999999\n", Limits{}},
		"small entries": {hd + "2 2 5\n", Limits{MaxEntries: 4}},
		"array":         {"%%MatrixMarket matrix array real general\n100000 100000\n", Limits{}},
		"array sym":     {"%%MatrixMarket matrix array real symmetric\n100000 100000\n", Limits{}},
		"array skew":    {"%%MatrixMarket matrix array real skew-symmetric\n100000 100000\n", Limits{}},
		"long line":     {hd + "% " + strings.Repeat("x", 200) + "\n1 1 0\n", Limits{MaxLine: 100}},
		"long banner":   {strings.Repeat("x", 200) + "\n", Limits{MaxLine: 100}},
		"long data":     {hd + "2 2 1\n1 1 " + strings.Repeat("1", 200) + "\n", Limits{MaxLine: 100}},
		"long size":     {hd + strings.Repeat(" ", 200) + "\n", Limits{MaxLine: 100}},
		"comments":      {hd + "%a\n%b\n%c\n1 1 0\n", Limits{MaxComments: 2}},
		"bytes":         {hd + strings.Repeat("% filler\n", 100) + "1 1 0\n", Limits{MaxBytes: 100}},
	}
	for name, c := range cases {
		_, err := ReadLimited(strings.NewReader(c.src), c.lim)
		if !errors.Is(err, ErrFormat) {
			t.Errorf("%s: %v", name, err)
		}
	}
	// within the limits the same shapes are fine
	if _, err := ReadLimited(strings.NewReader(hd+"%a\n1 1 0\n"), Limits{MaxLine: 100, MaxComments: 2, MaxEntries: 4, MaxDim: 4, MaxBytes: 1000}); err != nil {
		t.Fatal(err)
	}
	// MaxLine below the minimum is raised, not honoured
	if (Limits{MaxLine: 3}).fill().MaxLine != 64 {
		t.Fatal("minimum line length")
	}
}

func TestGzipBomb(t *testing.T) {
	var raw bytes.Buffer
	raw.WriteString("%%MatrixMarket matrix coordinate real general\n")
	raw.WriteString(strings.Repeat("\n", 10<<20)) // 10 MB of blank lines before the size line
	raw.WriteString("1 1 0\n")
	var gz bytes.Buffer
	w := gzip.NewWriter(&gz)
	w.Write(raw.Bytes())
	w.Close()
	if gz.Len() > 100<<10 {
		t.Fatalf("test input should compress well, got %d bytes", gz.Len())
	}
	// 1 MiB decompressed budget: the bomb is cut off, not expanded
	if _, err := ReadLimited(bytes.NewReader(gz.Bytes()), Limits{MaxBytes: 1 << 20}); !errors.Is(err, ErrFormat) {
		t.Fatalf("bomb accepted: %v", err)
	}
}
