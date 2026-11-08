//go:build !race

// The race detector's instrumentation allocates: the counts here are exact
// only without it (CI runs them in the plain test run of each lane).

package mtx

import (
	"bytes"
	"io"
	"testing"

	"github.com/go-sparse/sparse"
)

func band(n int) *sparse.CSR[float64] {
	c := sparse.NewCOO[float64](n, n)
	for i := 0; i < n; i++ {
		_ = c.Append(i, i, 2.5)
		if i > 0 {
			_ = c.Append(i, i-1, -1)
		}
	}
	return c.ToCSR()
}

// TestAllocsDoNotGrowWithEntries asserts that Read and Write allocate the same
// number of times for 100 and for 4000 entries. Before: Write boxed every
// operand (fmt.Fprintf) and formatted every float into a string, and Read made
// a string and a slice per line: about 3 allocations per entry.
func TestAllocsDoNotGrowWithEntries(t *testing.T) {
	small, big := band(50), band(2000)
	var bs, bb bytes.Buffer
	if err := Write(&bs, small, Options{}); err != nil {
		t.Fatal(err)
	}
	if err := Write(&bb, big, Options{}); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		max  float64
		run  func(a *sparse.CSR[float64], b []byte)
	}{
		{"Write", 10, func(a *sparse.CSR[float64], _ []byte) { _ = Write(io.Discard, a, Options{}) }},
		{"Read", 18, func(_ *sparse.CSR[float64], b []byte) { _, _ = Read(bytes.NewReader(b)) }},
	} {
		s := testing.AllocsPerRun(5, func() { c.run(small, bs.Bytes()) })
		l := testing.AllocsPerRun(5, func() { c.run(big, bb.Bytes()) })
		t.Logf("%s: %v allocations (100 entries), %v (4000)", c.name, s, l)
		// slack 3: the triplets grow once past the preallocation bound, and
		// the runtime's own allocations that the counter sees on a large run
		if l > s+3 || l > c.max {
			t.Errorf("%s: %v allocations for 4000 entries, %v for 100 (want at most 3 more, and at most %v)", c.name, l, s, c.max)
		}
	}
}
