package mtx

import (
	"bytes"
	"testing"
)

// FuzzRead feeds arbitrary bytes to the reader. It must never panic or
// allocate without bound, and whatever it accepts must be a valid matrix
// that survives a write/read round trip. The corpus under testdata/fuzz/
// holds the interesting inputs found so far.
func FuzzRead(f *testing.F) {
	for _, s := range []string{
		"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 2\n",
		"%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 1 2\n",
		"%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n2 1\n3 3\n",
		"%%MatrixMarket matrix coordinate integer skew-symmetric\n2 2 1\n2 1 5\n",
		"%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n",
		"%%MatrixMarket matrix array complex hermitian\n2 2\n1 0\n2 1\n4 0\n",
		"%%MatrixMarket matrix coordinate real general\n99999999999999999999 1 1\n",
		"%%MatrixMarket matrix coordinate real general\n2 2 -1\n",
		"%%MatrixMarket matrix coordinate real general\n2 2 1\n3 3 1\n",
		"%%MatrixMarket matrix array real symmetric\n134217728 134217728\n",
		"\x1f\x8b\x08\x00",
		"",
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := ReadLimited(bytes.NewReader(data), Limits{MaxDim: 1 << 12, MaxEntries: 1 << 14, MaxLine: 1 << 10, MaxBytes: 1 << 16, MaxComments: 64})
		if err != nil {
			return
		}
		if m.Real != nil {
			if m.Real.Validate() != nil {
				t.Fatal("invalid real COO")
			}
			a := m.Real.ToCSR()
			if a.Validate() != nil {
				t.Fatal("invalid CSR")
			}
			var buf bytes.Buffer
			if err := Write(&buf, a, Options{}); err != nil {
				t.Fatal(err)
			}
			b, err := Read(&buf)
			if err != nil {
				t.Fatal(err)
			}
			c := b.Real.ToCSR()
			if len(c.Data) != len(a.Data) {
				t.Fatal("round trip changed the entry count")
			}
			return
		}
		if m.Complex.Validate() != nil || m.Complex.ToCSR().Validate() != nil {
			t.Fatal("invalid complex matrix")
		}
	})
}
