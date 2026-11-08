package mtx

import (
	"strings"
	"testing"
)

// TestSplitFieldsIsStringsFields: splitFields splits exactly where
// strings.Fields does, on ASCII and on Unicode white space, and on bytes that
// are not UTF-8.
func TestSplitFieldsIsStringsFields(t *testing.T) {
	for _, s := range []string{
		"", " ", "a", "  a  b\tc\r\n", "1 2 3", " ", "x\xffy z", "\xff", "%c 1", "é è",
	} {
		want := strings.Fields(s)
		got := splitFields(nil, []byte(s))
		if len(got) != len(want) {
			t.Fatalf("%q: %d fields, want %d", s, len(got), len(want))
		}
		for i := range want {
			if string(got[i]) != want[i] {
				t.Errorf("%q: field %d is %q, want %q", s, i, got[i], want[i])
			}
		}
	}
}

// TestReadUnicodeSpace: an entry line separated by no-break spaces reads as it
// did when the line was split by strings.Fields.
func TestReadUnicodeSpace(t *testing.T) {
	m, err := Read(strings.NewReader("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 3.5\n2 2 4\n"))
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.RealCSR()
	if err != nil {
		t.Fatal(err)
	}
	if a.At(0, 0) != 3.5 || a.At(1, 1) != 4 {
		t.Fatalf("read %v", a.Data)
	}
}
