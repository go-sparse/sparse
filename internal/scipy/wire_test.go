package scipy

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/go-sparse/sparse"
)

func TestFromCSREmptyIsNotNull(t *testing.T) {
	empty := &sparse.CSR[float64]{Rows: 2, Cols: 2, Indptr: []int{0, 0, 0}}
	b, err := json.Marshal(FromCSR(empty))
	if err != nil || strings.Contains(string(b), "null") {
		t.Fatalf("%s %v", b, err)
	}
	full := FromCSR(sparse.Eye[float64](2))
	if len(full.Indices) != 2 || len(full.Data) != 2 {
		t.Fatal("full matrix altered")
	}
}
