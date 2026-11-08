package bench

import (
	"testing"

	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// TestPerf is the refbench entry point:
//
//	go test -run TestPerf .                          CI mode: allocations, table, denominator
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
func TestPerf(t *testing.T) {
	cs, err := allCases()
	if err != nil {
		t.Fatal(err)
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cs}, refbench.Options{
		Repo: "go-sparse/sparse",
		RefVersions: map[string]string{
			"QDLDL":  "v0.1.9 (Apache-2.0, copied verbatim in bench/qdldl.c), cgo -O3",
			"scipy":  "1.18.1 (scipy.sparse, one thread)",
			"python": refbench.PythonVersion,
		},
	})
}
