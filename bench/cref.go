// Package bench measures github.com/go-sparse/sparse against its references on
// one core, with refbench (github.com/go-scikit/refbench):
//
//   - the direct path (ldl) against C QDLDL v0.1.9, copied verbatim beside this
//     file (qdldl.c, qdldl.h, Apache-2.0), driven through cgo by the loops of
//     loops.c;
//   - everything else against scipy.sparse 1.18.1 (numpy 2.5.4) through the
//     refbench Python worker, pinned by refs.txt.
//
// It is a separate module so that cgo never enters the library's build.
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, denominator
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
package bench

/*
#cgo CFLAGS: -O3 -w
void *kkt_new(int n, const int *Ap, const int *Ai, const double *Ax, const double *b,
              const int *perm, const int *map, const double *Kx, int nKx);
int kkt_nnzl(void *h);
void kkt_solve_one(void *h, double *out);
void kkt_psolve_one(void *h, double *out);
double loop_etree(void *h, long iters);
double loop_factor(void *h, long iters);
double loop_pfactor(void *h, long iters);
double loop_solve(void *h, long iters);
double loop_lsolve(void *h, long iters);
double loop_ltsolve(void *h, long iters);
double loop_dsolve(void *h, long iters);
double loop_psolve(void *h, long iters);
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Ref is the QDLDL state for one matrix: the arrays the C loops work on.
type Ref struct{ h unsafe.Pointer }

// NewRef copies the upper triangle (indptr, indices, ax) and the right-hand
// side b into C memory and runs QDLDL's symbolic and numeric phases once.
// perm, pmap and kx describe a permuted factorisation (the factored matrix
// is ax = kx[pmap[k]], row i of the permuted system is row perm[i]); they are
// nil for the natural order.
func NewRef(indptr []int, indices []int32, ax, b []float64, perm []int, pmap []int32, kx []float64) (*Ref, error) {
	n := len(b)
	ap := make([]int32, len(indptr))
	for i, v := range indptr {
		ap[i] = int32(v)
	}
	var pp []int32
	if perm != nil {
		pp = make([]int32, len(perm))
		for i, v := range perm {
			pp[i] = int32(v)
		}
	}
	var perm0, map0 *C.int
	var kx0 *C.double
	if pp != nil {
		perm0 = (*C.int)(unsafe.Pointer(&pp[0]))
		map0 = (*C.int)(unsafe.Pointer(&pmap[0]))
		kx0 = (*C.double)(unsafe.Pointer(&kx[0]))
	}
	h := C.kkt_new(C.int(n), (*C.int)(unsafe.Pointer(&ap[0])), (*C.int)(unsafe.Pointer(&indices[0])),
		(*C.double)(unsafe.Pointer(&ax[0])), (*C.double)(unsafe.Pointer(&b[0])), perm0, map0, kx0, C.int(len(kx)))
	if h == nil {
		return nil, errors.New("QDLDL refused the matrix")
	}
	return &Ref{h}, nil
}

// NNZL is QDLDL's sum of column counts.
func (r *Ref) NNZL() int { return int(C.kkt_nnzl(r.h)) }

// Solve returns K⁻¹ b as QDLDL computes it.
func (r *Ref) Solve(n int) []float64 {
	out := make([]float64, n)
	C.kkt_solve_one(r.h, (*C.double)(unsafe.Pointer(&out[0])))
	return out
}

// PSolve returns the permuted solve of the state built with a permutation.
func (r *Ref) PSolve(n int) []float64 {
	out := make([]float64, n)
	C.kkt_psolve_one(r.h, (*C.double)(unsafe.Pointer(&out[0])))
	return out
}

var sink float64

// Loop runs the C kernel iters times and returns the sum of what it returned,
// so nothing is optimised away.
func (r *Ref) Loop(kind string, iters int) float64 {
	it := C.long(iters)
	switch kind {
	case "etree":
		return float64(C.loop_etree(r.h, it))
	case "factor":
		return float64(C.loop_factor(r.h, it))
	case "pfactor":
		return float64(C.loop_pfactor(r.h, it))
	case "solve":
		return float64(C.loop_solve(r.h, it))
	case "lsolve":
		return float64(C.loop_lsolve(r.h, it))
	case "ltsolve":
		return float64(C.loop_ltsolve(r.h, it))
	case "dsolve":
		return float64(C.loop_dsolve(r.h, it))
	case "psolve":
		return float64(C.loop_psolve(r.h, it))
	}
	panic("bench: unknown loop " + kind)
}
