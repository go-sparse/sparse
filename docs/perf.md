# Performance against SciPy

One core, Go go1.27.2 on darwin/arm64, SciPy 1.18.1, NumPy 2.5.4. Each cell is the median of 7 interleaved rounds (Go, SciPy, Go, SciPy, ...), measured one at a time; a round is skipped if the one-minute load average exceeds 8. Ratio = Go time / SciPy time; below 1 means Go is faster. CG runs a fixed number of iterations and GMRES(30) a fixed number of restart cycles (no stopping test), so both do identical work.

| matrix | nnz | operation | work | Go (ms) | SciPy (ms) | ratio | rounds used |
|---|---:|---|---|---:|---:|---:|---:|
| lap2d:150:150 | 111900 | spmv | 4000 products | 229.61 | 237.11 | 0.97 | 7 |
| lap2d:150:150 | 111900 | cg | 2000 iterations | 155.87 | 223.86 | 0.70 | 7 |
| lap2d:150:150 | 111900 | gmres | 30 cycles x 30 | 179.81 | 323.50 | 0.56 | 7 |
| lap2d:450:450 | 1010700 | spmv | 400 products | 202.88 | 206.24 | 0.98 | 7 |
| lap2d:450:450 | 1010700 | cg | 250 iterations | 178.35 | 200.95 | 0.89 | 7 |
| lap2d:450:450 | 1010700 | gmres | 4 cycles x 30 | 224.71 | 273.81 | 0.82 | 7 |

# `ldl` against C QDLDL v0.1.9

One core (`GOMAXPROCS=1`, single-threaded C built with `cc -O3`), Go go1.27.2 on darwin/arm64 (Apple M4 Max), measured on 2026-11-01 at one-minute load 2.3 before and after. Each cell is the median of 9 interleaved rounds (a C batch, a Go batch, ...) of `TestPerfVsQDLDL`; ratio = Go time / C time, so above 1 means Go is slower. The matrices are KKT systems of a 2-D grid Laplacian plus coupling constraints. In the AMD rows `order.AMD` is applied once and the C side receives the permuted upper triangle, so both sides factor the same pattern.

| order, nnz(K), nnz(L) | ordering | factor C (µs) | factor Go (µs) | ratio | solve C (µs) | solve Go (µs) | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 72, 204, 1516 | natural | 11.5 | 12.2 | 1.06 | 1.9 | 1.8 | 0.96 |
| 288, 840, 22474 | natural | 473.4 | 466.3 | 0.99 | 25.1 | 25.3 | 1.01 |
| 800, 2360, 168018 | natural | 8501.9 | 8530.8 | 1.00 | 179.3 | 180.5 | 1.01 |
| 800, 2360, 4885 | AMD | 29.8 | 36.0 | 1.21 | 5.3 | 6.0 | 1.13 |
| 3200, 9520, 28309 | AMD | 278.6 | 314.0 | 1.13 | 36.2 | 34.9 | 0.96 |
| 9800, 29260, 102299 | AMD | 1364.2 | 1520.0 | 1.11 | 112.7 | 123.7 | 1.10 |

## Open defect: AMD-ordered matrices, numeric factor 11 to 21% and solve up to 13% slower than C

With long columns of L (natural order) Go matches C. With AMD, L has 6 to 10 entries per column, and the time moves from the scatter loop to the per-row work around it.

What was opened and measured (pprof, one core, `BenchmarkRefactorAMD`; QDLDL `qdldl.c` read at v0.1.9; `cc -O3 -S` and `go tool objdump` of both loops):

- C's scatter loop is 7 instructions per entry (`ldr`, `ldrsw`, `ldr`, `fmsub`, `str`, `subs`, `b.ne`, post-incremented pointers, 32-bit `int` everywhere). Go's is about 10: index addressing, one bounds check on `yval[r]` that cannot be proven (the row index comes from data), and a loop-limit compare. Unrolling the loop by four did not change the time (within 1%), so loop overhead is not the limit; short columns are.
- The per-row work (3 slice-bound checks for `li[lp[c]:t]`, `lx[...]`, loads of `next`, `lp`, `dinv`, the stores to `li`, `lx`, `next`, `yval`, `ymark`) dominates, and `Refactor` had so many live slices that the register allocator spilled loop operands to the stack (`MOVD 128(RSP), R7` before `li[tmp] = k`). v0.2.1 splits the row loop into `pattern` (elimination-tree walk) and `eliminate` (the triangular solve): 3% at order 800, 6% at orders 3200 and 9800, accuracy tests unchanged and the results bit-identical.
- `samePattern` (the O(nnz) check that the matrix has the analysed pattern, which C does not do) costs 1.5% of the profile; it is a safety check on untrusted input and stays.
- The remainder is bounds checks on indices that come from data (`yval[r]`, `ymark[nx]`, `etree[nx]`, `yidx[nnzY]`, `elim[nnzE]`: about 40 in `Refactor`), 64-bit `lp` and `next` against C's 32-bit, and no scatter vectorisation available in Go or in C. Removing them needs `unsafe` (ruled out) or a different layout of L.

### v0.3.0 (issue #10)

- **Interleaved layout of L, tried and not kept.** Index and value in one 16-byte struct (a third more bytes per entry than the two arrays), as variants inside one process in interleaved rounds on one core, for the numeric phase and the solves, at orders 72, 800, 3200 and 9800: equal within noise at 72 and 800, 11% slower in the numeric phase at 3200, scattered at 9800 (the machine was busy; no run showed a consistent gain). The saved address computation does not pay for the extra memory traffic.
- **Solve fused into two sweeps.** D is applied inside the L sweep (`x[i]` is final when column `i` is reached) and `Permuted.Solve` scatters each entry back as the Lᵀ sweep makes it final. Bit-identical to the three separate passes (tested bit for bit); 2 to 4% faster in interleaved `go test -bench` rounds (load 6 to 7, orders 72 to 9800). The remaining time is the Lᵀ chain `v -= l*x[r]`, each step dependent on the last, which cannot be reassociated without changing the bits.
- **Numeric phase.** The scatter loop handles four entries per trip (same values). Measured at 0 to 4% on small factors, nothing on large ones, as the unrolling test of v0.2.1 found.
- **Symbolic side.** `order.AMDWith`/`COLAMDWith` (`order.Workspace`) and `ldl.Permuted.Reset`: AMD + `NewPermuted` on the KKT grids, `BenchmarkSetupAMD`, one core, busy machine so read the ratios: order 72 28.5 µs and 652 allocations in v0.2.1, 15.4 µs and 79 with fresh storage, 9.8 µs and 0 with a reused workspace; order 800 280 to 440 µs and 6 323 allocations, 160 µs and 94, 121 µs and 0; order 3200 1.40 ms and 24 952, 0.70 ms and 103, 0.54 ms and 0. The permutations are identical to v0.2.1's (3000 random matrices).

The ratios against QDLDL in the table above are v0.2.1's. They are re-measured with `TestPerfVsQDLDL` when the one-minute load is below 8 (it was above 60 for the whole session in which v0.3.0 was made), and the result is added here.

### Wave W3 of the performance campaign (issue #10)

**Benchmarks moved to refbench.** `bench/` is a module of its own (cgo stays out of the library) that registers one case per public function, or names the function with a reason in [perf-exempt.md](perf-exempt.md), and measures it with [go-scikit/refbench](https://github.com/go-scikit/refbench) v0.1.2 (`rbtest.Perf`: one core, interleaved rounds, gated on a 1-minute load below 8). The direct path (`ldl`) is timed against QDLDL v0.1.9 (`bench/qdldl.c`, copied verbatim, Apache-2.0, driven by the C loops of `bench/loops.c`; the numeric phase, the solves, `Analyze`, and the permuted path with the gather and scatter an OSQP-style caller adds); everything else against scipy.sparse 1.18.1 (numpy 2.5.4, the pinned `bench/refs.txt`) through the refbench Python worker. `TestParityWithQDLDL` checks the factor sizes and solutions bit for bit; `TestDataIdentity` (`RB_PYTHON=1`) checks that the matrices and vectors the Python side times hash like the Go side's.

```sh
cd bench
go test -run TestPerf .                         # CI mode: allocations, table, denominator (no Python, no timing)
go test -run TestPerf -cpu 1 -perf.run .        # measure; writes docs/perf.json and docs/perf-table.md
```

The CI job `perf` runs the first line. The committed [perf-table.md](perf-table.md) is **deferred** (allocation counts, no timing); the first run, kept in the history of this file at commit 50b53d2 (pull request 17), was an **invalid** run (2026-11-08, `-perf.onload=wait`): the gate opened at the start (load 6.3 to 7.1) but a round ended at 10.7, so refbench gives no verdict and files no issue. Read it as indicative only; the rows are not repeated here. What it suggests, to be confirmed by a valid run: `ldl` numeric phase 1.17 to 1.30 times QDLDL's and the solves 1.05 to 1.19 (unchanged from the v0.2.1 picture; `DSolve` about 2, a loop of 800 to 10 000 multiplications that C vectorises and Go does not), `CSR.NormFro` 18, `gallery.Laplacian1D` 9, `COO.ToCSC` 4.6, `Diags` 4.4, `CSR.Prune` 4.5, `CSR.Validate` 3.2, `CSR.Norm1` 2.2 and `CSR.Dense` 2.1 times SciPy's, while the Krylov solvers (0.17 to 0.47), `Tril`/`Triu`, `ScaleRows/Cols`, `FromDense` and `mtx.Write` are faster. These are candidates for the next wave, not claims. The ratios in this file and the README stay v0.2.1's until a run at load below 8 replaces them.

**Causes of the largest indicative gaps, and the changes (unmeasured, load 6 to 13; no timing claimed).** Read against scipy 1.18.1's source:

- `CSR.NormFro` (x18): `scipy.sparse.linalg.norm(A, 'fro')` is `np.linalg.norm(A.data)`, that is `sqrt(dot(x, x))`, a BLAS `ddot` with several accumulators; ours was one chain of dependent additions (about 4 cycles each). Float64 data now goes through `floats.Dot` (the fleet's SIMD kernel, as `iter.nrm2`). The sum differs from the old serial one in the last bits, as scipy's differs between BLAS builds; the conformance oracle compares with a tolerance, and a test bounds the difference at 1e-14 relative.
- `COO.ToCSC` (x4.6), `COO.ToCSR` (x1.7), `Diags` (x4.4): `compress` made two stable counting sorts, two random scatters of every entry (the second strided by column for a CSC), with an n-sized counter and an nnz-sized permutation. scipy's `coo_tocsr` scatters once by major index. It is now one scatter in input order, then each slice put in minor order by a stable sort that is skipped when the slice already is (always, for a COO made from a CSR) and an insertion sort when short; duplicates are still summed in input order, so the result is bit-identical (`TestCompressMatchesTwoPassSort` compares with the old function on random, shuffled and sorted input).
- `gallery.Laplacian1D/2D/3D` (x9 for 1D): assembled as triplets, then converted (sorted and summed again). A stencil never repeats a neighbour, so the CSR is now built directly: one counting sweep, one filling sweep, a few entries per row put in order (`TestDirectCSRMatchesTripletAssembly`, bit for bit).
- `CSR.Prune` (x4.5): the generic `abs` boxed and type-switched on every entry (about 2 ns); one type switch per matrix now, then loops on concrete `float64` or `complex128` data. The semantics (`|v| > tol` kept, so NaN is dropped) are unchanged.

Not fixed here: `CSR.Validate` (x3.2), `CSR.Norm1` (x2.2), `CSR.Dense` (x2.1), `CSR.Scale` (x2.0), `CSR.ToCOO` (x1.7), `Add`/`Sub` (x1.4), `CSC.MulVec` (x1.5), `ldl` `DSolve` (about x2) and `Refactor` (x1.2). They are tracked in a perf issue and wait for a valid run.

**Allocations, before and after** (exact counts; `testing.AllocsPerRun`; asserted by `alloc_test.go` in `order`, the root package, `gallery` and `mtx`: the count must not grow with the size of the input):

| call | before | after |
|---|---|---|
| `order.AMD`, fresh storage, grids of 8, 16, 32 | 50, 58, 66 | 28, 28, 28 |
| `order.COLAMD`, same | 49, 57, 62 | 26, 26, 26 |
| `order.RCM`, same | 88, 290, 1 077 (a swapper per vertex) | 9, 9, 9 |
| `CSR.Permute`, order 10 000 | 30 055 (three per row) | 6 |
| `CSR.Tril`, `CSR.Triu`, order 10 000 | 46 | 4 |
| `FromDense`, order 900 | 33 | 4 |
| `Diags`, order 10 000 | 76 | 10 |
| `Mul` (Laplacian times tridiagonal) | 59 | 15 |
| `gallery.Laplacian1D/2D/3D` | 10 081 / 10 186 / 11 266 | 15 |
| `mtx.Write`, 4 380 entries | 14 187 | 10 |
| `mtx.Read`, 4 380 entries | 8 821 | 18 |

`AMDWith`, `COLAMDWith`, `Permuted.Reset`, `Refactor` and every solve were already at 0. No timing is claimed for any of these changes (unmeasured, load 9 to 13); the permutations of `AMD`, `COLAMD` and `RCM` are unchanged (a stable sort has a unique result), and `mtx.Write` produces the same bytes.

**Profile of the numeric phase and the solves** (pprof, `BenchmarkRefactorAMD` and the solve benchmarks, load 9 to 13: a ranking, not a ratio). Refactor: `eliminate` 69% of the samples, `pattern` 15%, `math.IsInf` (the pivot check) 1%, the rest scheduler noise. Solve: `ltsolveTo` 51% and `lsolveD` 44% (cumulative), nothing else. The loop order is QDLDL's (`qdldl.c` v0.1.9, `QDLDL_factor` lines 126 to 259 and `QDLDL_solve`): the same row-by-row elimination-tree walk, the same scatter, the same dependent chain in the Lᵀ sweep, with D folded into the L sweep. No algorithmic difference was found to remove; what is left is the bounds checks on indices that come from data and the 64-bit `lp`, which need `unsafe` (ruled out) to go. The timing verdict is open.
