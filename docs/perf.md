# Performance against SciPy

One core, Go go1.27.2 on darwin/arm64, SciPy 1.18.1, NumPy 2.5.4. Each cell is the median of 7 interleaved rounds (Go, SciPy, Go, SciPy, ...), measured one at a time; a round is skipped if the one-minute load average exceeds 8. Ratio = Go time / SciPy time; below 1 means Go is faster. CG runs a fixed number of iterations and GMRES(30) a fixed number of restart cycles (no stopping test), so both do identical work.

| matrix | nnz | operation | work | Go (ms) | SciPy (ms) | ratio | rounds used |
|---|---:|---|---|---:|---:|---:|---:|
| lap2d:150:150 | 111900 | spmv | 400 products | 24.73 | 23.59 | 1.05 | 7 |
| lap2d:150:150 | 111900 | cg | 200 iterations | 18.62 | 22.16 | 0.84 | 7 |
| lap2d:150:150 | 111900 | gmres | 10 cycles x 30 | 80.50 | 107.94 | 0.75 | 7 |
| lap2d:450:450 | 1010700 | spmv | 50 products | 26.51 | 26.14 | 1.01 | 7 |
| lap2d:450:450 | 1010700 | cg | 40 iterations | 37.50 | 32.64 | 1.15 | 7 |
| lap2d:450:450 | 1010700 | gmres | 2 cycles x 30 | 156.39 | 137.31 | 1.14 | 7 |
