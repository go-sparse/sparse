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
