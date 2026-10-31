/* Timing driver for the reference QDLDL (v0.1.9), used by TestPerfVsQDLDL.
 *
 *   qdldl_bench matrix.bin reps_factor reps_solve
 *
 * matrix.bin: int64 n, int64 nnz, int64 Ap[n+1], int64 Ai[nnz], float64 Ax[nnz]
 * (little endian), the upper triangle in CSC form. After one untimed call of
 * each, it prints the mean nanoseconds of one QDLDL_factor and of one
 * QDLDL_solve (on a fresh copy of the right-hand side, as the Go side does).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "qdldl.h"

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 2;
    int64_t n, nnz;
    if (fread(&n, 8, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1) return 2;
    int64_t* t = malloc(8 * (size_t)(n + 1 + nnz));
    if (fread(t, 8, (size_t)(n + 1 + nnz), f) != (size_t)(n + 1 + nnz)) return 2;
    QDLDL_float* Ax = malloc(8 * (size_t)nnz);
    if (fread(Ax, 8, (size_t)nnz, f) != (size_t)nnz) return 2;
    fclose(f);
    int rf = atoi(argv[2]), rs = atoi(argv[3]);

    QDLDL_int* Ap = malloc(sizeof(QDLDL_int) * (n + 1));
    QDLDL_int* Ai = malloc(sizeof(QDLDL_int) * nnz);
    for (int64_t i = 0; i <= n; i++) Ap[i] = (QDLDL_int)t[i];
    for (int64_t i = 0; i < nnz; i++) Ai[i] = (QDLDL_int)t[n + 1 + i];

    QDLDL_int* etree = malloc(sizeof(QDLDL_int) * n);
    QDLDL_int* Lnz = malloc(sizeof(QDLDL_int) * n);
    QDLDL_int* Lp = malloc(sizeof(QDLDL_int) * (n + 1));
    QDLDL_float* D = malloc(sizeof(QDLDL_float) * n);
    QDLDL_float* Dinv = malloc(sizeof(QDLDL_float) * n);
    QDLDL_int* iwork = malloc(sizeof(QDLDL_int) * 3 * n);
    QDLDL_bool* bwork = malloc(sizeof(QDLDL_bool) * n);
    QDLDL_float* fwork = malloc(sizeof(QDLDL_float) * n);
    QDLDL_int sumLnz = QDLDL_etree(n, Ap, Ai, iwork, Lnz, etree);
    if (sumLnz < 0) return 3;
    QDLDL_int* Li = malloc(sizeof(QDLDL_int) * sumLnz);
    QDLDL_float* Lx = malloc(sizeof(QDLDL_float) * sumLnz);

    if (QDLDL_factor(n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork) < 0) return 3;
    double t0 = now_ns();
    for (int r = 0; r < rf; r++)
        QDLDL_factor(n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
    double tf = (now_ns() - t0) / rf;

    QDLDL_float* b = malloc(sizeof(QDLDL_float) * n);
    QDLDL_float* x = malloc(sizeof(QDLDL_float) * n);
    for (int64_t i = 0; i < n; i++) b[i] = 1.0 + (double)(i % 7);
    memcpy(x, b, sizeof(QDLDL_float) * n);
    QDLDL_solve(n, Lp, Li, Lx, Dinv, x);
    t0 = now_ns();
    for (int r = 0; r < rs; r++) {
        memcpy(x, b, sizeof(QDLDL_float) * n);
        QDLDL_solve(n, Lp, Li, Lx, Dinv, x);
    }
    double ts = (now_ns() - t0) / rs;
    printf("%.17g %.17g %.17g\n", tf, ts, x[0]);
    return 0;
}
