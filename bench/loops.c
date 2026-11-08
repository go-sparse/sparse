/* Timing loops over QDLDL v0.1.9 (Apache-2.0, copied verbatim beside this file
 * from https://github.com/osqp/qdldl). Every loop runs the reference
 * `iters` times inside this process, so no call boundary is in the interval.
 * The data are copies of the Go side's arrays: the same matrix, the same
 * right-hand side. */
#include <stdlib.h>
#include <string.h>
#include "qdldl.h"

typedef struct {
    int n, nnz, nl;
    int *Ap, *Ai;
    double *Ax;            /* upper triangle that is factored (permuted for the AMD cases) */
    int *etree, *Lnz, *Lp, *Li;
    double *Lx, *D, *Dinv;
    int *iwork;
    unsigned char *bwork;
    double *fwork;
    double *b, *x, *w;     /* right-hand side, solution, permutation scratch */
    int *perm;             /* perm[i] = original index of permuted row i */
    int *map;              /* Ax[k] = Kx[map[k]] */
    double *Kx;            /* values of the unpermuted matrix */
} kkt_t;

static void *xcopy(const void *p, size_t bytes) {
    void *q = malloc(bytes ? bytes : 1);
    memcpy(q, p, bytes);
    return q;
}

/* kkt_new builds the reference state for the upper triangle (Ap, Ai, Ax) of n
 * columns. perm, map and Kx may be NULL (natural order). It returns NULL when
 * QDLDL_etree or QDLDL_factor refuses the matrix. */
void *kkt_new(int n, const int *Ap, const int *Ai, const double *Ax, const double *b,
              const int *perm, const int *map, const double *Kx, int nKx) {
    kkt_t *k = calloc(1, sizeof(kkt_t));
    k->n = n;
    k->nnz = Ap[n];
    k->Ap = xcopy(Ap, sizeof(int) * (n + 1));
    k->Ai = xcopy(Ai, sizeof(int) * k->nnz);
    k->Ax = xcopy(Ax, sizeof(double) * k->nnz);
    k->b = xcopy(b, sizeof(double) * n);
    k->x = malloc(sizeof(double) * n);
    k->w = malloc(sizeof(double) * n);
    if (perm) k->perm = xcopy(perm, sizeof(int) * n);
    if (map) k->map = xcopy(map, sizeof(int) * k->nnz);
    if (Kx) k->Kx = xcopy(Kx, sizeof(double) * nKx);
    k->etree = malloc(sizeof(int) * n);
    k->Lnz = malloc(sizeof(int) * n);
    k->Lp = malloc(sizeof(int) * (n + 1));
    k->D = malloc(sizeof(double) * n);
    k->Dinv = malloc(sizeof(double) * n);
    k->iwork = malloc(sizeof(int) * 3 * n);
    k->bwork = malloc(n);
    k->fwork = malloc(sizeof(double) * n);
    int sum = QDLDL_etree(n, k->Ap, k->Ai, k->iwork, k->Lnz, k->etree);
    if (sum < 0) return NULL;
    k->nl = sum;
    k->Li = malloc(sizeof(int) * (sum ? sum : 1));
    k->Lx = malloc(sizeof(double) * (sum ? sum : 1));
    if (QDLDL_factor(n, k->Ap, k->Ai, k->Ax, k->Lp, k->Li, k->Lx, k->D, k->Dinv, k->Lnz, k->etree,
                     k->bwork, k->iwork, k->fwork) < 0)
        return NULL;
    return k;
}

int kkt_nnzl(void *h) { return ((kkt_t *)h)->nl; }

/* kkt_solve_one solves K x = b once and copies x out (for the parity check). */
void kkt_solve_one(void *h, double *out) {
    kkt_t *k = h;
    memcpy(k->x, k->b, sizeof(double) * k->n);
    QDLDL_solve(k->n, k->Lp, k->Li, k->Lx, k->Dinv, k->x);
    memcpy(out, k->x, sizeof(double) * k->n);
}

/* kkt_psolve_one is the permuted solve: x = P^T (K_p^{-1} (P b)). */
void kkt_psolve_one(void *h, double *out) {
    kkt_t *k = h;
    for (int i = 0; i < k->n; i++) k->w[i] = k->b[k->perm[i]];
    QDLDL_solve(k->n, k->Lp, k->Li, k->Lx, k->Dinv, k->w);
    for (int i = 0; i < k->n; i++) out[k->perm[i]] = k->w[i];
}

double loop_etree(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) s += QDLDL_etree(k->n, k->Ap, k->Ai, k->iwork, k->Lnz, k->etree);
    return s;
}

double loop_factor(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++)
        s += QDLDL_factor(k->n, k->Ap, k->Ai, k->Ax, k->Lp, k->Li, k->Lx, k->D, k->Dinv, k->Lnz,
                          k->etree, k->bwork, k->iwork, k->fwork);
    return s;
}

/* loop_pfactor: the numeric phase of a permuted factorisation as OSQP does it:
 * gather the values of the permuted upper triangle through the precomputed
 * map, then QDLDL_factor. */
double loop_pfactor(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        for (int p = 0; p < k->nnz; p++) k->Ax[p] = k->Kx[k->map[p]];
        s += QDLDL_factor(k->n, k->Ap, k->Ai, k->Ax, k->Lp, k->Li, k->Lx, k->D, k->Dinv, k->Lnz,
                          k->etree, k->bwork, k->iwork, k->fwork);
    }
    return s;
}

double loop_solve(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        memcpy(k->x, k->b, sizeof(double) * k->n);
        QDLDL_solve(k->n, k->Lp, k->Li, k->Lx, k->Dinv, k->x);
        s += k->x[0];
    }
    return s;
}

double loop_lsolve(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        memcpy(k->x, k->b, sizeof(double) * k->n);
        QDLDL_Lsolve(k->n, k->Lp, k->Li, k->Lx, k->x);
        s += k->x[0];
    }
    return s;
}

double loop_ltsolve(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        memcpy(k->x, k->b, sizeof(double) * k->n);
        QDLDL_Ltsolve(k->n, k->Lp, k->Li, k->Lx, k->x);
        s += k->x[0];
    }
    return s;
}

double loop_dsolve(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        memcpy(k->x, k->b, sizeof(double) * k->n);
        for (int i = 0; i < k->n; i++) k->x[i] *= k->Dinv[i];
        s += k->x[0];
    }
    return s;
}

double loop_psolve(void *h, long iters) {
    kkt_t *k = h;
    double s = 0;
    for (long r = 0; r < iters; r++) {
        /* in place, as Permuted.Solve is: copy b into x, gather, solve, scatter */
        memcpy(k->x, k->b, sizeof(double) * k->n);
        for (int i = 0; i < k->n; i++) k->w[i] = k->x[k->perm[i]];
        QDLDL_solve(k->n, k->Lp, k->Li, k->Lx, k->Dinv, k->w);
        for (int i = 0; i < k->n; i++) k->x[k->perm[i]] = k->w[i];
        s += k->x[0];
    }
    return s;
}
