package scipy

// script is the whole of the Python the oracle runs: a read-eval-print loop
// over JSON lines that calls scipy.sparse.linalg (and cvxopt's SuiteSparse
// AMD) and returns plain numbers. It contains no logic of its own beyond
// translating a request into the corresponding SciPy call.
const script = `
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee

store = {}

def csr(d):
    return sp.csr_array((np.array(d["data"], dtype=float), np.array(d["indices"], dtype=np.int32),
                         np.array(d["indptr"], dtype=np.int32)), shape=(d["m"], d["n"]))

def op(n, f):
    return sla.LinearOperator((n, n), matvec=f, dtype=float)

def precond(spec, A):
    k = spec.get("kind", "none")
    n = A.shape[0]
    if k == "none":
        return None
    if k == "jacobi":
        d = A.diagonal()
        return op(n, lambda r: r / d)
    if k == "ssor":
        w = spec["omega"]; d = A.diagonal()
        lo = (sp.diags_array(d) + w * sp.tril(A, -1)).tocsr()
        up = (sp.diags_array(d) + w * sp.triu(A, 1)).tocsr()
        return op(n, lambda r: w * (2 - w) * sla.spsolve_triangular(up, d * sla.spsolve_triangular(lo, r, lower=True), lower=False))
    if k == "ilu":
        L = (csr(spec["L"]) + sp.eye_array(n)).tocsr(); U = csr(spec["U"])
        return op(n, lambda r: sla.spsolve_triangular(U, sla.spsolve_triangular(L, r, lower=True), lower=False))
    if k == "ic":
        L = csr(spec["L"]); Lt = L.T.tocsr()
        return op(n, lambda r: sla.spsolve_triangular(Lt, sla.spsolve_triangular(L, r, lower=True), lower=False))
    raise ValueError(k)

def solve(req):
    A = csr(req["A"]); b = np.array(req["b"], dtype=float)
    M = precond(req.get("M") or {}, A)
    cnt = [0]
    def cb(x):
        cnt[0] += 1
    s = req["solver"]
    rtol = req["rtol"]; atol = req.get("atol", 0.0); maxiter = req.get("maxiter")
    if s == "cg":
        x, info = sla.cg(A, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M, callback=cb)
    elif s == "bicgstab":
        x, info = sla.bicgstab(A, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M, callback=cb)
    elif s == "gmres":
        x, info = sla.gmres(A, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M, restart=req["restart"],
                            callback=cb, callback_type="pr_norm")
    elif s == "minres":
        x, info = sla.minres(A, b, rtol=rtol, maxiter=maxiter, M=M, callback=cb)
    else:
        raise ValueError(s)
    return {"info": int(info), "iters": cnt[0], "res": float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))}

def lsqr(req):
    A = csr(req["A"]); b = np.array(req["b"], dtype=float)
    r = sla.lsqr(A, b, damp=req["damp"], atol=req["atol"], btol=req["btol"], conlim=req["conlim"], iter_lim=req["iter_lim"])
    return {"x": r[0].tolist(), "istop": int(r[1]), "itn": int(r[2]), "r1norm": float(r[3]), "r2norm": float(r[4]),
            "anorm": float(r[5]), "acond": float(r[6]), "arnorm": float(r[7]), "xnorm": float(r[8])}

def rcm(req):
    return {"perm": [int(v) for v in reverse_cuthill_mckee(csr(req["A"]), symmetric_mode=False)]}

def lu(req):
    A = csr(req["A"]); perm = req.get("perm")
    if perm is not None:
        A = A[:, perm] if req.get("symmetric") is not True else A[perm][:, perm]
    opts = {}
    if req.get("symmetric"):
        opts = {"SymmetricMode": True, "DiagPivotThresh": 0.0}
    f = sla.splu(sp.csc_array(A), permc_spec=req["spec"], options=opts)
    return {"nnz": int(f.L.nnz + f.U.nnz), "perm_c": [int(v) for v in f.perm_c]}

def amd(req):
    from cvxopt import amd as camd, spmatrix
    d = req["A"]
    A = csr(d).tocoo()
    keep = A.row >= A.col
    M = spmatrix(A.data[keep].tolist(), A.row[keep].tolist(), A.col[keep].tolist(), (d["m"], d["n"]))
    return {"perm": [int(v) for v in camd.order(M)]}

def load(req):
    store[req["name"]] = (csr(req["A"]), np.array(req["b"], dtype=float))
    return {}

def timeit(req):
    A, b = store[req["name"]]
    what = req["what"]; reps = req["reps"]
    if what == "spmv":
        t0 = time.perf_counter()
        for _ in range(reps):
            y = A @ b
        return {"seconds": time.perf_counter() - t0}
    if what == "cg":
        t0 = time.perf_counter()
        sla.cg(A, b, rtol=0.0, atol=0.0, maxiter=reps)
        return {"seconds": time.perf_counter() - t0}
    if what == "gmres":
        t0 = time.perf_counter()
        sla.gmres(A, b, rtol=0.0, atol=0.0, maxiter=reps, restart=req["restart"])
        return {"seconds": time.perf_counter() - t0}
    raise ValueError(what)

def version(req):
    import scipy, numpy
    return {"scipy": scipy.__version__, "numpy": numpy.__version__}

OPS = {"solve": solve, "lsqr": lsqr, "rcm": rcm, "lu": lu, "amd": amd, "load": load, "time": timeit, "version": version}

for line in sys.stdin:
    try:
        req = json.loads(line)
        resp = OPS[req["op"]](req)
    except Exception as e:
        resp = {"error": repr(e)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
`
