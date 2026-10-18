package conformance

import (
	"fmt"
	"strings"
)

// SolveTable renders solver rows as a Markdown table.
func SolveTable(rows []SolveRow) string {
	var b strings.Builder
	b.WriteString("| matrix | solver | preconditioner | SciPy it | Go it | SciPy info | Go info | SciPy ‖b−Ax‖/‖b‖ | Go ‖b−Ax‖/‖b‖ |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %s | %s | %s | %d | %d | %d | %d | %.2e | %.2e |\n",
			r.Matrix, r.Solver, r.Precond, r.Sci.Iters, r.GoIters, r.Sci.Info, r.GoInfo, r.Sci.Res, r.GoRes)
	}
	return b.String()
}

// LSQRTable renders least-squares rows.
func LSQRTable(rows []LSQRRow) string {
	var b strings.Builder
	b.WriteString("| system m:n:k:seed | damp | atol=btol | SciPy istop/itn | Go istop/itn | max rel. diff of x |\n|---|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %s | %g | %g | %d/%d | %d/%d | %.1e |\n", r.Matrix, r.Damp, r.ATol, r.Sci.Istop, r.Sci.Itn, r.GoIstop, r.GoItn, r.MaxXDiff)
	}
	return b.String()
}

// LSMRTable renders lsmr rows.
func LSMRTable(rows []LSMRRow) string {
	var b strings.Builder
	b.WriteString("| system m:n:k:seed | damp | atol=btol | SciPy istop/itn | Go istop/itn | max rel. diff of x |\n|---|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %s | %g | %g | %d/%d | %d/%d | %.1e |\n", r.Matrix, r.Damp, r.ATol, r.Sci.Istop, r.Sci.Itn, r.GoIstop, r.GoItn, r.MaxXDiff)
	}
	return b.String()
}

// OrderTable renders ordering rows: Go next to SciPy / SuiteSparse.
func OrderTable(rows []OrderRow) string {
	var b strings.Builder
	b.WriteString("| matrix | n | nnz | RCM bandwidth Go / SciPy | RCM profile Go / SciPy | Cholesky nnz, AMD Go / SuiteSparse AMD | LU nnz, AMD-ordered Go / SuperLU MMD(A+Aᵀ) | LU nnz, COLAMD Go / SuperLU COLAMD (natural) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		amd, amdLU := "–", "–"
		if r.SuiteSparseAMDChol > 0 {
			amd = fmt.Sprintf("%d / %d", r.GoAMDChol, r.SuiteSparseAMDChol)
			amdLU = fmt.Sprintf("%d / %d", r.GoAMDLU, r.SuperLUMMDLU)
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %d / %d | %d / %d | %s | %s | %d / %d (%d) |\n", r.Matrix, r.N, r.NNZ,
			r.GoRCMBandwidth, r.SciRCMBandwidth, r.GoRCMProfile, r.SciRCMProfile, amd, amdLU, r.GoCOLAMDLU, r.SciCOLAMDLU, r.NaturalLU)
	}
	return b.String()
}
