package scipy

// LSMRResponse is the tuple scipy.sparse.linalg.lsmr returns.
type LSMRResponse struct {
	X      []float64 `json:"x"`
	Istop  int       `json:"istop"`
	Itn    int       `json:"itn"`
	NormR  float64   `json:"normr"`
	NormAR float64   `json:"normar"`
	NormA  float64   `json:"norma"`
	CondA  float64   `json:"conda"`
	NormX  float64   `json:"normx"`
}

// LSMR runs scipy lsmr; the request has the shape of an LSQR request
// (IterLim is lsmr's maxiter).
func (s *Session) LSMR(r LSQRRequest) (LSMRResponse, error) {
	r.Op = "lsmr"
	var out LSMRResponse
	return out, s.Call(r, &out)
}
