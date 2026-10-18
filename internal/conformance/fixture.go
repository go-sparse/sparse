package conformance

import (
	"encoding/json"
	"os"

	"github.com/go-sparse/sparse/internal/scipy"
)

// SuiteSolve and SuiteOrder name the matrices of the committed suite.
var (
	SuiteSolve = []string{"lap2d:30:30", "lap3d:12:12:12", "cd2d:30:30:2", "rnd:1000:6:1:1", "rnd:1000:6:0:2"}
	SuiteOrder = []string{"lap2d:30:30+scr:1", "lap3d:12:12:12+scr:2", "cd2d:30:30:1+scr:3", "rnd:1000:3:1:4", "rnd:1000:3:0:5"}
)

// Suite builds the named matrices.
func Suite(names []string) ([]Named, error) {
	var out []Named
	for _, n := range names {
		m, err := Build(n)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}

// Fixture is the committed oracle record.
type Fixture struct {
	SciPy  string     `json:"scipy"`
	NumPy  string     `json:"numpy"`
	Solves []SolveFix `json:"solves"`
	LSQR   []LSQRFix  `json:"lsqr"`
	LSMR   []LSMRFix  `json:"lsmr"`
	Orders []OrderFix `json:"orders"`
}

// GenerateAll runs the whole suite against the oracle.
func GenerateAll(s *scipy.Session) (*Fixture, error) {
	f := &Fixture{}
	var err error
	if f.SciPy, f.NumPy, err = s.Versions(); err != nil {
		return nil, err
	}
	sm, _ := Suite(SuiteSolve) // the suite names are constants that Build accepts
	if f.Solves, err = Generate(s, sm); err != nil {
		return nil, err
	}
	if f.LSQR, err = GenerateLSQR(s); err != nil {
		return nil, err
	}
	if f.LSMR, err = GenerateLSMR(s); err != nil {
		return nil, err
	}
	om, _ := Suite(SuiteOrder)
	f.Orders, err = GenerateOrders(s, om)
	return f, err
}

// SaveFixture writes f as indented JSON.
func SaveFixture(path string, f *Fixture) error {
	b, _ := json.MarshalIndent(f, "", " ") // plain data: cannot fail
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// LoadFixture reads a fixture file.
func LoadFixture(path string) (*Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f Fixture
	return &f, json.Unmarshal(b, &f)
}
