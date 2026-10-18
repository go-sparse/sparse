package conformance

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-sparse/sparse/mtx"
)

// CorpusEntry names a matrix of the NIST Matrix Market Harwell-Boeing set.
type CorpusEntry struct{ Set, Name string }

// CorpusBase is where the matrices come from. They carry no stated licence,
// so they are never copied into this repository: they are downloaded into a
// cache directory outside every work tree and checked against pinned SHA-256
// sums.
const CorpusBase = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/"

// Corpus is the small named set (n <= 5000) used by the live comparison.
var Corpus = []CorpusEntry{
	{"bcsstruc1", "bcsstk01"}, {"bcsstruc1", "bcsstk04"}, {"bcsstruc1", "bcsstk05"},
	{"lanpro", "nos1"}, {"lanpro", "nos2"}, {"lanpro", "nos3"}, {"lanpro", "nos5"}, {"lanpro", "nos6"}, {"lanpro", "nos7"},
	{"chemwest", "west0067"}, {"chemwest", "west0479"},
	{"sherman", "sherman1"}, {"sherman", "sherman4"}, {"pores", "pores_3"},
	{"oilgen", "orsirr_1"}, {"oilgen", "orsreg_1"},
}

// CorpusDir returns the cache directory (GO_SPARSE_CORPUS overrides it) and
// refuses a directory inside a git work tree.
func CorpusDir(getenv func(string) string, userCache func() (string, error)) (string, error) {
	dir := getenv("GO_SPARSE_CORPUS")
	if dir == "" {
		c, err := userCache()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(c, "go-sparse-corpus")
	}
	dir, _ = filepath.Abs(dir) // fails only without a working directory
	for d := dir; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return "", fmt.Errorf("conformance: corpus directory %s is inside the git work tree %s", dir, d)
		}
		if filepath.Dir(d) == d {
			return dir, nil
		}
	}
}

// Fetch makes sure every entry is in dir (as <name>.mtx.gz) and returns the
// paths. A file already present is reused; a downloaded or present file whose
// SHA-256 is pinned in sums must match it. unpinned lists the entries that
// had no pinned sum, with the sum observed, for the maintainer to pin.
func Fetch(dir string, get func(string) ([]byte, error), entries []CorpusEntry, sums map[string]string) (paths []string, unpinned map[string]string, err error) {
	if err = os.MkdirAll(dir, 0o755); err != nil {
		return nil, nil, err
	}
	unpinned = map[string]string{}
	for _, e := range entries {
		p := filepath.Join(dir, e.Name+".mtx.gz")
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			if b, err = get(CorpusBase + e.Set + "/" + e.Name + ".mtx.gz"); err != nil {
				return nil, nil, err
			}
			if err = os.WriteFile(p, b, 0o644); err != nil {
				return nil, nil, err
			}
		}
		sum := sha256.Sum256(b)
		got := hex.EncodeToString(sum[:])
		want, pinned := sums[e.Name]
		switch {
		case !pinned:
			unpinned[e.Name] = got
		case want != got:
			return nil, nil, fmt.Errorf("conformance: %s has sha256 %s, pinned %s", p, got, want)
		}
		paths = append(paths, p)
	}
	return paths, unpinned, nil
}

// LoadCorpus reads the files; symmetric files are treated as SPD candidates.
func LoadCorpus(paths []string) ([]Named, error) {
	var out []Named
	for _, p := range paths {
		m, err := mtx.ReadFile(p)
		if err != nil {
			return nil, err
		}
		a, err := m.RealCSR()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		name := strings.TrimSuffix(filepath.Base(p), ".mtx.gz")
		out = append(out, Named{Name: name, A: a, SPD: m.Symmetry == mtx.Symmetric && a.Rows == a.Cols})
	}
	return out, nil
}
