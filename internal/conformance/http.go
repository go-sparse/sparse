//go:build !android && !ios

package conformance

import (
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPGet fetches a URL with a timeout. (Not built for android and ios,
// where importing net/http pulls in cgo and the corpus download is
// meaningless.)
func HTTPGet(url string) ([]byte, error) {
	c := http.Client{Timeout: 60 * time.Second}
	r, err := c.Get(url)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("conformance: GET %s: %s", url, r.Status)
	}
	return io.ReadAll(r.Body)
}
