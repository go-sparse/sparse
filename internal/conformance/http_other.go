//go:build android || ios

package conformance

import "errors"

// HTTPGet is unavailable on android and ios.
func HTTPGet(string) ([]byte, error) {
	return nil, errors.New("conformance: downloading is not supported on this platform")
}
