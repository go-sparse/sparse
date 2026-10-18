//go:build !android && !ios

package conformance

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHTTPGet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/missing" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte("hello"))
	}))
	b, err := HTTPGet(srv.URL + "/ok")
	if err != nil || string(b) != "hello" {
		t.Fatal(string(b), err)
	}
	if _, err := HTTPGet(srv.URL + "/missing"); err == nil {
		t.Fatal("404")
	}
	srv.Close()
	if _, err := HTTPGet(srv.URL + "/ok"); err == nil {
		t.Fatal("closed server")
	}
}
