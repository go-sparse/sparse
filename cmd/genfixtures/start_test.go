package main

import "os/exec"

func nil2() *exec.Cmd { return exec.Command("/nonexistent/oracle") }
