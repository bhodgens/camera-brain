// cmd/cbrain/correlate_test.go
package main

import "testing"

func TestEscapeILIKE(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"plain", "person", "person"},
		{"percent", "100%", `100\%`},
		{"underscore", "front_door", `front\_door`},
		{"backslash", `C:\path`, `C:\\path`},
		{"combined", `%foo_bar\`, `\%foo\_bar\\`},
		{"empty", "", ""},
		{"no special chars", "car", "car"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := escapeILIKE(tt.input)
			if got != tt.want {
				t.Errorf("escapeILIKE(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestCameraTransition(t *testing.T) {
	tests := []struct {
		name string
		from string
		to   string
		want string
	}{
		{"first entry", "", "front_door", "front_door"},
		{"same camera", "front_door", "front_door", "[front_door]"},
		{"camera change", "front_door", "driveway", "front_door→driveway"},
		{"empty to empty", "", "", ""},
		{"to empty", "front_door", "", "front_door→"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cameraTransition(tt.from, tt.to)
			if got != tt.want {
				t.Errorf("cameraTransition(%q, %q) = %q, want %q", tt.from, tt.to, got, tt.want)
			}
		})
	}
}
