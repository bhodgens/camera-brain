// cmd/cbrain/query_test.go
package main

import (
	"encoding/json"
	"testing"
)

func TestBuildQueryRequest(t *testing.T) {
	tests := []struct {
		query string
		want  string
	}{
		{"person", `{"query":"person"}`},
		{"vehicles in driveway", `{"query":"vehicles in driveway"}`},
	}

	for _, tt := range tests {
		got := buildQueryRequest(tt.query)
		if got != tt.want {
			t.Errorf("buildQueryRequest(%q) = %s, want %s", tt.query, got, tt.want)
		}
		// Verify it round-trips back to the same query
		var m map[string]string
		if err := json.Unmarshal([]byte(got), &m); err != nil {
			t.Fatalf("buildQueryRequest(%q) produced invalid JSON: %v", tt.query, err)
		}
		if m["query"] != tt.query {
			t.Errorf("buildQueryRequest(%q) round-trip got %q", tt.query, m["query"])
		}
	}
}
