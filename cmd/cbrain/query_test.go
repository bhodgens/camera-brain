// cmd/cbrain/query_test.go
package main

import (
	"testing"
)

func TestParseQueryRequest(t *testing.T) {
	tests := []struct {
		query    string
		wantJSON string
	}{
		{"person", `{"query":"person"}`},
		{"vehicles in driveway", `{"query":"vehicles in driveway"}`},
	}

	for _, tt := range tests {
		got := buildQueryRequest(tt.query)
		if got != tt.wantJSON {
			t.Errorf("buildQueryRequest(%q) = %v, want %v", tt.query, got, tt.wantJSON)
		}
	}
}
