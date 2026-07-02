// cmd/cbrain/query_test.go
package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestParseQueryRequest(t *testing.T) {
	tests := []struct {
		query    string
		expected map[string]string
	}{
		{"person", map[string]string{"query": "person"}},
		{"vehicles in driveway", map[string]string{"query": "vehicles in driveway"}},
	}

	for _, tt := range tests {
		got := buildQueryRequest(tt.query)
		var gotMap map[string]string
		json.Unmarshal([]byte(got), &gotMap)
		if !reflect.DeepEqual(gotMap, tt.expected) {
			t.Errorf("buildQueryRequest(%q) = %v, want %v", tt.query, gotMap, tt.expected)
		}
	}
}
