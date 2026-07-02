package main

import (
	"testing"
)

func TestValidateSQL(t *testing.T) {
	tests := []struct {
		sql   string
		valid bool
	}{
		{"SELECT * FROM observations", true},
		{"DROP TABLE observations", false},
		{"DELETE FROM observations", false},
		{"INSERT INTO observations", false},
		{"UPDATE observations SET", false},
		{"SELECT 1; DROP TABLE--", false},
		{"TRUNCATE TABLE observations", false},
		{"ALTER TABLE observations ADD", false},
		{"CREATE TABLE evil", false},
		{"SELECT 1 -- DROP TABLE", false},
	}

	for _, tt := range tests {
		t.Run(tt.sql, func(t *testing.T) {
			got := isValidSQL(tt.sql)
			if got != tt.valid {
				t.Errorf("isValidSQL(%q) = %v, want %v", tt.sql, got, tt.valid)
			}
		})
	}
}
