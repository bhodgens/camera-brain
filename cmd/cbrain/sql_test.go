package main

import (
	"testing"
)

func TestValidateSQL(t *testing.T) {
	tests := []struct {
		sql   string
		valid bool
	}{
		// Valid read-only queries.
		{"SELECT * FROM observations", true},
		{"SHOW timezone", true},
		{"WITH x AS (SELECT 1) SELECT * FROM x", true},
		{"VALUES (1), (2), (3)", true},
		{"EXPLAIN SELECT * FROM observations", true},
		{"  select 1", true}, // leading whitespace + lowercase

		// Destructive statements must be blocked.
		{"DROP TABLE observations", false},
		{"DELETE FROM observations", false},
		{"INSERT INTO observations", false},
		{"UPDATE observations SET", false},
		{"TRUNCATE TABLE observations", false},
		{"ALTER TABLE observations ADD", false},
		{"CREATE TABLE evil", false},
		{"GRANT SELECT ON observations TO public", false},

		// Multi-statement / comment-based injection.
		{"SELECT 1; DROP TABLE--", false},
		{"SELECT 1 -- DROP TABLE", false},
		{"SELECT 1; DROP TABLE x", false},

		// EXPLAIN ANALYZE *executes* the underlying statement, so it must
		// be blocked even when the inner verb is destructive — and also
		// when it is read-only, because ANALYZE semantically runs the query
		// and we have not audited every verb that can follow ANALYZE.
		{"EXPLAIN ANALYZE DELETE FROM observations", false},
		{"EXPLAIN ANALYZE UPDATE observations SET x=1", false},
		{"EXPLAIN ANALYZE INSERT INTO observations DEFAULT VALUES", false},
		{"EXPLAIN ANALYZE TRUNCATE observations", false},
		{"EXPLAIN ANALYZE SELECT * FROM observations", false},
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
