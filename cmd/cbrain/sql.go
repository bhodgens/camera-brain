package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	_ "github.com/lib/pq"
	"github.com/spf13/cobra"
)

var sqlCmd = &cobra.Command{
	Use:   "sql [SQL query]",
	Short: "Execute raw SQL queries",
	Long:  "Run direct SQL queries against the Camera Brain database. Destructive operations (DROP, DELETE, INSERT, UPDATE, TRUNCATE) are blocked for safety.",
	Example: `  cbrain sql "SELECT count(*) FROM observations"
  cbrain sql "SELECT class_name, count(*) FROM observations GROUP BY class_name" -o json
  cbrain sql "SELECT detected_at, camera_id FROM observations WHERE detected_at > NOW() - INTERVAL '1 hour'"

Example queries:
  # Visitors after 8 PM (temporal analysis)
  cbrain sql "SELECT date_trunc('day', detected_at) as date, count(*) as visitor_count FROM observations WHERE EXTRACT(HOUR FROM detected_at) >= 20 AND class_name = 'person' GROUP BY 1 ORDER BY 1 DESC"

  # Count detections by hour
  cbrain sql "SELECT date_trunc('hour', detected_at) as hour, class_name, count(*) FROM observations WHERE detected_at > NOW() - INTERVAL '24 hours' GROUP BY 1, 2 ORDER BY 1 DESC"

  # Cars in driveway this week
  cbrain sql "SELECT detected_at, attributes->>'color' as color FROM observations WHERE class_name ILIKE '%car%' AND camera_id = 'driveway' AND detected_at > NOW() - INTERVAL '7 days'"`,
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return fmt.Errorf("SQL query required")
		}

		query := strings.Join(args, " ")

		if !isValidSQL(query) {
			return fmt.Errorf("query blocked: destructive operations are not allowed")
		}

		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		outputFormat, _ := cmd.Flags().GetString("output")

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return fmt.Errorf("database connection failed: %w", err)
		}
		defer db.Close()

		rows, err := db.Query(query)
		if err != nil {
			return fmt.Errorf("query failed: %w", err)
		}
		defer rows.Close()

		return formatSQLResult(os.Stdout, outputFormat, rows)
	},
}

// isValidSQL checks whether a query contains only SELECT-like statements.
// It blocks destructive SQL operations.
func isValidSQL(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))

	blocked := []string{"DROP ", "DELETE ", "INSERT ", "UPDATE ", "TRUNCATE ", "ALTER ", "CREATE ", "GRANT ", "REVOKE "}
	for _, b := range blocked {
		if strings.Contains(upper, b) {
			return false
		}
	}

	// Check for statement injection via semicolons
	if strings.Contains(upper, ";") {
		return false
	}

	return true
}

func formatSQLResult(w io.Writer, format string, rows *sql.Rows) error {
	cols, err := rows.Columns()
	if err != nil {
		return err
	}

	if format == "json" {
		return formatSQLJSON(w, cols, rows)
	}

	return formatSQLTable(w, cols, rows)
}

func formatSQLJSON(w io.Writer, cols []string, rows *sql.Rows) error {
	var results []map[string]any
	for rows.Next() {
		row := make(map[string]any)
		vals := make([]any, len(cols))
		valPtrs := make([]any, len(cols))
		for i := range vals {
			valPtrs[i] = &vals[i]
		}
		if err := rows.Scan(valPtrs...); err != nil {
			return err
		}
		for i, col := range cols {
			row[col] = vals[i]
		}
		results = append(results, row)
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}

func formatSQLTable(w io.Writer, cols []string, rows *sql.Rows) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)

	// Header
	for i, col := range cols {
		if i > 0 {
			fmt.Fprint(tw, "\t")
		}
		fmt.Fprint(tw, col)
	}
	fmt.Fprintln(tw)

	// Separator
	for i := range cols {
		if i > 0 {
			fmt.Fprint(tw, "\t")
		}
		fmt.Fprint(tw, strings.Repeat("-", len(cols[i])))
	}
	fmt.Fprintln(tw)

	// Rows
	for rows.Next() {
		vals := make([]any, len(cols))
		valPtrs := make([]any, len(cols))
		for i := range vals {
			valPtrs[i] = &vals[i]
		}
		if err := rows.Scan(valPtrs...); err != nil {
			return err
		}
		for i := range cols {
			if i > 0 {
				fmt.Fprint(tw, "\t")
			}
			if vals[i] == nil {
				fmt.Fprint(tw, "NULL")
			} else {
				fmt.Fprintf(tw, "%v", vals[i])
			}
		}
		fmt.Fprintln(tw)
	}

	return tw.Flush()
}

func init() {
	rootCmd.AddCommand(sqlCmd)
}
