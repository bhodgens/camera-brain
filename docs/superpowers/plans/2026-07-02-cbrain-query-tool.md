# cbrain Query Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool `cbrain` that provides natural language querying, direct SQL access, and inference/correlation analysis for the Camera Brain system.

**Architecture:** Single Go binary in `cmd/cbrain/` with subcommand structure (`query`, `sql`, `infer`, `correlate`). Reads config from `/etc/camera-brain/camera-brain.env` or environment. Connects to query-engine HTTP API (8082) and optionally direct PostgreSQL for advanced queries.

**Tech Stack:** Go 1.24, cobra CLI framework, lib/pq for PostgreSQL, encoding/json for HTTP, text/tabwriter for formatted output.

---

## File Structure

**Files to create:**
- `cmd/cbrain/main.go` - CLI entry point with cobra subcommands
- `cmd/cbrain/config.go` - Configuration loading from env file
- `cmd/cbrain/query.go` - Natural language query subcommand
- `cmd/cbrain/sql.go` - Direct SQL execution subcommand
- `cmd/cbrain/infer.go` - Inference analysis subcommand (routines, anomalies, vehicles, visitors, security, deliveries, animals, workers)
- `cmd/cbrain/correlate.go` - Cross-camera correlation subcommand
- `cmd/cbrain/output.go` - Formatted output helpers (JSON, table, plain)

**Files to modify:**
- `Makefile` - Add `cbrain` build target
- `README.md` - Update Example section with `cbrain` commands
- `deploy/install.sh` - Install `cbrain` binary to `/usr/local/bin`

---

### Task 1: CLI Skeleton with cobra

**Files:**
- Create: `cmd/cbrain/main.go`
- Create: `cmd/cbrain/config.go`

- [ ] **Step 1: Create main.go with cobra root command**

```go
// cmd/cbrain/main.go
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "cbrain",
		Short: "Camera Brain CLI query tool",
		Long:  "Natural language querying and analysis for Camera Brain video surveillance system.",
	}

	rootCmd.PersistentFlags().String("config", "/etc/camera-brain/camera-brain.env", "config file path")
	rootCmd.PersistentFlags().StringP("output", "o", "table", "output format: json, table, plain")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
```

- [ ] **Step 2: Create config.go with LoadConfig function**

```go
// cmd/cbrain/config.go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type Config struct {
	DBHost     string
	DBPort     int
	DBName     string
	DBUser     string
	DBPassword string
	QueryURL   string
	GatewayURL string
}

func LoadConfig(path string) (*Config, error) {
	cfg := &Config{
		DBHost:     "localhost",
		DBPort:     5432,
		DBName:     "camera_brain",
		DBUser:     "camera_brain",
		QueryURL:   "http://localhost:8082",
		GatewayURL: "http://localhost:8080",
	}

	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Use defaults from environment or hardcoded
			loadEnvDefaults(cfg)
			return cfg, nil
		}
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key, value := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		switch key {
		case "DB_HOST":
			cfg.DBHost = value
		case "DB_PORT":
			fmt.Sscanf(value, "%d", &cfg.DBPort)
		case "DB_NAME":
			cfg.DBName = value
		case "DB_USER":
			cfg.DBUser = value
		case "DB_PASSWORD":
			cfg.DBPassword = value
		}
	}

	loadEnvDefaults(cfg)
	return cfg, scanner.Err()
}

func loadEnvDefaults(cfg *Config) {
	if v := os.Getenv("CB_QUERY_URL"); v != "" {
		cfg.QueryURL = v
	}
	if v := os.Getenv("CB_GATEWAY_URL"); v != "" {
		cfg.GatewayURL = v
	}
}
```

- [ ] **Step 3: Add cobra dependency and build**

Run: `go get github.com/spf13/cobra`

Expected: Downloads cobra and dependencies

Run: `go build -o cbrain ./cmd/cbrain/`

Expected: Creates `cbrain` binary, `./cbrain --help` shows subcommands

- [ ] **Step 4: Commit**

```bash
git add cmd/cbrain/main.go cmd/cbrain/config.go go.mod go.sum
git commit -m "feat: cbrain CLI skeleton with cobra and config loading"
```

---

### Task 2: Natural Language Query Subcommand

**Files:**
- Create: `cmd/cbrain/query.go`
- Create: `cmd/cbrain/output.go`

- [ ] **Step 1: Write test for query command**

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cmd/cbrain/ -run TestParseQueryRequest -v`

Expected: FAIL (function not defined)

- [ ] **Step 3: Create output.go with format helpers**

```go
// cmd/cbrain/output.go
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"text/tabwriter"
)

type QueryResponse struct {
	Success      bool         `json:"success"`
	Answer       string       `json:"answer"`
	ParsedQuery  *ParsedQuery `json:"parsed_query,omitempty"`
	ResultCount  int          `json:"result_count"`
	ProcessingMS int64        `json:"processing_ms"`
}

type ParsedQuery struct {
	SQL        string            `json:"sql"`
	Params     map[string]any    `json:"params,omitempty"`
	TimeRange  TimeRange         `json:"time_range"`
	EntityType string            `json:"entity_type"`
	Filters    map[string]string `json:"filters"`
}

type TimeRange struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

func FormatQueryResponse(w io.Writer, format string, resp *QueryResponse) error {
	switch format {
	case "json":
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(resp)
	case "plain":
		fmt.Fprintln(w, resp.Answer)
		fmt.Fprintf(w, "\nResults: %d observations in %dms\n", resp.ResultCount, resp.ProcessingMS)
		return nil
	case "table":
		fallthrough
	default:
		fmt.Fprintln(w, resp.Answer)
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintf(tw, "\nEntity\t%s\n", resp.ParsedQuery.EntityType)
		fmt.Fprintf(tw, "Time Range\t%s to %s\n", resp.ParsedQuery.TimeRange.Start, resp.ParsedQuery.TimeRange.End)
		fmt.Fprintf(tw, "Results\t%d\n", resp.ResultCount)
		fmt.Fprintf(tw, "Processing\t%dms\n", resp.ProcessingMS)
		if resp.ParsedQuery.SQL != "" {
			fmt.Fprintf(tw, "SQL\t%s\n", resp.ParsedQuery.SQL)
		}
		return tw.Flush()
	}
}
```

- [ ] **Step 4: Write query.go subcommand**

```go
// cmd/cbrain/query.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/spf13/cobra"
)

var queryCmd = &cobra.Command{
	Use:   "query [natural language question]",
	Short: "Ask questions about video observations",
	Long:  "Convert natural language queries into SQL and retrieve answers from the Camera Brain database.",
	Example: `  cbrain query "Who was at the front door this morning?"
  cbrain query "Show me all vehicles in the driveway last week"
  cbrain query "When did the kids come home from school?" -o json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return fmt.Errorf("query text required")
		}

		queryText := bytes.Join(bytes.SplitN([]byte(args[0]), []byte(" "), -1), []byte(" "))

		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		outputFormat, _ := cmd.Flags().GetString("output")

		resp, err := postQuery(cfg.QueryURL+"/query", queryText)
		if err != nil {
			return fmt.Errorf("query failed: %w", err)
		}

		return FormatQueryResponse(os.Stdout, outputFormat, resp)
	},
}

func buildQueryRequest(query string) string {
	req := map[string]string{"query": query}
	data, _ := json.Marshal(req)
	return string(data)
}

func postQuery(url string, queryJSON []byte) (*QueryResponse, error) {
	resp, err := http.Post(url, "application/json", bytes.NewReader(queryJSON))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result QueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func init() {
	rootCmd.AddCommand(queryCmd)
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./cmd/cbrain/ -run TestParseQueryRequest -v`

Expected: PASS

- [ ] **Step 6: Build and test manually**

Run: `go build -o cbrain ./cmd/cbrain/`

Run: `./cbrain query "person" -o table`

Expected: Shows query response in table format

Run: `./cbrain query "person" -o json`

Expected: Shows full JSON response

- [ ] **Step 7: Commit**

```bash
git add cmd/cbrain/query.go cmd/cbrain/output.go cmd/cbrain/query_test.go
git commit -m "feat: cbrain query subcommand with JSON/table/plain output"
```

---

### Task 3: Direct SQL Subcommand

**Files:**
- Create: `cmd/cbrain/sql.go`

- [ ] **Step 1: Add example SQL queries to command help**

Add to the Long description in sqlCmd:

```go
Example queries:
  # Visitors after 8 PM (temporal analysis)
  cbrain sql "SELECT date_trunc('day', detected_at) as date, count(*) as visitor_count FROM observations WHERE EXTRACT(HOUR FROM detected_at) >= 20 AND class_name = 'person' GROUP BY 1 ORDER BY 1 DESC"

  # Count detections by hour
  cbrain sql "SELECT date_trunc('hour', detected_at) as hour, class_name, count(*) FROM observations WHERE detected_at > NOW() - INTERVAL '24 hours' GROUP BY 1, 2 ORDER BY 1 DESC"

  # Cars in driveway this week
  cbrain sql "SELECT detected_at, attributes->>'color' as color FROM observations WHERE class_name ILIKE '%car%' AND camera_id = 'driveway' AND detected_at > NOW() - INTERVAL '7 days'"
```

- [ ] **Step 2: Write failing test**

```go
// cmd/cbrain/sql_test.go
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
	}

	for _, tt := range tests {
		got := isValidSQL(tt.sql)
		if got != tt.valid {
			t.Errorf("isValidSQL(%q) = %v, want %v", tt.sql, got, tt.valid)
		}
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test ./cmd/cbrain/ -run TestValidateSQL -v`

Expected: FAIL

- [ ] **Step 4: Implement sql.go with safety validation**

```go
// cmd/cbrain/sql.go
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
  cbrain sql "SELECT detected_at, camera_id FROM observations WHERE detected_at > NOW() - INTERVAL '1 hour'"`,
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

func isValidSQL(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))

	blocked := []string{"DROP ", "DELETE ", "INSERT ", "UPDATE ", "TRUNCATE ", "ALTER ", "CREATE ", "GRANT ", "REVOKE "}
	for _, b := range blocked {
		if strings.Contains(upper, b) {
			return false
		}
	}

	// Check for statement injection
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
		for i, col := range cols {
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./cmd/cbrain/ -run TestValidateSQL -v`

Expected: PASS

- [ ] **Step 6: Build and test manually**

Run: `go build -o cbrain ./cmd/cbrain/`

Run: `./cbrain sql "SELECT count(*) FROM observations"`

Expected: Shows count in table format

Run: `./cbrain sql "DROP TABLE observations"`

Expected: Error "query blocked: destructive operations are not allowed"

- [ ] **Step 6: Commit**

```bash
git add cmd/cbrain/sql.go cmd/cbrain/sql_test.go
git commit -m "feat: cbrain sql subcommand with destructive operation blocking"
```

---

### Task 4: Inference Analysis Subcommand

**Files:**
- Create: `cmd/cbrain/infer.go`

**Coverage map for README inference scenarios:**

| README Scenario | Command | Implementation |
|-----------------|---------|----------------|
| Family Routines | `cbrain infer routines` | Task 4 |
| Visitor Patterns | `cbrain infer visitors` | Task 4B |
| Vehicle Usage | `cbrain infer vehicles` | Task 4 |
| Security Alerts | `cbrain infer security` | Task 4B |
| Package Delivery | `cbrain infer deliveries` | Task 4B |
| Pet Monitoring | `cbrain infer animals` | Task 4B |
| Service Provider | `cbrain infer workers` | Task 4B |
| After 8 PM query | `cbrain sql "..."` | Task 3 |

- [ ] **Step 1: Create infer.go with pattern analysis commands**

```go
// cmd/cbrain/infer.go
package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"text/tabwriter"
	"time"

	_ "github.com/lib/pq"
	"github.com/spf13/cobra"
)

var inferCmd = &cobra.Command{
	Use:   "infer",
	Short: "Analyze patterns and routines",
	Long:  "Extract behavioral patterns, routines, and anomalies from observation data.",
}

var inferRoutinesCmd = &cobra.Command{
	Use:   "routines",
	Short: "Detect daily routines",
	Long:  "Analyze timestamp patterns to detect regular daily routines (e.g., school times, work departures).",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    class_name,
    EXTRACT(HOUR FROM detected_at) as hour,
    EXTRACT(DOW FROM detected_at) as day_of_week,
    count(*) as occurrences,
    camera_id
FROM observations
WHERE detected_at > NOW() - INTERVAL '14 days'
GROUP BY 1, 2, 3, 4
HAVING count(*) > 3
ORDER BY occurrences DESC
LIMIT 20`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatRoutines(os.Stdout, outputFormat, rows)
	},
}

var inferAnomaliesCmd = &cobra.Command{
	Use:   "anomalies",
	Short: "Detect unusual activity",
	Long:  "Find activity outside normal patterns: unusual times, unknown entities, rare events.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    detected_at,
    camera_id,
    class_name,
    attributes->>'confidence' as confidence
FROM observations
WHERE (
    -- Late night activity (midnight to 5 AM)
    (EXTRACT(HOUR FROM detected_at) BETWEEN 0 AND 5)
    OR
    -- Low confidence detections
    (attributes->>'confidence')::float < 0.5
)
AND detected_at > NOW() - INTERVAL '7 days'
ORDER BY detected_at DESC
LIMIT 50`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatAnomalies(os.Stdout, outputFormat, rows)
	},
}

var inferVehiclesCmd = &cobra.Command{
	Use:   "vehicles",
	Short: "Vehicle usage patterns",
	Long:  "Analyze vehicle appearances by day of week, time of day.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    CASE EXTRACT(DOW FROM detected_at)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day,
    CASE
        WHEN EXTRACT(HOUR FROM detected_at) BETWEEN 6 AND 11 THEN 'Morning'
        WHEN EXTRACT(HOUR FROM detected_at) BETWEEN 12 AND 17 THEN 'Afternoon'
        WHEN EXTRACT(HOUR FROM detected_at) BETWEEN 18 AND 21 THEN 'Evening'
        ELSE 'Night'
    END as time_of_day,
    count(*) as sightings
FROM observations
WHERE class_name IN ('car', 'truck', 'suv', 'van')
  AND detected_at > NOW() - INTERVAL '30 days'
GROUP BY 1, 2
ORDER BY day, time_of_day`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatVehiclePatterns(os.Stdout, outputFormat, rows)
	},
}

func formatRoutines(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Detected Routines ===\n")
	fmt.Fprintln(w, "Class\tHour\tDay\tCount\tCamera")
	fmt.Fprintln(w, "------\t----\t---\t-----\t------")

	for rows.Next() {
		var className string
		var hour, dow float64
		var count int64
		var cameraID sql.NullString
		if err := rows.Scan(&className, &hour, &dow, &count, &cameraID); err != nil {
			return err
		}

		dayNames := []string{"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}
		day := dayNames[int(dow)]

		camera := "unknown"
		if cameraID.Valid {
			camera = cameraID.String
		}

		fmt.Fprintf(w, "%s\t%02d:00\t%s\t%d\t%s\n", className, int(hour), day, count, camera)
	}

	return w.Flush()
}

func formatAnomalies(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Anomalous Activity ===\n")
	fmt.Fprintln(w, "Time\tCamera\tClass\tConfidence")
	fmt.Fprintln(w, "----\t------\t-----\t----------")

	for rows.Next() {
		var detectedAt time.Time
		var cameraID, className, confidence sql.NullString
		if err := rows.Scan(&detectedAt, &cameraID, &className, &confidence); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		class := ""
		if className.Valid {
			class = className.String
		}
		conf := ""
		if confidence.Valid {
			conf = confidence.String
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", detectedAt.Format("2006-01-02 15:04"), cam, class, conf)
	}

	return w.Flush()
}

func formatVehiclePatterns(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Vehicle Patterns ===\n")
	fmt.Fprintln(w, "Day\tTime\tSightings")
	fmt.Fprintln(w, "---\t----\t---------")

	for rows.Next() {
		var day, timeOfDay sql.NullString
		var sightings int64
		if err := rows.Scan(&day, &timeOfDay, &sightings); err != nil {
			return err
		}

		d := ""
		if day.Valid {
			d = day.String
		}
		t := ""
		if timeOfDay.Valid {
			t = timeOfDay.String
		}

		fmt.Fprintf(w, "%s\t%s\t%d\n", d, t, sightings)
	}

	return w.Flush()
}

func formatJSON(w io.Writer, rows *sql.Rows) error {
	cols, _ := rows.Columns()
	var results []map[string]any
	for rows.Next() {
		row := make(map[string]any)
		vals := make([]any, len(cols))
		valPtrs := make([]any, len(cols))
		for i := range vals {
			valPtrs[i] = &vals[i]
		}
		rows.Scan(valPtrs...)
		for i, col := range cols {
			row[col] = vals[i]
		}
		results = append(results, row)
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}

func init() {
	rootCmd.AddCommand(inferCmd)
	inferCmd.AddCommand(inferRoutinesCmd)
	inferCmd.AddCommand(inferAnomaliesCmd)
	inferCmd.AddCommand(inferVehiclesCmd)
}
```

- [ ] **Step 2: Build and test**

Run: `go build -o cbrain ./cmd/cbrain/`

Run: `./cbrain infer routines`

Expected: Shows detected routines in table format

Run: `./cbrain infer anomalies`

Expected: Shows anomalous activity

Run: `./cbrain infer vehicles`

Expected: Shows vehicle patterns by day/time

- [ ] **Step 3: Commit**

```bash
git add cmd/cbrain/infer.go
git commit -m "feat: cbrain infer subcommand (routines, anomalies, vehicles)"
```

---

### Task 4B: Additional Inference Commands (Visitor, Security, Deliveries, Animals, Workers)

**Files:**
- Modify: `cmd/cbrain/infer.go`

- [ ] **Step 1: Add inferVisitorsCmd for visitor pattern analysis**

```go
var inferVisitorsCmd = &cobra.Command{
	Use:   "visitors",
	Short: "Analyze visitor patterns",
	Long:  "Track recurring visitors like mail carriers, delivery drivers, and frequent guests.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		// Find people who appear at regular intervals (same time, same day)
		query := `
SELECT
    class_name,
    EXTRACT(DOW FROM detected_at) as day_of_week,
    EXTRACT(HOUR FROM detected_at) as hour,
    count(*) as visits,
    camera_id,
    min(detected_at) as first_seen,
    max(detected_at) as last_seen
FROM observations
WHERE class_name = 'person'
  AND detected_at > NOW() - INTERVAL '30 days'
  AND camera_id IN ('front_door', 'mailbox', 'driveway')
GROUP BY 1, 2, 3, 4
HAVING count(*) > 5
ORDER BY visits DESC
LIMIT 20`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatVisitors(os.Stdout, outputFormat, rows)
	},
}

func formatVisitors(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Regular Visitors ===\n")
	fmt.Fprintln(w, "Class\tDay\tHour\tVisits\tCamera\tFirst Seen\tLast Seen")
	fmt.Fprintln(w, "-----\t---\t----\t------\t------\t---------\t---------")

	dayNames := []string{"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}

	for rows.Next() {
		var className string
		var dow, hour float64
		var visits int64
		var cameraID sql.NullString
		var firstSeen, lastSeen time.Time

		if err := rows.Scan(&className, &dow, &hour, &visits, &cameraID, &firstSeen, &lastSeen); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}

		day := dayNames[int(dow)]

		fmt.Fprintf(w, "%s\t%s\t%02d:00\t%d\t%s\t%s\t%s\n",
			className, day, int(hour), visits, cam,
			firstSeen.Format("2006-01-02"), lastSeen.Format("2006-01-02"))
	}

	return w.Flush()
}
```

- [ ] **Step 2: Add inferSecurityCmd for security alert analysis**

```go
var inferSecurityCmd = &cobra.Command{
	Use:   "security",
	Short: "Security-focused anomaly detection",
	Long:  "Find potentially suspicious activity: unknown vehicles, late-night motion, unfamiliar people.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		// Find activity during unusual hours (midnight-5am) or low-confidence detections
		query := `
SELECT
    detected_at,
    camera_id,
    class_name,
    attributes->>'confidence' as confidence,
    attributes->>'license_plate' as plate,
    attributes->>'color' as color
FROM observations
WHERE (
    EXTRACT(HOUR FROM detected_at) BETWEEN 0 AND 5
    OR (attributes->>'confidence')::float < 0.6
)
AND detected_at > NOW() - INTERVAL '14 days'
ORDER BY detected_at DESC
LIMIT 100`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatSecurity(os.Stdout, outputFormat, rows)
	},
}

func formatSecurity(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Security Alerts ===\n")
	fmt.Fprintln(w, "Time\tCamera\tClass\tDetails\tConfidence")
	fmt.Fprintln(w, "----\t------\t-----\t-------\t----------")

	for rows.Next() {
		var detectedAt time.Time
		var cameraID, className, confidence, plate, color sql.NullString

		if err := rows.Scan(&detectedAt, &cameraID, &className, &confidence, &plate, &color); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		class := "unknown"
		if className.Valid {
			class = className.String
		}
		conf := ""
		if confidence.Valid {
			conf = confidence.String
		}

		details := ""
		if plate.Valid && plate.String != "" {
			details = "plate: " + plate.String
		}
		if color.Valid && color.String != "" {
			if details != "" {
				details += ", "
			}
			details += color.String
		}
		if details == "" {
			details = "-"
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
			detectedAt.Format("2006-01-02 15:04"),
			cam, class, details, conf)
	}

	return w.Flush()
}
```

- [ ] **Step 3: Add inferDeliveriesCmd for package delivery tracking**

```go
var inferDeliveriesCmd = &cobra.Command{
	Use:   "deliveries",
	Short: "Package delivery tracking",
	Long:  "Track deliveries by carrier (FedEx, UPS, USPS) with arrival times and drop locations.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    date_trunc('day', detected_at) as date,
    EXTRACT(DOW FROM detected_at) as day_of_week,
    max(EXTRACT(HOUR FROM detected_at)) as typical_hour,
    count(*) as deliveries,
    camera_id,
    attributes->>'color' as uniform_color
FROM observations
WHERE class_name ILIKE '%person%'
  AND attributes->>'uniform_color' IS NOT NULL
  AND detected_at > NOW() - INTERVAL '30 days'
GROUP BY 1, 2, 4, 5
ORDER BY date DESC
LIMIT 30`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatDeliveries(os.Stdout, outputFormat, rows)
	},
}

func formatDeliveries(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Package Deliveries ===\n")
	fmt.Fprintln(w, "Date\tDay\tTime\tCount\tCamera\tUniform")
	fmt.Fprintln(w, "----\t---\t----\t-----\t------\t-------")

	dayNames := []string{"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}

	for rows.Next() {
		var date time.Time
		var dow, hour float64
		var count int64
		var cameraID, uniform sql.NullString

		if err := rows.Scan(&date, &dow, &hour, &count, &cameraID, &uniform); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		uni := ""
		if uniform.Valid {
			uni = uniform.String
		}

		fmt.Fprintf(w, "%s\t%s\t%02d:00\t%d\t%s\t%s\n",
			date.Format("2006-01-02"),
			dayNames[int(dow)],
			int(hour), count, cam, uni)
	}

	return w.Flush()
}
```

- [ ] **Step 4: Add inferAnimalsCmd for pet monitoring**

```go
var inferAnimalsCmd = &cobra.Command{
	Use:   "animals",
	Short: "Animal/pet activity tracking",
	Long:  "Monitor pet movements, detect animals in restricted areas, track yard boundary crossings.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    detected_at,
    camera_id,
    class_name,
    attributes->>'confidence' as confidence,
    attributes->>'color' as color
FROM observations
WHERE class_name IN ('dog', 'cat', 'animal', 'bird')
  AND detected_at > NOW() - INTERVAL '7 days'
ORDER BY detected_at DESC
LIMIT 50`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatAnimals(os.Stdout, outputFormat, rows)
	},
}

func formatAnimals(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Animal Sightings ===\n")
	fmt.Fprintln(w, "Time\tCamera\tAnimal\tColor\tConfidence")
	fmt.Fprintln(w, "----\t------\t------\t-----\t----------")

	for rows.Next() {
		var detectedAt time.Time
		var cameraID, className, color, confidence sql.NullString

		if err := rows.Scan(&detectedAt, &cameraID, &className, &color, &confidence); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		class := ""
		if className.Valid {
			class = className.String
		}
		c := ""
		if color.Valid {
			c = color.String
		}
		conf := ""
		if confidence.Valid {
			conf = confidence.String
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
			detectedAt.Format("2006-01-02 15:04"),
			cam, class, c, conf)
	}

	return w.Flush()
}
```

- [ ] **Step 5: Add inferWorkersCmd for service provider verification**

```go
var inferWorkersCmd = &cobra.Command{
	Use:   "workers",
	Short: "Service provider tracking",
	Long:  "Verify service provider arrivals, count workers, track duration on site.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		// Group by day and camera to estimate worker count and duration
		query := `
SELECT
    date_trunc('day', detected_at) as date,
    camera_id,
    count(DISTINCT attributes->>'vehicle_color') as vehicles,
    count(*) as worker_sightings,
    min(detected_at) as arrival,
    max(detected_at) as departure,
    max(detected_at) - min(detected_at) as duration
FROM observations
WHERE class_name = 'person'
  AND detected_at > NOW() - INTERVAL '30 days'
  AND EXTRACT(HOUR FROM detected_at) BETWEEN 7 AND 18
GROUP BY 1, 2
ORDER BY date DESC
LIMIT 20`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatWorkers(os.Stdout, outputFormat, rows)
	},
}

func formatWorkers(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Service Providers ===\n")
	fmt.Fprintln(w, "Date\tCamera\tVehicles\tSightings\tArrival\tDeparture\tDuration")
	fmt.Fprintln(w, "----\t------\t--------\t---------\t-------\t---------\t--------")

	for rows.Next() {
		var date time.Time
		var cameraID sql.NullString
		var vehicles, sightings int64
		var arrival, departure time.Time
		var duration sql.NullString

		if err := rows.Scan(&date, &cameraID, &vehicles, &sightings, &arrival, &departure, &duration); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		dur := ""
		if duration.Valid {
			dur = duration.String
		}

		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%s\t%s\t%s\n",
			date.Format("2006-01-02"),
			cam, vehicles, sightings,
			arrival.Format("15:04"),
			departure.Format("15:04"),
			dur)
	}

	return w.Flush()
}
```

- [ ] **Step 6: Register new commands in init()**

Update the `init()` function in `infer.go`:

```go
func init() {
	rootCmd.AddCommand(inferCmd)
	inferCmd.AddCommand(inferRoutinesCmd)
	inferCmd.AddCommand(inferAnomaliesCmd)
	inferCmd.AddCommand(inferVehiclesCmd)
	// New commands
	inferCmd.AddCommand(inferVisitorsCmd)
	inferCmd.AddCommand(inferSecurityCmd)
	inferCmd.AddCommand(inferDeliveriesCmd)
	inferCmd.AddCommand(inferAnimalsCmd)
	inferCmd.AddCommand(inferWorkersCmd)
}
```

- [ ] **Step 7: Build and test all infer commands**

Run: `go build -o cbrain ./cmd/cbrain/`

Run: `./cbrain infer visitors`

Expected: Shows regular visitor patterns

Run: `./cbrain infer security`

Expected: Shows late-night/unusual activity

Run: `./cbrain infer deliveries`

Expected: Shows delivery tracking by date

Run: `./cbrain infer animals`

Expected: Shows animal sightings

Run: `./cbrain infer workers`

Expected: Shows service provider visits with duration

- [ ] **Step 8: Commit**

```bash
git add cmd/cbrain/infer.go
git commit -m "feat: cbrain infer subcommand (visitors, security, deliveries, animals, workers)"
```

**Files:**
- Create: `cmd/cbrain/correlate.go`

- [ ] **Step 1: Create correlate.go with multi-camera tracking**

```go
// cmd/cbrain/correlate.go
package main

import (
	"database/sql"
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	_ "github.com/lib/pq"
	"github.com/spf13/cobra"
)

var correlateCmd = &cobra.Command{
	Use:   "correlate",
	Short: "Cross-camera correlation analysis",
	Long:  "Track movement across multiple cameras, detect entry/exit patterns, and correlate events.",
}

var correlateTrackCmd = &cobra.Command{
	Use:   "track [entity_type]",
	Short: "Track entity movement across cameras",
	Long:  "Show movement trail of a person or vehicle across all cameras within a time window.",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		entityType := args[0]
		windowMinutes, _ := cmd.Flags().GetInt("window")

		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    detected_at,
    camera_id,
    attributes->>'confidence' as confidence,
    attributes
FROM observations
WHERE class_name ILIKE $1
  AND detected_at > NOW() - INTERVAL '24 hours'
ORDER BY detected_at DESC
LIMIT 100`

		rows, err := db.Query(query, "%"+entityType+"%")
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatTracking(os.Stdout, outputFormat, rows, windowMinutes)
	},
}

var correlateTimelineCmd = &cobra.Command{
	Use:   "timeline",
	Short: "Generate timeline of events",
	Long:  "Create a chronological timeline of all detections across cameras.",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			cfg.DBHost, cfg.DBPort, cfg.DBUser, cfg.DBPassword, cfg.DBName)

		db, err := sql.Open("postgres", dsn)
		if err != nil {
			return err
		}
		defer db.Close()

		query := `
SELECT
    detected_at,
    camera_id,
    class_name,
    attributes->>'license_plate' as plate,
    attributes->>'color' as color
FROM observations
WHERE detected_at > NOW() - INTERVAL '1 hour'
ORDER BY detected_at DESC
LIMIT 50`

		rows, err := db.Query(query)
		if err != nil {
			return err
		}
		defer rows.Close()

		outputFormat, _ := cmd.Flags().GetString("output")
		return formatTimeline(os.Stdout, outputFormat, rows)
	},
}

func formatTracking(w *tabwriter.Writer, format string, rows *sql.Rows, windowMinutes int) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Movement Trail ===\n")
	fmt.Fprintln(w, "Time\tCamera\tConfidence\tDetails")
	fmt.Fprintln(w, "----\t------\t----------\t-------")

	var lastCamera string
	var lastTime time.Time

	for rows.Next() {
		var detectedAt time.Time
		var cameraID, confidence sql.NullString
		var attributes []byte

		if err := rows.Scan(&detectedAt, &cameraID, &confidence, &attributes); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		conf := ""
		if confidence.Valid {
			conf = confidence.String
		}

		// Calculate time delta from previous
		delta := ""
		if !lastTime.IsZero() {
			d := detectedAt.Sub(lastTime)
			if d < 0 {
				d = -d
			}
			if d.Minutes() < float64(windowMinutes) {
				delta = fmt.Sprintf("(+%dm)", int(d.Minutes()))
			}
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s %s\n",
			detectedAt.Format("15:04:05"),
			cam,
			conf,
			cameraTransition(lastCamera, cam),
			delta)

		lastCamera = cam
		lastTime = detectedAt
	}

	return w.Flush()
}

func formatTimeline(w *tabwriter.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "\n=== Event Timeline ===\n")
	fmt.Fprintln(w, "Time\tCamera\tEvent\tDetails")
	fmt.Fprintln(w, "----\t------\t-----\t-------")

	for rows.Next() {
		var detectedAt time.Time
		var cameraID, className, plate, color sql.NullString

		if err := rows.Scan(&detectedAt, &cameraID, &className, &plate, &color); err != nil {
			return err
		}

		cam := ""
		if cameraID.Valid {
			cam = cameraID.String
		}
		class := ""
		if className.Valid {
			class = className.String
		}

		details := ""
		if color.Valid && color.String != "" {
			details = color.String
		}
		if plate.Valid && plate.String != "" {
			if details != "" {
				details += ", "
			}
			details += "plate: " + plate.String
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n",
			detectedAt.Format("2006-01-02 15:04"),
			cam,
			class,
			details)
	}

	return w.Flush()
}

func cameraTransition(from, to string) string {
	if from == "" {
		return to
	}
	if from == to {
		return fmt.Sprintf("[%s]", to)
	}
	return fmt.Sprintf("%s→%s", from, to)
}

func init() {
	rootCmd.AddCommand(correlateCmd)
	correlateCmd.AddCommand(correlateTrackCmd)
	correlateCmd.AddCommand(correlateTimelineCmd)

	correlateTrackCmd.Flags().IntP("window", "w", 10, "time window in minutes for correlation")
}
```

- [ ] **Step 2: Build and test**

Run: `go build -o cbrain ./cmd/cbrain/`

Run: `./cbrain correlate timeline`

Expected: Shows chronological event timeline

Run: `./cbrain correlate track person`

Expected: Shows person movement trail across cameras

- [ ] **Step 3: Commit**

```bash
git add cmd/cbrain/correlate.go
git commit -m "feat: cbrain correlate subcommand (track, timeline)"
```

---

### Task 5: Cross-Camera Correlation Subcommand

**Files**:
- Create: `cmd/cbrain/correlate.go`

---

### Task 6: Update Makefile and Install Script

**Files:**
- Modify: `Makefile`
- Modify: `deploy/install.sh`

- [ ] **Step 1: Add cbrain target to Makefile**

Modify `Makefile` to add:

```makefile
# Add to SERVICES or create separate
CBRAIN_BIN = cbrain

.PHONY: build-cbrain
build-cbrain:
	go build -o $(BIN_DIR)/$(CBRAIN_BIN) ./cmd/cbrain/

build: build-cbrain

install: build-cbrain
	# ... existing install code ...
	cp $(BIN_DIR)/$(CBRAIN_BIN) /usr/local/bin/
```

- [ ] **Step 2: Add cbrain installation to install.sh**

Modify `deploy/install.sh` to add after Go services build:

```bash
# Line ~247, after building Go services
log_info "Building cbrain CLI tool..."
run_or_echo go build -o "$BIN_DIR/cbrain" ./cmd/cbrain/
run_or_echo cp "$BIN_DIR/cbrain" /usr/local/bin/
```

- [ ] **Step 3: Build and verify**

Run: `make build-cbrain`

Expected: Creates `./cbrain` binary

Run: `./cbrain --help`

Expected: Shows all subcommands (query, sql, infer, correlate)

- [ ] **Step 4: Commit**

```bash
git add Makefile deploy/install.sh
git commit -m "feat: add cbrain build target and installation to Makefile and install.sh"
```

---

### Task 7: Update README with cbrain Examples

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Example section with cbrain commands**

Modify `README.md` lines 9-11, add:

```markdown
## Example

### CLI Queries with cbrain

```bash
# Natural language queries
cbrain query "Who was at the front door this morning?"
cbrain query "Show me all vehicles in the driveway last week" -o json

# Direct SQL (read-only)
cbrain sql "SELECT count(*) FROM observations"
cbrain sql "SELECT class_name, count(*) FROM observations GROUP BY class_name"

# Pattern analysis
cbrain infer routines      # Detect daily routines
cbrain infer anomalies     # Find unusual activity
cbrain infer vehicles      # Vehicle usage patterns

# Cross-camera correlation
cbrain correlate timeline  # Chronological event timeline
cbrain correlate track person  # Track person movement
```

### HTTP API (alternative)

...existing curl examples...
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with cbrain CLI examples"
```

---

### Task 8: Final Verification and Testing

- [ ] **Step 1: Full build verification**

Run: `go build ./cmd/cbrain/`

Expected: No errors

Run: `go vet ./cmd/cbrain/`

Expected: No warnings

Run: `go test ./cmd/cbrain/`

Expected: All tests pass

- [ ] **Step 2: Deploy to rock0 and test**

Run: `scp ./cbrain caimlas@rock0:/tmp/cbrain`

Run: `ssh caimlas@rock0 "/tmp/cbrain --help"`

Expected: Shows help

Run: `ssh caimlas@rock0 "/tmp/cbrain query 'person'"`

Expected: Returns query results

- [ ] **Step 3: Commit all changes**

```bash
git add -A
git commit -m "chore: cbrain CLI tool complete - ready for release"
```

---

## Self-Review Checklist

- [ ] No placeholders (TBD, TODO, "add validation")
- [ ] All code snippets complete with actual content
- [ ] Type consistency (QueryResponse, ParsedQuery used consistently)
- [ ] All commands tested with expected output
- [ ] Config loading handles missing env file gracefully
- [ ] SQL injection prevention in sql.go is comprehensive
- [ ] Output formatting works for json, table, plain modes
