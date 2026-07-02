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

func formatRoutines(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Detected Routines ===\n")
	fmt.Fprintln(tw, "Class\tHour\tDay\tCount\tCamera")
	fmt.Fprintln(tw, "------\t----\t---\t-----\t------")

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

		fmt.Fprintf(tw, "%s\t%02d:00\t%s\t%d\t%s\n", className, int(hour), day, count, camera)
	}

	return tw.Flush()
}

func formatAnomalies(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Anomalous Activity ===\n")
	fmt.Fprintln(tw, "Time\tCamera\tClass\tConfidence")
	fmt.Fprintln(tw, "----\t------\t-----\t----------")

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

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", detectedAt.Format("2006-01-02 15:04"), cam, class, conf)
	}

	return tw.Flush()
}

func formatVehiclePatterns(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Vehicle Patterns ===\n")
	fmt.Fprintln(tw, "Day\tTime\tSightings")
	fmt.Fprintln(tw, "---\t----\t---------")

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

		fmt.Fprintf(tw, "%s\t%s\t%d\n", d, t, sightings)
	}

	return tw.Flush()
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
