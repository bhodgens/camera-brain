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
	fmt.Fprintln(tw, "=== Detected Routines ===")
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
	fmt.Fprintln(tw, "=== Anomalous Activity ===")
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
	fmt.Fprintln(tw, "=== Vehicle Patterns ===")
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

func formatVisitors(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Regular Visitors ===")
	fmt.Fprintln(tw, "Class\tDay\tHour\tVisits\tCamera\tFirst Seen\tLast Seen")
	fmt.Fprintln(tw, "-----\t---\t----\t------\t------\t---------\t---------")

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

		fmt.Fprintf(tw, "%s\t%s\t%02d:00\t%d\t%s\t%s\t%s\n",
			className, day, int(hour), visits, cam,
			firstSeen.Format("2006-01-02"), lastSeen.Format("2006-01-02"))
	}

	return tw.Flush()
}

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

func formatSecurity(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Security Alerts ===")
	fmt.Fprintln(tw, "Time\tCamera\tClass\tDetails\tConfidence")
	fmt.Fprintln(tw, "----\t------\t-----\t-------\t----------")

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

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n",
			detectedAt.Format("2006-01-02 15:04"),
			cam, class, details, conf)
	}

	return tw.Flush()
}

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

func formatDeliveries(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Package Deliveries ===")
	fmt.Fprintln(tw, "Date\tDay\tTime\tCount\tCamera\tUniform")
	fmt.Fprintln(tw, "----\t---\t----\t-----\t------\t-------")

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

		fmt.Fprintf(tw, "%s\t%s\t%02d:00\t%d\t%s\t%s\n",
			date.Format("2006-01-02"),
			dayNames[int(dow)],
			int(hour), count, cam, uni)
	}

	return tw.Flush()
}

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

func formatAnimals(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Animal Sightings ===")
	fmt.Fprintln(tw, "Time\tCamera\tAnimal\tColor\tConfidence")
	fmt.Fprintln(tw, "----\t------\t------\t-----\t----------")

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

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n",
			detectedAt.Format("2006-01-02 15:04"),
			cam, class, c, conf)
	}

	return tw.Flush()
}

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

func formatWorkers(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "\n=== Service Providers ===")
	fmt.Fprintln(tw, "Date\tCamera\tVehicles\tSightings\tArrival\tDeparture\tDuration")
	fmt.Fprintln(tw, "----\t------\t--------\t---------\t-------\t---------\t--------")

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

		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%s\t%s\t%s\n",
			date.Format("2006-01-02"),
			cam, vehicles, sightings,
			arrival.Format("15:04"),
			departure.Format("15:04"),
			dur)
	}

	return tw.Flush()
}

func init() {
	rootCmd.AddCommand(inferCmd)
	inferCmd.AddCommand(inferRoutinesCmd)
	inferCmd.AddCommand(inferAnomaliesCmd)
	inferCmd.AddCommand(inferVehiclesCmd)
	// Task 4B: Additional inference commands
	inferCmd.AddCommand(inferVisitorsCmd)
	inferCmd.AddCommand(inferSecurityCmd)
	inferCmd.AddCommand(inferDeliveriesCmd)
	inferCmd.AddCommand(inferAnimalsCmd)
	inferCmd.AddCommand(inferWorkersCmd)
}
