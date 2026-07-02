// cmd/cbrain/correlate.go
package main

import (
	"database/sql"
	"fmt"
	"io"
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
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		return formatTracking(tw, outputFormat, rows, windowMinutes)
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
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		return formatTimeline(tw, outputFormat, rows)
	},
}

func formatTracking(w io.Writer, format string, rows *sql.Rows, windowMinutes int) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "=== Movement Trail ===")
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

	return w.(interface{ Flush() error }).Flush()
}

func formatTimeline(w io.Writer, format string, rows *sql.Rows) error {
	if format == "json" {
		return formatJSON(w, rows)
	}

	fmt.Fprintln(w, "=== Event Timeline ===")
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

	return w.(interface{ Flush() error }).Flush()
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
