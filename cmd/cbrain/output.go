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
