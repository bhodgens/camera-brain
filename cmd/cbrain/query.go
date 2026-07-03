// cmd/cbrain/query.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

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

		queryText := strings.Join(args, " ")

		cfg, err := LoadConfig(cmd.Flag("config").Value.String())
		if err != nil {
			return err
		}

		outputFormat, _ := cmd.Flags().GetString("output")

		resp, err := postQuery(cfg.QueryURL+"/query", []byte(queryText))
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
