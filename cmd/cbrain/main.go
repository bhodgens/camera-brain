// cmd/cbrain/main.go
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// rootCmd is the root command for cbrain. Made package-level so
// subcommand init() functions can register their commands.
var rootCmd = &cobra.Command{
	Use:   "cbrain",
	Short: "Camera Brain CLI query tool",
	Long:  "Natural language querying and analysis for Camera Brain video surveillance system.",
}

func init() {
	rootCmd.PersistentFlags().String("config", "/etc/camera-brain/camera-brain.env", "config file path")
	rootCmd.PersistentFlags().StringP("output", "o", "table", "output format: json, table, plain")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
