//go:build !linux || !arm64

// Package plugin initializes built-in plugins (non-RKNN platforms).
package plugin

import (
	"rock-cluster/pkg/plugin/analysis"
	llamacpp "rock-cluster/pkg/plugin/analysis/llamacpp"
)

// init registers all built-in plugins for non-RKNN platforms.
func init() {
	// Register analysis plugins
	RegisterAnalysis("llamacpp", func() analysis.Analyzer {
		return llamacpp.New()
	})
}
