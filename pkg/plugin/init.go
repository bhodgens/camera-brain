//go:build !linux || !arm64

// Package plugin initializes built-in plugins (non-RKNN platforms).
package plugin

import (
	"rock-cluster/pkg/plugin/analysis"
	llamacpp "rock-cluster/pkg/plugin/analysis/llamacpp"
	llamacpp_text "rock-cluster/pkg/plugin/analysis/llamacpp_text"
)

// init registers all built-in plugins for non-RKNN platforms.
func init() {
	// Register analysis plugins
	RegisterAnalysis("llamacpp", func() analysis.Analyzer {
		return llamacpp.New()
	})
	RegisterAnalysis("llamacpp-text", func() analysis.Analyzer {
		return llamacpp_text.New()
	})
}
