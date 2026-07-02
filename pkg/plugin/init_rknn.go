//go:build linux && arm64

// Package plugin initializes built-in plugins (RKNN platform).
package plugin

import (
	"rock-cluster/pkg/plugin/analysis"
	"rock-cluster/pkg/plugin/detection"
	llamacpp "rock-cluster/pkg/plugin/analysis/llamacpp"
	llamacpp_text "rock-cluster/pkg/plugin/analysis/llamacpp_text"
	rknn "rock-cluster/pkg/plugin/detection/rknn"
)

// init registers all built-in plugins for RKNN platforms.
func init() {
	// Register detection plugins
	RegisterDetection("rknn", func() detection.Detector {
		return rknn.New()
	})

	// Register analysis plugins
	RegisterAnalysis("llamacpp", func() analysis.Analyzer {
		return llamacpp.New()
	})
	RegisterAnalysis("llamacpp-text", func() analysis.Analyzer {
		return llamacpp_text.New()
	})
}
