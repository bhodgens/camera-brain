// Package analysis defines the plugin interface for image analysis backends.
package analysis

import (
	"context"
)

// AnalysisResult contains structured analysis output.
type AnalysisResult struct {
	// Attributes is a map of extracted attributes from the analysis.
	// Keys and values depend on the prompt and model capabilities.
	Attributes map[string]interface{}
	// RawResponse is the unprocessed model response.
	RawResponse string
	// Confidence is the overall confidence (0.0 - 1.0)
	Confidence float32
	// TokensUsed is the number of tokens consumed (for API/cost tracking)
	TokensUsed int
}

// Config contains analysis plugin configuration.
type Config struct {
	// Endpoint is the URL for API-based plugins (e.g., "http://localhost:8888")
	Endpoint string
	// ModelPath is the path to the model file (for local plugins)
	ModelPath string
	// MMProjPath is the path to the multimodal projector (for VLMs)
	MMProjPath string
	// APIKey is the API key for cloud services
	APIKey string
	// MaxTokens limits the response length
	MaxTokens int
	// Temperature controls randomness (0.0 - 1.0)
	Temperature float32
	// Timeout is the request timeout
	TimeoutSec int
	// ModelName identifies the model (for API plugins)
	ModelName string
}

// PluginInfo contains plugin metadata.
type PluginInfo struct {
	// Name is the plugin identifier
	Name string
	// Version is the plugin version
	Version string
	// Modality is the input type ("vision", "text", "multimodal")
	Modality string
	// IsLocal indicates if inference runs locally
	IsLocal bool
}

// Analyzer defines the interface for image analysis.
type Analyzer interface {
	// Initialize loads models and prepares resources.
	Initialize(ctx context.Context, cfg Config) error

	// Analyze processes an image with the given prompt and returns structured results.
	// The image should be JPEG-encoded bytes.
	Analyze(ctx context.Context, image []byte, prompt string) (*AnalysisResult, error)

	// Close releases all resources.
	Close() error

	// Info returns plugin metadata.
	Info() PluginInfo
}
