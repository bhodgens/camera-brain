// Package llamacpp_text provides a text-only Llama.cpp analysis plugin.
package llamacpp_text

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"rock-cluster/pkg/plugin/analysis"
)

// pluginInfo is the analysis.PluginInfo for this plugin.
var pluginInfo = analysis.PluginInfo{
	Name:     "llamacpp-text",
	Version:  "1.0.0",
	Modality: "text",
	IsLocal:  true,
}

// plugin implements the analysis.Analyzer interface for text-only LLM calls.
type plugin struct {
	endpoint    string
	maxTokens   int
	temperature float32
	timeout     time.Duration
	client      *http.Client
}

// New creates a new text-only plugin instance.
func New() analysis.Analyzer {
	return &plugin{}
}

// Initialize configures the plugin.
func (p *plugin) Initialize(ctx context.Context, cfg analysis.Config) error {
	p.endpoint = cfg.Endpoint
	p.maxTokens = cfg.MaxTokens
	p.temperature = cfg.Temperature

	timeout := time.Duration(cfg.TimeoutSec) * time.Second
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	p.timeout = timeout

	p.client = &http.Client{Timeout: p.timeout}

	return nil
}

// Analyze sends a text-only prompt to llama-server and returns structured results.
// The imgData parameter is ignored since this is a text-only plugin.
func (p *plugin) Analyze(ctx context.Context, imgData []byte, prompt string) (*analysis.AnalysisResult, error) {
	reqBody := map[string]interface{}{
		"messages": []map[string]interface{}{{
			"role":    "user",
			"content": prompt,
		}},
		"max_tokens":  p.maxTokens,
		"temperature": p.temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint+"/v1/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llama-server request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("llama-server returned HTTP %d", resp.StatusCode)
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty response from llama-server")
	}

	content := result.Choices[0].Message.Content
	var attrs map[string]interface{}
	var rawResponse string
	var confidence float32

	if err := json.Unmarshal([]byte(content), &attrs); err != nil {
		rawResponse = content
		attrs = map[string]interface{}{"raw_description": content}
	} else {
		if conf, ok := attrs["confidence"].(float64); ok {
			confidence = float32(conf)
		}
		rawResponse = content
	}

	return &analysis.AnalysisResult{Attributes: attrs, RawResponse: rawResponse, Confidence: confidence}, nil
}

// Close releases resources.
func (p *plugin) Close() error {
	if p.client != nil {
		p.client.CloseIdleConnections()
	}
	return nil
}

// Info returns plugin metadata.
func (p *plugin) Info() analysis.PluginInfo {
	return pluginInfo
}
