// Package llamacpp provides a llama.cpp-based VLM analysis plugin.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"time"

	"rock-cluster/pkg/plugin/analysis"
)

// pluginInfo is the analysis.PluginInfo for this plugin.
var pluginInfo = analysis.PluginInfo{
	Name:     "llamacpp-vision",
	Version:  "1.0.0",
	Modality: "multimodal",
	IsLocal:  true,
}

// vlmPlugin implements the analysis.Analyzer interface.
type vlmPlugin struct {
	endpoint    string
	modelPath   string
	mmprojPath  string
	maxTokens   int
	temperature float32
	timeout     time.Duration
	client      *http.Client
}

// New creates a new VLM plugin instance.
func New() analysis.Analyzer {
	return &vlmPlugin{}
}

// Initialize configures the plugin.
func (p *vlmPlugin) Initialize(ctx context.Context, cfg analysis.Config) error {
	p.endpoint = cfg.Endpoint
	p.modelPath = cfg.ModelPath
	p.mmprojPath = cfg.MMProjPath
	p.maxTokens = cfg.MaxTokens
	p.temperature = cfg.Temperature

	timeout := time.Duration(cfg.TimeoutSec) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	p.timeout = timeout

	p.client = &http.Client{Timeout: p.timeout}

	return nil
}

// Analyze processes an image with the given prompt.
func (p *vlmPlugin) Analyze(ctx context.Context, imgData []byte, prompt string) (*analysis.AnalysisResult, error) {
	img, err := decodeAndResizeImage(imgData)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	buf := new(bytes.Buffer)
	if err := jpeg.Encode(buf, img, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	base64Img := base64.StdEncoding.EncodeToString(buf.Bytes())

	reqBody := map[string]interface{}{
		"messages": []map[string]string{{
			"role":    "user",
			"content": fmt.Sprintf("data:image/jpeg;base64,%s\n\n%s", base64Img, prompt),
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
		resp.Body.Close()
		return nil, fmt.Errorf("llama-server returned HTTP %d", resp.StatusCode)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		return nil, fmt.Errorf("llama-server returned HTTP %d", resp.StatusCode)
	}

	var result struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
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
func (p *vlmPlugin) Close() error {
	if p.client != nil {
		p.client.CloseIdleConnections()
	}
	return nil
}

// Info returns plugin metadata.
func (p *vlmPlugin) Info() analysis.PluginInfo {
	return pluginInfo
}

// decodeAndResizeImage decodes JPEG bytes and resizes to max 512x512.
func decodeAndResizeImage(imgData []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(imgData))
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	maxDim := 512
	if bounds.Dx() > maxDim || bounds.Dy() > maxDim {
		scale := float64(maxDim) / float64(max(bounds.Dx(), bounds.Dy()))
		newW := int(float64(bounds.Dx()) * scale)
		newH := int(float64(bounds.Dy()) * scale)
		img = resizeImage(img, newW, newH)
	}
	return img, nil
}

// resizeImage resizes using nearest-neighbor sampling.
func resizeImage(img image.Image, newW, newH int) image.Image {
	resized := image.NewRGBA(image.Rect(0, 0, newW, newH))
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			srcX := x * img.Bounds().Dx() / newW
			srcY := y * img.Bounds().Dy() / newH
			resized.Set(x, y, img.At(srcX, srcY))
		}
	}
	return resized
}
