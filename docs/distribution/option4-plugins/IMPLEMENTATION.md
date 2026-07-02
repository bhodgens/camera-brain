# Option 4: Hybrid Core Plus Plugins Architecture - Implementation Plan

## Goal
Refactor camera-brain into a modular architecture where detection and analysis backends can be swapped based on hardware capabilities and deployment requirements.

## Scope
- Define plugin interfaces for detection and analysis
- Implement reference plugins (NPU, CPU, local LLM, remote API)
- Create configuration-driven plugin selection
- Document plugin development guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Camera Brain Core                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Worker    │  │   Gateway   │  │   Query Service     │ │
│  │  Manager    │  │    API      │  │   (Natural Lang)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
│  │              Plugin Manager (Registry)                  │ │
│  └──────┬────────────────┬─────────────────────┬──────────┘ │
└─────────┼────────────────┼─────────────────────┼────────────┘
          │                │                     │
   ┌──────┴──────┐  ┌──────┴──────┐      ┌──────┴──────┐
   │  Detection  │  │  Analysis   │      │   Storage   │
   │   Plugins   │  │   Plugins   │      │   Plugins   │
   └─────────────┘  └─────────────┘      └─────────────┘
```

## Plugin Interfaces

### Detection Plugin Interface

```go
// pkg/plugin/detection/interface.go
package detection

import "context"

// Detection represents a detected object with bounding box
type Detection struct {
    ClassID    int
    ClassName  string
    Confidence float32
    BBox       [4]int  // x1, y1, x2, y2
    Metadata   map[string]interface{}
}

// Config contains detection plugin configuration
type Config struct {
    ModelPath     string
    ModelType     string  // "rknn", "onnx", "yolov5", "yolov8"
    InputSize     [2]int  // width, height
    ConfidenceTh  float32
    NMSThreshold  float32
}

// Detector defines the interface for object detection
type Detector interface {
    // Initialize loads models and prepares resources
    Initialize(ctx context.Context, cfg Config) error

    // Detect runs inference on an image
    Detect(ctx context.Context, image []byte) ([]Detection, error)

    // Close releases resources
    Close() error

    // Info returns plugin metadata
    Info() PluginInfo
}

type PluginInfo struct {
    Name        string
    Version     string
    Backend     string  // "npu", "cpu", "cuda"
    ModelFormat string
}
```

### Analysis Plugin Interface

```go
// pkg/plugin/analysis/interface.go
package analysis

import "context"

// AnalysisResult contains structured analysis output
type AnalysisResult struct {
    Attributes  map[string]interface{}
    RawResponse string
    Confidence  float32
}

// Config contains analysis plugin configuration
type Config struct {
    Endpoint        string  // URL for API-based plugins
    ModelPath       string  // Path for local models
    MaxTokens       int
    Temperature     float32
    Timeout         time.Duration
}

// Analyzer defines the interface for image analysis
type Analyzer interface {
    Initialize(ctx context.Context, cfg Config) error
    Analyze(ctx context.Context, image []byte, prompt string) (*AnalysisResult, error)
    Close() error
    Info() PluginInfo
}

type PluginInfo struct {
    Name      string
    Version   string
    Modality  string  // "vision", "text", "multimodal"
    IsLocal   bool    // true = local inference, false = API
}
```

## Built-in Detection Plugins

### 1. RKNN (Rockchip NPU)
```go
// pkg/plugin/detection/rknn/plugin.go
package rknn

import "rock-cluster/pkg/plugin/detection"

type RKNNPlugin struct {
    model  *rknn.Model
    config detection.Config
}

func (p *RKNNPlugin) Initialize(ctx context.Context, cfg detection.Config) error {
    // Load RKNN model from .rknn file
    // Validate INT8/FP16 compatibility
    // Pre-allocate input buffers
}

func (p *RKNNPlugin) Detect(ctx context.Context, image []byte) ([]detection.Detection, error) {
    // Preprocess: decode JPEG, resize to input size, NHWC conversion
    // Run rknn_run()
    // Postprocess: decode YOLOv5 output, apply NMS
}
```

### 2. ONNX (CPU/GPU via ONNX Runtime)
```go
// pkg/plugin/detection/onnx/plugin.go
package onnx

type ONNXPlugin struct {
    session *onnxruntime.Session
}

func (p *ONNXPlugin) Initialize(ctx context.Context, cfg detection.Config) error {
    // Load .onnx model
    // Create session with CPU or CUDA execution provider
}

func (p *ONNXPlugin) Detect(ctx context.Context, image []byte) ([]detection.Detection, error) {
    // Preprocess: NCHW format, normalize
    // session.Run()
    // Postprocess
}
```

### 3. External API (for cloud-based detection)
```go
// pkg/plugin/detection/api/plugin.go
package api

type APIPlugin struct {
    client *http.Client
    cfg    detection.Config
}

func (p *APIPlugin) Initialize(ctx context.Context, cfg detection.Config) error {
    // Validate endpoint, API key
    // Test connection
}

func (p *APIPlugin) Detect(ctx context.Context, image []byte) ([]detection.Detection, error) {
    // POST image bytes to endpoint
    // Parse JSON response into Detection structs
}
```

## Built-in Analysis Plugins

### 1. Llama.cpp (Local VLM)
```go
// pkg/plugin/analysis/llamacpp/plugin.go
package llamacpp

type LlamaPlugin struct {
    serverURL string
    client    *http.Client
}

func (p *LlamaPlugin) Initialize(ctx context.Context, cfg analysis.Config) error {
    // Validate llama-server is running
    // Health check endpoint
}

func (p *LlamaPlugin) Analyze(ctx context.Context, image []byte, prompt string) (*analysis.AnalysisResult, error) {
    // Base64 encode image
    // POST to /v1/chat/completions with vision format
    // Parse JSON response
}
```

### 2. Anthropic API (Claude Vision)
```go
// pkg/plugin/analysis/anthropic/plugin.go
package anthropic

type AnthropicPlugin struct {
    apiKey   string
    client   *http.Client
    model    string  // "claude-sonnet-4-0"
}

func (p *AnthropicPlugin) Analyze(ctx context.Context, image []byte, prompt string) (*analysis.AnalysisResult, error) {
    // POST to https://api.anthropic.com/v1/messages
    // Handle pagination for large responses
}
```

### 3. OpenAI API (GPT-4 Vision)
```go
// pkg/plugin/analysis/openai/plugin.go
package openai

type OpenAIPlugin struct {
    apiKey string
    model  string  // "gpt-4o"
}
```

## Plugin Registry

```go
// pkg/plugin/registry.go
package plugin

type DetectionPluginFactory func() detection.Detector
type AnalysisPluginFactory func() analysis.Analyzer

var (
    detectionPlugins = map[string]DetectionPluginFactory{
        "rknn":   func() detection.Detector { return &rknn.RKNNPlugin{} },
        "onnx":   func() detection.Detector { return &onnx.ONNXPlugin{} },
        "api":    func() detection.Detector { return &api.APIPlugin{} },
    }

    analysisPlugins = map[string]AnalysisPluginFactory{
        "llamacpp": func() analysis.Analyzer { return &llamacpp.LlamaPlugin{} },
        "anthropic": func() analysis.Analyzer { return &anthropic.AnthropicPlugin{} },
        "openai":   func() analysis.Analyzer { return &openai.OpenAIPlugin{} },
    }
)

// GetDetectionPlugin creates a detection plugin by name
func GetDetectionPlugin(name string) (detection.Detector, error) {
    factory, exists := detectionPlugins[name]
    if !exists {
        return nil, fmt.Errorf("unknown detection plugin: %s", name)
    }
    return factory(), nil
}
```

## Configuration-Driven Plugin Selection

```yaml
# config.yaml
detection:
  plugin: rknn  # or "onnx", "api"
  config:
    model_path: /models/yolov5s_int8.rknn
    model_type: rknn
    input_size: [640, 640]
    confidence_threshold: 0.5
    nms_threshold: 0.45

analysis:
  plugin: llamacpp
  config:
    endpoint: http://localhost:8888
    model_path: /models/LFM2.5-VL-1.6B.Q8_0.gguf
    max_tokens: 256
    temperature: 0.1

storage:
  plugin: postgres
  config:
    host: localhost
    port: 5432
    database: camera_brain
```

## Plugin Configuration Validation

```go
// pkg/config/validate.go
package config

func Validate(cfg *Config) error {
    var errs []error

    // Validate detection plugin
    if _, exists := plugin.DetectionPlugins[cfg.Detection.Plugin]; !exists {
        errs = append(errs, fmt.Errorf("unknown detection plugin: %s", cfg.Detection.Plugin))
    }

    // Validate analysis plugin
    if _, exists := plugin.AnalysisPlugins[cfg.Analysis.Plugin]; !exists {
        errs = append(errs, fmt.Errorf("unknown analysis plugin: %s", cfg.Analysis.Plugin))
    }

    // Validate model paths exist
    if cfg.Detection.Config.ModelPath != "" {
        if _, err := os.Stat(cfg.Detection.Config.ModelPath); os.IsNotExist(err) {
            errs = append(errs, fmt.Errorf("model not found: %s", cfg.Detection.Config.ModelPath))
        }
    }

    return errors.Join(errs...)
}
```

## Plugin Development Guide

### Creating a New Detection Plugin

```go
// 1. Create package
package myplugin

import "rock-cluster/pkg/plugin/detection"

// 2. Implement interface
type MyPlugin struct {
    // plugin state
}

func (p *MyPlugin) Initialize(ctx context.Context, cfg detection.Config) error {
    // initialization code
}

func (p *MyPlugin) Detect(ctx context.Context, image []byte) ([]detection.Detection, error) {
    // inference code
}

func (p *MyPlugin) Close() error {
    // cleanup code
}

func (p *MyPlugin) Info() detection.PluginInfo {
    return detection.PluginInfo{
        Name: "myplugin",
        Version: "1.0.0",
        Backend: "cpu",
    }
}

// 3. Register in main.go
func main() {
    plugin.RegisterDetection("myplugin", func() detection.Detector {
        return &MyPlugin{}
    })
}
```

## Migration Strategy

### Phase 1: Extract Current Code
- Move current NPU code to `pkg/plugin/detection/rknn/`
- Move current VLM processor to `pkg/plugin/analysis/llamacpp/`

### Phase 2: Create Registry
- Implement plugin registry
- Add configuration parsing

### Phase 3: Add Alternative Plugins
- Add ONNX detection plugin (CPU fallback)
- Add Anthropic analysis plugin (API fallback)

### Phase 4: Update Main Entry Points
- Gateway creates plugins from config
- Services use plugin interfaces, not concrete types

### Phase 5: Testing
- Test each plugin independently
- Test plugin hot-swapping (restart with different config)

## Deployment Profiles

Create pre-configured profiles for common deployments:

```yaml
# profiles/raspberry-pi-npu.yaml
detection:
  plugin: rknn
  config:
    model_path: /usr/share/camera-brain/models/yolov5s_int8.rknn

analysis:
  plugin: llamacpp
  config:
    endpoint: http://remote-server:8888  # Offload to server

# profiles/desktop-cuda.yaml
detection:
  plugin: onnx
  config:
    model_path: /opt/camera-brain/models/yolov5s.onnx
    execution_provider: cuda

analysis:
  plugin: llamacpp
  config:
    endpoint: http://localhost:8888

# profiles/cloud-api.yaml
detection:
  plugin: api
  config:
    endpoint: https://detect.myapi.com/v1/detect
    api_key: ${DETECT_API_KEY}

analysis:
  plugin: anthropic
  config:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-0
```

## Validation

- [ ] Plugin interfaces compile
- [ ] RKNN plugin produces same results as current implementation
- [ ] ONNX plugin works on x86_64 with CPU
- [ ] Configuration validation catches errors
- [ ] Services start with plugins loaded from config
- [ ] Can swap plugins by changing config only (no code changes)

## Next Steps After Implementation

1. Document plugin API for third-party developers
2. Create plugin template repository
3. Add plugin versioning and compatibility checking
4. Consider plugin loading from external directories (dynamically load .so files)
