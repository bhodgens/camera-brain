# Camera Brain Plugin Development Guide

This guide explains how to create custom plugins for the Camera Brain system.

## Plugin Architecture

Camera Brain uses a plugin-based architecture for:

1. **Detection Plugins** - Object detection from video frames
2. **Analysis Plugins** - Image analysis and attribute extraction

Each plugin type implements a Go interface, allowing you to swap backends without changing application code.

## Creating a Detection Plugin

### Step 1: Create the Plugin Package

```go
// pkg/plugin/detection/myplugin/plugin.go
package myplugin

import (
    "context"
    "image"

    "rock-cluster/pkg/plugin/detection"
)

type MyPlugin struct {
    config detection.Config
    // Add your model/state fields here
}

// Ensure interface compliance
var _ detection.Detector = (*MyPlugin)(nil)
```

### Step 2: Implement the Interface

```go
func (p *MyPlugin) Initialize(ctx context.Context, cfg detection.Config) error {
    p.config = cfg
    // Load your model file from cfg.ModelPath
    // Initialize any resources
    return nil
}

func (p *MyPlugin) Detect(ctx context.Context, img image.Image) ([]detection.Detection, error) {
    // Preprocess image to model input format
    // Run inference
    // Post-process results

    return []detection.Detection{
        {
            ClassID:    0,
            ClassName:  "person",
            Confidence: 0.95,
            BBox:       [4]int{100, 100, 200, 300},
        },
    }, nil
}

func (p *MyPlugin) Close() error {
    // Release resources
    return nil
}

func (p *MyPlugin) Info() detection.PluginInfo {
    return detection.PluginInfo{
        Name:        "myplugin",
        Version:     "1.0.0",
        Backend:     "cpu", // or "npu", "cuda", "api"
        ModelFormat: "onnx", // or "rknn", "pt", etc.
    }
}
```

### Step 3: Register the Plugin

```go
// cmd/myapp/main.go
import "rock-cluster/pkg/plugin"

func main() {
    plugin.RegisterDetection("myplugin", func() detection.Detector {
        return &myplugin.MyPlugin{}
    })

    // ... rest of your application
}
```

### Step 4: Configure Usage

```yaml
# config.yaml
detection:
  plugin: myplugin
  config:
    model_path: /models/mymodel.onnx
    model_type: onnx
    input_width: 640
    input_height: 640
    confidence_threshold: 0.5
    nms_threshold: 0.45
```

## Creating an Analysis Plugin

### Step 1: Create the Plugin Package

```go
// pkg/plugin/analysis/myanalyzer/plugin.go
package myanalyzer

import (
    "context"

    "rock-cluster/pkg/plugin/analysis"
)

type MyAnalyzer struct {
    config analysis.Config
}

var _ analysis.Analyzer = (*MyAnalyzer)(nil)
```

### Step 2: Implement the Interface

```go
func (a *MyAnalyzer) Initialize(ctx context.Context, cfg analysis.Config) error {
    a.config = cfg
    // Setup API client or load local model
    return nil
}

func (a *MyAnalyzer) Analyze(ctx context.Context, image []byte, prompt string) (*analysis.AnalysisResult, error) {
    // image is JPEG-encoded bytes
    // Send to your API or run local inference

    return &analysis.AnalysisResult{
        Attributes: map[string]interface{}{
            "color": "red",
            "size":  "large",
        },
        RawResponse: "The object is red and large.",
        Confidence:  0.9,
        TokensUsed:  150,
    }, nil
}

func (a *MyAnalyzer) Close() error {
    return nil
}

func (a *MyAnalyzer) Info() analysis.PluginInfo {
    return analysis.PluginInfo{
        Name:      "myanalyzer",
        Version:   "1.0.0",
        Modality:  "vision", // or "text", "multimodal"
        IsLocal:   false,    // true for local inference
    }
}
```

### Step 3: Register and Configure

```go
plugin.RegisterAnalysis("myanalyzer", func() analysis.Analyzer {
    return &myanalyzer.MyAnalyzer{}
})
```

```yaml
analysis:
  plugin: myanalyzer
  config:
    endpoint: https://api.example.com/v1/analyze
    api_key: ${API_KEY}
    max_tokens: 256
    timeout_sec: 30
```

## Built-in Plugins

### Detection Plugins

| Plugin | Backend | Description |
|--------|---------|-------------|
| `rknn` | NPU | Rockchip NPU (RK3568, RK3588) |
| `onnx` | CPU/CUDA | ONNX Runtime (cross-platform) |
| `api` | HTTP | External detection API |

### Analysis Plugins

| Plugin | Backend | Description |
|--------|---------|-------------|
| `llamacpp` | Local | llama.cpp server (CPU/GPU) |
| `anthropic` | API | Anthropic Claude API |
| `openai` | API | OpenAI GPT-4 Vision API |

## Plugin Configuration Reference

### Detection Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | string | required | Path to model file |
| `model_type` | string | required | Format identifier |
| `input_width` | int | 640 | Model input width |
| `input_height` | int | 640 | Model input height |
| `confidence_threshold` | float | 0.5 | Min confidence |
| `nms_threshold` | float | 0.45 | NMS IoU threshold |
| `device_id` | int | 0 | Device selector |
| `threads` | int | 4 | CPU threads |

### Analysis Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint` | string | - | API URL |
| `model_path` | string | - | Local model path |
| `mmproj_path` | string | - | Vision projector |
| `api_key` | string | - | API authentication |
| `max_tokens` | int | 256 | Response length |
| `temperature` | float | 0.1 | Randomness |
| `timeout_sec` | int | 120 | Request timeout |
| `model_name` | string | - | API model ID |

## Testing Your Plugin

```go
// pkg/plugin/detection/myplugin/plugin_test.go
package myplugin

import (
    "context"
    "image"
    "testing"
)

func TestMyPlugin(t *testing.T) {
    p := &MyPlugin{}

    err := p.Initialize(context.Background(), detection.Config{
        ModelPath: "testdata/model.onnx",
    })
    if err != nil {
        t.Fatal(err)
    }
    defer p.Close()

    // Create test image
    img := image.NewRGBA(image.Rect(0, 0, 640, 480))

    detections, err := p.Detect(context.Background(), img)
    if err != nil {
        t.Fatal(err)
    }

    if len(detections) == 0 {
        t.Error("Expected detections")
    }
}
```

## Best Practices

1. **Context Awareness**: Always respect `context.Context` for cancellation and timeouts.

2. **Resource Management**: Release all resources in `Close()`. Use defer patterns.

3. **Error Handling**: Return descriptive errors with context.

4. **Thread Safety**: Plugins may be called from multiple goroutines. Use mutexes if needed.

5. **Configuration Validation**: Validate configuration in `Initialize()`, not later.

6. **Logging**: Use structured logging. Include plugin name in log entries.

7. **Metrics**: Consider exposing Prometheus metrics for inference latency, throughput.

## Example: Complete Detection Plugin

See `pkg/plugin/detection/rknn/plugin.go` for a complete working example using the Rockchip NPU.
