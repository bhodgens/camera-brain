// Package detection defines the plugin interface for object detection backends.
package detection

import (
	"context"
	"image"
)

// Detection represents a detected object with bounding box.
type Detection struct {
	// ClassID is the numeric class identifier
	ClassID int
	// ClassName is the human-readable class name (e.g., "person", "car")
	ClassName string
	// Confidence is the detection confidence (0.0 - 1.0)
	Confidence float32
	// BBox is the bounding box [x1, y1, x2, y2]
	BBox [4]int
	// Metadata contains additional detection-specific information
	Metadata map[string]interface{}
}

// Config contains detection plugin configuration.
type Config struct {
	// ModelPath is the path to the model file
	ModelPath string
	// ModelType identifies the model format (e.g., "rknn", "onnx", "yolov5")
	ModelType string
	// InputSize is the expected input dimensions [width, height]
	InputSize [2]int
	// ConfidenceThreshold filters low-confidence detections
	ConfidenceThreshold float32
	// NMSThreshold is the non-maximum suppression IoU threshold
	NMSThreshold float32
	// DeviceID specifies which device to use (for multi-device systems)
	DeviceID int
	// Threads is the number of CPU threads to use
	Threads int
}

// PluginInfo contains plugin metadata.
type PluginInfo struct {
	// Name is the plugin identifier
	Name string
	// Version is the plugin version
	Version string
	// Backend is the hardware backend ("npu", "cpu", "cuda", "api")
	Backend string
	// ModelFormat is the model file format
	ModelFormat string
}

// Detector defines the interface for object detection.
type Detector interface {
	// Initialize loads models and prepares resources.
	// Must be called before Detect().
	Initialize(ctx context.Context, cfg Config) error

	// Detect runs inference on an image and returns detections.
	// Returns nil, nil if no objects detected.
	Detect(ctx context.Context, image image.Image) ([]Detection, error)

	// Close releases all resources.
	// Should be called when the detector is no longer needed.
	Close() error

	// Info returns plugin metadata.
	Info() PluginInfo
}
