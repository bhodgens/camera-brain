//go:build linux && arm64

// Package rknn provides an RKNN-based object detection plugin using Rockchip NPU.
package rknn

/*
#cgo LDFLAGS: -L/usr/lib -lrknnrt
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <rknn_api.h>

// Helper struct to avoid Go keyword conflict with 'type' keyword.
// Must match the fields of rknn_input in rknn_api.h exactly.
typedef struct {
    uint32_t index;
    void* buf;
    uint32_t size;
    uint8_t pass_through;
    rknn_tensor_type tensor_type;
    rknn_tensor_format fmt;
} rknn_input_wrapper;

// Helper function to set input
static int rknn_set_input(rknn_context ctx, rknn_input_wrapper* w) {
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = w->index;
    input.buf = w->buf;
    input.size = w->size;
    input.pass_through = w->pass_through;
    input.type = w->tensor_type;
    input.fmt = w->fmt;
    return rknn_inputs_set(ctx, 1, &input);
}

// Read file helper
static int read_file(const char* path, void** out_buf, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc(size);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, size, f);
    fclose(f);
    if (got != size) {
        free(buf);
        return -2;
    }
    *out_buf = buf;
    *out_size = size;
    return 0;
}
*/
import "C"

import (
	"context"
	"fmt"
	"image"
	"math"
	"sync"
	"unsafe"

	"golang.org/x/image/draw"

	"rock-cluster/pkg/plugin/detection"
)

const (
	yoloInputSize  = 640
	yoloNumClasses = 80
)

var yoloStrides = []int{8, 16, 32}

var yoloAnchors = [][2]int{
	{10, 13}, {16, 30}, {33, 23},
	{30, 61}, {62, 45}, {59, 119},
	{116, 90}, {156, 198}, {373, 326},
}

// cocoClasses maps YOLOv5 class IDs to COCO class names (80 classes).
var cocoClasses = map[int]string{
	0:  "person",
	1:  "bicycle",
	2:  "car",
	3:  "motorcycle",
	4:  "airplane",
	5:  "bus",
	6:  "train",
	7:  "truck",
	8:  "boat",
	9:  "traffic light",
	10: "fire hydrant",
	11: "stop sign",
	12: "parking meter",
	13: "bench",
	14: "bird",
	15: "cat",
	16: "dog",
	17: "horse",
	18: "sheep",
	19: "cow",
	20: "elephant",
	21: "bear",
	22: "zebra",
	23: "giraffe",
	24: "backpack",
	25: "umbrella",
	26: "handbag",
	27: "tie",
	28: "suitcase",
	29: "frisbee",
	30: "skis",
	31: "snowboard",
	32: "sports ball",
	33: "kite",
	34: "baseball bat",
	35: "baseball glove",
	36: "skateboard",
	37: "surfboard",
	38: "tennis racket",
	39: "bottle",
	40: "wine glass",
	41: "cup",
	42: "fork",
	43: "knife",
	44: "spoon",
	45: "bowl",
	46: "banana",
	47: "apple",
	48: "sandwich",
	49: "orange",
	50: "broccoli",
	51: "carrot",
	52: "hot dog",
	53: "pizza",
	54: "donut",
	55: "cake",
	56: "chair",
	57: "couch",
	58: "potted plant",
	59: "bed",
	60: "dining table",
	61: "toilet",
	62: "tv",
	63: "laptop",
	64: "mouse",
	65: "remote",
	66: "keyboard",
	67: "cell phone",
	68: "microwave",
	69: "oven",
	70: "toaster",
	71: "sink",
	72: "refrigerator",
	73: "book",
	74: "clock",
	75: "vase",
	76: "scissors",
	77: "teddy bear",
	78: "hair drier",
	79: "toothbrush",
}

var pluginInfo = detection.PluginInfo{
	Name:        "rknn",
	Version:     "1.0.0",
	Backend:     "npu",
	ModelFormat: "rknn",
}

type detector struct {
	mu            sync.Mutex
	ctx           C.rknn_context
	modelPath     string
	inputW        int
	inputH        int
	confThreshold float32
	nmsThreshold  float32
}

func New() detection.Detector {
	return &detector{confThreshold: 0.3, nmsThreshold: 0.45}
}

func (d *detector) Initialize(_ context.Context, cfg detection.Config) error {
	if cfg.ModelPath == "" {
		return fmt.Errorf("ModelPath is required")
	}

	var buf unsafe.Pointer
	var size C.size_t
	cPath := C.CString(cfg.ModelPath)
	defer C.free(unsafe.Pointer(cPath))

	ret := C.read_file(cPath, &buf, &size)
	if ret < 0 {
		return fmt.Errorf("failed to read model file: %s", cfg.ModelPath)
	}
	defer C.free(buf)

	ret = C.rknn_init(&d.ctx, buf, C.uint(size), 0, nil)
	if ret < 0 {
		return fmt.Errorf("rknn_init failed: %d", ret)
	}

	var attr C.rknn_tensor_attr
	attr.index = 0
	ret = C.rknn_query(d.ctx, C.RKNN_QUERY_INPUT_ATTR, unsafe.Pointer(&attr), C.sizeof_rknn_tensor_attr)
	if ret < 0 {
		C.rknn_destroy(d.ctx)
		d.ctx = 0
		return fmt.Errorf("rknn_query input attr: %d", ret)
	}

	// dims layout depends on attr.fmt. For NHWC the dims are [N, H, W, C],
	// so dims[1]=H, dims[2]=W. NCHW would invert these and silently swap
	// the preprocessing resize — corrupting every inference.
	if attr.fmt != C.RKNN_TENSOR_NHWC {
		C.rknn_destroy(d.ctx)
		d.ctx = 0
		return fmt.Errorf("unsupported input tensor format: expected NHWC, got %d (model must be converted with NHWC input layout)", int(attr.fmt))
	}
	d.inputW = int(attr.dims[2])
	d.inputH = int(attr.dims[1])
	d.modelPath = cfg.ModelPath

	if cfg.ConfidenceThreshold > 0 {
		d.confThreshold = cfg.ConfidenceThreshold
	}
	if cfg.NMSThreshold > 0 {
		d.nmsThreshold = cfg.NMSThreshold
	}
	return nil
}

func (d *detector) Detect(ctx context.Context, img image.Image) ([]detection.Detection, error) {
	// RKNN contexts are single-tenant: rknn_inputs_set → rknn_run →
	// rknn_outputs_get share state on the context handle. Concurrent
	// calls would interleave inputs and outputs. Per PLUGIN-GUIDE #4
	// we serialize the whole inference pipeline.
	d.mu.Lock()
	defer d.mu.Unlock()

	preprocessed := d.preprocess(img)
	if preprocessed == nil {
		return nil, fmt.Errorf("preprocess failed")
	}

	var input C.rknn_input_wrapper
	input.index = 0
	input.buf = unsafe.Pointer(&preprocessed[0])
	input.size = C.uint(len(preprocessed))
	input.pass_through = 0
	input.tensor_type = C.RKNN_TENSOR_UINT8
	input.fmt = C.RKNN_TENSOR_NHWC

	ret := C.rknn_set_input(d.ctx, &input)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_inputs_set failed: %d", ret)
	}

	ret = C.rknn_run(d.ctx, nil)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_run failed: %d", ret)
	}

	// RKNN_QUERY_IN_OUT_NUM populates a rknn_input_output_num struct
	// (uint32_t n_input, uint32_t n_output) — see /tmp/rknn_api.h:269-275.
	// Passing a 4-byte buffer reads only n_input and corrupts the next
	// stack slot. Must pass the full 8-byte struct.
	var ioNum C.rknn_input_output_num
	ret = C.rknn_query(d.ctx, C.RKNN_QUERY_IN_OUT_NUM, unsafe.Pointer(&ioNum), C.sizeof_rknn_input_output_num)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_query in/out num: %d", ret)
	}
	numOutputs := uint32(ioNum.n_output)
	// Allow both 1 (concatenated) and 3 (separate) outputs for YOLOv5 models
	if numOutputs != 1 && int(numOutputs) != len(yoloStrides) {
		return nil, fmt.Errorf("model has %d outputs, expected 1 (concatenated) or 3 (separate stride heads)", numOutputs)
	}

	outputs := make([]C.rknn_output, int(numOutputs))
	for i := range outputs {
		outputs[i].want_float = 1 // Request float32 output, not int8
		outputs[i].is_prealloc = 0
	}

	ret = C.rknn_outputs_get(d.ctx, C.uint(numOutputs), &outputs[0], nil)
	if ret < 0 {
		// BUG PREVENTION: Do NOT add early returns above this line or after this
		// block. The defer below must fire before any early return, or the NPU
		// output buffers will leak (C.rknn_outputs_release will never run).
		return nil, fmt.Errorf("rknn_outputs_get failed: %d", ret)
	}
	defer C.rknn_outputs_release(d.ctx, C.uint(numOutputs), &outputs[0])

	return d.parseYOLOv5Output(outputs), nil
}

func (d *detector) Close() error {
	if d.ctx != 0 {
		C.rknn_destroy(d.ctx)
		d.ctx = 0
	}
	return nil
}

func (d *detector) Info() detection.PluginInfo {
	return pluginInfo
}

func (d *detector) preprocess(img image.Image) []byte {
	resized := image.NewRGBA(image.Rect(0, 0, d.inputW, d.inputH))
	draw.BiLinear.Scale(resized, resized.Rect, img, img.Bounds(), draw.Src, nil)
	buf := make([]byte, d.inputH*d.inputW*3)
	for y := 0; y < d.inputH; y++ {
		for x := 0; x < d.inputW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			offset := (y*d.inputW + x) * 3
			buf[offset] = uint8(r >> 8)
			buf[offset+1] = uint8(g >> 8)
			buf[offset+2] = uint8(b >> 8)
		}
	}
	return buf
}

type yoloDet struct {
	confidence float32
	classID    int
	x1, y1, x2, y2 int
}

func (d *detector) parseYOLOv5Output(outputs []C.rknn_output) []detection.Detection {
	var allDets []yoloDet

	// Check if we have a single concatenated output (2,142,000 bytes = all 3 strides combined)
	// This happens when the RKNN model exports a single tensor instead of 3 separate stride heads
	if len(outputs) == 1 && outputs[0].size == 2142000 {
		// Single concatenated output - split into 3 stride regions
		// stride 8: 80x80x3x85 = 1,632,000 bytes (offset 0)
		// stride 16: 40x40x3x85 = 408,000 bytes (offset 1,632,000)
		// stride 32: 20x20x3x85 = 102,000 bytes (offset 2,040,000)
		strideOffsets := []int{0, 1632000, 2040000}
		gridSizes := []int{80, 40, 20}

		// Try float32 first - YOLOv5 outputs are typically float32, not int8
		data := (*[1 << 30]float32)(outputs[0].buf)[:535500:535500]

		for outIdx := 0; outIdx < 3; outIdx++ {
			stride := yoloStrides[outIdx]
			gridSize := gridSizes[outIdx]
			numAnchors := 3
			// Convert byte offset to float32 element offset
			offset := strideOffsets[outIdx] / 4

			for anchorIdx := 0; anchorIdx < numAnchors; anchorIdx++ {
				for gy := 0; gy < gridSize; gy++ {
					for gx := 0; gx < gridSize; gx++ {
						baseIdx := offset + ((gy*gridSize+gx)*numAnchors+anchorIdx)*85
						if baseIdx+84 >= len(data) {
							continue
						}

						// YOLOv5 outputs after RKNN: values typically already decoded to 0-640 pixel space
						// But check if values are normalized (0-1) by looking at magnitude
						cx := data[baseIdx]
						cy := data[baseIdx+1]
						w := data[baseIdx+2]
						h := data[baseIdx+3]
						boxConf := data[baseIdx+4]

						// If cx is < 1, values are normalized - apply sigmoid and grid offset
						if cx < 1.0 && cx > 0.0 {
							// Standard YOLO decoding: apply sigmoid to box centers
							cx = float32(gx) + 1.0/(1.0+float32(math.Exp(-float64(cx))))
							cy = float32(gy) + 1.0/(1.0+float32(math.Exp(-float64(cy))))
							w = float32(yoloAnchors[outIdx*3+anchorIdx][0]) * float32(math.Exp(float64(w)))
							h = float32(yoloAnchors[outIdx*3+anchorIdx][1]) * float32(math.Exp(float64(h)))
							// Scale to pixel coordinates
							cx *= float32(stride)
							cy *= float32(stride)
							w *= float32(stride)
							h *= float32(stride)
						}

						if boxConf < d.confThreshold {
							continue
						}

						bestClass := 0
						bestScore := float32(0)
						for c := 0; c < yoloNumClasses; c++ {
							score := data[baseIdx+5+c] * boxConf
							if score > bestScore {
								bestScore = score
								bestClass = c
							}
						}

						if bestScore < d.confThreshold {
							continue
						}

						x1 := int(cx - w/2)
						y1 := int(cy - h/2)
						x2 := int(cx + w/2)
						y2 := int(cy + h/2)

						x1 = clampInt(x1, 0, yoloInputSize)
						y1 = clampInt(y1, 0, yoloInputSize)
						x2 = clampInt(x2, 0, yoloInputSize)
						y2 = clampInt(y2, 0, yoloInputSize)

						allDets = append(allDets, yoloDet{confidence: bestScore, classID: bestClass, x1: x1, y1: y1, x2: x2, y2: y2})
					}
				}
			}
		}
	} else {
		// Multiple separate outputs (original logic)
		for outIdx, output := range outputs {
			stride := yoloStrides[outIdx]
			gridSize := yoloInputSize / stride
			outputSize := int(output.size)
			if outputSize == 0 {
				continue
			}

			// Read as float32 instead of int8
			data := (*[1 << 30]float32)(output.buf)[:outputSize/4:outputSize/4]
			numAnchors := 3

			for anchorIdx := 0; anchorIdx < numAnchors; anchorIdx++ {
				for gy := 0; gy < gridSize; gy++ {
					for gx := 0; gx < gridSize; gx++ {
						baseIdx := ((gy*gridSize+gx)*numAnchors+anchorIdx)*85
						if baseIdx+84 >= len(data) {
							continue
						}

						cx := data[baseIdx]
						cy := data[baseIdx+1]
						w := data[baseIdx+2]
						h := data[baseIdx+3]
						boxConf := data[baseIdx+4]

						// If cx < 1, values are normalized - apply sigmoid and grid offset
						if cx < 1.0 && cx > 0.0 {
							cx = float32(gx) + 1.0/(1.0+float32(math.Exp(-float64(cx))))
							cy = float32(gy) + 1.0/(1.0+float32(math.Exp(-float64(cy))))
							w = float32(yoloAnchors[outIdx*3+anchorIdx][0]) * float32(math.Exp(float64(w)))
							h = float32(yoloAnchors[outIdx*3+anchorIdx][1]) * float32(math.Exp(float64(h)))
							cx *= float32(stride)
							cy *= float32(stride)
							w *= float32(stride)
							h *= float32(stride)
						}

						if boxConf < d.confThreshold {
							continue
						}

						bestClass := 0
						bestScore := float32(0)
						for c := 0; c < yoloNumClasses; c++ {
							score := data[baseIdx+5+c] * boxConf
							if score > bestScore {
								bestScore = score
								bestClass = c
							}
						}

						if bestScore < d.confThreshold {
							continue
						}

						x1 := int(cx - w/2)
						y1 := int(cy - h/2)
						x2 := int(cx + w/2)
						y2 := int(cy + h/2)

						x1 = clampInt(x1, 0, yoloInputSize)
						y1 = clampInt(y1, 0, yoloInputSize)
						x2 = clampInt(x2, 0, yoloInputSize)
						y2 = clampInt(y2, 0, yoloInputSize)

						allDets = append(allDets, yoloDet{confidence: bestScore, classID: bestClass, x1: x1, y1: y1, x2: x2, y2: y2})
					}
				}
			}
		}
	}

	allDets = nms(allDets, d.nmsThreshold)

	var detections []detection.Detection
	for _, r := range allDets {
		name, ok := cocoClasses[r.classID]
		if !ok {
			// Use "unknown" for unrecognized class IDs instead of dropping
			name = "unknown"
		}
		detections = append(detections, detection.Detection{
			ClassID: r.classID, ClassName: name, Confidence: r.confidence,
			BBox: [4]int{r.x1, r.y1, r.x2, r.y2},
		})
	}
	return detections
}

func nms(dets []yoloDet, iouThresh float32) []yoloDet {
	for i := 0; i < len(dets)-1; i++ {
		for j := i + 1; j < len(dets); j++ {
			if dets[j].confidence > dets[i].confidence {
				dets[i], dets[j] = dets[j], dets[i]
			}
		}
	}
	keep := make([]bool, len(dets))
	for i := range keep {
		keep[i] = true
	}
	for i := 0; i < len(dets); i++ {
		if !keep[i] {
			continue
		}
		for j := i + 1; j < len(dets); j++ {
			if !keep[j] || dets[i].classID != dets[j].classID {
				continue
			}
			if calcIoU(dets[i], dets[j]) > iouThresh {
				keep[j] = false
			}
		}
	}
	var result []yoloDet
	for i, k := range keep {
		if k {
			result = append(result, dets[i])
		}
	}
	return result
}

func calcIoU(a, b yoloDet) float32 {
	x1 := maxInt(a.x1, b.x1)
	y1 := maxInt(a.y1, b.y1)
	x2 := minInt(a.x2, b.x2)
	y2 := minInt(a.y2, b.y2)
	if x2 <= x1 || y2 <= y1 {
		return 0
	}
	inter := float32((x2 - x1) * (y2 - y1))
	areaA := float32((a.x2 - a.x1) * (a.y2 - a.y1))
	areaB := float32((b.x2 - b.x1) * (b.y2 - b.y1))
	return inter / (areaA + areaB - inter)
}

func maxInt(a, b int) int { if a > b { return a }; return b }
func minInt(a, b int) int { if a < b { return a }; return b }
func clampInt(v, lo, hi int) int { if v < lo { return lo }; if v > hi { return hi }; return v }
