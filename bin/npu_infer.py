#!/usr/bin/env python3
"""
NPU Inference script for YOLOv5 RKNN models.
Reads JPEG image from stdin, outputs JSON detections.
"""

import sys
import json
import struct
from rknnlite.api import RKNNLite

# COCO 80 class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_PATH = "/home/camera-brain/models/yolov5s_int8.rknn"
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
INPUT_SIZE = 640

def main():
    # Read JPEG from stdin
    img_data = sys.stdin.buffer.read()
    if not img_data:
        print(json.dumps({"success": False, "error": "No image data"}))
        return

    try:
        # Initialize RKNN
        rknn = RKNNLite()
        ret = rknn.load_rknn(path=MODEL_PATH)
        if ret != 0:
            print(json.dumps({"success": False, "error": f"Load RKNN failed: {ret}"}))
            return

        # Run inference
        ret = rknn.inference(inputs=[img_data])
        if ret is None:
            print(json.dumps({"success": False, "error": "Inference failed"}))
            return

        # Parse outputs (assuming concatenated single output)
        outputs = ret[0] if isinstance(ret, list) else ret

        # Simple YOLOv5 output parsing
        # Output shape: (1, 25200, 85) = (batch, anchors, [box(4) + conf(1) + classes(80)])
        detections = parse_yolo_outputs(outputs)

        print(json.dumps({
            "success": True,
            "detections": detections,
            "count": len(detections)
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def parse_yolo_outputs(outputs):
    """Parse YOLOv5 RKNN outputs and apply NMS."""
    import numpy as np

    detections = []

    # Handle different output shapes
    if outputs.ndim == 3:
        # Single concatenated output: (1, 25200, 85)
        outputs = outputs[0]  # (25200, 85)
    elif outputs.ndim == 2:
        # Already flattened
        pass

    # YOLOv5 anchors
    anchors = [
        [10, 13], [16, 30], [33, 23],
        [30, 61], [62, 45], [59, 119],
        [116, 90], [156, 198], [373, 326]
    ]
    strides = [8, 16, 32]

    for stride_idx, stride in enumerate(strides):
        grid_size = INPUT_SIZE // stride
        anchor_start = stride_idx * 3

        for anchor_idx in range(3):
            for gy in range(grid_size):
                for gx in range(grid_size):
                    base_idx = ((gy * grid_size + gx) * 3 + anchor_idx) * 85

                    if base_idx + 84 >= len(outputs):
                        continue

                    # Parse box outputs
                    x = outputs[base_idx]
                    y = outputs[base_idx + 1]
                    w = outputs[base_idx + 2]
                    h = outputs[base_idx + 3]
                    conf = outputs[base_idx + 4]

                    # Check if values need sigmoid (normalized 0-1)
                    if 0 < x < 1:
                        import math
                        x = gx + 1.0 / (1.0 + math.exp(-x))
                        y = gy + 1.0 / (1.0 + math.exp(-y))
                        w = anchors[anchor_start + anchor_idx][0] * math.exp(w)
                        h = anchors[anchor_start + anchor_idx][1] * math.exp(h)
                        x *= stride
                        y *= stride
                        w *= stride
                        h *= stride

                    if conf < CONF_THRESHOLD:
                        continue

                    # Find best class
                    best_class = 0
                    best_score = 0.0
                    for c in range(80):
                        score = outputs[base_idx + 5 + c] * conf
                        if score > best_score:
                            best_score = score
                            best_class = c

                    if best_score < CONF_THRESHOLD:
                        continue

                    # Convert to x1, y1, x2, y2
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    detections.append({
                        "class_id": best_class,
                        "class_name": COCO_CLASSES[best_class] if 0 <= best_class < len(COCO_CLASSES) else "unknown",
                        "confidence": float(best_score),
                        "bbox": [max(0, x1), max(0, y1), min(INPUT_SIZE, x2), min(INPUT_SIZE, y2)]
                    })

    # Apply NMS
    detections = apply_nms(detections, NMS_THRESHOLD)
    return detections

def apply_nms(detections, iou_threshold):
    """Apply non-maximum suppression."""
    if not detections:
        return []

    # Sort by confidence descending
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        # Remove detections with high IoU
        remaining = []
        for det in detections:
            if det["class_id"] != best["class_id"]:
                remaining.append(det)
                continue

            iou = calc_iou(best["bbox"], det["bbox"])
            if iou < iou_threshold:
                remaining.append(det)

        detections = remaining

    return keep

def calc_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 + area2 - inter == 0:
        return 0
    return inter / (area1 + area2 - inter)

if __name__ == "__main__":
    main()
