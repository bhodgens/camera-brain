#!/usr/bin/env python3
"""
RKNN YOLOv5 inference script for use with Go worker.
Reads image from stdin as raw RGB bytes, outputs detections as JSON.

Usage:
    echo "<base64_encoded_image>" | python3 npu_infer.py
    or
    python3 npu_infer.py --image <path> --output json
"""

import sys
import json
import base64
import numpy as np
from io import BytesIO

try:
    from rknn.api import RKNN
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

MODEL_PATH = "/home/camera-brain/models/yolov5s_fp16.rknn"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant",
}

_rknn = None

def init_rknn():
    global _rknn
    if _rknn is None:
        _rknn = RKNN()
        ret = _rknn.load_rknn(MODEL_PATH)
        if ret != 0:
            print(f"ERROR: load_rknn failed: {ret}", file=sys.stderr)
            return None
        ret = _rknn.init_runtime(target='rk3568')
        if ret != 0:
            print(f"ERROR: init_runtime failed: {ret}", file=sys.stderr)
            return None
    return _rknn

def preprocess(img_bytes):
    """Decode image and resize to INPUT_SIZE x INPUT_SIZE."""
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

def parse_output(outputs):
    """
    Parse YOLOv5 output tensor (ultralytics format) to detections.

    Model outputs single tensor: (1, 84, 8400)
    Format per anchor: [cx, cy, w, h, conf, class_scores...]
    - Columns 0-3: box coordinates (center x, center y, width, height)
    - Column 4: objectness confidence
    - Columns 5-84: 80 COCO class scores

    Final confidence = objectness * max(class_scores)
    """
    if len(outputs) != 1:
        print(f"ERROR: Expected 1 output, got {len(outputs)}", file=sys.stderr)
        return []

    # Reshape to (8400, 84)
    out = outputs[0].reshape(84, 8400).T

    # Extract components
    boxes = out[:, 0:4]  # (8400, 4) - cx, cy, w, h
    objectness = out[:, 4]  # (8400,) - obj conf
    class_scores = out[:, 5:85]  # (8400, 80) - class scores

    # Final confidence = objectness * best class score
    best_class_idx = class_scores.argmax(axis=1)  # (8400,)
    best_class_score = class_scores.max(axis=1)  # (8400,)
    confidence = objectness * best_class_score  # (8400,)

    detections = []
    for i in range(len(boxes)):
        if confidence[i] < CONF_THRESHOLD:
            continue

        cx, cy, w, h = boxes[i]
        best_class = int(best_class_idx[i])
        best_score = float(confidence[i])

        # Convert to corner coordinates
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)

        # Clamp to image bounds
        x1 = max(0, min(INPUT_SIZE, x1))
        y1 = max(0, min(INPUT_SIZE, y1))
        x2 = max(0, min(INPUT_SIZE, x2))
        y2 = max(0, min(INPUT_SIZE, y2))

        detections.append({
            "class_id": best_class,
            "class_name": COCO_CLASSES.get(best_class, "unknown"),
            "confidence": best_score,
            "bbox": [x1, y1, x2, y2]
        })

    # Simple NMS
    detections = nms(detections, NMS_THRESHOLD)
    return detections

def nms(dets, iou_thresh):
    """Non-maximum suppression."""
    if not dets:
        return []

    # Sort by confidence descending
    dets.sort(key=lambda x: x["confidence"], reverse=True)

    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best["bbox"], d["bbox"]) <= iou_thresh]

    return keep

def iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 + area2 - inter <= 0:
        return 0
    return inter / (area1 + area2 - inter)

def run_inference(image_bytes):
    """Run RKNN inference on image bytes."""
    rknn = init_rknn()
    if rknn is None:
        return None

    img = preprocess(image_bytes)
    outputs = rknn.inference(inputs=[img])

    if not outputs or len(outputs) == 0:
        return []

    return parse_output(outputs)

def main():
    if len(sys.argv) < 2:
        # Read from stdin (for subprocess call)
        img_bytes = sys.stdin.buffer.read()
    elif sys.argv[1] == "--image" and len(sys.argv) > 2:
        with open(sys.argv[2], "rb") as f:
            img_bytes = f.read()
    elif sys.argv[1] == "--base64" and len(sys.argv) > 2:
        img_bytes = base64.b64decode(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [--image <path>|--base64 <data>]", file=sys.stderr)
        sys.exit(1)

    detections = run_inference(img_bytes)

    result = {
        "success": detections is not None,
        "detections": detections if detections else [],
        "count": len(detections) if detections else 0
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
