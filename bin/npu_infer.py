#!/usr/bin/env python3
"""
NPU Inference script for YOLOv5 RKNN models.
Reads JPEG image from stdin, outputs JSON detections.
"""
import sys
import json

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

def main():
    img_data = sys.stdin.buffer.read()
    if not img_data:
        print(json.dumps({"success": False, "error": "No image data"}))
        return

    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(path=MODEL_PATH)
        if ret != 0:
            print(json.dumps({"success": False, "error": f"Load RKNN failed: {ret}"}))
            return

        ret = rknn.inference(inputs=[img_data])
        if ret is None:
            print(json.dumps({"success": False, "error": "Inference failed"}))
            return

        outputs = ret[0] if isinstance(ret, list) else ret
        detections = []

        # Simple parsing: outputs shape (25200, 85)
        for i in range(len(outputs)):
            conf = float(outputs[i][4])
            if conf < CONF_THRESHOLD:
                continue

            best_class, best_score = 0, conf
            for c in range(80):
                score = float(outputs[i][5 + c]) * conf
                if score > best_score:
                    best_score = score
                    best_class = c

            if best_score < CONF_THRESHOLD:
                continue

            class_name = COCO_CLASSES[best_class] if 0 <= best_class < len(COCO_CLASSES) else "unknown"
            detections.append({
                "class_id": best_class,
                "class_name": class_name,
                "confidence": float(best_score),
                "bbox": [0, 0, 640, 640]  # Placeholder - full parsing needs anchor logic
            })

        print(json.dumps({"success": True, "detections": detections, "count": len(detections)}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main()
