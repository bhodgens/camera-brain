#!/usr/bin/env python3
"""
Vehicle Type Classifier (Pre-trained / Heuristic-based)

This script classifies vehicles into types: sedan, SUV, truck, van, pickup, bus, motorcycle, bicycle

Two modes:
1. HEURISTIC MODE (default): Uses size ratios + color - no dependencies, works immediately
2. PRE-TRAINED MODE: Uses EfficientNet trained on Stanford Cars - higher accuracy

Usage:
    # Heuristic mode (works immediately)
    python detect.py --crop car_crop.jpg

    # With pre-trained model (higher accuracy, requires model file)
    python detect.py --crop car_crop.jpg --model efficientnet_vehicle.onnx
"""

import argparse
import json
import sys
import os

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Vehicle class definitions
VEHICLE_CLASSES = ['sedan', 'SUV', 'truck', 'van', 'pickup', 'bus', 'motorcycle', 'bicycle']

# Typical size ratios for different vehicle types (width/height in crop)
# These are approximate and used for heuristic classification
VEHICLE_SIZE_RATIOS = {
    'sedan': (1.5, 2.0),      # Wide, low profile
    'SUV': (1.2, 1.6),        # More square
    'truck': (1.4, 2.2),      # Long, varied height
    'van': (1.3, 1.8),        # Boxy
    'pickup': (1.5, 2.0),     # Like sedan but taller
    'bus': (1.8, 3.0),        # Very long
    'motorcycle': (1.5, 3.0), # Long, narrow
    'bicycle': (1.2, 2.5)     # Side view: long, front: narrow
}

# Color associations (optional heuristic)
COLOR_VEHICLE_ASSOCIATIONS = {
    'sedan': ['white', 'black', 'silver', 'gray', 'red', 'blue'],
    'SUV': ['white', 'black', 'silver', 'gray', 'red', 'blue'],
    'truck': ['white', 'black', 'silver', 'red'],
    'van': ['white', 'silver'],
    'pickup': ['white', 'black', 'silver', 'red'],
    'bus': ['yellow', 'white', 'blue'],
    'motorcycle': ['black', 'red', 'blue'],
    'bicycle': ['black', 'red', 'blue', 'green']
}


def get_dominant_color(image: np.ndarray) -> str:
    """
    Get dominant color from image using HSV histogram.
    Returns color name or 'unknown'.
    """
    if len(image.shape) < 3:
        return 'unknown'

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        'red': ([0, 50, 50], [10, 255, 255]),
        'orange': ([11, 50, 50], [20, 255, 255]),
        'yellow': ([21, 50, 50], [30, 255, 255]),
        'green': ([31, 50, 50], [70, 255, 255]),
        'blue': ([71, 50, 50], [130, 255, 255]),
        'purple': ([131, 50, 50], [160, 255, 255]),
        'white': ([0, 0, 200], [20, 20, 255]),
        'gray': ([0, 0, 50], [180, 20, 200]),
        'black': ([0, 0, 0], [180, 20, 50]),
        'silver': ([0, 10, 180], [20, 50, 230]),
    }

    dominant = 'unknown'
    max_pixels = 0
    total_pixels = image.shape[0] * image.shape[1]

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = np.sum(mask > 0)
        if count > max_pixels:
            max_pixels = count
            dominant = color

    # Only return color if it covers significant portion
    if max_pixels < total_pixels * 0.1:
        return 'unknown'

    return dominant


def estimate_vehicle_type_heuristic(image: np.ndarray) -> dict:
    """
    Estimate vehicle type using heuristics (size ratio + color).
    Quick but less accurate than ML model.
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Get dominant color
    color = get_dominant_color(image)

    # Score each vehicle type based on aspect ratio match
    scores = {}
    for vehicle_type, (ratio_min, ratio_max) in VEHICLE_SIZE_RATIOS.items():
        if ratio_min <= aspect_ratio <= ratio_max:
            scores[vehicle_type] = 0.5  # Base score for ratio match
            # Bonus for color match
            if color in COLOR_VEHICLE_ASSOCIATIONS.get(vehicle_type, []):
                scores[vehicle_type] += 0.3
        else:
            # Penalty for ratio mismatch
            ratio_mid = (ratio_min + ratio_max) / 2
            penalty = abs(aspect_ratio - ratio_mid) * 0.2
            scores[vehicle_type] = max(0, 0.2 - penalty)

    # Normalize scores to probabilities
    total = sum(scores.values())
    if total > 0:
        probs = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        probs = {k: round(1 / len(VEHICLE_CLASSES), 3) for k in VEHICLE_CLASSES}

    # Get prediction
    predicted_type = max(scores, key=scores.get)
    confidence = scores[predicted_type]

    return {
        'success': True,
        'vehicle_type': predicted_type,
        'confidence': round(min(confidence * 2, 0.95), 3),  # Scale confidence
        'all_scores': probs,
        'color': color,
        'aspect_ratio': round(aspect_ratio, 2),
        'method': 'heuristic'
    }


def classify_vehicle_ml(image: np.ndarray, model_path: str) -> dict:
    """
    Classify vehicle type using pre-trained ONNX model.
    Higher accuracy than heuristics.
    """
    if not HAS_ONNX:
        print("ERROR: ONNX runtime not installed", file=sys.stderr)
        return None

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        return None

    # Load model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Preprocess image
    img_resized = cv2.resize(image, (224, 224))
    img_float = img_resized.astype(np.float32)

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float / 255.0 - mean) / std

    # Rearrange to NCHW format
    img_nchw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_nchw, 0)

    # Run inference
    outputs = session.run(None, {input_name: img_batch})
    logits = outputs[0][0]

    # Convert to probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)

    predicted_idx = np.argmax(probs)
    predicted_type = VEHICLE_CLASSES[predicted_idx]

    scores_dict = {cls: round(float(p), 3) for cls, p in zip(VEHICLE_CLASSES, probs)}

    return {
        'success': True,
        'vehicle_type': predicted_type,
        'confidence': round(float(probs[predicted_idx]), 3),
        'all_scores': scores_dict,
        'method': 'ml'
    }


def detect_vehicle_type(crop_path: str, model_path: str = None) -> dict:
    """
    Detect vehicle type from crop image.
    Falls back to heuristic if no model provided.
    """
    if not HAS_CV:
        return {
            'success': False,
            'error': 'OpenCV not installed',
            'vehicle_type': None
        }

    image = cv2.imread(crop_path)
    if image is None:
        return {
            'success': False,
            'error': f'Could not read image: {crop_path}',
            'vehicle_type': None
        }

    # Use ML model if available, otherwise heuristic
    if model_path:
        result = classify_vehicle_ml(image, model_path)
        if result:
            return result

    # Fall back to heuristic
    return estimate_vehicle_type_heuristic(image)


def main():
    parser = argparse.ArgumentParser(description='Vehicle type detection')
    parser.add_argument('--crop', required=True, help='Path to vehicle crop image')
    parser.add_argument('--model', help='Path to ONNX model (optional, uses heuristics if not provided)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    result = detect_vehicle_type(args.crop, args.model)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result['success']:
            print(f"Vehicle Type: {result['vehicle_type']} (confidence: {result['confidence']:.1%})")
            if result.get('color'):
                print(f"Color: {result['color']}")
            if args.verbose and result.get('all_scores'):
                print(f"All scores: {result['all_scores']}")
            print(f"Method: {result.get('method', 'unknown')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == '__main__':
    main()
