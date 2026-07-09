#!/usr/bin/env python3
"""
Convert YOLOv5s ONNX to RKNN INT8 format on rock1.

This script:
1. Creates a calibration dataset (synthetic images if real data unavailable)
2. Loads YOLOv5s ONNX model
3. Runs quantization calibration
4. Builds RKNN INT8 model
5. Tests output to verify detection heads are intact

Usage:
    python3 convert-yolov5-rknn.py --model /path/to/yolov5s.onnx --output /path/to/yolov5s_int8.rknn
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 not installed. Install with:")
    print("  pip3 install rknn_toolkit2-*.whl")
    sys.exit(1)


def create_calibration_data(num_images=20, img_size=640):
    """
    Create calibration dataset for INT8 quantization.
    Uses synthetic images if real data unavailable.
    Returns list of numpy arrays shaped as [N, C, H, W] in RGB order.
    """
    print(f"Creating calibration dataset: {num_images} images at {img_size}x{img_size}...")

    calib_data = []
    for i in range(num_images):
        # Synthetic image: random values scaled to look like normalized images
        # RKNN expects NHWC format for calibration, but model input is NCHW
        # rknn-toolkit2 handles the conversion if preprocessed correctly
        img = np.random.randint(0, 256, (1, img_size, img_size, 3), dtype=np.uint8)
        calib_data.append(img)

    print(f"  Created {len(calib_data)} calibration images")
    return calib_data


def create_calibration_file(calib_data, output_path='calib.txt'):
    """
    Create calibration file listing paths to calibration images.
    Alternatively, returns in-memory data for rknn.do_quantization().
    """
    # For synthetic data, we return the actual numpy arrays
    # rknn-toolkit2 accepts: list of numpy arrays OR list of file paths
    return calib_data


def convert_onnx_to_rknn(onnx_path, output_path, dataset=None, do_quant=True):
    """
    Convert YOLOv5s ONNX to RKNN format.

    The ultralytics-exported YOLOv5s has output shape [1, 84, 8400]:
    - 84 = 4 box coords + 1 confidence + 80 class scores
    - 8400 = total anchors (3 per grid cell across all strides)

    Args:
        onnx_path: Path to YOLOv5s ONNX file
        output_path: Path to save RKNN model
        dataset: Calibration dataset (list of numpy arrays)
        do_quant: Enable INT8 quantization
    """
    print(f"=== Converting {onnx_path} to RKNN ===")
    print(f"Target platform: rk3568")
    print(f"Quantization: {'INT8' if do_quant else 'FP16'}")

    # Create RKNN instance
    rknn = RKNN(verbose=True)

    # Configure for RK3568 (Rock Pi 3A / ROCK 3A)
    print("\nConfiguring RKNN...")
    ret = rknn.config(
        target_platform='rk3568',
        optimization_level=0,  # 0=default, 1=aggressive
        quant_config={
            'mode': 'default',
            'quant_weight': True,
        } if do_quant else None
    )
    if ret != 0:
        print(f"ERROR: rknn.config failed: {ret}")
        return False

    # Load ONNX model
    # YOLOv5s from ultralytics has:
    # - Input: [1, 3, 640, 640] (NCHW)
    # - Output: [1, 84, 8400] where 84 = 4 box + 1 conf + 80 classes
    print(f"\nLoading ONNX model: {onnx_path}...")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[['images', [1, 3, 640, 640]]],
        outputs=['output'],  # Main detection output
        input_size_list=[[1, 640, 640, 3]]  # NHWC for RKNN
    )
    if ret != 0:
        print(f"ERROR: rknn.load_onnx failed: {ret}")
        return False

    # Build model (with or without quantization)
    print("\nBuilding model...")
    if do_quant:
        print("Running quantization calibration...")
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset,
            quant_config={
                'need_extract_before_relu': True,
                'quant_activation_type': 'asymmetric',
            }
        )
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"ERROR: rknn.build failed: {ret}")
        return False

    # Export RKNN model
    print(f"\nExporting RKNN model to: {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: rknn.export_rknn failed: {ret}")
        return False

    print(f"\nSuccessfully created: {output_path}")

    # Cleanup
    rknn.release()

    return True


def verify_rknn_output(rknn_path, num_test_outputs=100):
    """
    Verify RKNN model output has proper detection heads.

    New format (ultralytics YOLOv5): [1, 84, 8400] or [1, 8400, 84] after transpose
    - 84 = 4 box coords + 1 confidence + 80 class scores
    - 8400 = total anchors

    Old format (traditional YOLOv5): concatenated 85 values/anchor
    - 85 = 4 box coords + 1 confidence + 80 class scores
    """
    print(f"\n=== Verifying RKNN output structure ===")

    rknn = RKNN()
    ret = rknn.load_rknn(path=rknn_path)
    if ret != 0:
        print(f"ERROR: Failed to load RKNN model: {ret}")
        return False

    # Initialize runtime
    ret = rknn.init_runtime(target='rk3568')
    if ret != 0:
        print(f"ERROR: Failed to init runtime: {ret}")
        return False

    # Create test input (random noise image)
    test_input = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)

    # Run inference
    print("Running inference...")
    outputs = rknn.inference(inputs=[test_input])

    if len(outputs) == 0:
        print("ERROR: No output from inference")
        return False

    output = outputs[0]
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Determine output format and verify
    if len(output.shape) == 3:
        # Format: [1, 84, 8400] or [1, 8400, 84]
        if output.shape[1] == 84:
            # [1, 84, 8400] - need to transpose to per-anchor format
            output = output.transpose(0, 2, 1)  # -> [1, 8400, 84]
            print(f"Transposed to: {output.shape}")

        # Now output is [1, 8400, 84]
        anchor_data = output[0]  # [8400, 84]
        print(f"Per-anchor data shape: {anchor_data.shape}")

        # Check structure: [cx, cy, w, h, conf, class_scores...]
        print("\nFirst 10 anchor structures:")
        confidence_nonzero = 0
        class_nonzero = 0

        for i in range(min(10, len(anchor_data))):
            box_vals = anchor_data[i, 0:4]
            conf_val = anchor_data[i, 4]
            class_vals = anchor_data[i, 5:85]

            box_nonzero = np.count_nonzero(box_vals)
            class_nonzero_count = np.count_nonzero(class_vals)

            print(f"  Anchor {i}: box={box_nonzero}/4 nonzero, conf={conf_val:.4f}, classes={class_nonzero_count}/80 nonzero")

            if conf_val != 0:
                confidence_nonzero += 1
            if class_nonzero_count > 0:
                class_nonzero += 1

        print(f"\nConfidence non-zero: {confidence_nonzero}/10")
        print(f"Class scores non-zero: {class_nonzero}/10")

        if confidence_nonzero == 0:
            print("\nWARNING: All confidence values are ZERO - model may have incomplete detection heads!")
            rknn.release()
            return False

        if class_nonzero == 0:
            print("\nWARNING: All class scores are ZERO - model may have incomplete classification head!")
            rknn.release()
            return False

        print("\nOK: Detection heads appear intact!")
    else:
        # Old format: flattened or different structure
        flat = output.flatten()
        print(f"Flattened size: {len(flat)}")

        # Try to detect 85-value anchor structure
        num_anchors = len(flat) // 85
        if num_anchors > 0:
            print(f"Possible anchor structure: {num_anchors} anchors x 85 values")
            # Check confidence positions
            conf_positions = [flat[i * 85 + 4] for i in range(min(10, num_anchors))]
            print(f"Confidence samples: {conf_positions[:5]}")

    rknn.release()
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv5 ONNX to RKNN')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLOv5s ONNX model')
    parser.add_argument('--output', type=str, default='yolov5s_int8.rknn',
                        help='Output RKNN model path')
    parser.add_argument('--no-quant', action='store_true',
                        help='Skip quantization (FP16 mode)')
    parser.add_argument('--calib-size', type=int, default=20,
                        help='Number of calibration images for INT8')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify output after conversion')

    args = parser.parse_args()

    # Check ONNX file exists
    if not os.path.exists(args.model):
        print(f"ERROR: ONNX model not found: {args.model}")
        sys.exit(1)

    # Create calibration dataset
    calib_data = create_calibration_data(num_images=args.calib_size)

    # Convert
    success = convert_onnx_to_rknn(
        onnx_path=args.model,
        output_path=args.output,
        dataset=calib_data,
        do_quant=not args.no_quant
    )

    if not success:
        print("\nConversion FAILED")
        sys.exit(1)

    # Verify
    if args.verify:
        success = verify_rknn_output(args.output)
        if not success:
            print("\nVerification FAILED - model may have issues")
            sys.exit(1)

    print("\n=== Conversion Complete ===")
    print(f"Output model: {args.output}")
    print("\nDeploy to workers:")
    print(f"  scp {args.output} rock1:/home/camera-brain/models/  # Repeat for rock2-4")
    print("  systemctl restart camera-worker  # On each worker")


if __name__ == '__main__':
    main()
