#!/usr/bin/env python3
"""
Convert YOLOv5s ONNX to RKNN INT8 format.
Simplified script for rknn-toolkit2 v2.3.2 API.

Uses existing calibration dataset if available.

Usage:
    python3 convert-yolov5-rknn-simple.py --model yolov5s.onnx --output yolov5s_int8.rknn
"""

import sys
import os
import argparse

try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 not installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='yolov5s_int8.rknn', help='Output RKNN path')
    parser.add_argument('--dataset', type=str, default='/home/rock/dataset.txt', help='Calibration dataset file')
    parser.add_argument('--verify', action='store_true', default=True, help='Verify output')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Check for existing calibration dataset
    if not os.path.exists(args.dataset):
        print(f"ERROR: Calibration dataset not found: {args.dataset}")
        print("Please create calibration images (JPEG format) and a dataset.txt file")
        sys.exit(1)

    # Count calibration images
    with open(args.dataset, 'r') as f:
        num_calib = sum(1 for line in f if line.strip())
    print(f"Using calibration dataset: {args.dataset} ({num_calib} images)")

    print(f"=== Converting {args.model} to RKNN ===")
    print(f"Target: rk3568, INT8 quantization")

    # Create RKNN instance
    rknn = RKNN(verbose=False)

    # Configure
    print("Configuring RKNN...")
    rknn.config(target_platform='rk3568')

    # Load ONNX
    print(f"Loading ONNX: {args.model}...")
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        print(f"ERROR: load_onnx failed: {ret}")
        sys.exit(1)

    # Build with INT8 quantization
    print("Building RKNN model (INT8)...")
    ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if ret != 0:
        print(f"ERROR: build failed: {ret}")
        sys.exit(1)

    # Export
    print(f"Exporting to {args.output}...")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"ERROR: export failed: {ret}")
        sys.exit(1)

    print(f"Created: {args.output}")

    # Verify if requested
    if args.verify:
        print("\n=== Verifying output ===")
        rknn2 = RKNN()
        ret = rknn2.load_rknn(path=args.output)
        if ret != 0:
            print(f"ERROR: load_rknn failed: {ret}")
            sys.exit(1)

        ret = rknn2.init_runtime(target='rk3568')
        if ret != 0:
            print(f"ERROR: init_runtime failed: {ret}")
            sys.exit(1)

        # Test inference
        import numpy as np
        test_input = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)
        outputs = rknn2.inference(inputs=[test_input])

        if len(outputs) > 0:
            output = outputs[0]
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            print(f"Non-zero elements: {np.count_nonzero(output)}/{output.size}")
            print(f"Min: {output.min():.4f}, Max: {output.max():.4f}")

            # Reshape and check anchor structure for [1, 84, 8400] format
            if len(output.shape) == 3:
                if output.shape[1] == 84:
                    output = output.transpose(0, 2, 1)  # -> [1, 8400, 84]

                if len(output.shape) == 3 and output.shape[1] == 8400 and output.shape[2] == 84:
                    anchor_data = output[0]  # [8400, 84]
                    conf_vals = anchor_data[:, 4]
                    class_vals = anchor_data[:, 5:85]

                    conf_nz = np.count_nonzero(conf_vals)
                    class_nz = np.count_nonzero(np.any(class_vals != 0, axis=1))

                    print(f"\nAnchor structure verification (84 = 4 box + 1 conf + 80 class):")
                    print(f"  Confidence non-zero: {conf_nz}/8400 ({100*conf_nz/8400:.1f}%)")
                    print(f"  Anchors with class scores: {class_nz}/8400 ({100*class_nz/8400:.1f}%)")

                    # Sample first 5 anchors
                    print("\n  First 5 anchors:")
                    for i in range(5):
                        box = anchor_data[i, 0:4]
                        conf = anchor_data[i, 4]
                        max_class = np.max(anchor_data[i, 5:85])
                        print(f"    [{i}] box=[{box[0]:.1f}, {box[1]:.1f}, ...], conf={conf:.4f}, max_class={max_class:.4f}")

                    if conf_nz > 0:
                        print("\nOK: Detection heads intact!")
                    else:
                        print("\nWARNING: All confidence values are zero!")

        rknn2.release()

    rknn.release()
    print("\n=== Conversion Complete ===")
    print(f"\nModel saved to: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
