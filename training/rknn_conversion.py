#!/usr/bin/env python3
"""
Convert ONNX models to RKNN format for RK3568 NPU deployment

Usage:
    python rknn_conversion.py --onnx vehicle_classifier.onnx --output vehicle_classifier.rknn
    python rknn_conversion.py --onnx person_attr.onnx --output person_attr.rknn --dataset /path/to/calibration

Requirements:
    - rknn-toolkit2 installed (pip install rknn-toolkit2)
    - ONNX model file
    - Calibration dataset (for INT8 quantization)
"""

import argparse
import os
import sys

try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 not installed")
    print("Install with: pip install rknn-toolkit2")
    sys.exit(1)


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    dataset_path: str = None,
    do_quantization: bool = True,
    target_platform: str = "rk3568"
):
    """
    Convert ONNX model to RKNN format.

    Args:
        onnx_path: Path to input ONNX file
        output_path: Path for output RKNN file
        dataset_path: Path to calibration dataset (txt file with image paths)
        do_quantization: Whether to apply INT8 quantization
        target_platform: Rockchip target platform
    """

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        sys.exit(1)

    print(f"=== RKNN Conversion ===")
    print(f"Input:  {onnx_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_platform}")
    print(f"Quantization: {'INT8' if do_quantization else 'FP16'}")
    print()

    # Initialize RKNN
    rknn = RKNN(verbose=True)

    # Configure
    print("Configuring RKNN...")
    config = rknn.config(
        target_platform=target_platform,
        optimization_level=3
    )

    # Load ONNX
    print(f"Loading ONNX model: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("ERROR: Failed to load ONNX model")
        sys.exit(1)

    # Build (with optional quantization)
    print("\nBuilding RKNN model...")
    if do_quantization and dataset_path:
        print(f"Using calibration dataset: {dataset_path}")
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_path
        )
    else:
        if do_quantization:
            print("WARNING: Quantization enabled but no dataset provided, using FP16")
        ret = rknn.build(
            do_quantization=False
        )

    if ret != 0:
        print("ERROR: Failed to build RKNN model")
        sys.exit(1)

    # Export
    print(f"\nExporting to: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("ERROR: Failed to export RKNN")
        sys.exit(1)

    print(f"\n✓ Conversion complete!")
    print(f"  RKNN file: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    # Inference test (optional)
    print("\n=== Inference Test ===")
    ret = rknn.init_runtime(target=target_platform, perf_debug=True)
    if ret != 0:
        print("WARNING: Could not init runtime (expected if not on RK3568)")
    else:
        # Would add inference test here with sample input
        print("Runtime initialized successfully")

    rknn.release()
    return output_path


def create_calibration_dataset_txt(image_dir: str, output_txt: str, max_samples: int = 100):
    """
    Create calibration dataset text file for RKNN quantization.

    Format: Each line is a path to an image file
    """
    import glob

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext), recursive=True))

    if not image_files:
        print(f"WARNING: No images found in {image_dir}")
        return None

    # Limit samples
    image_files = image_files[:max_samples]

    # Write text file
    with open(output_txt, 'w') as f:
        for img_path in image_files:
            f.write(img_path + '\n')

    print(f"Created calibration dataset: {output_txt} ({len(image_files)} images)")
    return output_txt


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to RKNN for RK3568')
    parser.add_argument('--onnx', required=True, help='Input ONNX file')
    parser.add_argument('--output', required=True, help='Output RKNN file')
    parser.add_argument('--dataset', help='Calibration dataset (txt or image dir)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 instead of INT8')
    parser.add_argument('--target', default='rk3568', help='Target platform (default: rk3568)')

    args = parser.parse_args()

    # Handle dataset path
    dataset_path = args.dataset
    if dataset_path and os.path.isdir(dataset_path):
        # Create calibration txt from directory
        dataset_path = create_calibration_dataset_txt(
            dataset_path,
            os.path.join(args.output.replace('.rknn', ''), '_calib.txt')
        )

    # Determine quantization
    do_quantization = not args.fp16

    # Convert
    convert_onnx_to_rknn(
        onnx_path=args.onnx,
        output_path=args.output,
        dataset_path=dataset_path,
        do_quantization=do_quantization,
        target_platform=args.target
    )


if __name__ == '__main__':
    main()
