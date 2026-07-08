#!/usr/bin/env python3
"""
Convert ONNX model to RKNN INT8 format for RK3568.

Usage:
    python convert-onnx-to-rknn.py /path/to/model.onnx [--output /path/to/output.rknn]

Requires:
    pip install rknn-toolkit2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rknn.api import RKNN


def convert(onnx_path, output_path=None, quantize=True, dataset_path=None):
    """Convert ONNX to RKNN."""

    onnx_path = Path(onnx_path)
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = onnx_path.parent / f"{onnx_path.stem}_int8.rknn"

    print(f"=== Converting {onnx_path.name} to RKNN ===")
    print(f"Output: {output_path}")
    print(f"Quantization: {'INT8' if quantize else 'FP16'}")

    rknn = RKNN()

    # Configure for RK3568
    print("\nConfiguring RKNN...")
    rknn.config(
        target_platform='rk3568',
        optimization_level=3,
    )

    # Load ONNX
    print(f"Loading ONNX: {onnx_path}...")
    ret = rknn.load_onnx(
        model=str(onnx_path),
        inputs=[{'name': 'images', 'n_dims': 4, 'shape': [1, 3, 640, 640]}]
    )
    if ret != 0:
        print(f"ERROR: load_onnx failed: {ret}")
        return None

    # Check graph info
    print("\nModel info:")
    inputs = rknn.graph_info.get('inputs', [])
    outputs = rknn.graph_info.get('outputs', [])
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {outputs}")

    # Build
    if quantize:
        print("\nBuilding with INT8 quantization...")
        if dataset_path:
            print(f"Using calibration dataset: {dataset_path}")
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            # Use random data for calibration
            print("Using random calibration data...")
            ret = rknn.build(do_quantization=True)
    else:
        print("\nBuilding without quantization (FP16)...")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"ERROR: build failed: {ret}")
        return None

    # Export
    print(f"\nExporting to {output_path}...")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        print(f"ERROR: export_rknn failed: {ret}")
        return None

    print(f"\n✓ RKNN model saved to: {output_path}")

    # Test inference
    print("\n=== Testing RKNN inference ===")
    ret = rknn.init_runtime(target='rk3568')
    if ret != 0:
        print(f"WARNING: init_runtime failed: {ret}")
    else:
        # Check output format
        test_input = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
        outputs = rknn.inference(inputs=[test_input])

        print(f"Output tensors: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  [{i}] shape={out.shape}, dtype={out.dtype}")
            print(f"      min={out.min():.6f}, max={out.max():.6f}")

        # Verify ultralytics format
        if len(outputs) == 1 and outputs[0].shape == (1, 84, 8400):
            out = outputs[0]
            conf_col = out[:, 4, :].reshape(-1)
            print(f"\n✓ Ultralytics format confirmed")
            print(f"  Confidence: min={conf_col.min():.6f}, max={conf_col.max():.6f}")
            print(f"  Non-zero: {(conf_col > 0).sum()} / {len(conf_col)}")
        else:
            print(f"\nWARNING: Unexpected format {[o.shape for o in outputs]}")

    rknn.release()
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to RKNN for RK3568')
    parser.add_argument('onnx_path', type=str, help='Path to ONNX model')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output RKNN path (default: same as ONNX with .rknn ext)')
    parser.add_argument('--no-quant', action='store_true',
                        help='Skip INT8 quantization (use FP16)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset for INT8 calibration')

    args = parser.parse_args()

    output = convert(
        args.onnx_path,
        output_path=args.output,
        quantize=not args.no_quant,
        dataset_path=args.dataset
    )

    if output:
        print(f"\n✓ Conversion complete: {output}")
        return 0
    else:
        print("\n✗ Conversion failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
