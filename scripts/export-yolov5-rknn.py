#!/usr/bin/env python3
"""
Export YOLOv5s to RKNN format with correct output heads for RK3568 NPU.

This script:
1. Loads YOLOv5s from ultralytics (auto-downloads pretrained weights)
2. Exports to ONNX with the detection head consolidated into single output [1, 84, 8400]
3. Converts to RKNN INT8 format for RK3568
4. Saves the RKNN model ready for deployment

Requirements:
    pip install ultralytics onnx rknn-toolkit2 opencv-python

Usage:
    python export-yolov5-rknn.py --output ./models
"""

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from ultralytics import YOLO


def export_onnx(model_name='yolov5s', img_size=640, opset=11, output_dir='./models'):
    """
    Export YOLOv5 to ONNX with ultralytics format [1, 84, 8400].

    The ultralytics YOLOv5 format outputs:
    - [cx, cy, w, h, conf, class_scores...] per anchor
    - 8400 anchors = 80x80/4 + 40x40/2 + 20x20 = 3200 + 800 + 400 per 3 anchors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exporting {model_name} to ONNX ===")
    print(f"Input size: {img_size}x{img_size}")
    print(f"ONNX opset: {opset}")

    # Load model
    print(f"\nLoading {model_name}.pt (auto-downloads if not present)...")
    model = YOLO(f"{model_name}.pt")
    model.model.eval()

    # Export using ultralytics built-in exporter
    onnx_path = output_dir / f"{model_name}_rknn.onnx"
    print(f"\nExporting to {onnx_path}...")

    # Use ultralytics export which handles the detection head properly
    model.export(
        format='onnx',
        imgsz=img_size,
        opset=opset,
        simplify=True,  # Fuse BatchNorm into Conv
        dynamic=False,  # Static batch size
    )

    # The exported model is in models/yolov5s.onnx
    # Move to our target path
    src = Path(f"{model_name}.onnx")
    if src.exists():
        shutil.move(str(src), str(onnx_path))

    # Verify ONNX
    print(f"\n=== Verifying ONNX model ===")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")

    for output in onnx_model.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"Output shape: {shape}")

    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        print("\n=== Testing ONNX inference ===")
        sess = ort.InferenceSession(str(onnx_path))

        test_input = np.random.randint(0, 255, (1, 3, img_size, img_size)).astype(np.float32) / 255.0
        outputs = sess.run(None, {'images': test_input})

        print(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}")
            print(f"    min={out.min():.6f}, max={out.max():.6f}, nonzero={(out != 0).sum()}")

        # Check for ultralytics format
        if len(outputs) == 1 and outputs[0].shape == (1, 84, 8400):
            print("\n✓ Ultralytics format confirmed: [1, 84, 8400]")
            out = outputs[0]
            # Check confidence column (index 4)
            conf_col = out[:, 4, :].reshape(-1)
            print(f"  Confidence values: min={conf_col.min():.6f}, max={conf_col.max():.6f}")
            print(f"  Non-zero confidence: {(conf_col > 0).sum()} / {len(conf_col)}")
        else:
            print(f"\nWARNING: Unexpected output format: {[o.shape for o in outputs]}")

    except Exception as e:
        print(f"ONNX Runtime test failed: {e}")

    print(f"\n✓ ONNX export complete: {onnx_path}")
    return str(onnx_path)


def convert_to_rknn(onnx_path, dataset_path=None, output_dir='./models'):
    """
    Convert ONNX to RKNN INT8 format for RK3568.

    Requires rknn-toolkit2 installed (only works on aarch64 Linux).
    """
    from rknn.api import RKNN

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Converting {onnx_path} to RKNN ===")

    rknn = RKNN()

    # Configure
    print("Configuring RKNN...")
    rknn.config(
        target_platform='rk3568',
        optimization_level=3,
        quant_config={
            'enabled': True,
            'type': 'w8a8sym',  # Weight-Activation INT8 symmetric
            'export_quant_config': False,
        }
    )

    # Load ONNX
    print(f"Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[{'name': 'images', 'n_dims': 4, 'shape': [1, 3, 640, 640]}]
    )
    if ret != 0:
        print(f"ERROR: load_onnx failed: {ret}")
        return None

    # Build (quantization)
    print("Building RKNN model (INT8 quantization)...")
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print(f"ERROR: build failed: {ret}")
        return None

    # Export
    rknn_path = output_dir / f"{Path(onnx_path).stem}_int8.rknn"
    print(f"Exporting to {rknn_path}...")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print(f"ERROR: export_rknn failed: {ret}")
        return None

    print(f"\n✓ RKNN export complete: {rknn_path}")

    # Test inference
    print("\n=== Testing RKNN inference ===")
    ret = rknn.init_runtime(target='rk3568')
    if ret != 0:
        print(f"WARNING: init_runtime failed: {ret}")
    else:
        test_input = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
        outputs = rknn.inference(inputs=[test_input])

        print(f"RKNN outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}")

    rknn.release()
    return str(rknn_path)


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv5 to RKNN for RK3568')
    parser.add_argument('--model', type=str, default='yolov5s',
                        help='YOLOv5 model variant (yolov5s, yolov5m, etc.)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset for INT8 calibration')
    parser.add_argument('--onnx-only', action='store_true',
                        help='Only export ONNX, skip RKNN conversion')

    args = parser.parse_args()

    # Export ONNX
    onnx_path = export_onnx(
        model_name=args.model,
        img_size=args.img_size,
        opset=args.opset,
        output_dir=args.output_dir
    )

    if args.onnx_only:
        print("\n✓ ONNX-only export complete")
        print(f"Transfer to RK3568 device and run:")
        print(f"  scp {onnx_path} rock@rock1:/tmp/")
        return 0

    # Convert to RKNN (requires aarch64)
    if sys.platform != 'linux' or 'aarch64' not in sys.platform:
        print("\nNOTE: RKNN conversion requires aarch64 Linux")
        print(f"Transfer ONNX to rock1 and convert there:")
        print(f"  scp {onnx_path} rock@rock1:/tmp/")
        print(f"  ssh rock@rock1 'cd /home/camera-brain && python3 scripts/convert-onnx-to-rknn.py /tmp/{Path(onnx_path).name}'")
        return 0

    rknn_path = convert_to_rknn(onnx_path, dataset_path=args.dataset, output_dir=args.output_dir)

    if rknn_path:
        print(f"\n✓ Full export pipeline complete:")
        print(f"  ONNX: {onnx_path}")
        print(f"  RKNN: {rknn_path}")

    return 0 if rknn_path else 1


if __name__ == '__main__':
    main()
