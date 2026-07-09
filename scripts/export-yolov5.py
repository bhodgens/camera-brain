#!/usr/bin/env python3
"""
Export YOLOv5s to ONNX with proper detection heads intact.

Uses the ultralytics package (already has YOLOv5 code bundled).
"""

import torch
import onnx
import argparse
import os
from pathlib import Path


def export_yolov5(model_name='yolov5s', img_size=640, opset=12, output_dir='./models'):
    """
    Export YOLOv5 model to ONNX with detection heads preserved.
    """
    print(f"=== Exporting {model_name} to ONNX ===")
    print(f"Input size: {img_size}x{img_size}")
    print(f"ONNX opset: {opset}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model using ultralytics
    print(f"\nLoading {model_name} from ultralytics...")
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")  # Auto-downloads pretrained weights

    # CRITICAL: Set to eval mode
    model.model.eval()

    # Create dummy input
    dummy_input = torch.zeros(1, 3, img_size, img_size)

    # Export to ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    print(f"\nExporting to {onnx_path}...")

    torch.onnx.export(
        model.model,  # Export the underlying detection model
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
    )

    print(f"Export complete: {onnx_path}")

    # Verify ONNX model
    print("\n=== Verifying ONNX model ===")
    onnx_model = onnx.load(str(onnx_path))

    # Check graph inputs/outputs
    print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")

    # Check output shape
    for output in onnx_model.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"Output '{output.name}' shape: {shape}")

    # Test inference to verify outputs are non-zero
    print("\n=== Testing ONNX inference ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    test_input = torch.randn(1, 3, img_size, img_size).numpy()
    outputs = sess.run(None, {'images': test_input})

    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
        print(f"  Min: {out.min():.6f}, Max: {out.max():.6f}, Mean: {out.mean():.6f}")
        print(f"  Non-zero elements: {(out != 0).sum()} / {out.size}")

        # Check structure: if output has 85 channels, verify confidence positions
        if len(out.shape) == 3 and out.shape[2] == 85:
            sample = out[0, :10, :]
            conf_values = sample[:, 4]
            print(f"  Confidence values (idx 4) sample: {conf_values[:5]}")
            print(f"  Confidence non-zero: {(conf_values != 0).sum()} / {len(conf_values)}")

    print("\n=== Export Complete ===")
    print(f"Model saved to: {onnx_path}")
    print("\nNext steps:")
    print("1. Transfer ONNX to rock1: scp {onnx_path} rock1:/tmp/")
    print("2. Convert to RKNN using rknn-toolkit2 on rock1")
    print("3. Verify RKNN output has non-zero confidence at index 4")

    return str(onnx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOv5 to ONNX')
    parser.add_argument('--model', type=str, default='yolov5s',
                        help='YOLOv5 model variant')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Output directory')

    args = parser.parse_args()
    export_yolov5(args.model, args.img_size, args.opset, args.output_dir)
