#!/bin/bash
#
# YOLOv5s Re-export and Deploy Script
#
# This script:
# 1. Exports YOLOv5s from PyTorch to ONNX (requires PyTorch installed)
# 2. Converts ONNX to RKNN INT8 (requires rknn-toolkit2 on rock1)
# 3. Deploys to all workers (rock1-4)
# 4. Verifies detections appear
#
# Usage: ./reexport-yolov5.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
OUTPUT_RKNN="${MODELS_DIR}/yolov5s_int8.rknn"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2
    exit 1
}

# Step 1: Export YOLOv5s to ONNX
export_onnx() {
    log "=== Step 1: Export YOLOv5s to ONNX ==="

    if ! command -v python3 &> /dev/null; then
        error "python3 not found"
    fi

    # Check if PyTorch is available
    if ! python3 -c "import torch" 2>/dev/null; then
        log "PyTorch not available locally. Trying rock0..."

        # Try to run export on rock0 (if it has PyTorch)
        if ssh rock0 "python3 -c 'import torch' 2>/dev/null" 2>/dev/null; then
            log "Running export on rock0..."
            ssh rock0 << 'EOF'
set -e
mkdir -p /tmp/yolov5-export
cd /tmp/yolov5-export
python3 -c "
import torch
import onnx

print('Loading YOLOv5s...')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

print('Exporting to ONNX...')
dummy = torch.zeros(1, 3, 640, 640)
torch.onnx.export(model, dummy, '/tmp/yolov5s.onnx',
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['images'],
                  output_names=['output'])
print('ONNX export complete: /tmp/yolov5s.onnx')
"
EOF
            log "Copying ONNX from rock0..."
            scp rock0:/tmp/yolov5s.onnx "${MODELS_DIR}/yolov5s.onnx"
        else
            error "PyTorch not available on rock0 or local machine. Install with: pip install torch onnx"
        fi
    else
        # Run locally
        log "Running export locally..."
        python3 "${SCRIPT_DIR}/export-yolov5.py" \
            --model yolov5s \
            --img-size 640 \
            --opset 12 \
            --output-dir "${MODELS_DIR}"
    fi

    log "ONNX export complete"
}

# Step 2: Convert ONNX to RKNN on rock1
convert_rknn() {
    log "=== Step 2: Convert ONNX to RKNN (on rock1) ==="

    ONNX_PATH="${MODELS_DIR}/yolov5s.onnx"

    if [ ! -f "${ONNX_PATH}" ]; then
        error "ONNX model not found: ${ONNX_PATH}"
    fi

    log "Copying ONNX to rock1..."
    scp "${ONNX_PATH}" rock1:/tmp/yolov5s.onnx

    log "Copying conversion script to rock1..."
    scp "${SCRIPT_DIR}/convert-yolov5-rknn.py" rock1:/tmp/

    log "Running conversion on rock1..."
    ssh rock1 << 'EOF'
set -e
cd /tmp
echo "Starting conversion..."
python3 /tmp/convert-yolov5-rknn.py \
    --model /tmp/yolov5s.onnx \
    --output /tmp/yolov5s_int8.rknn \
    --calib-size 20 \
    --verify
EOF

    log "Copying RKNN model back..."
    mkdir -p "${MODELS_DIR}"
    scp rock1:/tmp/yolov5s_int8.rknn "${OUTPUT_RKNN}"

    log "RKNN conversion complete: ${OUTPUT_RKNN}"
}

# Step 3: Deploy to all workers
deploy_to_workers() {
    log "=== Step 3: Deploy to workers (rock1-4) ==="

    for i in 1 2 3 4; do
        log "Deploying to rock${i}..."
        scp "${OUTPUT_RKNN}" "rock${i}:/home/camera-brain/models/yolov5s_int8.rknn"

        # Update worker config if needed
        ssh "rock${i}" << 'EOF'
# Kill any running workers
pkill -f camera-worker || true
pkill -f "go run.*worker" || true
sleep 1
EOF
        log "  Deployed to rock${i}"
    done

    log "All workers updated"
}

# Step 4: Test detections
test_detections() {
    log "=== Step 4: Test detections on rock1 ==="

    log "Starting worker on rock1..."
    ssh rock1 << 'EOF'
cd /home/camera-brain/bin
# Start worker in background, capture output
nohup ./camera-worker > /tmp/worker.log 2>&1 &
echo "Worker started, PID: $!"
sleep 5
EOF

    log "Waiting for detections (10 seconds)..."
    sleep 10

    log "Checking worker logs on rock1..."
    ssh rock1 "tail -50 /tmp/worker.log | grep -E 'detected|NPU|detections'" || log "No detections yet - may need more time"

    log "Check worker logs manually: ssh rock1 'tail -f /tmp/worker.log'"
}

# Main
main() {
    log "=== YOLOv5s Re-export and Deploy ==="
    log "This script will:"
    log "  1. Export YOLOv5s to ONNX"
    log "  2. Convert to RKNN INT8 on rock1"
    log "  3. Deploy to workers rock1-4"
    log "  4. Test detections"
    log ""

    mkdir -p "${MODELS_DIR}"

    export_onnx
    convert_rknn
    deploy_to_workers
    test_detections

    log "=== Complete ==="
    log "Model deployed: ${OUTPUT_RKNN}"
    log "Check worker logs for detections: ssh rock1 'tail -f /tmp/worker.log'"
}

main "$@"
