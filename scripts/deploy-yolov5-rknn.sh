#!/bin/bash
#
# Deploy YOLOv5s RKNN model to all workers
#
# This script:
# 1. Copies ONNX model to rock1
# 2. Runs RKNN conversion on rock1
# 3. Copies RKNN model back
# 4. Deploys to all workers (rock1-4)
# 5. Restarts workers and verifies detections
#
# Usage: ./deploy-yolov5-rknn.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"
ONNX_PATH="${MODELS_DIR}/yolov5s.onnx"

# Worker configuration
WORKERS=("rock1.local" "rock2.local" "rock3.local" "rock4.local")
SSH_USER="rock"
WORK_DIR="/home/camera-brain"
MODEL_DIR="${WORK_DIR}/models"
BIN_DIR="${WORK_DIR}/bin"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2
    exit 1
}

# Check prerequisites
check_prereqs() {
    log "Checking prerequisites..."

    # Check if ONNX model exists
    if [ ! -f "${ONNX_PATH}" ] && [ ! -f "${ONNX_PATH}.data" ]; then
        error "ONNX model not found: ${ONNX_PATH}"
        log "Run: python3 scripts/export-yolov5.py first"
        exit 1
    fi

    if [ ! -f "${SCRIPT_DIR}/convert-yolov5-rknn-simple.py" ]; then
        error "Conversion script not found: ${SCRIPT_DIR}/convert-yolov5-rknn-simple.py"
        exit 1
    fi

    # Test SSH to rock1
    log "Testing SSH to rock1.local..."
    if ! ssh -o ConnectTimeout=5 "${SSH_USER}@rock1.local" "echo 'SSH OK'" >/dev/null 2>&1; then
        error "Cannot SSH to rock1.local"
        exit 1
    fi

    log "Prerequisites OK"
}

# Step 1: Transfer files to rock1
transfer_to_rock1() {
    log "=== Step 1: Transferring files to rock1 ==="

    # Copy ONNX model
    log "Copying ONNX model..."
    scp "${ONNX_PATH}" "${SSH_USER}@rock1.local:/tmp/yolov5s.onnx"
    scp "${ONNX_PATH}.data" "${SSH_USER}@rock1.local:/tmp/yolov5s.onnx.data"

    # Copy conversion script
    log "Copying conversion script..."
    scp "${SCRIPT_DIR}/convert-yolov5-rknn-simple.py" "${SSH_USER}@rock1.local:/tmp/convert_yolov5.py"

    log "Files transferred successfully"
}

# Step 2: Run conversion on rock1
convert_on_rock1() {
    log "=== Step 2: Converting ONNX to RKNN on rock1 ==="

    ssh "${SSH_USER}@rock1.local" << 'REMOTE_EOF'
set -e
cd /tmp

log() {
    echo "[RKNN] $*"
}

log "Checking rknn-toolkit2..."
if ! python3 -c "from rknn.api import RKNN" 2>/dev/null; then
    log "ERROR: rknn-toolkit2 not installed"
    exit 1
fi

log "Starting conversion..."
python3 /tmp/convert_yolov5.py \
    --model /tmp/yolov5s.onnx \
    --output /tmp/yolov5s_int8.rknn \
    --dataset /home/rock/dataset.txt \
    --verify

if [ ! -f /tmp/yolov5s_int8.rknn ]; then
    log "ERROR: Conversion failed - RKNN model not created"
    exit 1
fi

log "Conversion successful!"
ls -lh /tmp/yolov5s_int8.rknn
REMOTE_EOF

    log "Conversion complete"
}

# Step 3: Copy RKNN model back to Mac (optional backup)
copy_rknn_back() {
    log "=== Step 3: Copying RKNN model backup to Mac ==="

    scp "${SSH_USER}@rock1.local:/tmp/yolov5s_int8.rknn" "${MODELS_DIR}/"

    log "RKNN model backup saved to: ${MODELS_DIR}/yolov5s_int8.rknn"
}

# Step 4: Deploy to all workers
deploy_to_workers() {
    log "=== Step 4: Deploying to workers ==="

    for worker in "${WORKERS[@]}"; do
        log "Deploying to ${worker}..."

        # Create destination directory if needed
        ssh "${SSH_USER}@${worker}" "mkdir -p ${MODEL_DIR}" 2>/dev/null || true

        # Copy RKNN model
        scp "${MODELS_DIR}/yolov5s_int8.rknn" "${SSH_USER}@${worker}:${MODEL_DIR}/yolov5s_int8.rknn.new"

        # Replace old model
        ssh "${SSH_USER}@${worker}" "mv ${MODEL_DIR}/yolov5s_int8.rknn ${MODEL_DIR}/yolov5s_int8.rknn.bak 2>/dev/null || true"
        ssh "${SSH_USER}@${worker}" "mv ${MODEL_DIR}/yolov5s_int8.rknn.new ${MODEL_DIR}/yolov5s_int8.rknn"

        log "  Deployed to ${worker}"
    done

    log "All workers updated"
}

# Step 5: Restart workers
restart_workers() {
    log "=== Step 5: Restarting workers ==="

    for worker in "${WORKERS[@]}"; do
        log "Restarting worker on ${worker}..."

        ssh "${SSH_USER}@${worker}" << 'REMOTE_EOF'
# Kill any running worker processes
pkill -9 -f camera-worker || true

sleep 1

# Start fresh worker
cd /home/camera-brain/bin
nohup ./camera-worker > /tmp/worker.log 2>&1 &
echo "Worker started with PID: $!"

sleep 3
REMOTE_EOF

        log "  Worker restarted on ${worker}"
    done
}

# Step 6: Verify detections
verify_detections() {
    log "=== Step 6: Verifying detections ==="

    log "Waiting 5 seconds for workers to initialize..."
    sleep 5

    for worker in "${WORKERS[@]}"; do
        log "Checking ${worker} logs..."

        ssh "${SSH_USER}@${worker}" << 'REMOTE_EOF'
echo "=== Last 50 lines of worker log ==="
tail -50 /tmp/worker.log | grep -E 'detected|NPU|detections|final|stride' | tail -20

echo ""
echo "=== Detection count ==="
grep -c "detected.*objects" /tmp/worker.log 2>/dev/null || echo "0"
REMOTE_EOF
    done
}

# Main
main() {
    log "=== YOLOv5s RKNN Deployment Script ==="
    log ""
    log "Configuration:"
    log "  SSH User: ${SSH_USER}"
    log "  Workers: ${WORKERS[*]}"
    log "  Work Dir: ${WORK_DIR}"
    log "  Model Dir: ${MODEL_DIR}"
    log ""

    check_prereqs
    transfer_to_rock1
    convert_on_rock1
    copy_rknn_back
    deploy_to_workers
    restart_workers
    verify_detections

    log ""
    log "=== Deployment Complete ==="
    log ""
    log "RKNN model deployed to: ${MODEL_DIR}/yolov5s_int8.rknn"
    log ""
    log "To monitor workers:"
    log "  ssh ${SSH_USER}@rock1.local 'tail -f /tmp/worker.log'"
    log ""
}

main "$@"
