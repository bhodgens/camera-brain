#!/bin/bash
# Deploy camera-worker to rock1-5 nodes
# Run this script from rock0 to deploy the worker binary to all NPU nodes

set -e

WORKER_NODES=("rock1" "rock2" "rock3" "rock4" "rock5")
BINARY_URL="http://rock0:9999/camera-worker"
SERVICE_NAME="camera-brain-worker"

echo "=== Camera Worker Deployment Script ==="
echo "This script deploys the camera-worker binary to rock1-5 nodes."
echo ""
echo "IMPORTANT: SSH keys must be authorized on all worker nodes."
echo ""

# Copy SSH key to all nodes (requires password once per node)
for node in "${WORKER_NODES[@]}"; do
    echo "Setting up SSH access to $node..."
    ssh-copy-id -i ~/.ssh/id_ed25519.pub "caimlas@$node" 2>/dev/null || {
        echo "Failed to copy SSH key to $node. Please run manually:"
        echo "  ssh-copy-id caimlas@$node"
    }
done

echo ""
echo "Deploying worker binary to nodes..."

for node in "${WORKER_NODES[@]}"; do
    echo ""
    echo "--- Deploying to $node ---"

    # Download binary
    if ssh "caimlas@$node" "curl -sL '$BINARY_URL' -o /home/caimlas/camera-worker-new && chmod +x /home/caimlas/camera-worker-new"; then
        echo "  Downloaded binary successfully"

        # Stop existing service
        ssh "caimlas@$node" "sudo systemctl stop $SERVICE_NAME 2>/dev/null || true"

        # Backup old binary and install new one
        ssh "caimlas@$node" "
            if [ -f /home/camera-brain/bin/camera-worker ]; then
                cp /home/camera-brain/bin/camera-worker /home/camera-brain/bin/camera-worker.bak
            fi
            mv /home/caimlas/camera-worker-new /home/camera-brain/bin/camera-worker
            chmod +x /home/camera-brain/bin/camera-worker
        "
        echo "  Binary installed"

        # Restart service
        if ssh "caimlas@$node" "sudo systemctl restart $SERVICE_NAME"; then
            echo "  Service restarted"
        else
            echo "  Failed to restart service - may need manual intervention"
        fi

        # Check status
        ssh "caimlas@$node" "systemctl status $SERVICE_NAME --no-pager | head -10" || true
    else
        echo "  FAILED to download binary to $node"
    fi
done

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To check worker status on any node:"
echo "  ssh caimlas@rockN 'systemctl status camera-brain-worker'"
echo ""
echo "To view worker logs:"
echo "  ssh caimlas@rockN 'journalctl -fu camera-brain-worker'"
