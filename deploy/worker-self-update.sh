#!/bin/bash
# Camera Worker Self-Update Script
# Run this on each worker node (rock1-5) to update to the latest binary
#
# Usage: curl -sL http://rock0:9999/worker-self-update.sh | sudo bash

set -e

BINARY_URL="http://rock0:9999/camera-worker"
TARGET="/home/camera-brain/bin/camera-worker"
SERVICE="camera-brain-worker"
BACKUP="${TARGET}.bak.$(date +%Y%m%d-%H%M%S)"

echo "=== Camera Worker Self-Update ==="
echo "Downloading binary from: $BINARY_URL"

# Download new binary
curl -sL "$BINARY_URL" -o /tmp/camera-worker.new
if [ $? -ne 0 ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

# Verify it's an executable
chmod +x /tmp/camera-worker.new
if ! file /tmp/camera-worker.new | grep -q "ELF"; then
    echo "ERROR: Downloaded file is not a valid binary!"
    rm /tmp/camera-worker.new
    exit 1
fi

echo "Downloaded: $(ls -la /tmp/camera-worker.new)"

# Stop existing service
echo "Stopping $SERVICE service..."
systemctl stop "$SERVICE" 2>/dev/null || true

# Backup existing binary
if [ -f "$TARGET" ]; then
    echo "Backing up existing binary to: $BACKUP"
    cp "$TARGET" "$BACKUP"
fi

# Install new binary
echo "Installing new binary..."
mv /tmp/camera-worker.new "$TARGET"
chmod +x "$TARGET"

# Verify installation
if ! "$TARGET" --help >/dev/null 2>&1; then
    echo "WARNING: Binary verification failed (may need manual check)"
fi

# Restart service
echo "Starting $SERVICE service..."
systemctl restart "$SERVICE"

# Show status
echo ""
echo "=== Service Status ==="
systemctl status "$SERVICE" --no-pager | head -15

echo ""
echo "=== Update Complete ==="
echo "Backup saved to: $BACKUP"
echo ""
echo "To verify NPU detection is working:"
echo "  journalctl -fu $SERVICE"
echo ""
echo "Look for: 'Listening for assignments' and detection logs"
