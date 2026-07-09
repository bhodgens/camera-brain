#!/bin/bash
# Deploy chat application to rock0 on port 80

set -e

echo "=== Deploying Camera Brain Chat to rock0 ==="

# Create Python virtual environment
ssh rock0 "cd /home/camera-brain && python3 -m venv chat-env || true"

# Install dependencies
echo "Installing Python dependencies..."
ssh rock0 "cd /home/camera-brain && ./chat-env/bin/pip install -q flask requests"

# Copy application
echo "Copying application files..."
scp -r chat/ rock0:/home/camera-brain/

# Stop existing service if running
echo "Stopping existing service..."
ssh rock0 "sudo systemctl stop camera-brain-chat 2>/dev/null || true"

# Install systemd service
echo "Installing systemd service..."
scp deploy/camera-brain-chat.service rock0:/tmp/camera-brain-chat.service
ssh rock0 "sudo mv /tmp/camera-brain-chat.service /etc/systemd/system/camera-brain-chat.service"

# Reload and start
echo "Starting service..."
ssh rock0 "sudo systemctl daemon-reload && sudo systemctl enable camera-brain-chat && sudo systemctl start camera-brain-chat"

# Wait for startup
sleep 2

# Grant CAP_NET_BIND_SERVICE capability to Python binary (allows binding port 80)
echo "Setting capabilities for port 80..."
ssh rock0 "sudo setcap 'cap_net_bind_service=+ep' /home/camera-brain/chat-env/bin/python3"

# Restart service to pick up port 80
echo "Restarting service on port 80..."
ssh rock0 "sudo systemctl restart camera-brain-chat"

# Check status
echo ""
echo "=== Service Status ==="
ssh rock0 "sudo systemctl status camera-brain-chat --no-pager -l"

echo ""
echo "=== Health Check ==="
ssh rock0 "curl -s http://localhost/health | python3 -m json.tool"

echo ""
echo "✓ Chat application deployed successfully!"
echo "  Web interface: http://rock0/"
echo "  Logs: ssh rock0 'sudo journalctl -u camera-brain-chat -f'"
