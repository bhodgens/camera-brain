#!/bin/bash
# Deploy chat application to rock0 with port 80 access via iptables redirect

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

# Setup iptables redirect from port 80 to 8083
echo "Setting up port redirect (80 -> 8083)..."
ssh rock0 "sudo iptables -t nat -F PREROUTING 2>/dev/null || true"
ssh rock0 "sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8083"
ssh rock0 "sudo iptables-save | sudo grep -q '80.*8083' || sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null"

# Reload and start
echo "Starting service..."
ssh rock0 "sudo systemctl daemon-reload && sudo systemctl enable camera-brain-chat && sudo systemctl start camera-brain-chat"

# Wait for startup
sleep 2

# Check status
echo ""
echo "=== Service Status ==="
ssh rock0 "sudo systemctl status camera-brain-chat --no-pager -l"

echo ""
echo "=== Health Check ==="
ssh rock0 "curl -s http://localhost/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost/health"

echo ""
echo "✓ Chat application deployed successfully!"
echo "  Web interface: http://rock0/ (redirects to port 8083)"
echo "  Direct: http://rock0:8083"
echo "  Logs: ssh rock0 'sudo journalctl -u camera-brain-chat -f'"
