# Option 1: Minimal Viable Distribution - Implementation Plan

## Goal
Create a basic installer that makes camera-brain deployable to similar ARM64/Linux systems with minimal manual work.

## Scope
- Single `install.sh` script as the entry point
- Configuration via `camera-brain.env` template
- Systemd service templates
- Pre and post-install validation

## File Structure

```
camera-brain/
├── deploy/
│   ├── install.sh                      # Main installer entry point
│   ├── camera-brain.env.template       # Configuration template
│   └── systemd/
│       ├── llama-server.service.template
│       ├── vlm-processor.service.template
│       ├── query-engine.service.template
│       └── camera-brain-gateway.service.template
├── cmd/
│   ├── vlm-processor/
│   │   └── main.go
│   ├── query-engine/
│   │   └── main.go
│   └── camera-brain-gateway/
│       └── main.go
├── storage/
│   └── schema.sql                      # PostgreSQL schema
├── Makefile                            # Build automation
└── README.md                           # Quick start
```

## install.sh Checklist

- [ ] Shebang and strict mode (`set -euo pipefail`)
- [ ] Color output helpers
- [ ] Platform detection (OS, arch, systemd availability)
- [ ] Dependency check (Go 1.21+, cmake, git, pkg-config)
- [ ] Configuration directory setup (`/etc/camera-brain/`)
- [ ] Data directory setup (`/var/lib/camera-brain/`)
- [ ] Log directory setup (`/var/log/camera-brain/`)
- [ ] Clone and build llama.cpp with LFM2 support
- [ ] Build Go services
- [ ] Install binaries to `/usr/local/bin/`
- [ ] Generate config from template
- [ ] Install systemd services from templates
- [ ] Initialize database (create user, database, run schema)
- [ ] Enable and start services
- [ ] Post-install status check
- [ ] Output next steps

## Configuration Template (camera-brain.env.template)

```bash
# Camera Brain Configuration
# Copy to /etc/camera-brain/camera-brain.env and customize

# Installation paths
CB_DATA_DIR=/var/lib/camera-brain
CB_MODEL_DIR=${CB_DATA_DIR}/models
CB_STORAGE_DIR=${CB_DATA_DIR}/storage

# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=camera_brain
DB_USER=camera_brain
DB_PASSWORD=<generate or user-provided>

# NATS configuration
NATS_URL=nats://localhost:4222

# LLM Server configuration
LLAMA_SERVER_URL=http://localhost:8888
LLAMA_MODEL=${CB_MODEL_DIR}/LFM2.5-VL-1.6B.Q8_0.gguf
LLAMA_MMProj=${CB_MODEL_DIR}/LFM2.5-VL-1.6B.mmproj-f16.gguf

# Service ports
VLM_PROCESSOR_PORT=8081
QUERY_ENGINE_PORT=8082
GATEWAY_PORT=8080

# Worker configuration
WORKER_ID=auto  # auto = generate from hostname
WORKER_HEARTBEAT_INTERVAL=30
```

## Systemd Service Template Pattern

```ini
[Unit]
Description=Camera Brain {{SERVICE_NAME}}
After=network.target {{DEPENDS_ON}}

[Service]
Type=simple
User=camera-brain
WorkingDirectory={{BIN_DIR}}
EnvironmentFile=/etc/camera-brain/camera-brain.env
Environment="PORT={{PORT}}"
ExecStart={{BIN_DIR}}/{{BINARY_NAME}}
Restart=always
RestartSec=5
StandardOutput=append:/var/log/camera-brain/{{SERVICE_NAME}}.log
StandardError=append:/var/log/camera-brain/{{SERVICE_NAME}}.log

[Install]
WantedBy=multi-user.target
```

## Validation

### Pre-install Checks
```bash
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is required but not installed"
        return 1
    fi
}

check_command go
check_command cmake
check_command git
check_command pkg-config
```

### Post-install Verification
```bash
verify_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        echo "✓ $service is running"
    else
        echo "✗ $service is not running"
        return 1
    fi
}

verify_service llama-server
verify_service vlm-processor
verify_service query-engine
```

## Error Handling

- Trap errors and provide helpful messages
- Log all output to `/var/log/camera-brain/install.log`
- Support `--dry-run` flag for testing
- Support `--uninstall` for cleanup

## Next Steps After Implementation

1. Test on rock1-5 (identical hardware)
2. Test on a fresh Debian/Ubuntu VM
3. Document known issues in README
4. Create v0.1.0 GitHub release
