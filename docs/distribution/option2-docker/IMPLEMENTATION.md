# Option 2: Container-First Approach - Implementation Plan

## Goal
Create a Docker Compose-based distribution that runs camera-brain on any system with Docker, abstracting away dependencies and configuration.

## Scope
- Single `docker-compose.yml` defining all services
- `.env` file for configuration
- `start.sh` entry point with validation
- Named volumes for persistent data
- Bind mounts for models (user-managed)

## File Structure

```
camera-brain/
├── docker/
│   ├── docker-compose.yml              # Main compose file
│   ├── .env.example                    # Environment template
│   ├── start.sh                        # Entry point with validation
│   └── Dockerfile*                     # Custom images if needed
├── cmd/
│   ├── vlm-processor/
│   │   └── main.go
│   ├── query-engine/
│   │   └── main.go
│   └── camera-brain-gateway/
│       └── main.go
├── storage/
│   └── schema.sql
└── Makefile                            # build-docker, run-docker targets
```

## Docker Compose Services

### 1. llama-server
```yaml
llama-server:
  image: ghcr.io/ggerganov/llama.cpp:server
  ports:
    - "8888:8888"
  volumes:
    - ${MODEL_DIR}:/models:ro
    - llama-data:/data
  environment:
    - MODEL=/models/${LLAMA_MODEL}
    - MMPROJ=/models/${LLAMA_MMPROJ}
  command: >
    --model /models/${LLAMA_MODEL}
    --mmproj /models/${LLAMA_MMPROJ}
    --host 0.0.0.0
    --port 8888
    --ctx-size 4096
    --threads ${CPU_THREADS}
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia  # Optional GPU passthrough
            count: 1
            capabilities: [gpu]
```

### 2. vlm-processor
```yaml
vlm-processor:
  build:
    context: .
    dockerfile: docker/Dockerfile.vlm-processor
  ports:
    - "${VLM_PORT:-8081}:8081"
  environment:
    - LLAMA_SERVER_URL=http://llama-server:8888
    - PORT=8081
    - DB_HOST=postgres
  depends_on:
    - llama-server
    - postgres
  restart: unless-stopped
```

### 3. query-engine
```yaml
query-engine:
  build:
    context: .
    dockerfile: docker/Dockerfile.query-engine
  ports:
    - "${QUERY_PORT:-8082}:8082"
  environment:
    - LLAMA_SERVER_URL=http://llama-server:8888
    - DB_HOST=postgres
    - DB_USER=${DB_USER}
    - DB_PASSWORD=${DB_PASSWORD}
  depends_on:
    - llama-server
    - postgres
  restart: unless-stopped
```

### 4. postgres (TimescaleDB)
```yaml
postgres:
  image: timescale/timescaledb-ha:pg16
  ports:
    - "${DB_PORT:-5432}:5432"
  volumes:
    - postgres-data:/var/lib/postgresql/data
    - ./storage/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
  environment:
    - POSTGRES_USER=${DB_USER}
    - POSTGRES_PASSWORD=${DB_PASSWORD}
    - POSTGRES_DB=${DB_NAME}
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
    interval: 5s
    timeout: 5s
    retries: 5
```

### 5. nats
```yaml
nats:
  image: nats:2.10-alpine
  ports:
    - "4222:4222"
    - "8222:8222"
  command: ["-js"]
  volumes:
    - nats-data:/data
  restart: unless-stopped
```

### 6. Grafana (optional)
```yaml
grafana:
  image: grafana/grafana:10.4.0
  ports:
    - "3000:3000"
  volumes:
    - grafana-data:/var/lib/grafana
    - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    - ./grafana/datasources:/etc/grafana/provisioning/datasources
  environment:
    - GF_SECURITY_ADMIN_USER=admin
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
  depends_on:
    - postgres
```

## Volumes

```yaml
volumes:
  postgres-data:
  nats-data:
  llama-data:
  grafana-data:
```

## .env.example

```bash
# Camera Brain Docker Configuration

# Database
DB_USER=camera_brain
DB_PASSWORD=change_me_in_production
DB_NAME=camera_brain
DB_PORT=5432

# Model paths (bind mount from host)
MODEL_DIR=/path/to/models
LLAMA_MODEL=LFM2.5-VL-1.6B.Q8_0.gguf
LLAMA_MMPROJ=LFM2.5-VL-1.6B.mmproj-f16.gguf

# CPU/Memory tuning
CPU_THREADS=4

# Ports
VLM_PORT=8081
QUERY_PORT=8082
GATEWAY_PORT=8080

# Optional: Grafana
GRAFANA_PASSWORD=admin
```

## start.sh Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# Validate .env exists
if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found"
    echo "Copy .env.example to .env and configure"
    exit 1
fi

# Validate MODEL_DIR exists and has models
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory ${MODEL_DIR} does not exist"
    exit 1
fi

# Check Docker is running
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running"
    exit 1
fi

# Create directories
mkdir -p data/{postgres,nats,grafana}

# Start services
docker compose up -d

# Wait for postgres to be ready
echo "Waiting for database..."
sleep 10

# Check service health
docker compose ps
```

## Dockerfile Patterns

### Go Services (multi-stage build)
```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
RUN apk add --no-cache git pkgconfig
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/myapp ./cmd/myapp

# Runtime stage
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /out/myapp /usr/local/bin/
CMD ["myapp"]
```

## Makefile Targets

```makefile
.PHONY: build-docker run-docker stop-docker clean-docker

build-docker:
	docker compose build

run-docker:
	./docker/start.sh

stop-docker:
	docker compose down

clean-docker:
	docker compose down -v
```

## Validation

- [ ] All services start with `docker compose up -d`
- [ ] Services restart automatically after crash
- [ ] Data persists across restarts
- [ ] VLM processor can reach llama-server
- [ ] Query engine can reach postgres
- [ ] Services log to stdout/stderr (visible via `docker compose logs`)

## Next Steps After Implementation

1. Test on x86_64 desktop
2. Test on ARM64 SBC (different from rock0)
3. Add GPU passthrough documentation for NVIDIA
4. Create GitHub Container Registry workflow
5. Publish pre-built images to GHCR
