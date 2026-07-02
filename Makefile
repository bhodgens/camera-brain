# Camera Brain Makefile
# Build, test, and deployment automation

.PHONY: help build clean test run-docker stop-docker clean-docker build-docker install

# ============================================================================
# Variables
# ============================================================================
GO := go
GOFLAGS := -v
LDFLAGS := -ldflags="-w -s"
CMD_DIR := cmd
BIN_DIR := bin

# Services
SERVICES := vlm-processor query-engine gateway

# ============================================================================
# Help
# ============================================================================
help:
	@echo "Camera Brain Build System"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build all Go services"
	@echo "  clean          Remove build artifacts"
	@echo "  test           Run tests"
	@echo "  run-docker     Start Docker Compose services"
	@echo "  stop-docker    Stop Docker Compose services"
	@echo "  clean-docker   Remove Docker volumes and containers"
	@echo "  build-docker   Build Docker images"
	@echo "  install        Run installer (requires sudo)"
	@echo ""

# ============================================================================
# Go Builds
# ============================================================================
build: $(SERVICES)

$(SERVICES):
	@echo "Building $@..."
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$@ $(CMD_DIR)/$@

clean:
	@echo "Cleaning..."
	rm -rf $(BIN_DIR)
	rm -rf docker/build
	go clean -cache

test:
	$(GO) test -v ./...

# ============================================================================
# Docker
# ============================================================================
build-docker:
	@echo "Building Docker images..."
	cd docker && docker compose build

run-docker:
	@echo "Starting services..."
	cd docker && ./start.sh

stop-docker:
	@echo "Stopping services..."
	cd docker && docker compose down

clean-docker:
	@echo "Removing volumes and containers..."
	cd docker && docker compose down -v

# ============================================================================
# Installation
# ============================================================================
install:
	@echo "Running installer..."
	sudo ./deploy/install.sh

install-dry-run:
	@echo "Running installer (dry run)..."
	./deploy/install.sh --dry-run

# ============================================================================
# Development
# ============================================================================
fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

lint: fmt vet

# ============================================================================
# Model Management
# ============================================================================
download-models:
	@echo "Model downloads must be done manually due to size"
	@echo "Place these files in your MODEL_DIR:"
	@echo "  - LFM2.5-VL-1.6B.Q8_0.gguf"
	@echo "  - LFM2.5-VL-1.6B.mmproj-f16.gguf"
	@echo "  - yolov5s_int8.rknn (for NPU detection)"
