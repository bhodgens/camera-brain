#!/usr/bin/env bash
# Camera Brain Docker Start Script
# Usage: ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Validation
# ============================================================================
validate_env() {
    if [[ ! -f .env ]]; then
        log_error ".env file not found"
        log_error "Copy .env.example to .env and configure:"
        log_error "  cp .env.example .env"
        log_error "  # Edit .env with your settings"
        exit 1
    fi

    # Load and validate
    # shellcheck source=/dev/null
    source .env

    if [[ "${MODEL_DIR:-}" == "/path/to/your/models" ]] || [[ -z "${MODEL_DIR:-}" ]]; then
        log_error "MODEL_DIR not configured in .env"
        log_error "Set MODEL_DIR to the path containing your GGUF model files"
        exit 1
    fi

    if [[ ! -d "$MODEL_DIR" ]]; then
        log_error "Model directory does not exist: $MODEL_DIR"
        exit 1
    fi

    # Check for required model files
    local llama_model="${LLAMA_MODEL:-LFM2.5-VL-1.6B.Q8_0.gguf}"
    local mmproj="${LLAMA_MMPROJ:-LFM2.5-VL-1.6B.mmproj-f16.gguf}"

    if [[ ! -f "$MODEL_DIR/$llama_model" ]]; then
        log_error "Model file not found: $MODEL_DIR/$llama_model"
        exit 1
    fi

    if [[ ! -f "$MODEL_DIR/$mmproj" ]]; then
        log_error "MMProj file not found: $MODEL_DIR/$mmproj"
        exit 1
    fi

    log_success "Environment validated"
}

validate_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose plugin not found"
        log_error "Install with: apt install docker-compose-plugin"
        exit 1
    fi

    log_success "Docker validated"
}

# ============================================================================
# Setup
# ============================================================================
setup_directories() {
    log_info "Creating data directories..."
    mkdir -p "${DATA_DIR:-./data}"/{postgres,nats,grafana,storage}
    log_success "Directories created"
}

# ============================================================================
# Start Services
# ============================================================================
start_services() {
    log_info "Starting services..."
    docker compose up -d

    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check service status
    echo ""
    log_info "Service status:"
    docker compose ps

    echo ""
    log_success "Camera Brain is running!"
    echo ""
    echo "Access points:"
    echo "  Gateway:     http://localhost:${GATEWAY_PORT:-8080}"
    echo "  VLM API:     http://localhost:${VLM_PORT:-8081}"
    echo "  Query API:   http://localhost:${QUERY_PORT:-8082}"
    echo "  llama.cpp:   http://localhost:8888"
    echo "  Grafana:     http://localhost:3000 (admin/admin)"
    echo "  PostgreSQL:  localhost:${DB_PORT:-5432}"
    echo "  NATS:        localhost:4222"
    echo ""
    echo "View logs:"
    echo "  docker compose logs -f [service-name]"
    echo ""
    echo "Stop services:"
    echo "  docker compose down"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo "=============================================="
    echo "  Camera Brain - Docker Start"
    echo "=============================================="
    echo ""

    validate_env
    validate_docker
    setup_directories
    start_services
}

main "$@"
