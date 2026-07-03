#!/usr/bin/env bash
# Camera Brain Installer - Minimal Viable Distribution
# Usage: ./install.sh [--dry-run] [--uninstall]
set -euo pipefail

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log_error "Installation failed with exit code $rc. Manual cleanup may be needed."
    fi
    exit $rc
}
trap cleanup EXIT

# ============================================================================
# Configuration
# ============================================================================
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
CONFIG_DIR="/etc/camera-brain"
DATA_DIR="/var/lib/camera-brain"
LOG_DIR="/var/log/camera-brain"
BIN_DIR="${INSTALL_PREFIX}/bin"
LLAMA_DIR="/opt/llama.cpp"
CB_USER="camera-brain"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Dry run mode
DRY_RUN=false
UNINSTALL=false

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_or_echo() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would execute: $*"
    else
        "$@"
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --prefix=*)
            INSTALL_PREFIX="${1#*=}"
            shift
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--uninstall] [--prefix=/path]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-install Checks
# ============================================================================
check_command() {
    local cmd=$1
    local package=$2
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$cmd is required but not installed. Install with: apt install $package"
        return 1
    fi
    log_success "Found: $cmd"
}

pre_install_checks() {
    log_info "Running pre-install checks..."

    check_command "go" "golang-go" || return 1
    check_command "cmake" "cmake" || return 1
    check_command "git" "git" || return 1
    check_command "pkg-config" "pkg-config" || return 1

    # Check Go version (need 1.21+)
    local go_version
    go_version=$(go version | awk '{print $3}' | sed 's/go//')
    if [[ "$(printf '%s\n' "1.21" "$go_version" | sort -V | head -n1)" != "1.21" ]]; then
        log_error "Go 1.21 or higher required, found: $go_version"
        return 1
    fi
    log_success "Go version: $go_version"
}

# ============================================================================
# Directory Setup
# ============================================================================
setup_directories() {
    log_info "Creating directories..."

    run_or_echo mkdir -p "$CONFIG_DIR"
    run_or_echo mkdir -p "$DATA_DIR/models"
    run_or_echo mkdir -p "$DATA_DIR/storage"
    run_or_echo mkdir -p "$LOG_DIR"
    run_or_echo mkdir -p "$BIN_DIR"
    run_or_echo mkdir -p "$LLAMA_DIR"

    # Set permissions
    if [[ "$DRY_RUN" == "false" ]]; then
        chown -R "$CB_USER:$CB_USER" "$DATA_DIR" "$LOG_DIR" 2>/dev/null || true
        chmod 755 "$DATA_DIR" "$LOG_DIR"
        # Config dir contains DB_PASSWORD — restrict to owner only.
        chmod 700 "$CONFIG_DIR"
        chown "$CB_USER:$CB_USER" "$CONFIG_DIR" 2>/dev/null || true
    fi

    log_success "Directories created"
}

# ============================================================================
# Configuration
# ============================================================================
generate_config() {
    local config_file="$CONFIG_DIR/camera-brain.env"

    log_info "Generating configuration..."

    if [[ -f "$config_file" ]]; then
        log_warning "Configuration already exists at $config_file"
        log_warning "Backing up to ${config_file}.bak"
        run_or_echo cp "$config_file" "${config_file}.bak"
    fi

    # Get script directory for template location
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local template="$script_dir/camera-brain.env.template"

    if [[ -f "$template" ]]; then
        log_info "Using configuration template: $template"
        run_or_echo cp "$template" "$config_file"

        # Generate random password if installing
        if [[ "$DRY_RUN" == "false" ]]; then
            local random_pass
            random_pass=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c20)
            if [[ -z "$random_pass" ]]; then
                log_error "Failed to generate random password"
                exit 1
            fi
            sed -i "s/DB_PASSWORD=change_me_in_production/DB_PASSWORD=$random_pass/" "$config_file"
            log_success "Generated random database password"
        fi
    else
        log_warning "Template not found, creating default config"
        if [[ "$DRY_RUN" == "false" ]]; then
            cat > "$config_file" << 'EOF'
# Camera Brain Configuration
CB_DATA_DIR=/var/lib/camera-brain
CB_MODEL_DIR=${CB_DATA_DIR}/models
DB_HOST=localhost
DB_PORT=5432
DB_NAME=camera_brain
DB_USER=camera_brain
DB_PASSWORD=change_me
NATS_URL=nats://localhost:4222
LLAMA_SERVER_URL=http://localhost:8888
LLAMA_TEXT_SERVER_URL=http://localhost:8889
VLM_PROCESSOR_PORT=8081
QUERY_ENGINE_PORT=8082
GATEWAY_PORT=8080
CPU_THREADS=4
EOF
        fi
    fi

    # Lock down the env file — it contains DB_PASSWORD.
    if [[ "$DRY_RUN" == "false" ]]; then
        chmod 600 "$config_file"
        chown root:root "$config_file"
    fi

    log_success "Configuration created at $config_file"
}

# ============================================================================
# Build llama.cpp
# ============================================================================
build_llama_cpp() {
    log_info "Building llama.cpp..."

    if [[ -d "$LLAMA_DIR" ]] && [[ -f "$LLAMA_DIR/build/bin/llama-server" ]]; then
        log_info "llama.cpp already built, skipping..."
        return 0
    fi

    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        log_info "Cloning llama.cpp..."
        run_or_echo git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    else
        log_info "llama.cpp directory already exists, skipping clone"
    fi

    log_info "Building llama.cpp with -j$(nproc)..."
    run_or_echo cmake -B "$LLAMA_DIR/build" -DLLAMA_BLAS=OFF "$LLAMA_DIR"
    run_or_echo cmake --build "$LLAMA_DIR/build" -j"$(nproc)"

    # Create symlink in bin directory
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$LLAMA_DIR/build/bin/llama-server" "$BIN_DIR/llama-server"
    fi

    log_success "llama.cpp built successfully"
}

# ============================================================================
# Build Go Services
# ============================================================================
build_go_services() {
    log_info "Building Go services..."

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"

    # Check if Go source exists
    if [[ ! -d "$project_root/cmd" ]]; then
        log_error "Go source not found at $project_root/cmd"
        log_error "Make sure you cloned the full repository"
        return 1
    fi

    cd "$project_root"

    # Download dependencies
    run_or_echo go mod download

    # Build services — target names (hyphenated) differ from cmd/ dir names.
    # Keep this mapping in sync with Makefile CMD_DIRS.
    local targets=("vlm-processor" "query-engine" "gateway")
    local dirs=("vlmprocessor" "queryengine" "gateway")
    for i in "${!targets[@]}"; do
        local service="${targets[$i]}"
        local cmd_subdir="${dirs[$i]}"
        if [[ -d "$project_root/cmd/$cmd_subdir" ]]; then
            log_info "Building $service (cmd/$cmd_subdir)..."
            run_or_echo go build -o "$BIN_DIR/$service" "$project_root/cmd/$cmd_subdir"
            if ! [[ -x "$BIN_DIR/$service" ]]; then
                log_error "Build verification failed: $BIN_DIR/$service does not exist"
                return 1
            fi
            log_success "Built: $BIN_DIR/$service"
        else
            log_warning "Service source not found: cmd/$cmd_subdir"
        fi
    done
}

# ============================================================================
# Build cbrain CLI Tool
# ============================================================================
build_cbrain() {
    log_info "Building cbrain CLI tool..."

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"

    # Check if Go source exists
    if [[ ! -d "$project_root/cmd/cbrain" ]]; then
        log_warning "cbrain source not found at $project_root/cmd/cbrain, skipping..."
        return 0
    fi

    cd "$project_root"

    # Build the binary
    run_or_echo go build -o "$BIN_DIR/cbrain" ./cmd/cbrain/

    # Verify binary was produced
    if ! [[ -x "$BIN_DIR/cbrain" ]]; then
        log_error "Build verification failed: $BIN_DIR/cbrain does not exist or is not executable"
        return 1
    fi

    if [[ "$DRY_RUN" == "false" ]]; then
        run_or_echo chmod 755 "$BIN_DIR/cbrain"
        # Copy to /usr/local/bin only if INSTALL_PREFIX is /usr/local
        if [[ "$INSTALL_PREFIX" == "/usr/local" ]]; then
            run_or_echo cp "$BIN_DIR/cbrain" /usr/local/bin/
        fi
        log_success "Built: $BIN_DIR/cbrain"
    fi
}

# ============================================================================
# Install Systemd Services
# ============================================================================
install_systemd_services() {
    log_info "Installing systemd services..."

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local template_dir="$script_dir/systemd"

    # Load environment for variable substitution
    local config_file="$CONFIG_DIR/camera-brain.env"
    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
    fi

    # Service definitions: (template_name, output_name, binary_name)
    local services=(
        "llama-server.service.template:llama-server.service:llama-server"
        "vlm-processor.service.template:vlm-processor.service:vlm-processor"
        "query-engine.service.template:query-engine.service:query-engine"
        "camera-brain-gateway.service.template:camera-brain-gateway.service:gateway"
    )

    for service_def in "${services[@]}"; do
        IFS=':' read -r template_file output_file binary_name <<< "$service_def"
        local template_path="$template_dir/$template_file"

        if [[ ! -f "$template_path" ]]; then
            log_warning "Template not found: $template_path"
            continue
        fi

        log_info "Installing $output_file..."

        # Substitute template variables
        local service_content
        service_content=$(cat "$template_path")
        service_content="${service_content//\{\{USER\}\}/$CB_USER}"
        service_content="${service_content//\{\{PORT\}\}/${VLM_PROCESSOR_PORT:-8081}}"
        service_content="${service_content//\{\{BIN_DIR\}\}/$BIN_DIR}"
        service_content="${service_content//\{\{CONFIG_DIR\}\}/$CONFIG_DIR}"
        service_content="${service_content//\{\{LOG_DIR\}\}/$LOG_DIR}"
        service_content="${service_content//\{\{LLAMA_DIR\}\}/$LLAMA_DIR}"
        service_content="${service_content//\{\{LLAMA_SERVER_URL\}\}/${LLAMA_SERVER_URL:-http://localhost:8888}}"
        service_content="${service_content//\{\{LLAMA_BIN\}\}/$BIN_DIR/llama-server}"
        service_content="${service_content//\{\{MODEL_PATH\}\}/${CB_MODEL_DIR:-/var/lib/camera-brain/models}/LFM2.5-VL-1.6B.Q8_0.gguf}"
        service_content="${service_content//\{\{MMProj_PATH\}\}/${CB_MODEL_DIR:-/var/lib/camera-brain/models}/LFM2.5-VL-1.6B.mmproj-f16.gguf}"
        service_content="${service_content//\{\{CPU_THREADS\}\}/${CPU_THREADS:-4}}"
        service_content="${service_content//\{\{DB_HOST\}\}/${DB_HOST:-localhost}}"
        service_content="${service_content//\{\{DB_PORT\}\}/${DB_PORT:-5432}}"
        service_content="${service_content//\{\{DB_USER\}\}/${DB_USER:-camera_brain}}"
        service_content="${service_content//\{\{DB_PASSWORD\}\}/${DB_PASSWORD:-change_me}}"
        service_content="${service_content//\{\{DB_NAME\}\}/${DB_NAME:-camera_brain}}"
        service_content="${service_content//\{\{NATS_URL\}\}/${NATS_URL:-nats://localhost:4222}}"
        service_content="${service_content//\{\{GATEWAY_PORT\}\}/${GATEWAY_PORT:-8080}}"
        service_content="${service_content//\{\{QUERY_ENGINE_PORT\}\}/${QUERY_ENGINE_PORT:-8082}}"

        if [[ "$DRY_RUN" == "false" ]]; then
            echo "$service_content" > "$SYSTEMD_DIR/$output_file"
        else
            echo "[DRY-RUN] Would create: $SYSTEMD_DIR/$output_file"
        fi

        log_success "Installed: $output_file"
    done

    # Reload systemd once after all units are written, not per-service.
    if [[ "$DRY_RUN" == "false" ]]; then
        systemctl daemon-reload
        for service_def in "${services[@]}"; do
            IFS=':' read -r _ output_file _ <<< "$service_def"
            systemctl enable "$output_file" 2>/dev/null || true
        done
    fi
}

# ============================================================================
# Database Initialization
# ============================================================================
init_database() {
    log_info "Initializing database..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would initialize PostgreSQL database"
        return 0
    fi

    local config_file="$CONFIG_DIR/camera-brain.env"
    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
    fi

    # Check if PostgreSQL is running
    if ! command -v psql &> /dev/null; then
        log_warning "psql not found, skipping database initialization"
        log_info "Manually run the schema script after installing PostgreSQL"
        return 0
    fi

    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL..."
    local retries=10
    while ! pg_isready -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" &> /dev/null; do
        ((retries--)) || {
            log_error "PostgreSQL not available"
            return 1
        }
        sleep 1
    done

    # Create user and database if they don't exist
    if sudo -u postgres psql -c "\du" | grep -q "^ $DB_USER "; then
        log_info "Database user $DB_USER already exists"
    else
        log_info "Creating database user: $DB_USER"
        sudo -u postgres psql <<EOF
CREATE USER "${DB_USER}" WITH PASSWORD '${DB_PASSWORD}';
CREATE DATABASE "${DB_NAME}" OWNER "${DB_USER}";
EOF
    fi

    if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log_info "Database $DB_NAME already exists"
    else
        log_info "Creating database: $DB_NAME"
        sudo -u postgres psql -c "CREATE DATABASE \"${DB_NAME}\" OWNER \"${DB_USER}\";"
    fi

    # Run schema
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local schema_file="$(dirname "$script_dir")/storage/schema.sql"

    if [[ -f "$schema_file" ]]; then
        log_info "Running schema..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$schema_file"
        log_success "Database initialized"
    else
        log_warning "Schema file not found: $schema_file"
    fi
}

# ============================================================================
# Start Services
# ============================================================================

# Wait for a systemd service to reach active state, with timeout.
wait_service_active() {
    local service=$1
    local timeout=${2:-60}
    local elapsed=0
    log_info "Waiting for $service to become active (timeout ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_error "$service did not become active within ${timeout}s"
    return 1
}

start_services() {
    log_info "Starting services..."

    # Ordered start with health-check waits so downstream services don't race
    # ahead of upstream ones (e.g. vlmprocessor before llama-server has loaded
    # its multi-GB model).
    local services=("llama-server.service" "vlm-processor.service" "query-engine.service" "camera-brain-gateway.service")

    for service in "${services[@]}"; do
        if [[ "$DRY_RUN" == "false" ]]; then
            systemctl start "$service" || {
                log_error "Failed to start $service"
                systemctl status "$service" --no-pager
                return 1
            }
            systemctl enable "$service"
            log_success "Started: $service"
            # Wait for upstream services before starting dependents.
            case "$service" in
                llama-server.service)
                    wait_service_active "$service" 120 || return 1
                    ;;
                vlm-processor.service)
                    wait_service_active "$service" 60 || return 1
                    ;;
            esac
        else
            echo "[DRY-RUN] Would start: $service"
        fi
    done
}

# ============================================================================
# Post-install Verification
# ============================================================================
verify_install() {
    log_info "Verifying installation..."

    local all_ok=true
    local services=("llama-server.service" "vlm-processor.service" "query-engine.service" "camera-brain-gateway.service")

    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            log_success "$service is running"
        else
            log_error "$service is not running"
            all_ok=false
        fi
    done

    if [[ "$all_ok" == "true" ]]; then
        log_success "All services are running!"
    else
        log_warning "Some services are not running. Check logs at $LOG_DIR/"
    fi
}

# ============================================================================
# Uninstall
# ============================================================================
do_uninstall() {
    log_info "Uninstalling Camera Brain..."

    # Stop and disable services
    local services=("llama-server.service" "vlm-processor.service" "query-engine.service" "camera-brain-gateway.service")
    for service in "${services[@]}"; do
        systemctl stop "$service" 2>/dev/null || true
        systemctl disable "$service" 2>/dev/null || true
        rm -f "$SYSTEMD_DIR/$service"
        log_info "Removed: $service"
    done

    # Remove binaries
    rm -f "$BIN_DIR/llama-server" "$BIN_DIR/vlm-processor" "$BIN_DIR/query-engine" "$BIN_DIR/gateway"
    rm -f "$BIN_DIR/cbrain"
    rm -f /usr/local/bin/cbrain
    log_info "Removed binaries"

    # Remove directories (keep data)
    rm -rf "$CONFIG_DIR" "$LOG_DIR"
    log_info "Removed configuration and logs"

    # Remove llama.cpp
    rm -rf "$LLAMA_DIR"
    log_info "Removed llama.cpp"

    systemctl daemon-reload

    log_success "Uninstall complete. Data preserved at $DATA_DIR"
    log_info "To remove data as well, run: rm -rf $DATA_DIR"
}

# ============================================================================
# System User
# ============================================================================
ensure_system_user() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    if ! getent group "$CB_USER" &>/dev/null; then
        groupadd --system "$CB_USER"
    fi
    if ! id "$CB_USER" &>/dev/null; then
        useradd --system --no-create-home --shell /usr/sbin/nologin --gid "$CB_USER" "$CB_USER"
    fi
    log_success "System user '$CB_USER' ready"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo "=============================================="
    echo "  Camera Brain Installer"
    echo "=============================================="
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN MODE - No changes will be made"
        echo ""
    fi

    if [[ "$UNINSTALL" == "true" ]]; then
        do_uninstall
        exit 0
    fi

    # Refuse to run without sudo unless dry-running.
    if [[ "$DRY_RUN" == "false" && "${EUID:-$(id -u)}" -ne 0 ]]; then
        log_error "Installer must be run as root (use sudo)"
        exit 1
    fi

    pre_install_checks
    ensure_system_user
    setup_directories
    generate_config
    build_llama_cpp
    build_go_services
    build_cbrain
    install_systemd_services
    init_database
    start_services
    verify_install

    echo ""
    echo "=============================================="
    log_success "Installation complete!"
    echo "=============================================="
    echo ""
    echo "Configuration: $CONFIG_DIR/camera-brain.env"
    echo "Data:          $DATA_DIR"
    echo "Logs:          $LOG_DIR"
    echo "Binaries:      $BIN_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Download LFM2.5-VL-1.6B models to ${CB_MODEL_DIR:-/var/lib/camera-brain/models}/"
    echo "     - LFM2.5-VL-1.6B.Q8_0.gguf (vision-language model)"
    echo "     - LFM2.5-VL-1.6B.mmproj-f16.gguf (multi-modal projector)"
    echo "  2. [Optional] Download LFM2.5-1.2B-Instruct.Q4_K_M.gguf for text-only query interpretation"
    echo "  3. Place yolov5s_int8.rknn model in the models directory"
    echo "  4. Update configuration at $CONFIG_DIR/camera-brain.env"
    echo "  5. View logs: journalctl -fu camera-brain-gateway.service"
    echo ""
}

main "$@"
