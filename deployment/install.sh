#!/bin/bash
#
# SAI Inference Service - Production Installation Script
# Formalizes the deployment of the SAI inference ecosystem
#
set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SERVICE_USER="service"
readonly SERVICE_GROUP="service" 
readonly INSTALL_DIR="/opt/sai-inference"
readonly SERVICE_NAME="sai-inference"
readonly LOG_DIR="/var/log/sai-inference"
readonly CONFIG_DIR="/etc/sai-inference"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null || ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ is required"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required"
        exit 1
    fi
    
    # Check systemctl
    if ! command -v systemctl &> /dev/null; then
        log_error "systemd is required"
        exit 1
    fi
    
    log_success "System requirements met"
}

# Create service user if not exists
setup_user() {
    log_info "Setting up service user..."
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d "$INSTALL_DIR" -c "SAI Inference Service" "$SERVICE_USER"
        log_success "Created service user: $SERVICE_USER"
    else
        log_info "Service user already exists: $SERVICE_USER"
    fi
}

# Create directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    # Create main directories
    mkdir -p "$INSTALL_DIR"/{src,models,logs,tmp}
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
    chown -R root:root "$CONFIG_DIR"
    chmod 755 "$CONFIG_DIR"
    
    log_success "Directory structure created"
}

# Deploy application code
deploy_code() {
    log_info "Deploying application code..."
    
    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_info "Stopping existing service..."
        systemctl stop "$SERVICE_NAME"
    fi
    
    # Copy application files
    cp -r "$SCRIPT_DIR"/../src "$INSTALL_DIR/"
    cp "$SCRIPT_DIR"/../requirements.txt "$INSTALL_DIR/"
    cp "$SCRIPT_DIR"/../run.py "$INSTALL_DIR/"
    
    # Copy additional scripts
    if [[ -f "$SCRIPT_DIR/../scripts/process_images.py" ]]; then
        cp "$SCRIPT_DIR/../scripts/process_images.py" "$INSTALL_DIR/"
    fi

    # Remove local .env file if exists (production uses systemd environment)
    if [[ -f "$INSTALL_DIR/.env" ]]; then
        log_warning "Found local .env file - backing up to .env.backup"
        mv "$INSTALL_DIR/.env" "$INSTALL_DIR/.env.backup"
        log_info "Production configuration comes from $CONFIG_DIR/production.env via systemd"
    fi

    # Set permissions
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"/src "$INSTALL_DIR"/*.py
    chmod +x "$INSTALL_DIR/run.py"

    log_success "Application code deployed"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    # Remove old venv if exists
    if [[ -d "$INSTALL_DIR/venv" ]]; then
        rm -rf "$INSTALL_DIR/venv"
    fi
    
    # Create new virtual environment
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    
    # Upgrade pip
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    
    # Install requirements
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
    
    log_success "Python environment configured"
}

# Setup models
setup_models() {
    log_info "Setting up SAI models..."
    
    # Check for local model in the repo's models directory
    local local_model="$SCRIPT_DIR/../models/last.pt"
    local sai_model_dest="$INSTALL_DIR/models/last.pt"
    
    if [[ -f "$local_model" ]]; then
        cp "$local_model" "$sai_model_dest"
        chown "$SERVICE_USER:$SERVICE_GROUP" "$sai_model_dest"
        log_success "SAI model deployed: $sai_model_dest"
    else
        log_warning "SAI model not found at $local_model"
        log_warning "Please place the model file 'last.pt' in the models/ directory and re-run installation"
    fi
}

# Create production configuration
create_config() {
    log_info "Creating production configuration..."

    # Create production config from scratch if template doesn't exist
    if [[ -f "$SCRIPT_DIR/../config/production.env.template" ]]; then
        cp "$SCRIPT_DIR/../config/production.env.template" "$CONFIG_DIR/production.env"
    else
        # Create basic production config
        cat > "$CONFIG_DIR/production.env" <<EOF
# SAI Inference Production Configuration
# Generated by install.sh on $(date)

# No config/production.env.template file found.
# Place your env settings there and re-run install
# or place them here and restart service.
EOF
    fi

    # Update paths in the configuration
    sed -i "s|SAI_MODEL_DIR=.*|SAI_MODEL_DIR=$INSTALL_DIR/models|" "$CONFIG_DIR/production.env"
    sed -i "s|SAI_LOG_FILE=.*|SAI_LOG_FILE=$LOG_DIR/service.log|" "$CONFIG_DIR/production.env" 2>/dev/null || true

    chmod 644 "$CONFIG_DIR/production.env"
    log_success "Production configuration created: $CONFIG_DIR/production.env"
}

# Update systemd service
update_systemd_service() {
    log_info "Updating systemd service configuration..."
    
    # Copy service file from repo
    cp "$SCRIPT_DIR/systemd/sai-inference.service" "/etc/systemd/system/$SERVICE_NAME.service"
    
    # Reload systemd
    systemctl daemon-reload
    log_success "Systemd service updated"
}

# Setup database
setup_database() {
    log_info "Checking PostgreSQL for enhanced alert system..."

    # Check if PostgreSQL is available
    if command -v psql &> /dev/null; then
        log_info "PostgreSQL found"

        # Check if we can connect to PostgreSQL
        if sudo -u postgres psql -c "\l" &> /dev/null; then
            # Check if sai_dashboard database exists
            if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw sai_dashboard; then
                log_success "Database 'sai_dashboard' found - enhanced schema will be auto-created on first run"
            else
                log_warning "Database 'sai_dashboard' not found"
                log_warning "Create with: sudo -u postgres psql -c 'CREATE DATABASE sai_dashboard;'"
                log_warning "Enhanced alerts will be disabled until database is created"
            fi
        else
            log_warning "Cannot connect to PostgreSQL as postgres user"
            log_warning "Enhanced alerts may not work - check PostgreSQL configuration"
        fi
    else
        log_warning "PostgreSQL not found - enhanced alerts will be disabled"
        log_warning "Install PostgreSQL for full functionality"
    fi

    log_success "Database check completed"
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/$SERVICE_NAME" <<EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $SERVICE_USER $SERVICE_GROUP
    postrotate
        systemctl reload-or-restart $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF

    log_success "Log rotation configured"
}

# Create management scripts
create_management_scripts() {
    log_info "Creating management scripts..."
    
    # Service control script
    cat > "$INSTALL_DIR/sai-service" <<'EOF'
#!/bin/bash
# SAI Inference Service Management Script

SERVICE_NAME="sai-inference"

case "$1" in
    start)
        echo "Starting SAI Inference Service..."
        sudo systemctl start "$SERVICE_NAME"
        ;;
    stop)
        echo "Stopping SAI Inference Service..."
        sudo systemctl stop "$SERVICE_NAME"
        ;;
    restart)
        echo "Restarting SAI Inference Service..."
        sudo systemctl restart "$SERVICE_NAME"
        ;;
    reload)
        echo "Reloading SAI Inference Service..."
        sudo systemctl reload "$SERVICE_NAME"
        ;;
    status)
        sudo systemctl status "$SERVICE_NAME" --no-pager -l
        ;;
    logs)
        sudo journalctl -u "$SERVICE_NAME" -f --no-pager
        ;;
    health)
        curl -s http://localhost:8888/api/v1/health | jq . 2>/dev/null || curl -s http://localhost:8888/api/v1/health
        ;;
    test)
        echo "Running service health check..."
        if curl -sf http://localhost:8888/api/v1/health > /dev/null; then
            echo "✅ Service is healthy"
        else
            echo "❌ Service is not responding"
            exit 1
        fi
        ;;
    db-setup)
        echo "Enhanced schema is auto-created by the application"
        echo "To recreate database, use: sudo -u postgres psql -d sai_dashboard"
        echo "Then restart service: sudo systemctl restart sai-inference"
        ;;
    db-status)
        echo "Checking database connection..."
        if sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*) FROM camera_detections;" 2>/dev/null; then
            echo "✅ Database connection successful"
        else
            echo "❌ Cannot connect to database"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|status|logs|health|test|db-setup|db-status}"
        exit 1
        ;;
esac
EOF

    chmod +x "$INSTALL_DIR/sai-service"
    ln -sf "$INSTALL_DIR/sai-service" /usr/local/bin/sai-service
    
    log_success "Management scripts created"
}

# Main installation function
main() {
    log_info "Starting SAI Inference Service installation..."
    
    check_root
    check_requirements
    setup_user
    setup_directories
    deploy_code
    setup_python_env
    setup_models
    create_config
    setup_database
    update_systemd_service
    setup_log_rotation
    create_management_scripts
    
    log_success "Installation completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Review configuration: $CONFIG_DIR/production.env"
    echo "  2. Enable service: sudo systemctl enable $SERVICE_NAME"
    echo "  3. Start service: sudo systemctl start $SERVICE_NAME"
    echo "  4. Check status: sai-service status"
    echo "  5. Test API: sai-service health"
    echo
    log_info "Management commands available:"
    echo "  sai-service {start|stop|restart|reload|status|logs|health|test|db-setup|db-status}"
    echo
    log_info "Service will run on: http://localhost:8888"
}

# Run main function
main "$@"
