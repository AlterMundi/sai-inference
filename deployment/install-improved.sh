#!/bin/bash
#
# SAI Inference Service - Improved Production Installation Script
# Handles both fresh installations and updates intelligently
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
readonly BACKUP_DIR="/opt/sai-inference-backups"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Installation mode
INSTALL_MODE="unknown"  # Will be: fresh, update, or repair

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m'

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

log_action() {
    echo -e "${MAGENTA}[ACTION]${NC} $*" >&2
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Detect installation mode
detect_mode() {
    log_info "Detecting installation mode..."
    
    if [[ -d "$INSTALL_DIR/src" ]] && [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
        INSTALL_MODE="update"
        log_info "Detected existing installation - UPDATE MODE"
    elif [[ -d "$INSTALL_DIR" ]] || [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
        INSTALL_MODE="repair"
        log_warning "Detected partial installation - REPAIR MODE"
    else
        INSTALL_MODE="fresh"
        log_info "No existing installation found - FRESH INSTALL MODE"
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
    
    # Check disk space (need at least 2GB for venv)
    local available_space=$(df "$INSTALL_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || df /opt | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 2097152 ]]; then
        log_error "Insufficient disk space. At least 2GB required."
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
    mkdir -p "$BACKUP_DIR"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
    chown -R root:root "$CONFIG_DIR"
    chmod 755 "$CONFIG_DIR"
    
    log_success "Directory structure ready"
}

# Backup existing installation
backup_existing() {
    if [[ "$INSTALL_MODE" == "update" ]]; then
        log_action "Creating backup of existing installation..."
        
        local backup_path="$BACKUP_DIR/backup_$TIMESTAMP"
        mkdir -p "$backup_path"
        
        # Backup code
        if [[ -d "$INSTALL_DIR/src" ]]; then
            cp -r "$INSTALL_DIR/src" "$backup_path/"
            log_info "Backed up source code"
        fi
        
        # Backup configuration
        if [[ -f "$CONFIG_DIR/production.env" ]]; then
            cp "$CONFIG_DIR/production.env" "$backup_path/"
            log_info "Backed up configuration"
        fi
        
        # Backup requirements for comparison
        if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
            cp "$INSTALL_DIR/requirements.txt" "$backup_path/"
        fi
        
        log_success "Backup created at: $backup_path"
    fi
}

# Deploy application code
deploy_code() {
    log_info "Deploying application code..."
    
    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_action "Stopping service for update..."
        systemctl stop "$SERVICE_NAME"
        sleep 2  # Give service time to fully stop
    fi
    
    # Deploy code with backup of old files
    if [[ -d "$INSTALL_DIR/src" ]]; then
        # Keep .old backups for manual recovery
        rm -rf "$INSTALL_DIR/src.old" 2>/dev/null || true
        mv "$INSTALL_DIR/src" "$INSTALL_DIR/src.old"
        log_info "Previous source backed up to src.old"
    fi
    
    # Copy new application files
    cp -r "$SCRIPT_DIR"/../src "$INSTALL_DIR/"
    cp "$SCRIPT_DIR"/../requirements.txt "$INSTALL_DIR/"
    cp "$SCRIPT_DIR"/../run.py "$INSTALL_DIR/"
    
    # Copy additional scripts if present
    for script in process_images.py; do
        if [[ -f "$SCRIPT_DIR/../scripts/$script" ]]; then
            cp "$SCRIPT_DIR/../scripts/$script" "$INSTALL_DIR/"
        fi
    done
    
    # Set permissions
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"/src "$INSTALL_DIR"/*.py
    chmod +x "$INSTALL_DIR/run.py"
    
    log_success "Application code deployed"
}

# Smart Python environment update
update_python_env() {
    log_info "Updating Python environment..."
    
    local needs_recreation=false
    
    # Check if venv exists and is healthy
    if [[ -d "$INSTALL_DIR/venv" ]]; then
        if [[ -f "$INSTALL_DIR/venv/bin/python" ]]; then
            # Check if requirements changed significantly
            if [[ -f "$BACKUP_DIR/backup_$TIMESTAMP/requirements.txt" ]]; then
                if ! diff -q "$BACKUP_DIR/backup_$TIMESTAMP/requirements.txt" "$INSTALL_DIR/requirements.txt" > /dev/null; then
                    log_info "Requirements changed - updating packages"
                    
                    # Try to update in-place first
                    if sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" --upgrade; then
                        log_success "Python packages updated successfully"
                        return 0
                    else
                        log_warning "In-place update failed, will recreate venv"
                        needs_recreation=true
                    fi
                else
                    log_info "Requirements unchanged - skipping package update"
                    return 0
                fi
            else
                # No backup to compare, just update
                sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" --upgrade
                log_success "Python packages updated"
                return 0
            fi
        else
            needs_recreation=true
        fi
    else
        needs_recreation=true
    fi
    
    if [[ "$needs_recreation" == true ]]; then
        log_action "Creating new Python virtual environment..."
        
        # Remove old venv if it exists
        [[ -d "$INSTALL_DIR/venv" ]] && rm -rf "$INSTALL_DIR/venv"
        
        # Create new virtual environment
        sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
        
        # Upgrade pip
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
        
        # Install requirements
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
        
        log_success "Python environment created"
    fi
}

# Setup models (preserve existing)
setup_models() {
    log_info "Managing models..."
    
    # Check for existing models
    local has_models=false
    if [[ -d "$INSTALL_DIR/models" ]] && ls "$INSTALL_DIR/models"/*.pt 2>/dev/null | grep -q .; then
        has_models=true
        log_success "Existing models found and preserved:"
        ls -lh "$INSTALL_DIR/models"/*.pt | awk '{print "  - " $9 " (" $5 ")"}'
    fi
    
    # Check for new model in repo
    local local_model="$SCRIPT_DIR/../models/last.pt"
    if [[ -f "$local_model" ]]; then
        local dest_model="$INSTALL_DIR/models/last.pt"
        
        if [[ -f "$dest_model" ]]; then
            # Compare checksums
            if ! cmp -s "$local_model" "$dest_model"; then
                # Backup old model
                mv "$dest_model" "$INSTALL_DIR/models/last.pt.backup_$TIMESTAMP"
                cp "$local_model" "$dest_model"
                chown "$SERVICE_USER:$SERVICE_GROUP" "$dest_model"
                log_success "Model updated (old version backed up)"
            else
                log_info "Model unchanged - keeping existing"
            fi
        else
            cp "$local_model" "$dest_model"
            chown "$SERVICE_USER:$SERVICE_GROUP" "$dest_model"
            log_success "Model deployed: $dest_model"
        fi
    elif [[ "$has_models" == false ]]; then
        log_warning "No models found - please add model files to $INSTALL_DIR/models/"
    fi
}

# Smart configuration update
update_config() {
    log_info "Managing configuration..."
    
    local config_file="$CONFIG_DIR/production.env"
    local template_file="$SCRIPT_DIR/../config/production.env.template"
    
    # If no template exists, create from current config
    if [[ ! -f "$template_file" ]]; then
        template_file="/tmp/sai-config-template.env"
        cat > "$template_file" <<'EOF'
# SAI Inference Service - Production Configuration

# Service Configuration
SAI_HOST=0.0.0.0
SAI_PORT=8888
SAI_WORKERS=1

# Model Configuration
SAI_MODEL_DIR=/opt/sai-inference/models
SAI_DEFAULT_MODEL=last.pt
SAI_DEVICE=cpu
SAI_INPUT_SIZE=864
SAI_CONFIDENCE_THRESHOLD=0.13
SAI_IOU_THRESHOLD=0.4

# Performance
SAI_BATCH_SIZE=1
SAI_MAX_QUEUE=100

# Logging
SAI_LOG_LEVEL=INFO
SAI_LOG_FILE=/var/log/sai-inference/service.log

# Security (uncomment and set if needed)
# SAI_API_KEY=your-secure-api-key-here

# Monitoring
SAI_ENABLE_METRICS=true
SAI_METRICS_PORT=9090
EOF
    fi
    
    if [[ -f "$config_file" ]]; then
        log_action "Preserving existing configuration..."
        
        # Create merged config
        local merged_config="/tmp/sai-merged-config.env"
        cp "$config_file" "$merged_config"
        
        # Update paths if needed
        sed -i "s|SAI_MODEL_DIR=.*|SAI_MODEL_DIR=$INSTALL_DIR/models|" "$merged_config"
        sed -i "s|SAI_LOG_FILE=.*|SAI_LOG_FILE=$LOG_DIR/service.log|" "$merged_config"
        
        # Add any new required variables from template
        while IFS= read -r line; do
            if [[ "$line" =~ ^[A-Z_]+= ]]; then
                var_name=$(echo "$line" | cut -d'=' -f1)
                if ! grep -q "^$var_name=" "$merged_config"; then
                    echo "$line" >> "$merged_config"
                    log_info "Added new config variable: $var_name"
                fi
            fi
        done < "$template_file"
        
        # Backup old config and install merged
        cp "$config_file" "$config_file.backup_$TIMESTAMP"
        mv "$merged_config" "$config_file"
        
        log_success "Configuration updated (backup: $config_file.backup_$TIMESTAMP)"
    else
        # Fresh install - use template
        cp "$template_file" "$config_file"
        
        # Update paths
        sed -i "s|SAI_MODEL_DIR=.*|SAI_MODEL_DIR=$INSTALL_DIR/models|" "$config_file"
        sed -i "s|SAI_LOG_FILE=.*|SAI_LOG_FILE=$LOG_DIR/service.log|" "$config_file"
        
        # Set appropriate device
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            sed -i "s|SAI_DEVICE=.*|SAI_DEVICE=cuda|" "$config_file"
            log_info "CUDA detected - configured for GPU"
        fi
        
        log_success "Configuration created: $config_file"
    fi
    
    chmod 644 "$config_file"
}

# Update systemd service
update_systemd_service() {
    log_info "Updating systemd service configuration..."
    
    # Check if service file exists and compare
    local service_file="/etc/systemd/system/$SERVICE_NAME.service"
    local new_service_file="$SCRIPT_DIR/systemd/sai-inference.service"
    
    # If repo doesn't have service file, create it
    if [[ ! -f "$new_service_file" ]]; then
        new_service_file="/tmp/sai-inference.service"
        cat > "$new_service_file" <<EOF
[Unit]
Description=SAI Inference Service - Fire/Smoke Detection API
After=network.target

[Service]
Type=notify
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$CONFIG_DIR/production.env
ExecStart=$INSTALL_DIR/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8888 --workers 1
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/service.log
StandardError=append:$LOG_DIR/service-error.log

# Watchdog
WatchdogSec=60
NotifyAccess=all

# Security
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF
    fi
    
    # Install service file
    if [[ -f "$service_file" ]]; then
        if ! diff -q "$service_file" "$new_service_file" > /dev/null 2>&1; then
            cp "$service_file" "$service_file.backup_$TIMESTAMP"
            cp "$new_service_file" "$service_file"
            log_info "Service file updated (backup: $service_file.backup_$TIMESTAMP)"
        else
            log_info "Service file unchanged"
        fi
    else
        cp "$new_service_file" "$service_file"
        log_success "Service file installed"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    log_success "Systemd configuration updated"
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
    backup)
        echo "Creating backup..."
        sudo cp -r /opt/sai-inference /opt/sai-inference.backup.$(date +%Y%m%d_%H%M%S)
        echo "Backup created"
        ;;
    rollback)
        if [[ -d /opt/sai-inference.old ]]; then
            echo "Rolling back to previous version..."
            sudo systemctl stop "$SERVICE_NAME"
            sudo mv /opt/sai-inference /opt/sai-inference.failed
            sudo mv /opt/sai-inference.old /opt/sai-inference
            sudo systemctl start "$SERVICE_NAME"
            echo "Rollback completed"
        else
            echo "No previous version found for rollback"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|status|logs|health|test|backup|rollback}"
        exit 1
        ;;
esac
EOF

    chmod +x "$INSTALL_DIR/sai-service"
    
    # Create symlink if not exists
    if [[ ! -L /usr/local/bin/sai-service ]]; then
        ln -sf "$INSTALL_DIR/sai-service" /usr/local/bin/sai-service
    fi
    
    log_success "Management scripts ready"
}

# Post-installation verification
verify_installation() {
    log_info "Verifying installation..."
    
    local errors=0
    
    # Check directories
    for dir in "$INSTALL_DIR/src" "$INSTALL_DIR/models" "$CONFIG_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Missing directory: $dir"
            ((errors++))
        fi
    done
    
    # Check configuration
    if [[ ! -f "$CONFIG_DIR/production.env" ]]; then
        log_error "Missing configuration file"
        ((errors++))
    fi
    
    # Check Python environment
    if [[ ! -f "$INSTALL_DIR/venv/bin/python" ]]; then
        log_error "Python virtual environment not found"
        ((errors++))
    fi
    
    # Check service file
    if [[ ! -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
        log_error "SystemD service file not found"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Installation verified successfully"
        return 0
    else
        log_error "Installation verification failed with $errors errors"
        return 1
    fi
}

# Main installation function
main() {
    echo -e "${MAGENTA}=== SAI Inference Service - Intelligent Installer ===${NC}"
    echo
    
    check_root
    detect_mode
    
    echo -e "${BLUE}Installation mode: ${YELLOW}$INSTALL_MODE${NC}"
    echo
    
    check_requirements
    setup_user
    setup_directories
    
    if [[ "$INSTALL_MODE" == "update" ]]; then
        backup_existing
    fi
    
    deploy_code
    update_python_env
    setup_models
    update_config
    update_systemd_service
    setup_log_rotation
    create_management_scripts
    
    if verify_installation; then
        echo
        log_success "Installation completed successfully!"
        echo
        
        if [[ "$INSTALL_MODE" == "update" ]]; then
            log_info "Update Summary:"
            echo "  - Previous version backed up to: $BACKUP_DIR/backup_$TIMESTAMP"
            echo "  - Configuration preserved with updates"
            echo "  - Models preserved"
            echo "  - Service ready to restart"
            echo
            log_action "Next steps:"
            echo "  1. Review configuration: $CONFIG_DIR/production.env"
            echo "  2. Start service: sudo systemctl start $SERVICE_NAME"
            echo "  3. Verify health: sai-service health"
            echo "  4. If issues occur: sai-service rollback"
        else
            log_info "Next steps:"
            echo "  1. Review configuration: $CONFIG_DIR/production.env"
            echo "  2. Add your model file to: $INSTALL_DIR/models/"
            echo "  3. Enable service: sudo systemctl enable $SERVICE_NAME"
            echo "  4. Start service: sudo systemctl start $SERVICE_NAME"
            echo "  5. Check status: sai-service status"
            echo "  6. Test API: sai-service health"
        fi
        
        echo
        log_info "Management commands available:"
        echo "  sai-service {start|stop|restart|reload|status|logs|health|test|backup|rollback}"
        echo
        log_info "Service endpoint: http://localhost:8888"
    else
        log_error "Installation failed - please check the errors above"
        if [[ "$INSTALL_MODE" == "update" ]]; then
            log_info "Your backup is available at: $BACKUP_DIR/backup_$TIMESTAMP"
        fi
        exit 1
    fi
}

# Run main function
main "$@"