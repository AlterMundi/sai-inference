#!/bin/bash
#
# SAI Daily Test Service - Installation Script
# Sets up automated daily testing for the alert system chain
#
set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly SERVICE_USER="service"
readonly SERVICE_GROUP="service" 
readonly INSTALL_DIR="/opt/sai-inference"
readonly CONFIG_DIR="/etc/sai-inference"
readonly LOG_DIR="/var/log/sai-inference"
readonly CRON_USER="service"

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

# Check if main service is installed and daily test script exists
check_main_service() {
    if [[ ! -d "$INSTALL_DIR" ]]; then
        log_error "Main SAI inference service not found at $INSTALL_DIR"
        log_error "Please install the main service first using deployment/install.sh"
        exit 1
    fi
    
    # Check for daily test script in project root (before copying)
    if [[ ! -f "$PROJECT_ROOT/src/daily_test.py" ]]; then
        log_error "Daily test script not found at $PROJECT_ROOT/src/daily_test.py"
        log_error "Please ensure the script exists in the project src/ directory"
        exit 1
    fi
}

# Create directories and set permissions
setup_directories() {
    log_info "Setting up directories and permissions..."
    
    # Ensure config directory exists
    mkdir -p "$CONFIG_DIR"
    chown root:$SERVICE_GROUP "$CONFIG_DIR"
    chmod 755 "$CONFIG_DIR"
    
    # Ensure log directory exists
    mkdir -p "$LOG_DIR"
    chown $SERVICE_USER:$SERVICE_GROUP "$LOG_DIR"
    chmod 755 "$LOG_DIR"
    
    # Create log file with proper permissions
    touch "$LOG_DIR/daily-test.log"
    chown $SERVICE_USER:$SERVICE_GROUP "$LOG_DIR/daily-test.log"
    chmod 644 "$LOG_DIR/daily-test.log"
    
    log_success "Directories created successfully"
}

# Install configuration file
install_config() {
    log_info "Installing configuration file..."
    
    local config_source="$PROJECT_ROOT/config/daily-test.env"
    local config_dest="$CONFIG_DIR/daily-test.env"
    
    if [[ -f "$config_source" ]]; then
        cp "$config_source" "$config_dest"
        chown root:$SERVICE_GROUP "$config_dest"
        chmod 640 "$config_dest"
        log_success "Configuration installed at $config_dest"
    else
        log_error "Configuration template not found at $config_source"
        log_error "Please ensure config/daily-test.env exists in the project"
        exit 1
    fi
}

# Setup test images and labels directory structure
setup_test_images() {
    log_info "Setting up test images and labels directory structure..."
    
    local test_images_dir="$INSTALL_DIR/tests/images"
    local test_labels_dir="$INSTALL_DIR/tests/labels"
    
    # Create subdirectories for images and labels
    for subdir in fire smoke both; do
        mkdir -p "$test_images_dir/$subdir"
        mkdir -p "$test_labels_dir/$subdir"
        chown $SERVICE_USER:$SERVICE_GROUP "$test_images_dir/$subdir"
        chown $SERVICE_USER:$SERVICE_GROUP "$test_labels_dir/$subdir"
        chmod 755 "$test_images_dir/$subdir"
        chmod 755 "$test_labels_dir/$subdir"
    done
    
    # Create README files for each subdirectory
    cat > "$test_images_dir/fire/README.md" << 'EOF'
# Fire Detection Test Images

Place images with known fire detections in this directory.
These images will be used to test the fire detection alert chain.

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
EOF

    cat > "$test_images_dir/smoke/README.md" << 'EOF'
# Smoke Detection Test Images

Place images with known smoke detections in this directory.
These images will be used to test the smoke detection alert chain.

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
EOF

    cat > "$test_images_dir/both/README.md" << 'EOF'
# Fire and Smoke Detection Test Images

Place images with both fire and smoke detections in this directory.
These images will be used to test combined detection alert chains.

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
EOF

    chown $SERVICE_USER:$SERVICE_GROUP "$test_images_dir"/*.md
    chmod 644 "$test_images_dir"/*.md
    
    # Create README files for labels subdirectories
    cat > "$test_labels_dir/fire/README.md" << 'EOF'
# Fire Detection Test Labels

YOLO format annotation files (.txt) for fire detection test images.
Must match image filenames exactly (e.g., test_fire.jpg → test_fire.txt).
Contains YOLO annotations parsed into metadata for n8n requests.

Format: class_id x_center y_center width height [confidence]
Fire class ID: 0

Example:
0 0.5 0.4 0.3 0.2
0 0.7 0.6 0.1 0.15
EOF

    cat > "$test_labels_dir/smoke/README.md" << 'EOF'
# Smoke Detection Test Labels

YOLO format annotation files (.txt) for smoke detection test images.
Must match image filenames exactly (e.g., test_smoke.jpg → test_smoke.txt).
Contains YOLO annotations parsed into metadata for n8n requests.

Format: class_id x_center y_center width height [confidence]
Smoke class ID: 1

Example:
1 0.45 0.35 0.4 0.3
1 0.75 0.55 0.2 0.25
EOF

    cat > "$test_labels_dir/both/README.md" << 'EOF'
# Fire and Smoke Detection Test Labels

YOLO format annotation files (.txt) for combined detection test images.
Must match image filenames exactly (e.g., emergency.jpg → emergency.txt).
Contains YOLO annotations parsed into metadata for n8n requests.

Format: class_id x_center y_center width height [confidence]
Class IDs: 0=fire, 1=smoke

Example:
0 0.4 0.3 0.25 0.2
1 0.6 0.5 0.35 0.3
1 0.8 0.7 0.15 0.1
EOF

    chown $SERVICE_USER:$SERVICE_GROUP "$test_labels_dir"/*.md
    chmod 644 "$test_labels_dir"/*.md
    
    log_success "Test images and labels directory structure created"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies for daily test service..."
    
    # Activate virtual environment and install packages
    if [[ -f "$INSTALL_DIR/venv/bin/activate" ]]; then
        source "$INSTALL_DIR/venv/bin/activate"
        
        # Install additional dependencies for daily test service
        pip install --quiet requests python-dotenv
        
        deactivate
        log_success "Dependencies installed successfully"
    else
        log_error "Virtual environment not found at $INSTALL_DIR/venv"
        exit 1
    fi
}

# Copy daily test script and test data to installation directory
copy_daily_test_files() {
    log_info "Copying daily test script and test data to installation directory..."
    
    # Copy the daily test script
    local source_script="$PROJECT_ROOT/src/daily_test.py"
    local dest_script="$INSTALL_DIR/src/daily_test.py"
    cp "$source_script" "$dest_script"
    chown $SERVICE_USER:$SERVICE_GROUP "$dest_script"
    chmod 755 "$dest_script"
    
    # Copy test images and labels if they exist
    if [[ -d "$PROJECT_ROOT/tests/images" ]]; then
        log_info "Copying test images..."
        cp -r "$PROJECT_ROOT/tests/images"/* "$INSTALL_DIR/tests/images/" 2>/dev/null || true
        chown -R $SERVICE_USER:$SERVICE_GROUP "$INSTALL_DIR/tests/images"
        chmod -R 644 "$INSTALL_DIR/tests/images"
        find "$INSTALL_DIR/tests/images" -type d -exec chmod 755 {} \;
    fi
    
    if [[ -d "$PROJECT_ROOT/tests/labels" ]]; then
        log_info "Copying test labels..."
        cp -r "$PROJECT_ROOT/tests/labels"/* "$INSTALL_DIR/tests/labels/" 2>/dev/null || true
        chown -R $SERVICE_USER:$SERVICE_GROUP "$INSTALL_DIR/tests/labels"
        chmod -R 644 "$INSTALL_DIR/tests/labels"
        find "$INSTALL_DIR/tests/labels" -type d -exec chmod 755 {} \;
    fi
    
    log_success "Daily test files copied successfully"
}

# Setup dynamic cron scheduling
setup_cron() {
    log_info "Setting up dynamic cron scheduling for daily test service..."
    
    # Check if service user exists
    if ! id "$CRON_USER" &>/dev/null; then
        log_error "User '$CRON_USER' does not exist"
        exit 1
    fi
    
    # Use the service's built-in cron management
    local script_path="$INSTALL_DIR/src/daily_test.py"
    local config_path="/etc/sai-inference/daily-test.env"
    
    log_info "Installing initial cron schedule from configuration..."
    
    # Run the update-cron command as the service user
    if sudo -u "$CRON_USER" "$INSTALL_DIR/venv/bin/python" "$script_path" "$config_path" --update-cron; then
        log_success "Dynamic cron scheduling installed successfully"
        
        # Show the current schedule
        log_info "Current schedule:"
        sudo -u "$CRON_USER" "$INSTALL_DIR/venv/bin/python" "$script_path" "$config_path" --show-cron | while read -r line; do
            log_info "   $line"
        done
    else
        log_error "Failed to setup dynamic cron scheduling"
        exit 1
    fi
}

# Test the installation
test_installation() {
    log_info "Testing daily test service installation..."
    
    # Test configuration loading
    if sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/python" -c "
import sys
sys.path.append('$INSTALL_DIR/src')
from daily_test import DailyTestConfig
config = DailyTestConfig('/etc/sai-inference/daily-test.env')
errors = config.validate()
if errors:
    print('Configuration errors:', ', '.join(errors))
    exit(1)
else:
    print('Configuration valid')
"; then
        log_success "Configuration test passed"
    else
        log_error "Configuration test failed"
        return 1
    fi
    
    # Test permissions
    if sudo -u "$SERVICE_USER" test -r "$CONFIG_DIR/daily-test.env"; then
        log_success "Configuration file permissions OK"
    else
        log_error "Service user cannot read configuration file"
        return 1
    fi
    
    if sudo -u "$SERVICE_USER" test -w "$LOG_DIR/daily-test.log"; then
        log_success "Log file permissions OK"
    else
        log_error "Service user cannot write to log file"
        return 1
    fi
    
    log_success "Installation test completed successfully"
}

# Show post-installation instructions
show_instructions() {
    echo
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ SAI Daily Test Service Installation Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${YELLOW}NEXT STEPS:${NC}"
    echo
    echo -e "1. ${BLUE}Configure the service:${NC}"
    echo -e "   Edit: ${GREEN}$CONFIG_DIR/daily-test.env${NC}"
    echo -e "   Required: N8N_WEBHOOK_URL and N8N_AUTH_TOKEN"
    echo
    echo -e "2. ${BLUE}Add test images:${NC}"
    echo -e "   Fire images:  ${GREEN}$INSTALL_DIR/tests/images/fire/${NC}"
    echo -e "   Smoke images: ${GREEN}$INSTALL_DIR/tests/images/smoke/${NC}"
    echo -e "   Both images:  ${GREEN}$INSTALL_DIR/tests/images/both/${NC}"
    echo
    echo -e "3. ${BLUE}Add test labels (optional):${NC}"
    echo -e "   Fire labels:  ${GREEN}$INSTALL_DIR/tests/labels/fire/${NC}"
    echo -e "   Smoke labels: ${GREEN}$INSTALL_DIR/tests/labels/smoke/${NC}"
    echo -e "   Both labels:  ${GREEN}$INSTALL_DIR/tests/labels/both/${NC}"
    echo -e "   ${YELLOW}Labels must match image names: image.jpg → image.txt${NC}"
    echo
    echo -e "4. ${BLUE}Test manually (optional):${NC}"
    echo -e "   ${GREEN}sudo -u service $INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/daily_test.py${NC}"
    echo
    echo -e "5. ${BLUE}Check cron schedule:${NC}"
    echo -e "   ${GREEN}sudo -u service $INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/daily_test.py --show-cron${NC}"
    echo
    echo -e "6. ${BLUE}Monitor logs:${NC}"
    echo -e "   ${GREEN}tail -f $LOG_DIR/daily-test.log${NC}"
    echo
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}DYNAMIC CRON SCHEDULING:${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BLUE}Current Schedule:${NC} From configuration (CRON_SCHEDULE)"
    echo -e "  ${BLUE}Change Schedule:${NC} Edit ${GREEN}$CONFIG_DIR/daily-test.env${NC}"
    echo -e "  ${BLUE}Apply Changes:${NC} ${GREEN}sudo -u service $INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/daily_test.py --update-cron${NC}"
    echo
    echo -e "  ${BLUE}Quick Schedule Examples:${NC}"
    echo -e "  • Weekdays 9 AM:   ${GREEN}CRON_SCHEDULE=\"0 9 * * 1-5\"${NC}"
    echo -e "  • Every 6 hours:    ${GREEN}CRON_SCHEDULE=\"0 */6 * * *\"${NC}"
    echo -e "  • Twice daily:      ${GREEN}CRON_SCHEDULE=\"0 9,21 * * *\"${NC}"
    echo -e "  • Disable:          ${GREEN}CRON_ENABLED=false${NC}"
    echo
    echo -e "  ${BLUE}Management Commands:${NC}"
    echo -e "  • Show status:      ${GREEN}--show-cron${NC}"
    echo -e "  • Update schedule:  ${GREEN}--update-cron${NC}"
    echo -e "  • Disable cron:     ${GREEN}--disable-cron${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# Main installation function
main() {
    echo -e "${BLUE}🔥 SAI Daily Test Service Installer${NC}"
    echo -e "${BLUE}====================================${NC}"
    echo
    
    # Pre-installation checks
    check_root
    check_main_service
    
    # Installation steps
    setup_directories
    install_config
    setup_test_images
    install_dependencies
    copy_daily_test_files
    setup_cron
    
    # Post-installation verification
    if test_installation; then
        show_instructions
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# Handle script interruption
trap 'log_error "Installation interrupted"; exit 130' INT TERM

# Run main installation
if main; then
    log_success "Daily test service installation completed successfully"
    exit 0
else
    log_error "Daily test service installation failed"
    exit 1
fi