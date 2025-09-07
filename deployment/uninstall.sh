#!/bin/bash
#
# SAI Inference Service - Uninstall Script
# Safely removes the SAI inference ecosystem
#
set -euo pipefail

# Configuration
readonly SERVICE_NAME="sai-inference"
readonly INSTALL_DIR="/opt/sai-inference"
readonly LOG_DIR="/var/log/sai-inference"
readonly CONFIG_DIR="/etc/sai-inference"
readonly SERVICE_USER="service"

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

# Confirm uninstallation
confirm_uninstall() {
    echo
    log_warning "This will completely remove the SAI Inference Service and all associated files."
    log_warning "The following will be removed:"
    echo "  - Service: $SERVICE_NAME"
    echo "  - Installation directory: $INSTALL_DIR"
    echo "  - Configuration: $CONFIG_DIR"
    echo "  - Logs: $LOG_DIR"
    echo "  - Management scripts: /usr/local/bin/sai-service"
    echo
    read -p "Are you sure you want to continue? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstallation cancelled"
        exit 0
    fi
    echo
}

# Stop and disable service
stop_service() {
    log_info "Stopping and disabling service..."
    
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
        log_success "Service disabled"
    fi
    
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl stop "$SERVICE_NAME"
        log_success "Service stopped"
    fi
}

# Remove systemd service
remove_systemd_service() {
    log_info "Removing systemd service..."
    
    if [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
        rm -f "/etc/systemd/system/$SERVICE_NAME.service"
        systemctl daemon-reload
        log_success "Systemd service removed"
    fi
}

# Remove directories and files
remove_files() {
    log_info "Removing installation files..."
    
    # Remove installation directory
    if [[ -d "$INSTALL_DIR" ]]; then
        rm -rf "$INSTALL_DIR"
        log_success "Installation directory removed: $INSTALL_DIR"
    fi
    
    # Remove configuration directory
    if [[ -d "$CONFIG_DIR" ]]; then
        rm -rf "$CONFIG_DIR"
        log_success "Configuration directory removed: $CONFIG_DIR"
    fi
    
    # Remove log directory (optional, preserve logs by default)
    read -p "Remove log directory $LOG_DIR? [y/N]: " -r
    if [[ $REPLY =~ ^[Yy]$ ]] && [[ -d "$LOG_DIR" ]]; then
        rm -rf "$LOG_DIR"
        log_success "Log directory removed: $LOG_DIR"
    fi
    
    # Remove management script
    if [[ -L "/usr/local/bin/sai-service" ]]; then
        rm -f "/usr/local/bin/sai-service"
        log_success "Management script removed"
    fi
    
    # Remove log rotation config
    if [[ -f "/etc/logrotate.d/$SERVICE_NAME" ]]; then
        rm -f "/etc/logrotate.d/$SERVICE_NAME"
        log_success "Log rotation configuration removed"
    fi
}

# Optionally remove service user
remove_user() {
    if id "$SERVICE_USER" &>/dev/null; then
        read -p "Remove service user '$SERVICE_USER'? [y/N]: " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            userdel "$SERVICE_USER" 2>/dev/null || true
            log_success "Service user removed: $SERVICE_USER"
        fi
    fi
}

# Main uninstall function
main() {
    log_info "SAI Inference Service Uninstaller"
    
    check_root
    confirm_uninstall
    stop_service
    remove_systemd_service
    remove_files
    remove_user
    
    echo
    log_success "SAI Inference Service has been completely uninstalled"
    log_info "Thank you for using SAI Inference Service!"
}

# Run main function
main "$@"