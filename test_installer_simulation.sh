#!/bin/bash
#
# SAI Inference Service - Installation Simulation Script
# Tests what the installer would do on the current system
#
set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m'

# Script configuration (matching installer)
readonly SERVICE_USER="service"
readonly SERVICE_GROUP="service" 
readonly INSTALL_DIR="/opt/sai-inference"
readonly SERVICE_NAME="sai-inference"
readonly LOG_DIR="/var/log/sai-inference"
readonly CONFIG_DIR="/etc/sai-inference"

echo -e "${MAGENTA}=== SAI Inference Service - Installation Simulation ===${NC}"
echo -e "${MAGENTA}This is a DRY RUN - no changes will be made${NC}\n"

# Function to simulate actions
simulate() {
    echo -e "${YELLOW}[SIMULATE]${NC} $*"
}

# Function to check current state
check() {
    echo -e "${BLUE}[CHECK]${NC} $*"
}

# Function to report issues
issue() {
    echo -e "${RED}[ISSUE]${NC} $*"
}

# Function to report success
success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

echo "=== Phase 1: Pre-flight Checks ==="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    issue "Script needs root - would fail immediately"
else
    success "Running as root"
fi

# Check Python version
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    success "Python 3.8+ available"
else
    issue "Python 3.8+ not found"
fi

# Check if service is running
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    check "Service is currently RUNNING"
    simulate "Would stop service before deployment"
else
    check "Service is not running"
fi

echo -e "\n=== Phase 2: User & Directory Setup ==="

# Check service user
if id "$SERVICE_USER" &>/dev/null; then
    success "Service user exists: $SERVICE_USER"
else
    simulate "Would create user: $SERVICE_USER"
fi

# Check directories
for dir in "$INSTALL_DIR" "$LOG_DIR" "$CONFIG_DIR"; do
    if [[ -d "$dir" ]]; then
        check "Directory exists: $dir"
        if [[ "$dir" == "$INSTALL_DIR" ]]; then
            # Check what would be overwritten
            echo -e "  ${YELLOW}Contents that would be affected:${NC}"
            ls -la "$dir" | head -5 | sed 's/^/    /'
        fi
    else
        simulate "Would create directory: $dir"
    fi
done

echo -e "\n=== Phase 3: Code Deployment Analysis ==="

# Check what would be replaced
if [[ -d "$INSTALL_DIR/src" ]]; then
    check "Existing src/ directory would be REPLACED"
    echo -e "  ${YELLOW}Current version info:${NC}"
    if [[ -f "$INSTALL_DIR/src/config.py" ]]; then
        grep -m1 "default=" "$INSTALL_DIR/src/config.py" | head -3 | sed 's/^/    /'
    fi
fi

# Check for backup files
if ls "$INSTALL_DIR"/*.old 2>/dev/null | grep -q .; then
    issue "Found .old backup files - may indicate previous manual updates"
    ls "$INSTALL_DIR"/*.old | sed 's/^/    /'
fi

echo -e "\n=== Phase 4: Virtual Environment ==="

if [[ -d "$INSTALL_DIR/venv" ]]; then
    issue "Existing venv would be DELETED and recreated"
    check "Current venv size: $(du -sh "$INSTALL_DIR/venv" 2>/dev/null | cut -f1)"
    simulate "Would remove and recreate venv (may take 2-3 minutes)"
else
    simulate "Would create new venv"
fi

echo -e "\n=== Phase 5: Model Management ==="

# Check models
if [[ -d "$INSTALL_DIR/models" ]]; then
    check "Models directory exists"
    echo -e "  ${BLUE}Current models:${NC}"
    ls -lh "$INSTALL_DIR/models"/*.pt 2>/dev/null | sed 's/^/    /' || echo "    No .pt files found"
fi

# Check if repo has model
if [[ -f "$(dirname "$0")/models/last.pt" ]]; then
    success "Repository has model file ready"
else
    issue "No model in repository - would show warning"
fi

echo -e "\n=== Phase 6: Configuration ==="

if [[ -f "$CONFIG_DIR/production.env" ]]; then
    issue "Existing production.env would be OVERWRITTEN"
    echo -e "  ${YELLOW}Current configuration highlights:${NC}"
    grep -E "SAI_INPUT_SIZE|SAI_CONFIDENCE|SAI_DEVICE" "$CONFIG_DIR/production.env" | sed 's/^/    /'
    simulate "Would replace with new template (backup recommended)"
else
    simulate "Would create new production.env"
fi

echo -e "\n=== Phase 7: SystemD Service ==="

if [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
    check "Service file exists"
    simulate "Would overwrite service file and reload systemd"
else
    simulate "Would create new service file"
fi

echo -e "\n=== Phase 8: Impact Summary ==="

echo -e "${MAGENTA}Critical Changes That Would Occur:${NC}"
echo "1. ❌ Service would be stopped (downtime)"
echo "2. ❌ Virtual environment would be deleted and recreated"
echo "3. ❌ Source code in src/ would be completely replaced"
echo "4. ⚠️  Configuration file would be overwritten"
echo "5. ✅ Models would be preserved"
echo "6. ✅ Logs would be preserved"

echo -e "\n${MAGENTA}Estimated Downtime:${NC}"
echo "- Minimum: 2-3 minutes (venv recreation + pip install)"
echo "- Maximum: 5-10 minutes (if dependencies need compilation)"

echo -e "\n${MAGENTA}Data Loss Risk:${NC}"
if [[ -f "$CONFIG_DIR/production.env" ]]; then
    echo -e "${RED}HIGH${NC} - Custom configuration in production.env would be lost"
    echo "  Recommendation: Backup $CONFIG_DIR/production.env first"
fi

echo -e "\n${MAGENTA}Recommended Pre-Update Actions:${NC}"
echo "1. Backup configuration: cp $CONFIG_DIR/production.env $CONFIG_DIR/production.env.backup"
echo "2. Backup current code: cp -r $INSTALL_DIR $INSTALL_DIR.backup"
echo "3. Note current settings: sai-service health > /tmp/sai-health-before.json"
echo "4. Schedule maintenance window (5-10 minutes)"

echo -e "\n${MAGENTA}=== End of Simulation ===${NC}"