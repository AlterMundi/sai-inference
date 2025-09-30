#!/bin/bash
# Install sai-inference CLI tool system-wide
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../sai-inference.sh"
INSTALL_PATH="/usr/local/bin/sai-inference"

# Check if source script exists
if [ ! -f "$SOURCE_SCRIPT" ]; then
    echo "Error: sai-inference.sh not found at $SOURCE_SCRIPT"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
for cmd in bash curl jq find basename; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: Required dependency '$cmd' not found"
        exit 1
    fi
done

echo "✓ All dependencies present"

# Install script
echo "Installing sai-inference to $INSTALL_PATH..."
cp "$SOURCE_SCRIPT" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"

# Verify installation
if command -v sai-inference &> /dev/null; then
    echo ""
    echo "✓ Installation successful!"
    echo ""
    echo "Usage:"
    echo "  sai-inference --health"
    echo "  sai-inference image.jpg"
    echo "  sai-inference /path/to/images/"
    echo ""
    echo "See 'sai-inference --help' for all options"
else
    echo "Error: Installation failed"
    exit 1
fi
