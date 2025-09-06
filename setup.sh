#!/bin/bash
set -e

echo "🔥 SAI Inference Service Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}Warning: Running as root. Consider using a dedicated user for production.${NC}"
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Copy SAI model if available
echo -e "${YELLOW}Looking for SAI models...${NC}"
if [ -f "/mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt" ]; then
    echo -e "${GREEN}Copying SAINet2.1 model...${NC}"
    cp /mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt models/sai_v2.1.pt
elif [ -f "/mnt/n8n-data/SAINet_v1.0/run_stage2/weights/best.pt" ]; then
    echo -e "${GREEN}Copying stage2 model...${NC}"
    cp /mnt/n8n-data/SAINet_v1.0/run_stage2/weights/best.pt models/sai_stage2.pt
else
    echo -e "${RED}No SAI models found. Please copy a model to the models/ directory.${NC}"
    echo "Expected locations:"
    echo "  - /mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt"
    echo "  - /mnt/n8n-data/SAINet_v1.0/run_stage2/weights/best.pt"
fi

# Create configuration file
echo -e "${YELLOW}Creating configuration...${NC}"
cp .env.example .env

# Test installation
echo -e "${YELLOW}Testing installation...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit .env file to configure the service"
echo "2. Run the service: python run.py"
echo "3. Or install as system service:"
echo "   sudo cp sai-inference.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable sai-inference"
echo "   sudo systemctl start sai-inference"
echo
echo -e "${YELLOW}API Documentation will be available at:${NC}"
echo "http://localhost:8888/api/v1/docs"
echo
echo -e "${YELLOW}n8n Webhook endpoint:${NC}"
echo "http://localhost:8888/webhook/sai"