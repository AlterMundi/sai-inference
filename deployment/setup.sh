#!/bin/bash
set -e

echo "🔥 SAI Inference Service Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}Warning: Running as root. Consider using a dedicated user for production.${NC}"
fi

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version meets requirements (>= $required_version)${NC}"
else
    echo -e "${RED}✗ Python $python_version is below required version $required_version${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Check for existing models
echo -e "${YELLOW}Checking models directory...${NC}"
echo
model_count=$(find models -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)

if [ $model_count -gt 0 ]; then
    echo -e "${GREEN}✓ Found $model_count model(s) in models/ directory:${NC}"
    for model in models/*.pt models/*.pth; do
        if [ -f "$model" ]; then
            size=$(du -h "$model" | cut -f1)
            echo -e "  ${BLUE}→ $(basename $model)${NC} ($size)"
        fi
    done
else
    echo -e "${YELLOW}⚠ No models found in models/ directory${NC}"
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}HOW TO ADD MODELS:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "Copy your YOLO model files (.pt) to the ${GREEN}models/${NC} directory:"
echo
echo -e "  ${BLUE}Model setup:${NC}"
echo -e "  • Place your model file as 'last.pt' in the models/ directory"
echo -e "  • Or copy any YOLO .pt model to the models/ directory"
echo
echo -e "  ${BLUE}Copy command:${NC}"
echo -e "  ${GREEN}cp /path/to/your/model.pt models/your_model_name.pt${NC}"
echo
echo -e "  ${BLUE}Configure default model (optional):${NC}"
echo -e "  Set via environment: ${GREEN}export SAI_DEFAULT_MODEL=your_model_name.pt${NC}"
echo -e "  Or create .env file: ${GREEN}cp .env.example .env${NC} and edit"
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Configuration setup
echo -e "${YELLOW}Configuration setup...${NC}"

# Auto-detect CUDA availability for user information
DETECTED_DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DETECTED_DEVICE="cuda"
    echo -e "${GREEN}✓ CUDA detected - GPU acceleration available${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not available - will use CPU inference${NC}"
fi

if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file already exists (keeping existing configuration)${NC}"
else
    echo -e "${BLUE}ℹ Configuration via environment variables (no .env file needed)${NC}"
    echo -e "  ${GREEN}Detected device: $DETECTED_DEVICE${NC}"
    echo -e "  ${BLUE}Optional:${NC} Copy .env.example to .env if you want to customize settings"
    echo -e "  ${BLUE}Command:${NC} cp .env.example .env"
fi

# Test installation
echo
echo -e "${YELLOW}Testing installation...${NC}"
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ PyTorch installed${NC}" || echo -e "${RED}✗ PyTorch installation failed${NC}"
python -c "import ultralytics; print(f'  ✓ Ultralytics: {ultralytics.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ Ultralytics installed${NC}" || echo -e "${RED}✗ Ultralytics installation failed${NC}"
python -c "import fastapi; print(f'  ✓ FastAPI: {fastapi.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ FastAPI installed${NC}" || echo -e "${RED}✗ FastAPI installation failed${NC}"

# Check CUDA availability
echo
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'  CUDA Available: {cuda_available}'); print(f'  CUDA Version: {torch.version.cuda if cuda_available else \"N/A\"}'); print(f'  Device Count: {torch.cuda.device_count() if cuda_available else 0}')" 2>/dev/null

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo
echo -e "1. ${BLUE}Add a model:${NC}"
if [ $model_count -eq 0 ]; then
    echo -e "   ${RED}⚠ REQUIRED: Copy a YOLO model to models/ directory${NC}"
    echo -e "   ${GREEN}cp /path/to/model.pt models/${NC}"
else
    echo -e "   ${GREEN}✓ Models already available${NC}"
fi
echo
echo -e "2. ${BLUE}Configure service (optional):${NC}"
echo -e "   ${BLUE}cp .env.example .env${NC} (if you want custom settings)"
echo -e "   Or use environment variables: ${GREEN}export SAI_DEFAULT_MODEL=your_model.pt${NC}"
echo
echo -e "3. ${BLUE}Run the service:${NC}"
echo -e "   ${GREEN}python run.py${NC} (development)"
echo -e "   ${GREEN}uvicorn src.main:app --host 0.0.0.0 --port 8888${NC} (production)"
echo
echo -e "4. ${BLUE}Optional - Install as system service:${NC}"
echo -e "   ${GREEN}sudo cp sai-inference.service /etc/systemd/system/${NC}"
echo -e "   ${GREEN}sudo systemctl daemon-reload${NC}"
echo -e "   ${GREEN}sudo systemctl enable sai-inference${NC}"
echo -e "   ${GREEN}sudo systemctl start sai-inference${NC}"
echo
echo -e "5. ${BLUE}Optional - Install daily test service:${NC}"
echo -e "   ${GREEN}sudo ./deployment/install-daily-test.sh${NC}"
echo -e "   ${YELLOW}(Automated testing for alert system validation)${NC}"
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}SERVICE ENDPOINTS:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  ${BLUE}Health Check:${NC}      http://localhost:8888/api/v1/health"
echo -e "  ${BLUE}Inference API:${NC}     http://localhost:8888/api/v1/infer"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"