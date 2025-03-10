#!/bin/bash

# Colors for better output formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pneumonia Detection Project Setup ===${NC}"
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Set up virtual environment
echo -e "${BLUE}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created in ./venv${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
else
    echo -e "${YELLOW}Unsupported OS. Please activate the virtual environment manually.${NC}"
    exit 1
fi

# Check if activation was successful
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Failed to activate virtual environment. Please activate it manually:${NC}"
    echo "source venv/bin/activate"
    exit 1
fi

# Install dependencies from requirements.txt
echo -e "${BLUE}Installing required dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}requirements.txt not found. Creating a basic one...${NC}"
    cat > requirements.txt << EOF
tensorflow>=2.5.0
keras>=2.5.0
numpy>=1.19.5
matplotlib>=3.4.2
pandas>=1.3.0
scikit-learn>=0.24.2
pillow>=8.2.0
streamlit>=1.0.0
seaborn>=0.11.1
EOF
    pip install -r requirements.txt
    echo -e "${GREEN}Basic dependencies installed${NC}"
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p models plots temp
echo -e "${GREEN}Created directories: models, plots, temp${NC}"

# Done
echo
echo -e "${GREEN}Setup complete!${NC}"
echo
echo -e "${BLUE}=== How to run the application ===${NC}"
echo
echo -e "1. Ensure your virtual environment is activated:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo
echo -e "2. Train the model (if not already trained):"
echo -e "   ${YELLOW}python scripts/train_model.py --data_path <path_to_data>${NC}"
echo
echo -e "3. Run the Streamlit app:"
echo -e "   ${YELLOW}streamlit run app.py${NC}"
echo
echo -e "4. To make predictions on individual images:"
echo -e "   ${YELLOW}python scripts/predict.py --image <path_to_image>${NC}"
echo
echo -e "${BLUE}Happy diagnosing!${NC}"

