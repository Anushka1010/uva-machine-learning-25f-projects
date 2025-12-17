#!/bin/bash
# run_local.sh - Quick local development runner
#
# Usage:
#   ./scripts/run_local.sh          # Run full pipeline + start API
#   ./scripts/run_local.sh train    # Just train the model
#   ./scripts/run_local.sh api      # Just start the API
#   ./scripts/run_local.sh test     # Run tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================="
echo "  Fair Credit Score Prediction - Local Dev"
echo "============================================="
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Run this script from the project root directory${NC}"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    pip install -q fastapi uvicorn pytest
    touch venv/.deps_installed
fi

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Handle command
COMMAND=${1:-all}

case $COMMAND in
    train)
        echo -e "${GREEN}Training model...${NC}"
        
        # Generate sample data if it doesn't exist
        if [ ! -f "data/sample_credit_data.csv" ]; then
            echo -e "${YELLOW}Generating sample data...${NC}"
            python scripts/generate_sample_data.py
        fi
        
        python -m src.main run --data-path data/sample_credit_data.csv
        ;;
        
    api)
        echo -e "${GREEN}Starting API server...${NC}"
        
        # Check if model exists
        if [ ! -f "models/credit_model.pkl" ]; then
            echo -e "${YELLOW}Model not found. Training first...${NC}"
            $0 train
        fi
        
        echo -e "${GREEN}API running at: http://127.0.0.1:8000${NC}"
        echo -e "${GREEN}Docs at: http://127.0.0.1:8000/docs${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
        ;;
        
    test)
        echo -e "${GREEN}Running tests...${NC}"
        pytest -v
        ;;
        
    all)
        echo -e "${GREEN}Running full pipeline...${NC}"
        
        # Generate sample data if it doesn't exist
        if [ ! -f "data/sample_credit_data.csv" ]; then
            echo -e "${YELLOW}Generating sample data...${NC}"
            python scripts/generate_sample_data.py
        fi
        
        # Train model
        echo -e "${GREEN}Training model...${NC}"
        python -m src.main run --data-path data/sample_credit_data.csv
        
        # Start API
        echo ""
        echo -e "${GREEN}Starting API server...${NC}"
        echo -e "${GREEN}API running at: http://127.0.0.1:8000${NC}"
        echo -e "${GREEN}Docs at: http://127.0.0.1:8000/docs${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
        ;;
        
    clean)
        echo -e "${YELLOW}Cleaning generated files...${NC}"
        rm -rf models/*.pkl
        rm -rf explanations/*.txt
        rm -rf reports/figures/*.png
        rm -rf outputs/*.json
        rm -rf logs/*.log
        echo -e "${GREEN}Done!${NC}"
        ;;
        
    *)
        echo "Usage: $0 {train|api|test|all|clean}"
        echo ""
        echo "Commands:"
        echo "  train  - Train the model"
        echo "  api    - Start the API server"
        echo "  test   - Run tests"
        echo "  all    - Train + start API (default)"
        echo "  clean  - Remove generated files"
        exit 1
        ;;
esac
