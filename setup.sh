#!/bin/bash

# First-time setup

echo "Setting up..."
echo ""

# Backend
echo "Backend stuff:"
cd team-07/src/backend

if [ ! -d "venv" ]; then
    echo "  Making virtual env..."
    python3 -m venv venv
fi

echo "  Activating it..."
source venv/bin/activate

echo "  Installing packages..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Done"

# Frontend
cd ../frontend
echo ""
echo "Frontend stuff:"

if [ ! -d "node_modules" ]; then
    echo "  Installing npm packages..."
    npm install
fi
echo "  Done"

# Check Ollama
cd ../../..
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "  Ollama is installed"
    
    echo "  Checking for llama3.2 model..."
    if ollama list | grep -q llama3.2; then
        echo "  Got it"
    else
        echo "  Model not found"
        echo "  Run: ollama pull llama3.2"
    fi
else
    echo "Ollama is not installed"
    echo "Install with: brew install ollama"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the application:"
echo ""
echo "1. Start backend:"
echo "   cd team-07/src/backend"
echo "   source venv/bin/activate"
echo "   python -m app.main"
echo ""
echo "2. Start Ollama (in another terminal):"
echo "   ollama serve"
echo ""
echo "3. Start frontend (in another terminal):"
echo "   cd team-07/src/frontend"
echo "   npm run dev"
echo ""
echo "4. Open browser: http://localhost:5173"