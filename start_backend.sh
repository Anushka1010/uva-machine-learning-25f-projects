#!/bin/bash

# Start the backend

echo "Starting backend..."
echo ""

cd "$(dirname "$0")/team-07/src/backend"

# Make venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual env first..."
    python3 -m venv venv
    echo ""
fi

# Turn it on
source venv/bin/activate

# Install packages if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing packages..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo ""
fi

echo "Starting server..."
echo "http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo ""
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m app.main