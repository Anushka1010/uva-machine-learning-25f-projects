#!/bin/bash

# Start the frontend

echo "Starting frontend..."
echo ""

cd "$(dirname "$0")/team-07/src/frontend"

# Install packages if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm stuff first..."
    npm install
    echo ""
fi

echo "Starting dev server..."
echo "http://localhost:5173"
echo ""
echo ""

npm run dev
