#!/bin/bash

# Quick test to see if everything's running

echo "Testing the system..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

# Test an endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "$name... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$response" -eq "$expected" ]; then
        echo -e "${GREEN}ok${NC}"
        ((PASSED++))
    else
        echo -e "${RED}nope${NC} (got $response)"
        ((FAILED++))
    fi
}

# Run tests
test_endpoint "Backend" "http://localhost:8000/" 200
test_endpoint "Files endpoint" "http://localhost:8000/api/files" 200

echo -n "Frontend... "
if curl -s http://localhost:5173 | grep -q "RAG"; then
    echo -e "${GREEN}ok${NC}"
    ((PASSED++))
else
    echo -e "${RED}nope${NC}"
    ((FAILED++))
fi

echo -n "Ollama... "
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}ok${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}not running${NC}"
fi

echo ""
echo "Results: ${GREEN}$PASSED ok${NC}"
if [ $FAILED -gt 0 ]; then
    echo "         ${RED}$FAILED failed${NC}"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}System looks good!${NC}"
    echo "Open: http://localhost:5173"
    exit 0
else
    echo -e "${RED}Something's not working${NC}"
    echo ""
    echo "Try:"
    echo "  Backend: cd team-07/src/backend && python -m app.main"
    echo "  Frontend: cd team-07/src/frontend && npm run dev"
    echo "  Check ports: lsof -i :8000,5173"
    exit 1
fi
