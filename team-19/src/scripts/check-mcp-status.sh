#!/usr/bin/env bash

echo "🔍 CanvasGPT MCP Setup Status"
echo "=============================="
echo

# Check launcher script
if [ -x "./bin/canvasgpt-mcp" ]; then
  echo "✅ Launcher script exists and is executable"
else
  echo "❌ Launcher script missing or not executable"
  echo "   Run: chmod +x ./bin/canvasgpt-mcp"
fi

# Check standalone server
if [ -f "./mcp-server-standalone.ts" ]; then
  echo "✅ Standalone server file exists"
else
  echo "❌ Standalone server file missing"
fi

# Check database
DB_PATH="$HOME/Library/Application Support/canvasgpt/database.db"
if [ -f "$DB_PATH" ]; then
  echo "✅ Database exists"
  SIZE=$(ls -lh "$DB_PATH" | awk '{print $5}')
  echo "   Size: $SIZE"
  
  # Count items
  ITEM_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM universal_items;" 2>/dev/null || echo "0")
  echo "   Items: $ITEM_COUNT"
else
  echo "⚠️  Database not found (will use test data)"
  echo "   Expected: $DB_PATH"
fi

# Check API key
CONFIG_PATH="$HOME/Library/Application Support/canvasgpt/config.json"
if [ -f "$CONFIG_PATH" ]; then
  if grep -q "OPENAI_API_KEY" "$CONFIG_PATH" 2>/dev/null; then
    echo "✅ OpenAI API key configured"
  else
    echo "⚠️  Config exists but no API key found"
  fi
else
  echo "⚠️  API key not configured"
  echo "   Expected: $CONFIG_PATH"
fi

# Check Claude Desktop config
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -f "$CLAUDE_CONFIG" ]; then
  echo "✅ Claude Desktop config exists"
  if grep -q "canvasgpt" "$CLAUDE_CONFIG" 2>/dev/null; then
    echo "   CanvasGPT server is configured"
  fi
else
  echo "❌ Claude Desktop config not found"
  echo "   Expected: $CLAUDE_CONFIG"
fi

# Check if tsx is installed
if command -v npx &> /dev/null; then
  if npx tsx --version &> /dev/null; then
    echo "✅ tsx is installed"
  else
    echo "❌ tsx not installed"
    echo "   Run: npm install"
  fi
else
  echo "❌ npx not found"
fi

echo
echo "📝 Next Steps:"
echo "=============="

if [ ! -f "$DB_PATH" ]; then
  echo "1. Create test database: npx tsx scripts/create-test-db.ts"
  echo "2. Or run the Electron app to create real database"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "3. Set up OpenAI API key (run the Electron app)"
fi

if [ ! -f "$CLAUDE_CONFIG" ]; then
  echo "4. Configure Claude Desktop (see MCP-SETUP-COMPLETE.md)"
fi

echo "5. Restart Claude Desktop"
echo "6. Ask Claude: 'What assignments are due this week?'"

echo
echo "💡 Test the server:"
echo "   ./bin/canvasgpt-mcp"
