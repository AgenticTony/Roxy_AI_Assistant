#!/bin/bash
set -euo pipefail

# =============================================================================
# Ollama Setup Script for Roxy
# =============================================================================
# Ensures Ollama is installed, running, and has all required models.
#
# Usage:
#   bash scripts/setup_ollama.sh
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Required models
MODELS=(
    "qwen3:8b"
    "qwen3:0.6b"
    "nomic-embed-text"
)

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Ollama Setup for Roxy${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# =============================================================================
# Check/Install Ollama
# =============================================================================
echo -e "${BLUE}[1/4] Checking Ollama installation...${NC}"

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found, installing via Homebrew...${NC}"

    if ! command -v brew &> /dev/null; then
        echo -e "${RED}✗ Homebrew not found${NC}"
        echo "  Please install Homebrew first: https://brew.sh"
        exit 1
    fi

    brew install ollama
    echo -e "${GREEN}✓ Ollama installed${NC}"
else
    # Get version if possible
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Ollama installed ($OLLAMA_VERSION)${NC}"
fi

echo ""

# =============================================================================
# Start Ollama Server
# =============================================================================
echo -e "${BLUE}[2/4] Starting Ollama server...${NC}"

# Check if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✓ Ollama is already running${NC}"
else
    echo "Starting Ollama server..."

    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!

    # Save PID for later cleanup
    echo $OLLAMA_PID > /tmp/roxy_ollama.pid

    echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"

    # Wait for Ollama to be ready
    echo -n "  Waiting for Ollama to be ready"

    for i in {1..30}; do
        if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi

        if [ $i -eq 30 ]; then
            echo -e " ${RED}✗${NC}"
            echo ""
            echo -e "${RED}Error: Ollama failed to start within 30 seconds${NC}"
            echo "  Check logs: tail -f /tmp/ollama.log"
            exit 1
        fi

        sleep 1
        echo -n "."
    done

    echo ""
fi

echo ""

# =============================================================================
# Verify Ollama Connection
# =============================================================================
echo -e "${BLUE}[3/4] Verifying Ollama connection...${NC}"

# Test API endpoint
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama API is responding at $OLLAMA_HOST${NC}"
else
    echo -e "${RED}✗ Cannot connect to Ollama at $OLLAMA_HOST${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if Ollama is running: pgrep -x ollama"
    echo "  2. Check logs: tail -f /tmp/ollama.log"
    echo "  3. Try restarting: pkill ollama && ollama serve &"
    exit 1
fi

# Get currently installed models
echo ""
echo "Currently installed models:"
ollama list 2>/dev/null || echo "  (none yet)"

echo ""

# =============================================================================
# Pull/Update Models
# =============================================================================
echo -e "${BLUE}[4/4] Ensuring required models are available...${NC}"

for model in "${MODELS[@]}"; do
    echo -n "  Checking $model... "

    # Check if model exists
    if ollama list | grep -q "$model"; then
        # Check if we should update
        echo -e "${GREEN}✓ Already installed${NC}"

        # Optionally check for updates
        # For now, we'll skip updates to save time
        # Uncomment to enable auto-update:
        # echo "    Updating..."
        # ollama pull "$model"
    else
        echo "Pulling..."
        if ollama pull "$model"; then
            echo -e "    ${GREEN}✓ Installed${NC}"
        else
            echo -e "    ${RED}✗ Failed to pull $model${NC}"
        fi
    fi
done

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Ollama Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Installed Models:${NC}"
for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo -e "  ${GREEN}✓${NC} $model"
    fi
done
echo ""
echo -e "${CYAN}Ollama Server:${NC}"
echo "  Host: $OLLAMA_HOST"
echo "  Status: $(pgrep -x ollama > /dev/null && echo "Running" || echo "Stopped")"
echo ""
echo -e "${CYAN}Useful Commands:${NC}"
echo "  • List models:    ollama list"
echo "  • Pull model:     ollama pull <model>"
echo "  • Show logs:      tail -f /tmp/ollama.log"
echo "  • Stop server:    pkill ollama"
echo "  • Start server:   ollama serve &"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Start Roxy:   uv run roxy"
echo "  2. Voice mode:    uv run roxy --voice"
echo "  3. Server mode:   uv run roxy --server"
echo ""
echo -e "${GREEN}Happy chatting! 🤖${NC}"
echo ""
