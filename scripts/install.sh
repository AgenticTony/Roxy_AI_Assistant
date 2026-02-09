#!/bin/bash
set -euo pipefail

# =============================================================================
# Roxy Installation Script
# =============================================================================
# Run: curl -sSL https://raw.githubusercontent.com/anthonyforan/roxy/main/scripts/install.sh | bash
# Or:  cd ~/roxy && bash scripts/install.sh
#
# This script installs all dependencies for Roxy on macOS.
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$HOME/roxy/data"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Roxy Installation Script${NC}"
echo -e "${CYAN}  Local-first AI Assistant for macOS${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# =============================================================================
# Prerequisites Check
# =============================================================================
echo -e "${BLUE}[1/8] Checking prerequisites...${NC}"

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)

if [ "$MACOS_MAJOR" -lt 13 ]; then
    echo -e "${RED}✗ macOS 13.0 (Ventura) or higher required${NC}"
    echo "  Current version: $MACOS_VERSION"
    exit 1
fi
echo -e "${GREEN}✓ macOS $MACOS_VERSION${NC}"

# Check Apple Silicon
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "${GREEN}✓ Apple Silicon detected${NC}"
else
    echo -e "${YELLOW}⚠ Intel Mac detected - some features will be slower${NC}"
fi

# Check Xcode Command Line Tools
if ! command -v clang &> /dev/null; then
    echo -e "${YELLOW}⚠ Xcode Command Line Tools not found${NC}"
    echo "  Installing..."
    xcode-select --install
else
    echo -e "${GREEN}✓ Xcode Command Line Tools${NC}"
fi

# Check for Python 3.12+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.12+ from python.org or brew"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo -e "${RED}✗ Python 3.12+ required, found $PYTHON_VERSION${NC}"
    echo "  Install with: brew install python@3.12"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠ uv not found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo -e "${GREEN}✓ uv installed${NC}"

# Check for Docker (optional but recommended)
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker not found (optional for SearXNG)${NC}"
    echo "  Install with: brew install --cask docker"
    HAS_DOCKER=false
else
    echo -e "${GREEN}✓ Docker available${NC}"
    HAS_DOCKER=true
fi

echo ""

# =============================================================================
# Install System Dependencies
# =============================================================================
echo -e "${BLUE}[2/8] Installing system dependencies...${NC}"

# Check and install Homebrew packages
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠ Homebrew not found${NC}"
    echo "  Please install Homebrew from https://brew.sh"
    echo "  Then run this script again"
    exit 1
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama
else
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

# Check for Hammerspoon (optional)
if command -v hammerspoon &> /dev/null; then
    echo -e "${GREEN}✓ Hammerspoon installed${NC}"
else
    echo -e "${YELLOW}⚠ Hammerspoon not found (optional for window management)${NC}"
    echo "  Install with: brew install hammerspoon"
fi

# PortAudio for microphone access
if brew list portaudio 2>/dev/null | grep -q .; then
    echo -e "${GREEN}✓ PortAudio installed${NC}"
else
    echo "Installing PortAudio..."
    brew install portaudio
fi

echo ""

# =============================================================================
# Set Up Ollama Models
# =============================================================================
echo -e "${BLUE}[3/8] Setting up Ollama models...${NC}"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 5  # Give Ollama time to start
else
    echo -e "${GREEN}✓ Ollama already running${NC}"
    OLLAMA_PID=""
fi

# Pull required models
echo "Ensuring Ollama models are available..."
MODELS=(
    "qwen3:8b"
    "qwen3:0.6b"
    "nomic-embed-text"
)

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}✓ $model already available${NC}"
    else
        echo "Pulling $model..."
        ollama pull "$model"
    fi
done

echo ""

# =============================================================================
# Set Up Python Environment
# =============================================================================
echo -e "${BLUE}[4/8] Setting up Python environment...${NC}"

cd "$PROJECT_ROOT"

echo "Installing Python dependencies..."
uv sync

echo "Installing Playwright browsers..."
uv run playwright install chromium

echo ""

# =============================================================================
# Set Up Configuration
# =============================================================================
echo -e "${BLUE}[5/8] Setting up configuration...${NC}"

# Create .env if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Creating .env file from template..."
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    else
        cat > "$PROJECT_ROOT/.env" << 'EOF'
# Roxy Environment Configuration

# LLM Providers
OLLAMA_HOST=http://localhost:11434

# Cloud LLM (optional - leave empty for local-only mode)
ZAI_API_KEY=
ZAI_BASE_URL=https://api.z.ai/api/paas/v4

# Web Search (optional)
BRAVE_SEARCH_API_KEY=

# Privacy
ROXY_CLOUD_CONSENT_MODE=ask
ROXY_PII_REDACTION=true

# Voice
ROXY_WAKE_WORD=hey roxy
ROXY_TTS_VOICE=af_heart
ROXY_TTS_SPEED=1.1

# General
ROXY_LOG_LEVEL=INFO
ROXY_DATA_DIR=$HOME/roxy/data
EOF
    fi
    echo -e "${GREEN}✓ .env created${NC}"
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

# Create data directory
mkdir -p "$DATA_DIR"
echo -e "${GREEN}✓ Data directory: $DATA_DIR${NC}"

echo ""

# =============================================================================
# Prompt for API Keys (Optional)
# =============================================================================
echo -e "${BLUE}[6/8] Optional API keys...${NC}"
echo ""
echo "Roxy works great with just local Ollama! For web search and other features,"
echo "you can optionally add API keys:"
echo ""

# Check for existing Brave Search API key
if grep -q "BRAVE_SEARCH_API_KEY=" "$PROJECT_ROOT/.env" 2>/dev/null; then
    EXISTING_BRAVE=$(grep "BRAVE_SEARCH_API_KEY=" "$PROJECT_ROOT/.env" | cut -d'=' -f2)
    if [ -n "$EXISTING_BRAVE" ]; then
        echo -e "${GREEN}✓ Brave Search API key already configured${NC}"
    else
        echo -e "${YELLOW}• Brave Search API key (for web search):${NC}"
        echo "  Get free key at: https://search.brave.com/register/api"
        echo ""
        read -p "Enter Brave Search API key (press Enter to skip): " BRAVE_KEY
        if [ -n "$BRAVE_KEY" ]; then
            sed -i.bak "s/BRAVE_SEARCH_API_KEY=.*/BRAVE_SEARCH_API_KEY=$BRAVE_KEY/" "$PROJECT_ROOT/.env"
            rm -f "$PROJECT_ROOT/.env.bak"
            echo -e "${GREEN}✓ Brave Search API key added${NC}"
        fi
    fi
else
    echo "Add later with: export BRAVE_SEARCH_API_KEY=your_key_here"
fi

echo ""

# Check for Z.ai API key
if grep -q "ZAI_API_KEY=" "$PROJECT_ROOT/.env" 2>/dev/null; then
    EXISTING_ZAI=$(grep "ZAI_API_KEY=" "$PROJECT_ROOT/.env" | cut -d'=' -f2)
    if [ -n "$EXISTING_ZAI" ]; then
        echo -e "${GREEN}✓ Z.ai API key already configured${NC}"
    else
        echo -e "${YELLOW}• Z.ai API key (for cloud LLM fallback):${NC}"
        echo "  Get key at: https://open.bigmodel.cn/usercenter/apikeys"
        echo ""
        read -p "Enter Z.ai API key (press Enter to skip - local-only mode works great): " ZAI_KEY
        if [ -n "$ZAI_KEY" ]; then
            sed -i.bak "s/ZAI_API_KEY=.*/ZAI_API_KEY=$ZAI_KEY/" "$PROJECT_ROOT/.env"
            rm -f "$PROJECT_ROOT/.env.bak"
            echo -e "${GREEN}✓ Z.ai API key added${NC}"
        fi
    fi
else
    echo "Add later with: export ZAI_API_KEY=your_key_here"
fi

echo ""

# =============================================================================
# Set Up Optional Components
# =============================================================================
echo -e "${BLUE}[7/8] Setting up optional components...${NC}"

# SearXNG
if [ "$HAS_DOCKER" = true ]; then
    echo "SearXNG provides privacy-focused local search as a fallback"
    read -p "Install SearXNG? (y/N): " INSTALL_SEARXNG
    if [[ "$INSTALL_SEARXNG" =~ ^[Yy]$ ]]; then
        bash "$PROJECT_ROOT/scripts/setup_searxng.sh"
    else
        echo "Skipping SearXNG (install later with: bash scripts/setup_searxng.sh)"
    fi
else
    echo "Docker not available - skipping SearXNG"
fi

# Talon Voice scripts
if [ -d "$HOME/.talon/user" ]; then
    echo "Installing Talon Voice scripts..."
    mkdir -p "$HOME/.talon/user/roxy"
    if [ -d "$PROJECT_ROOT/talon" ]; then
        cp -R "$PROJECT_ROOT/talon/"* "$HOME/.talon/user/roxy/" 2>/dev/null || true
        echo -e "${GREEN}✓ Talon scripts installed${NC}"
    fi
else
    echo "Talon Voice not found - skipping voice scripts"
    echo "Install from: https://talonvoice.com"
fi

echo ""

# =============================================================================
# Verification
# =============================================================================
echo -e "${BLUE}[8/8] Verifying installation...${NC}"

# Test Ollama
echo "Testing Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is responding${NC}"
else
    echo -e "${RED}✗ Ollama is not responding${NC}"
    echo "  Start with: ollama serve"
fi

# Test Python imports
echo "Testing Python imports..."
if uv run python -c "import agno; import httpx; print('✓ Core dependencies OK')" 2>&1 | tee /dev/null; then
    echo -e "${GREEN}✓ Python dependencies working${NC}"
else
    echo -e "${RED}✗ Python dependencies failed${NC}"
fi

echo ""
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Roxy Installation Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Getting Started:${NC}"
echo ""
echo -e "  ${YELLOW}• Start text mode:${NC}"
echo -e "    ${GREEN}cd $PROJECT_ROOT && uv run roxy${NC}"
echo ""
echo -e "  ${YELLOW}• Start voice mode:${NC}"
echo -e "    ${GREEN}cd $PROJECT_ROOT && uv run roxy --voice${NC}"
echo ""
echo -e "  ${YELLOW}• Start background service:${NC}"
echo -e "    ${GREEN}cd $PROJECT_ROOT && uv run roxy --server${NC}"
echo ""
echo -e "  ${YELLOW}• Enable verbose logging:${NC}"
echo -e "    ${GREEN}uv run roxy --verbose${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo ""
echo "  Config file:   $PROJECT_ROOT/config/default.yaml"
echo "  Environment:   $PROJECT_ROOT/.env"
echo "  Data dir:     $DATA_DIR"
echo ""
echo -e "${CYAN}Optional Setup:${NC}"
echo ""
echo "  • Add API keys to .env for web search and cloud LLM"
echo "  • Install Talon Voice from https://talonvoice.com"
echo "  • Grant microphone access in System Settings > Privacy"
echo "  • Set Roxy to start automatically on login"
echo ""
echo -e "${CYAN}Troubleshooting:${NC}"
echo ""
echo "  • Ollama not starting:  brew uninstall ollama && brew install ollama"
echo "  • Python errors:      uv sync --reinstall"
echo "  • Permission denied:  chmod +x scripts/*.sh"
echo "  • View logs:          tail -f $DATA_DIR/roxy.log"
echo ""

# Make scripts executable
chmod +x "$PROJECT_ROOT/scripts/"*.sh 2>/dev/null || true

echo -e "${GREEN}Happy coding with Roxy! 🤖${NC}"
echo ""

# Reminder about cleanup
if [ -n "$OLLAMA_PID" ]; then
    echo "Note: Ollama was started by this script."
    echo "      It will continue running in the background."
    echo "      Stop with: kill $OLLAMA_PID"
    echo ""
fi
