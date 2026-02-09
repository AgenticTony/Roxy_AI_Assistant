#!/bin/bash
set -euo pipefail

# ============================================
# Roxy Uninstaller
# Completely removes Roxy and all associated data from your system
# Run: bash scripts/uninstall.sh
# ============================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

ROXY_DIR="$HOME/roxy"
TALON_DIR="$HOME/.talon/user/roxy"
LAUNCH_AGENT="$HOME/Library/LaunchAgents/com.anthonyforan.roxy.plist"
DATA_DIR="$ROXY_DIR/data"

echo ""
echo -e "${RED}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${RED}${BOLD}║         Roxy Uninstaller                 ║${NC}"
echo -e "${RED}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Summary of what will be removed ────────────────────────────────

echo -e "${BOLD}Scanning for Roxy installation...${NC}"
echo ""

ITEMS_FOUND=0

if [ -d "$ROXY_DIR" ]; then
    ROXY_SIZE=$(du -sh "$ROXY_DIR" 2>/dev/null | cut -f1)
    echo -e "  ${YELLOW}●${NC} Project directory: $ROXY_DIR ($ROXY_SIZE)"
    ITEMS_FOUND=$((ITEMS_FOUND + 1))
fi

if [ -d "$DATA_DIR" ]; then
    DATA_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    echo -e "  ${YELLOW}●${NC} Memory & conversation data: $DATA_DIR ($DATA_SIZE)"
    echo -e "    ${RED}⚠  This includes all of Roxy's memories and conversation history${NC}"
fi

if [ -f "$LAUNCH_AGENT" ]; then
    echo -e "  ${YELLOW}●${NC} LaunchAgent (auto-start): $LAUNCH_AGENT"
    ITEMS_FOUND=$((ITEMS_FOUND + 1))
fi

if [ -d "$TALON_DIR" ]; then
    echo -e "  ${YELLOW}●${NC} Talon Voice scripts: $TALON_DIR"
    ITEMS_FOUND=$((ITEMS_FOUND + 1))
fi

# Check for Docker containers
if command -v docker &> /dev/null; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "searxng"; then
        echo -e "  ${YELLOW}●${NC} SearXNG Docker container"
        ITEMS_FOUND=$((ITEMS_FOUND + 1))
    fi
fi

# Check for Ollama models
if command -v ollama &> /dev/null; then
    echo ""
    echo -e "  ${BLUE}Ollama models (optional removal):${NC}"
    for model in "qwen3:8b" "qwen3:0.6b" "nomic-embed-text"; do
        if ollama list 2>/dev/null | grep -q "$model"; then
            MODEL_SIZE=$(ollama list 2>/dev/null | grep "$model" | awk '{print $3, $4}')
            echo -e "    ${YELLOW}●${NC} $model ($MODEL_SIZE)"
        fi
    done
fi

if [ "$ITEMS_FOUND" -eq 0 ]; then
    echo -e "  ${GREEN}Nothing found — Roxy doesn't appear to be installed.${NC}"
    exit 0
fi

# ── Confirmation ─────────────────────────────────────────────────

echo ""
echo -e "${RED}${BOLD}This action is irreversible. All memories and data will be permanently deleted.${NC}"
echo ""
read -p "Are you sure you want to uninstall Roxy? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo ""
    echo -e "${GREEN}Uninstall cancelled. Roxy lives another day.${NC}"
    exit 0
fi

echo ""

# ── Optional: Export data before deletion ────────────────────────────

read -p "Export your data (memories, logs) before deletion? (y/n): " EXPORT_DATA
if [ "$EXPORT_DATA" = "y" ] || [ "$EXPORT_DATA" = "Y" ]; then
    EXPORT_DIR="$HOME/Desktop/roxy-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$EXPORT_DIR"

    if [ -f "$DATA_DIR/memory.db" ]; then
        cp "$DATA_DIR/memory.db" "$EXPORT_DIR/"
    fi
    if [ -d "$DATA_DIR/chromadb" ]; then
        cp -r "$DATA_DIR/chromadb" "$EXPORT_DIR/"
    fi
    if [ -d "$DATA_DIR/mem0" ]; then
        cp -r "$DATA_DIR/mem0" "$EXPORT_DIR/"
    fi
    if [ -f "$DATA_DIR/cloud_requests.log" ]; then
        cp "$DATA_DIR/cloud_requests.log" "$EXPORT_DIR/"
    fi
    if [ -f "$DATA_DIR/roxy.log" ]; then
        cp "$DATA_DIR/roxy.log" "$EXPORT_DIR/"
    fi

    echo -e "  ${GREEN}✓${NC} Data exported to: $EXPORT_DIR"
fi

# ── Step 1: Stop Roxy if running ─────────────────────────────────────

echo -e "${BOLD}Stopping Roxy...${NC}"

# Unload LaunchAgent if loaded
if [ -f "$LAUNCH_AGENT" ]; then
    launchctl unload "$LAUNCH_AGENT" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} LaunchAgent unloaded"
fi

# Kill any running Roxy processes
if pgrep -f "roxy.main" > /dev/null 2>&1; then
    pkill -f "roxy.main" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Roxy processes stopped"
fi

# Kill Talon bridge socket if active
if [ -S "/tmp/roxy_talon.sock" ]; then
    rm -f /tmp/roxy_talon.sock
    echo -e "  ${GREEN}✓${NC} Talon bridge socket removed"
fi

# ── Step 2: Remove LaunchAgent ───────────────────────────────────

if [ -f "$LAUNCH_AGENT" ]; then
    rm -f "$LAUNCH_AGENT"
    echo -e "  ${GREEN}✓${NC} LaunchAgent removed"
fi

# ── Step 3: Remove Talon scripts ─────────────────────────────────

if [ -d "$TALON_DIR" ]; then
    rm -rf "$TALON_DIR"
    echo -e "  ${GREEN}✓${NC} Talon Voice scripts removed"
fi

# ── Step 4: Remove SearXNG Docker container ───────────────────────

if command -v docker &> /dev/null; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "searxng"; then
        docker stop searxng 2>/dev/null || true
        docker rm searxng 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} SearXNG Docker container removed"
    fi
    # Remove docker-compose file if it exists
    if [ -f "$ROXY_DIR/docker-compose.yml" ]; then
        (cd "$ROXY_DIR" && docker compose down 2>/dev/null) || true
    fi
fi

# ── Step 5: Remove Ollama models (optional) ──────────────────────

if command -v ollama &> /dev/null; then
    echo ""
    read -p "Remove Ollama models used by Roxy? (y/n): " REMOVE_MODELS
    if [ "$REMOVE_MODELS" = "y" ] || [ "$REMOVE_MODELS" = "Y" ]; then
        for model in "qwen3:8b" "qwen3:0.6b" "nomic-embed-text"; do
            if ollama list 2>/dev/null | grep -q "$model"; then
                ollama rm "$model" 2>/dev/null || true
                echo -e "  ${GREEN}✓${NC} Removed model: $model"
            fi
        done
    else
        echo -e "  ${BLUE}ℹ${NC} Keeping Ollama models (you can remove them later with: ollama rm <model>)"
    fi

    echo ""
    read -p "Uninstall Ollama itself? (y/n): " REMOVE_OLLAMA
    if [ "$REMOVE_OLLAMA" = "y" ] || [ "$REMOVE_OLLAMA" = "Y" ]; then
        brew uninstall ollama 2>/dev/null || true
        rm -rf "$HOME/.ollama" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Ollama uninstalled"
    else
        echo -e "  ${BLUE}ℹ${NC} Keeping Ollama installed"
    fi
fi

# ── Step 6: Remove Hammerspoon config (optional) ─────────────────

if [ -f "$HOME/.hammerspoon/init.lua" ]; then
    if grep -q "roxy" "$HOME/.hammerspoon/init.lua" 2>/dev/null; then
        echo ""
        echo -e "  ${YELLOW}⚠${NC}  Your Hammerspoon config contains Roxy references."
        echo -e "     Edit ~/.hammerspoon/init.lua to remove Roxy-specific sections."
        echo -e "     (Not removing automatically to avoid breaking your other Hammerspoon config)"
    fi
fi

# ── Step 7: Remove Roxy project directory ────────────────────────

echo ""
echo -e "${BOLD}Removing Roxy files...${NC}"

if [ -d "$ROXY_DIR" ]; then
    # Safety check: make sure we're not deleting something unexpected
    if [ -f "$ROXY_DIR/CLAUDE.md" ] || [ -f "$ROXY_DIR/pyproject.toml" ]; then
        rm -rf "$ROXY_DIR"
        echo -e "  ${GREEN}✓${NC} Project directory removed: $ROXY_DIR"
    else
        echo -e "  ${RED}⚠${NC}  $ROXY_DIR doesn't look like a Roxy installation (no CLAUDE.md found)"
        read -p "  Remove it anyway? (y/n): " FORCE_REMOVE
        if [ "$FORCE_REMOVE" = "y" ]; then
            rm -rf "$ROXY_DIR"
            echo -e "  ${GREEN}✓${NC} Directory removed"
        fi
    fi
fi

# ── Step 8: Clean up Python cache ───────────────────────────────

# Remove any cached uv environments for roxy
if [ -d "$HOME/.cache/uv" ]; then
    # Just note it — don't nuke the whole uv cache as it's shared
    echo -e "  ${BLUE}ℹ${NC} Python package cache at ~/.cache/uv/ (shared with other projects, not removed)"
fi

# ── Step 9: Remove Playwright browsers (optional) ─────────────────

echo ""
read -p "Remove Playwright browsers? May affect other projects (y/n): " REMOVE_PLAYWRIGHT
if [ "$REMOVE_PLAYWRIGHT" = "y" ] || [ "$REMOVE_PLAYWRIGHT" = "Y" ]; then
    if command -v playwright &> /dev/null; then
        uv run playwright uninstall --all 2>/dev/null || true
    fi
    rm -rf "$HOME/Library/Caches/ms-playwright" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Playwright browsers removed"
else
    echo -e "  ${BLUE}ℹ${NC} Keeping Playwright browsers"
fi

# ── Done ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║       Roxy has been uninstalled           ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

if [ -d "${EXPORT_DIR:-}" ]; then
    echo -e "Your data backup is at: ${BLUE}$EXPORT_DIR${NC}"
    echo ""
fi

echo -e "Not removed (may be shared with other projects):"
echo -e "  ${BLUE}●${NC} Homebrew (brew)"
echo -e "  ${BLUE}●${NC} Python / uv"
echo -e "  ${BLUE}●${NC} Docker"
echo -e "  ${BLUE}●${NC} Hammerspoon (config may need manual cleanup)"
echo ""
echo -e "Goodbye from Roxy 👋"
