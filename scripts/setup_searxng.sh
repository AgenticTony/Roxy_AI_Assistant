#!/bin/bash
# SearXNG Setup Script for Roxy
# Sets up a local privacy-focused search engine as a fallback

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SEARXNG_PORT=8888
SEARXNG_DIR="$HOME/roxy/searxng"
COMPOSE_FILE="$SEARXNG_DIR/docker-compose.yml"

echo -e "${GREEN}=== Roxy SearXNG Setup ===${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first:"
    echo "  brew install --cask docker"
    echo ""
    echo "Or download from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker is not running${NC}"
    echo "Please start Docker Desktop and run this script again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    echo "Please install docker-compose:"
    echo "  brew install docker-compose"
    exit 1
fi

# Use docker compose or docker-compose
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✓ Docker is available${NC}"
echo ""

# Create SearXNG directory
echo "Creating SearXNG configuration directory..."
mkdir -p "$SEARXNG_DIR"
cd "$SEARXNG_DIR"

# Create docker-compose.yml
echo "Creating docker-compose.yml..."
cat > "$COMPOSE_FILE" << 'EOF'
version: '3.8'

services:
  searxng:
    image: searxng/searxng:latest
    container_name: roxy-searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_SECRET=changemePleaseSetARandomSecretKey
      - SEARXNG_BASE_URL=http://localhost:8888
      - SEARXNG_PORT=8080
    restart: unless-stopped
    networks:
      - roxy-network
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  roxy-network:
    driver: bridge
EOF

# Create settings.yml for SearXNG
echo "Creating SearXNG settings..."
mkdir -p searxng
cat > searxng/settings.yml << 'EOF'
# SearXNG Configuration for Roxy
# Privacy-focused local search

# General settings
use_default_settings: true

# Server settings
server:
  # Must be 127.0.0.1 to prevent external access
  bind_address: "0.0.0.0"
  port: 8080
  secret_key: "changemePleaseSetARandomSecretKey"
  method: "GET"
  http_proxy_url:
  https_proxy_url:

# Search settings
search:
  # Number of results per page
  results_per_page: 10

  # Safe search (0: disabled, 1: moderate, 2: strict)
  safe_search: 0

  # Autocomplete
  autocomplete: ""

  # Image proxy
  image_proxy: true

  # Limit search time
  max_request_timeout: 3.0

# UI settings
ui:
  # Theme
  theme_args:
    simple_style: "auto"

# Privacy settings
# Disable telemetry and statistics
general:
  instance_name: "Roxy Local Search"
  contact_url: false
  enable_stats: false
  debug: false

# Search engines to use
engines:
  # DuckDuckGo (default, no API key needed)
  - name: duckduckgo
    engine: duckduckgo
    shortcut: dd

  # Google (optional, requires API key)
  # - name: google
  #   engine: google
  #   shortcut: go

  # Bing (optional, requires API key)
  # - name: bing
  #   engine: bing
  #   shortcut: bi

# Disable outgoing requests for stats
outgoing:
  request_timeout: 3.0
  max_request_timeout: 10.0
  useragent_suffix: []

# Enable limiter (prevent abuse)
limiter:
  botdetection:
    ip_limit: 0
  # No rate limiting for local use
EOF

echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

# Generate a random secret key
echo "Generating random secret key..."
SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "fallbackSecretKey$(date +%s)")
sed -i.bak "s/changemePleaseSetARandomSecretKey/$SECRET_KEY/g" "$COMPOSE_FILE"
sed -i.bak "s/changemePleaseSetARandomSecretKey/$SECRET_KEY/g" searxng/settings.yml
rm -f "$COMPOSE_FILE.bak" searxng/settings.yml.bak

# Start the container
echo ""
echo "Starting SearXNG container..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d

# Wait for health check
echo ""
echo "Waiting for SearXNG to be healthy..."
for i in {1..30}; do
    if curl -s "http://localhost:$SEARXNG_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ SearXNG is running and healthy!${NC}"
        break
    fi

    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}Warning: SearXNG may not be fully started yet${NC}"
        echo "Check status with: docker logs roxy-searxng"
    fi

    sleep 2
    echo -n "."
done

echo ""
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "SearXNG is now running at: http://localhost:$SEARXNG_PORT"
echo ""
echo "Useful commands:"
echo "  - Check status:  cd $SEARXNG_DIR && $DOCKER_COMPOSE -f $COMPOSE_FILE ps"
echo "  - View logs:    docker logs roxy-searxng"
echo "  - Stop:         cd $SEARXNG_DIR && $DOCKER_COMPOSE -f $COMPOSE_FILE down"
echo "  - Restart:      cd $SEARXNG_DIR && $DOCKER_COMPOSE -f $COMPOSE_FILE restart"
echo ""
echo "Note: SearXNG will start automatically on system reboot (Docker autostart)."
echo ""
echo -e "${YELLOW}To enable SearXNG in Roxy, edit config/mcp_servers.yaml and set:${NC}"
echo "  fallback_servers:"
echo "    searxng:"
echo "      enabled: true"
echo ""
