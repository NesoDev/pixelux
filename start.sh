#!/bin/bash

# Pixelux - Quick Start Script
# Requirements: Docker + NVIDIA GPU drivers

set -e

echo "Starting Pixelux..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to find available port starting from given port
find_available_port() {
    local start_port=$1
    local port=$start_port
    while ! check_port $port; do
        port=$((port + 1))
        if [ $port -gt $((start_port + 100)) ]; then
            echo -e "${RED}[ERROR] Could not find available port near $start_port${NC}"
            exit 1
        fi
    done
    echo $port
}

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker not found${NC}"
    echo -e "${YELLOW}Please install Docker first:${NC}"
    echo -e "   https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker and try again${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Docker is running${NC}"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[ERROR] NVIDIA GPU drivers not found${NC}"
    echo -e "${YELLOW}Please install NVIDIA drivers first${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] NVIDIA GPU detected:${NC} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null 2>&1; then
    echo -e "${RED}[ERROR] NVIDIA Container Toolkit not configured${NC}"
    echo -e "${YELLOW}Please install NVIDIA Container Toolkit:${NC}"
    echo -e "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo -e "${GREEN}[OK] NVIDIA Container Toolkit configured${NC}"

# Check and assign ports
FRONTEND_PORT=5173
API_PORT=8000

if ! check_port $FRONTEND_PORT; then
    echo -e "${YELLOW}[WARN] Port $FRONTEND_PORT is in use${NC}"
    FRONTEND_PORT=$(find_available_port $FRONTEND_PORT)
    echo -e "${BLUE}[INFO] Using alternative port: $FRONTEND_PORT${NC}"
fi

if ! check_port $API_PORT; then
    echo -e "${YELLOW}[WARN] Port $API_PORT is in use${NC}"
    API_PORT=$(find_available_port $API_PORT)
    echo -e "${BLUE}[INFO] Using alternative port: $API_PORT${NC}"
fi

# Export ports for docker-compose
export PIXELUX_FRONTEND_PORT=$FRONTEND_PORT
export PIXELUX_API_PORT=$API_PORT

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if containers exist
CONTAINERS_EXIST=false
if docker ps -a --format '{{.Names}}' | grep -q "pixelux-api"; then
    CONTAINERS_EXIST=true
fi

if [ "$CONTAINERS_EXIST" = true ]; then
    echo -e "${BLUE}[INFO] Containers exist. Checking status...${NC}"
    
    if docker ps --format '{{.Names}}' | grep -q "pixelux-api"; then
        echo -e "${GREEN}[OK] Containers already running${NC}"
        
        # Get current ports
        CURRENT_FRONTEND_PORT=$(docker port pixelux-frontend 5173 2>/dev/null | cut -d: -f2)
        CURRENT_API_PORT=$(docker port pixelux-api 8000 2>/dev/null | cut -d: -f2)
        
        if [ ! -z "$CURRENT_FRONTEND_PORT" ]; then
            FRONTEND_PORT=$CURRENT_FRONTEND_PORT
        fi
        if [ ! -z "$CURRENT_API_PORT" ]; then
            API_PORT=$CURRENT_API_PORT
        fi
    else
        echo -e "${BLUE}[INFO] Starting containers...${NC}"
        docker compose start
    fi
else
    echo -e "${BLUE}[INFO] Building and starting all services...${NC}"
    echo -e "${YELLOW}[WAIT] This may take a few minutes on first run...${NC}"
    docker compose up --build -d
fi

echo -e "${BLUE}[INFO] Waiting for services...${NC}"
sleep 10

# Check API
echo -e "${BLUE}[INFO] Checking API...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:$API_PORT/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK] API is ready${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "${YELLOW}[WAIT] Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}[ERROR] API not responding${NC}"
    echo -e "${YELLOW}Check logs: docker compose logs api${NC}"
    exit 1
fi

# Check Frontend
echo -e "${BLUE}[INFO] Checking Frontend...${NC}"
sleep 5

if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    echo -e "${GREEN}[OK] Frontend is ready${NC}"
else
    echo -e "${YELLOW}[WARN] Frontend may still be starting${NC}"
    echo -e "${YELLOW}Check logs: docker compose logs frontend${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pixelux is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Access:${NC}"
echo -e "   Frontend:  ${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "   API:       ${GREEN}http://localhost:$API_PORT${NC}"
echo -e "   API Docs:  ${GREEN}http://localhost:$API_PORT/api/docs${NC}"
echo ""
echo -e "${BLUE}Commands:${NC}"
echo -e "   Logs:     ${YELLOW}docker compose logs -f${NC}"
echo -e "   Stop:     ${YELLOW}docker compose down${NC}"
echo -e "   Restart:  ${YELLOW}docker compose restart${NC}"
echo ""
