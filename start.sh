#!/bin/bash

################################################################################
# Pixelux - Production Start Script
# Verifies Docker, NVIDIA drivers, and CUDA requirements
# For machines with NVIDIA GPU and CUDA drivers
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

ask_yes_no() {
    local question="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        local prompt="[Y/n]"
    else
        local prompt="[y/N]"
    fi
    
    while true; do
        read -p "$(echo -e ${CYAN}${question}${NC} ${prompt}: )" response
        response=${response:-$default}
        case $response in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || \
       netstat -tuln 2>/dev/null | grep -q ":$port " || \
       ss -tuln 2>/dev/null | grep -q ":$port "; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

find_available_port() {
    local start_port=$1
    local port=$start_port
    while ! check_port $port; do
        port=$((port + 1))
        if [ $port -gt $((start_port + 100)) ]; then
            log_error "Could not find available port near $start_port"
            exit 1
        fi
    done
    echo $port
}

################################################################################
# Verification Functions
################################################################################



################################################################################
# System Checks
################################################################################

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pixelux - Startup Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
    log_info "Detected OS: $PRETTY_NAME"
else
    log_error "Cannot detect OS version"
    exit 1
fi

# Check if running on Ubuntu (recommended)
if [ "$OS" != "ubuntu" ]; then
    log_warn "This script is optimized for Ubuntu. Your OS: $OS"
    if ! ask_yes_no "Continue anyway?"; then
        exit 0
    fi
fi

################################################################################
# Docker Check and Installation
################################################################################

echo ""
log_info "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    log_info "Please install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

log_success "Docker is installed: $(docker --version)"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_warn "Docker daemon is not running"
    log_info "Starting Docker..."
    sudo systemctl start docker
    sleep 2
    
    if ! docker info &> /dev/null; then
        log_error "Could not start Docker daemon"
        exit 1
    fi
fi

log_success "Docker daemon is running"

################################################################################
# NVIDIA Driver Check and Installation
################################################################################

echo ""
log_info "Checking NVIDIA GPU drivers..."

NVIDIA_INSTALLED=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        NVIDIA_INSTALLED=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
        log_success "NVIDIA GPU detected: $GPU_NAME"
        log_info "Driver version: $GPU_DRIVER"
    fi
fi

if [ "$NVIDIA_INSTALLED" = false ]; then
    log_error "NVIDIA drivers not found or not working"
    
    # Check if GPU exists
    if lspci | grep -i nvidia &> /dev/null; then
        log_info "NVIDIA GPU hardware detected but drivers not loaded"
        log_info "Please install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx"
    else
        log_error "No NVIDIA GPU detected in system"
        log_info "Pixelux requires an NVIDIA GPU with CUDA support"
    fi
    exit 1
fi

################################################################################
# NVIDIA Container Toolkit Check and Installation
################################################################################

echo ""
log_info "Checking NVIDIA Container Toolkit..."

TOOLKIT_INSTALLED=false
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    TOOLKIT_INSTALLED=true
    log_success "NVIDIA Container Toolkit is configured"
else
    log_error "NVIDIA Container Toolkit not configured"
    log_info "Please install NVIDIA Container Toolkit:"
    log_info "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

################################################################################
# Port Configuration
################################################################################

echo ""
log_info "Configuring ports..."

FRONTEND_PORT=5173
API_PORT=8000

if ! check_port $FRONTEND_PORT; then
    log_warn "Port $FRONTEND_PORT is in use"
    FRONTEND_PORT=$(find_available_port $FRONTEND_PORT)
    log_info "Using alternative frontend port: $FRONTEND_PORT"
fi

if ! check_port $API_PORT; then
    log_warn "Port $API_PORT is in use"
    API_PORT=$(find_available_port $API_PORT)
    log_info "Using alternative API port: $API_PORT"
fi

export PIXELUX_FRONTEND_PORT=$FRONTEND_PORT
export PIXELUX_API_PORT=$API_PORT

################################################################################
# Start Pixelux
################################################################################

echo ""
log_info "Starting Pixelux services..."

cd "$SCRIPT_DIR"

# Check if containers exist
CONTAINERS_EXIST=false
if docker ps -a --format '{{.Names}}' | grep -q "pixelux-api"; then
    CONTAINERS_EXIST=true
fi

if [ "$CONTAINERS_EXIST" = true ]; then
    log_info "Existing containers found"
    
    if docker ps --format '{{.Names}}' | grep -q "pixelux-api"; then
        log_success "Containers are already running"
        
        # Get current ports
        CURRENT_FRONTEND_PORT=$(docker port pixelux-frontend 5173 2>/dev/null | cut -d: -f2 || echo "")
        CURRENT_API_PORT=$(docker port pixelux-api 8000 2>/dev/null | cut -d: -f2 || echo "")
        
        if [ ! -z "$CURRENT_FRONTEND_PORT" ]; then
            FRONTEND_PORT=$CURRENT_FRONTEND_PORT
        fi
        if [ ! -z "$CURRENT_API_PORT" ]; then
            API_PORT=$CURRENT_API_PORT
        fi
    else
        log_info "Starting stopped containers..."
        docker compose start
    fi
else
    log_info "Building and starting services (first run)..."
    log_warn "This will take several minutes..."
    
    # Build backend first
    cd backend
    docker compose build --no-cache
    docker compose up -d
    cd ..
fi

################################################################################
# Health Checks
################################################################################

echo ""
log_info "Waiting for services to be ready..."
sleep 15

# Check API
log_info "Checking API health..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:$API_PORT/api/health > /dev/null 2>&1; then
        log_success "API is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -ne "${YELLOW}[WAIT]${NC} Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)\r"
    sleep 2
done
echo ""

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "API not responding"
    log_info "Check logs: docker compose logs api"
    exit 1
fi

# Get server IP for local network access
SERVER_IP=$(hostname -I | awk '{print $1}')

################################################################################
# Success Message
################################################################################

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pixelux is Running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Local Access:${NC}"
echo -e "   Frontend:  ${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "   API:       ${GREEN}http://localhost:$API_PORT${NC}"
echo -e "   API Docs:  ${GREEN}http://localhost:$API_PORT/api/docs${NC}"
echo ""

if [ ! -z "$SERVER_IP" ] && [ "$SERVER_IP" != "127.0.0.1" ]; then
    echo -e "${CYAN}Network Access (from other devices on your network):${NC}"
    echo -e "   Frontend:  ${GREEN}http://$SERVER_IP:$FRONTEND_PORT${NC}"
    echo -e "   API:       ${GREEN}http://$SERVER_IP:$API_PORT${NC}"
    echo -e "   API Docs:  ${GREEN}http://$SERVER_IP:$API_PORT/api/docs${NC}"
    echo ""
fi

echo -e "${CYAN}Useful Commands:${NC}"
echo -e "   View logs:       ${YELLOW}docker compose logs -f${NC}"
echo -e "   View API logs:   ${YELLOW}docker compose logs -f api${NC}"
echo -e "   Stop services:   ${YELLOW}docker compose down${NC}"
echo -e "   Restart:         ${YELLOW}docker compose restart${NC}"
echo -e "   GPU status:      ${YELLOW}nvidia-smi${NC}"
echo ""
