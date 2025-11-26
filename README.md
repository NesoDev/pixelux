# Pixelux

High-performance pixel art conversion system leveraging CUDA-accelerated GPU processing and distributed MPI computing.

## Overview

Pixelux is a production-grade image processing pipeline that transforms standard images into pixel art using GPU-accelerated algorithms. The system employs a distributed architecture with CUDA kernels for parallel processing, MPI for cluster coordination, and a modern web interface for user interaction.

**📚 Deployment Guides:**
- **[Quick Start (Español)](QUICKSTART.md)** - 5-minute deployment guide
- **[Full Deployment Guide](DEPLOYMENT.md)** - Comprehensive setup and troubleshooting

**Key Features:**
- GPU-accelerated image processing with CUDA
- Distributed computing via MPI cluster
- Multiple dithering algorithms (Floyd-Steinberg, Ordered)
- Configurable color quantization
- RESTful API with FastAPI
- Containerized deployment with Docker

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Layer                          │
│                 React + Vite (Port 5173)                    |
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                       API Gateway                           │
│               FastAPI Server (Port 8000)                    │
│               - Request validati(Pydantic)                  │
│               - CORS handling                               │
│               - Base64 encoding/decoding                    │
└────────────────────────────┬────────────────────────────────┘
                             │ Subprocess
┌────────────────────────────▼────────────────────────────────┐
│                    Processing Backend                       │
│                   C++/CUDA/MPI Cluster                      │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│        │  Master  │  │ Worker 1 │  │ Worker 2 │             │
│        │   Node   │  │   Node   │  │   Node   │             │
│        └──────────┘  └──────────┘  └──────────┘             │
│             │              │              │                 │
│             └──────────────┴──────────────┘                 │
│                       MPI Network                           │
│                  GPU Processing (CUDA)                      │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

The following must be installed on the host system:

1. **Docker Engine** (20.10+)
   - [Installation Guide](https://docs.docker.com/engine/install/)
   
2. **NVIDIA GPU Drivers**
   - Compatible with your GPU model
   - Minimum: CUDA 11.0 support
   
3. **NVIDIA Container Toolkit**
   - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

All other dependencies (Node.js, Python, C++ libraries) are containerized.

## Quick Start

```bash
./start.sh
```

The startup script will:
1. Verify system prerequisites
2. Detect and resolve port conflicts automatically
3. Build and deploy all services
4. Perform health checks
5. Display access URLs

**Default Access Points:**
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/api/docs`

If default ports are occupied, the script automatically assigns alternative ports.

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support (compute capability 3.5+)
- Minimum 4GB GPU memory
- 8GB system RAM (16GB recommended)
- 10GB available disk space

### Software
- Linux operating system (Ubuntu 20.04+ recommended)
- Docker Engine 20.10+
- NVIDIA Driver 470+
- NVIDIA Container Toolkit

## API Reference

### Process Image

**Endpoint:** `POST /api/process`

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KG...",
  "algorithm": "dithering",
  "scale": 5,
  "palette": "free"
}
```

**Parameters:**
- `image` (string, required): Base64-encoded image data
- `algorithm` (string): Processing algorithm (`dithering` | `no-dithering`)
- `scale` (integer): Pixel size (1-20)
- `palette` (string): Color palette (`free` | `grayscale`)

**Response:**
```json
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KG...",
  "message": "Image processed successfully",
  "processing_time_ms": 1234.56,
  "metadata": {
    "pixel_size": 5,
    "algorithm": "dithering",
    "palette": "free",
    "output_size_bytes": 45678
  }
}
```

### Health Check

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T12:34:56.789Z",
  "version": "1.0.0",
  "cuda_available": true
}
```

## Container Services

| Service | Description | Port | GPU |
|---------|-------------|------|-----|
| `master` | MPI master node with CUDA | 2222 | Yes |
| `worker1` | MPI worker node with CUDA | 2223 | Yes |
| `worker2` | MPI worker node with CUDA | 2224 | Yes |
| `api` | FastAPI REST server | 8000 | No |
| `frontend` | Vite development server | 5173 | No |

## Development

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f frontend
```

### Restart Services

```bash
# All services
docker compose restart

# Specific service
docker compose restart api
```

### Rebuild After Code Changes

```bash
# Backend C++/CUDA
docker exec -it master bash
cd /home/mpiuser/shared/pixelart
make clean && make mpi
exit

# API Server
docker compose restart api

# Frontend (auto-reloads via hot module replacement)
# No action needed
```

### Stop All Services

```bash
docker compose down
```

## Configuration

### Environment Variables

**API Server** (`docker-compose.yml`):
```yaml
environment:
  - PIXELART_BINARY=/home/mpiuser/shared/pixelart/pixelart_mpi
  - TEMP_DIR=/tmp/pixelux
  - ALLOWED_ORIGINS=http://localhost:5173
  - DEBUG=false
  - MAX_IMAGE_SIZE=10485760  # 10MB
```

**Frontend** (`docker-compose.yml`):
```yaml
environment:
  - VITE_API_URL=http://localhost:8000
```

### Port Configuration

Ports are automatically assigned by the startup script. To manually specify:

```bash
export PIXELUX_FRONTEND_PORT=5173
export PIXELUX_API_PORT=8000
./start.sh
```

## Performance Optimization

### GPU Utilization

Monitor GPU usage:
```bash
docker exec -it master nvidia-smi
```

### Batch Processing

For processing multiple images, use the MPI batch mode:
```bash
docker exec -it master bash
cd /home/mpiuser/shared/pixelart
mpirun -np 3 ./pixelart_mpi --mpi-batch ./input ./output 8 6 1 0 4
```

## Troubleshooting

### API Not Responding

```bash
# Check API logs
docker compose logs api

# Verify binary exists
docker exec -it master ls -lh /home/mpiuser/shared/pixelart/pixelart_mpi

# Recompile if necessary
docker exec -it master bash -c "cd /home/mpiuser/shared/pixelart && make clean && make mpi"
```

### GPU Not Detected

```bash
# Verify NVIDIA drivers
nvidia-smi

# Test GPU in container
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check NVIDIA Container Toolkit
docker info | grep -i nvidia
```

### Port Conflicts

The startup script automatically resolves port conflicts. To check current ports:
```bash
docker port pixelux-frontend
docker port pixelux-api
```

## Production Deployment

For production environments:

1. **Build optimized frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Update CORS origins** in `docker-compose.yml`
3. **Enable HTTPS** with reverse proxy (nginx/traefik)
4. **Configure resource limits** in `docker-compose.yml`
5. **Set up monitoring** (Prometheus/Grafana)
6. **Implement rate limiting** in API server

## Technical Stack

- **Backend Processing:** C++17, CUDA 11.0, OpenCV 4.x, OpenMPI
- **API Server:** Python 3.11, FastAPI 0.115, Uvicorn
- **Frontend:** React 19, Vite 7, JavaScript ES2022
- **Containerization:** Docker 20.10+, Docker Compose 2.x
- **GPU Runtime:** NVIDIA Container Toolkit

## License

[Specify License]

## Contributing

[Specify Contribution Guidelines]
