# Pixelux - Pixel Art Converter

Sistema completo de conversión de imágenes a pixel art usando CUDA/MPI para procesamiento acelerado por GPU.

**🐳 Completamente Dockerizado** - Solo requiere Docker + NVIDIA drivers instalados

## 🚀 Inicio Rápido

```bash
./start.sh
```

**Acceso:**
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

## 📋 Requisitos Previos

**Debes tener instalado:**
1. **Docker** - [Guía de instalación](https://docs.docker.com/engine/install/)
2. **NVIDIA GPU Drivers** - Para tu tarjeta gráfica
3. **NVIDIA Container Toolkit** - [Guía de instalación](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

El script `start.sh` verificará estos requisitos y te guiará si falta algo.

**NO necesitas instalar:**
- ❌ Node.js
- ❌ Python
- ❌ npm
- ❌ Ninguna dependencia global

Todo corre dentro de contenedores Docker.

## 🏗️ Arquitectura

```
┌─────────────────────┐
│   React Frontend    │  ← Contenedor Node.js
│   localhost:5173    │
└──────────┬──────────┘
           │
           │ HTTP/REST
           │
┌──────────▼──────────┐
│   FastAPI Server    │  ← Contenedor Python
│   localhost:8000    │
└──────────┬──────────┘
           │
           │ subprocess
           │
┌──────────▼──────────┐
│  C++/CUDA Backend   │  ← Contenedores CUDA
│  MPI Cluster        │
│  (master + workers) │
└─────────────────────┘
```

## 📦 Servicios Docker

El proyecto incluye 6 contenedores:

1. **master** - Nodo principal MPI con CUDA
2. **worker1** - Nodo worker MPI con CUDA
3. **worker2** - Nodo worker MPI con CUDA
4. **api** - Servidor FastAPI (Python)
5. **frontend** - Servidor Vite (Node.js)

Todos se levantan automáticamente con `./start.sh`

## 🎯 Uso

1. **Abrir**: http://localhost:5173
2. **Cargar imagen**: Click en "Examinar..."
3. **Configurar**:
   - Dithering: On/Off
   - Scale: 1-20
   - Palette: free/grayscale
4. **Procesar**: Click en "Procesar"
5. **Descargar**: Click en "Descargar"

## 🛠️ Comandos Útiles

```bash
# Ver logs de todos los servicios
docker compose logs -f

# Ver logs de un servicio específico
docker compose logs -f frontend
docker compose logs -f api

# Detener todos los servicios
docker compose down

# Reiniciar servicios
docker compose restart

# Reconstruir y reiniciar
docker compose up --build -d

# Ver estado de contenedores
docker compose ps
```

## 🔧 Desarrollo

### Modificar Frontend

Los cambios en `frontend/src/` se reflejan automáticamente gracias a hot-reload de Vite.

```bash
# Editar archivos en frontend/src/
# El navegador se recarga automáticamente
```

### Modificar API

```bash
# Editar backend/api_server.py
docker compose restart api
```

### Modificar Backend C++/CUDA

```bash
# Editar archivos en backend/shared/pixelux/
# Recompilar dentro del contenedor
docker exec -it master bash
cd /home/mpiuser/shared/pixelart
make clean && make mpi
```

## 🐛 Troubleshooting

### Error: "NVIDIA GPU drivers not found"
```bash
# Instalar drivers NVIDIA
sudo apt-get install nvidia-driver-535
sudo reboot
```

### Error: "Permission denied" al ejecutar Docker
```bash
# Agregar usuario al grupo docker
sudo usermod -aG docker $USER
# Cerrar sesión y volver a entrar
```

### Frontend no carga
```bash
# Ver logs
docker compose logs frontend

# Reiniciar
docker compose restart frontend
```

### API no responde
```bash
# Ver logs
docker compose logs api

# Verificar que master esté corriendo
docker compose ps

# Reiniciar
docker compose restart api master
```

## 📊 Estructura del Proyecto

```
pixelux/
├── docker-compose.yml          # Orquestación de todos los servicios
├── start.sh                    # Script de inicio automático
├── README.md
│
├── backend/
│   ├── dockerfile              # Imagen CUDA/MPI
│   ├── Dockerfile.api          # Imagen API Python
│   ├── docker-compose.yml      # (legacy, usar root)
│   ├── api_server.py           # Servidor FastAPI
│   ├── requirements.txt
│   └── shared/
│       └── pixelart/
│           ├── pixelart_mpi.cpp
│           ├── cuda_kernels.cu
│           └── makefile
│
└── frontend/
    ├── Dockerfile.dev          # Imagen Node.js dev
    ├── .dockerignore
    ├── package.json
    └── src/
        ├── App.jsx
        ├── services/
        │   └── api.js
        └── componentes/
            ├── ProcessBox.jsx
            ├── UploadBox.jsx
            └── Menu.jsx
```

## 🚀 Producción

Para producción, considera:

1. **Build frontend estático**:
```bash
# Crear Dockerfile.prod para frontend
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

2. **Configurar HTTPS** con nginx/traefik
3. **Actualizar ALLOWED_ORIGINS** en docker-compose.yml
4. **Configurar límites de recursos**
5. **Implementar logging centralizado**

## 📝 Licencia

[Especificar licencia]

## 🤝 Contribuciones

[Especificar guías de contribución]
