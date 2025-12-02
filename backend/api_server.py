"""
Production-ready FastAPI server for Pixelux image processing.
Connects React frontend with C++/CUDA backend.
"""

import os
import sys
import base64
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/pixelux_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PIXELART_BINARY = os.getenv('PIXELART_BINARY', '/home/mpiuser/shared/pixelart/pixelart_mpi')
TEMP_DIR = Path(os.getenv('TEMP_DIR', '/tmp/pixelux'))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', 10 * 1024 * 1024))  # 10MB

# Initialize FastAPI app
app = FastAPI(
    title="Pixelux API",
    description="API for converting images to pixel art using CUDA acceleration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for production
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ProcessImageRequest(BaseModel):
    """Request model for image processing"""
    image: str = Field(..., description="Base64 encoded image data (with or without data URI prefix)")
    algorithm: str = Field(default="no-dithering", description="Processing algorithm: 'dithering' or 'no-dithering'")
    scale: int = Field(default=5, ge=1, le=20, description="Pixel size (1-20)")
    palette: str = Field(default="free", description="Color palette: 'free', 'grayscale', etc.")
    
    @field_validator('image')
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate base64 image data"""
        if not v:
            raise ValueError("Image data cannot be empty")
        
        # Remove data URI prefix if present
        if ',' in v and v.startswith('data:'):
            v = v.split(',', 1)[1]
        
        # Validate base64
        try:
            decoded = base64.b64decode(v)
            if len(decoded) > MAX_IMAGE_SIZE:
                raise ValueError(f"Image size exceeds maximum allowed size of {MAX_IMAGE_SIZE} bytes")
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")
        
        return v
    
    @field_validator('algorithm')
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate algorithm choice"""
        allowed = ['dithering', 'no-dithering']
        if v not in allowed:
            raise ValueError(f"Algorithm must be one of: {allowed}")
        return v


class ProcessImageResponse(BaseModel):
    """Response model for image processing"""
    success: bool
    image: Optional[str] = Field(None, description="Base64 encoded processed image")
    message: str
    processing_time_ms: Optional[float] = None
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    cuda_available: bool


# Helper functions
def check_binary_health() -> tuple[bool, str]:
    """Check if binary exists and is executable with required libraries"""
    if not Path(PIXELART_BINARY).exists():
        return False, f"Binary not found: {PIXELART_BINARY}"
    
    if not os.access(PIXELART_BINARY, os.X_OK):
        return False, f"Binary not executable: {PIXELART_BINARY}"
    
    # Try to run ldd to check libraries
    try:
        result = subprocess.run(
            ['ldd', PIXELART_BINARY],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'not found' in result.stdout:
            return False, f"Binary has missing libraries: {result.stdout}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # ldd not available or timeout, skip library check
        pass
    
    return True, "Binary is ready"


def decode_base64_image(base64_str: str) -> bytes:
    """Decode base64 image string to bytes"""
    # Remove data URI prefix if present
    if ',' in base64_str and base64_str.startswith('data:'):
        base64_str = base64_str.split(',', 1)[1]
    
    return base64.b64decode(base64_str)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def get_dither_type(algorithm: str) -> int:
    """Map algorithm name to dither type integer"""
    mapping = {
        'no-dithering': 0,
        'dithering': 1,  # Floyd-Steinberg
    }
    return mapping.get(algorithm, 0)


def get_color_bits(palette: str) -> int:
    """Map palette name to color bits"""
    mapping = {
        'free': 6,
        'grayscale': 4,
        'limited': 4,
    }
    return mapping.get(palette, 6)


def is_grayscale(palette: str) -> bool:
    """Check if palette is grayscale"""
    return palette == 'grayscale'


def process_image_with_cuda(
    input_path: Path,
    output_path: Path,
    pixel_size: int,
    color_bits: int,
    dither_type: int,
    grayscale: bool
) -> tuple[bool, str]:
    """
    Execute the C++/CUDA binary to process the image.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Build command
        cmd = [
            str(PIXELART_BINARY),
            str(input_path),
            str(output_path),
            str(pixel_size),
            str(color_bits),
            str(dither_type),
            str(1 if grayscale else 0)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # FIX: Usar capture_output=False y manejar la salida manualmente
        # O usar subprocess.PIPE con decode apropiado
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,  # IMPORTANTE: No intentar decodificar como texto
            timeout=60,
            check=False
        )
        
        # Decodificar stderr de forma segura
        stderr_text = ""
        if result.stderr:
            try:
                stderr_text = result.stderr.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                stderr_text = result.stderr.decode('latin-1', errors='replace')
        
        if result.returncode != 0:
            error_msg = f"Processing failed with return code {result.returncode}: {stderr_text}"
            logger.error(error_msg)
            return False, error_msg
        
        if not output_path.exists():
            error_msg = "Processing completed but output file was not created"
            logger.error(error_msg)
            return False, error_msg
        
        # Verificar que el archivo de salida no esté vacío
        if output_path.stat().st_size == 0:
            error_msg = "Processing completed but output file is empty"
            logger.error(error_msg)
            return False, error_msg
        
        logger.info(f"Processing successful: {output_path} (size: {output_path.stat().st_size} bytes)")
        return True, "Image processed successfully"
        
    except subprocess.TimeoutExpired:
        error_msg = "Processing timeout exceeded (120 seconds)"
        logger.error(error_msg)
        return False, error_msg
    except FileNotFoundError:
        error_msg = f"Binary not found: {PIXELART_BINARY}"
        logger.error(error_msg)
        return False, error_msg
    except OSError as e:
        if e.errno == 8:  # Exec format error
            error_msg = f"Binary format error: Binary may be compiled for different architecture"
        else:
            error_msg = f"OS error executing binary: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


# API Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Verificar si el binario existe y es ejecutable
    binary_exists = Path(PIXELART_BINARY).exists()
    binary_executable = os.access(PIXELART_BINARY, os.X_OK)
    
    cuda_available = binary_exists and binary_executable
    
    status_msg = "healthy" if cuda_available else "degraded"
    if not binary_exists:
        status_msg = "binary_not_found"
    elif not binary_executable:
        status_msg = "binary_not_executable"
    
    return HealthResponse(
        status=status_msg,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        cuda_available=binary_ok
    )


@app.post("/api/process", response_model=ProcessImageResponse)
async def process_image(request: ProcessImageRequest):
    """
    Process an image to convert it to pixel art.
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Processing request: scale={request.scale}, algorithm={request.algorithm}, palette={request.palette}")
    
    # Verificar que el binario exista
    if not Path(PIXELART_BINARY).exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing binary not found: {PIXELART_BINARY}"
        )
    
    # Crear archivos temporales
    input_path = TEMP_DIR / f"{request_id}_input.png"
    output_path = TEMP_DIR / f"{request_id}_output.png"
    
    try:
        # Decodificar y guardar imagen de entrada
        try:
            image_data = decode_base64_image(request.image)
            with open(input_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"[{request_id}] Input image saved: {input_path} ({len(image_data)} bytes)")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode image: {str(e)}"
            )
        
        # Procesar imagen
        dither_type = get_dither_type(request.algorithm)
        color_bits = get_color_bits(request.palette)
        grayscale = is_grayscale(request.palette)
        
        success, message = process_image_with_cuda(
            input_path=input_path,
            output_path=output_path,
            pixel_size=request.scale,
            color_bits=color_bits,
            dither_type=dither_type,
            grayscale=grayscale
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
        
        # Codificar imagen de salida
        try:
            output_base64 = encode_image_to_base64(output_path)
            output_size = output_path.stat().st_size
            logger.info(f"[{request_id}] Output image encoded: {output_size} bytes")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to encode output image: {str(e)}"
            )
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"[{request_id}] Processing completed in {processing_time:.2f}ms")
        
        return ProcessImageResponse(
            success=True,
            image=f"data:image/png;base64,{output_base64}",
            message="Image processed successfully",
            processing_time_ms=round(processing_time, 2),
            metadata={
                "pixel_size": request.scale,
                "algorithm": request.algorithm,
                "palette": request.palette,
                "output_size_bytes": output_size
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Limpieza de archivos temporales
        try:
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()
            logger.debug(f"[{request_id}] Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to cleanup temporary files: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.getenv('DEBUG', 'false').lower() == 'true' else None
        }
    )


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info("="*60)
    logger.info("Starting Pixelux API server")
    logger.info(f"Server: {host}:{port}")
    logger.info(f"PIXELART_BINARY: {PIXELART_BINARY}")
    logger.info(f"TEMP_DIR: {TEMP_DIR}")
    logger.info(f"ALLOWED_ORIGINS: {ALLOWED_ORIGINS}")
    logger.info("="*60)
    
    # Startup validation
    logger.info("Running startup validation...")
    binary_ok, message = check_binary_health()
    if binary_ok:
        logger.info(f"✅ Binary check passed: {message}")
    else:
        logger.error(f"❌ Binary check failed: {message}")
        logger.error("API will start but image processing will fail!")
    
    # Check CUDA availability
    cuda_path = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if Path(cuda_path).exists():
        logger.info(f"✅ CUDA found at: {cuda_path}")
    else:
        logger.warning(f"⚠️  CUDA not found at: {cuda_path}")
    
    logger.info("="*60)
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=os.getenv('DEBUG', 'false').lower() == 'true',
        log_level="info"
    )