# Define compiler and flags
NVCC = nvcc
GXX = g++
MPICXX = mpicxx
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
CUDA_LIBS = -lcudart
OPENMP_LINKER_FLAG = -lgomp
MPI_LIBS = -lmpi

# Encontrar rutas de MPI automáticamente
MPI_CFLAGS = $(shell mpicxx --showme:compile)
MPI_LDFLAGS = $(shell mpicxx --showme:link)

# Encontrar ruta de CUDA automáticamente
CUDA_PATH = $(shell dirname $(shell which nvcc))/..
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

COMMON_FLAGS = -O3 -std=c++17 -fopenmp
NVCC_FLAGS = -Xcompiler "$(COMMON_FLAGS)" $(OPENCV_CFLAGS) -arch=sm_86
MPI_NVCC_FLAGS = -Xcompiler "$(COMMON_FLAGS)" $(OPENCV_CFLAGS) -arch=sm_86 $(MPI_CFLAGS)

TARGET = pixelart_converter
TARGET_MPI = pixelart_mpi
CPP_SOURCE = pixelart.cpp
MPI_SOURCE = pixelart_mpi.cpp
CU_SOURCE = cuda_kernels.cu
OBJ_FILES = $(CPP_SOURCE:.cpp=.o) $(CU_SOURCE:.cu=.o)
MPI_OBJ_FILES = $(MPI_SOURCE:.cpp=.o) $(CU_SOURCE:.cu=.o)

.PHONY: all clean mpi

all: $(TARGET)

mpi: $(TARGET_MPI)

# Compilar CUDA kernels
$(CU_SOURCE:.cu=.o): $(CU_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compilar archivo CPP principal
$(CPP_SOURCE:.cpp=.o): $(CPP_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compilar archivo MPI
$(MPI_SOURCE:.cpp=.o): $(MPI_SOURCE)
	$(NVCC) $(MPI_NVCC_FLAGS) -c $< -o $@

# Enlazar versión normal
$(TARGET): $(OBJ_FILES)
	$(NVCC) $^ -o $@ $(OPENCV_LIBS) $(CUDA_LIBS) $(OPENMP_LINKER_FLAG)

# Enlazar versión MPI
$(TARGET_MPI): $(MPI_OBJ_FILES)
	$(MPICXX) $^ -o $@ $(OPENCV_LIBS) -L$(CUDA_LIB_PATH) $(CUDA_LIBS) $(OPENMP_LINKER_FLAG) $(MPI_LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_MPI) $(OBJ_FILES) $(MPI_OBJ_FILES)