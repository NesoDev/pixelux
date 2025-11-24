#include <stdio.h>

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("Device: %s\n\n", prop.name);
  printf("Número de SM: %d\n", prop.multiProcessorCount);
  printf("Maximo numero de Bloques por SM: %d\n", prop.maxBlocksPerMultiProcessor);
  printf("Máximo numero de Threads por SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Memoria compartida total por SM en Kilobytes: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);
  printf("Máximo tamaño de cada dimension de un grid: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Máximo tamaño de cada dimension de un bloque: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Máximo numero de threads por bloque: %d\n\n", prop.maxThreadsPerBlock);
  printf("Memoria compartida reservada por CUDA driver por bloque en bytes: %zu\n", prop.reservedSharedMemPerBlock);
  printf("Memoria compartida disponible por bloque en Kilobytes: %zu KB\n", prop.sharedMemPerBlock / 1024);
  printf("Tamaño de warp en threads: %d\n", prop.warpSize);

  cudaDeviceSynchronize();

  return 0;
}