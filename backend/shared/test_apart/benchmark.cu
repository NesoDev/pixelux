#include <stdio.h>
#include <cuda_runtime.h>

__global__ void benchmarkKernel(float *data, int N, int work_factor)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    // Trabajo variable simulado
    float value = data[idx];
    for (int i = 0; i < (idx % work_factor) + 1; i++)
    {
      value = sin(value) + cos(value);
    }
    data[idx] = value;
  }
}

void testConfiguration(int blockSize, int N, int work_factor)
{
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  dim3 blockDim(blockSize);
  dim3 gridDim((N + blockSize - 1) / blockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  benchmarkKernel<<<gridDim, blockDim>>>(d_data, N, work_factor);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("BlockSize: %3d | Blocks: %4d | Time: %.3f ms | ",
         blockSize, gridDim.x, milliseconds);
  printf("Theoretical Occupancy: %.1f%%\n",
         (blockSize * min(16, 1536 / blockSize) / 1536.0) * 100);

  cudaFree(d_data);
}

int main()
{
  int N = 1000000;
  int work_factors[] = {1000, 10000, 100000}; // Diferentes cargas

  int size = sizeof(work_factors) / sizeof(work_factors[0]);

  printf("=== RTX 3060 Scheduling Analysis ===\n");
  printf("SMs: 28, Max Threads/SM: 1536, Max Blocks/SM: 16\n\n");

  for (int wf = 0; wf < size; wf++)
  {
    printf("Work Factor: %d\n", work_factors[wf]);
    printf("---------------------------------\n");

    int blockSizes[] = {32, 64, 96, 128, 256, 512, 1024};
    for (int bs = 0; bs < 6; bs++)
    {
      testConfiguration(blockSizes[bs], N, work_factors[wf]);
    }
    printf("\n");
  }

  return 0;
}