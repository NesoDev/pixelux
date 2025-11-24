#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void orderedDitherKernel(unsigned char *image, int width, int height, int channels, int colorLevels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = (y * width + x) * channels;
    int step = 256 / colorLevels;

    // Matriz de Bayer 4x4 para dithering ordenado
    const float bayerMatrix[4][4] = {
        {0, 8, 2, 10},
        {12, 4, 14, 6},
        {3, 11, 1, 9},
        {15, 7, 13, 5}};

    float threshold = bayerMatrix[y % 4][x % 4] / 16.0f - 0.5f;

    for (int c = 0; c < channels; c++)
    {
      float pixel = image[idx + c];
      float ditheredPixel = pixel + threshold * step;

      ditheredPixel = fmaxf(0.0f, fminf(255.0f, ditheredPixel));

      int newPixel = (int(ditheredPixel) / step) * step + step / 2;
      if (newPixel > 255)
        newPixel = 255;

      image[idx + c] = static_cast<unsigned char>(newPixel);
    }
  }
}

__global__ void floydSteinbergKernel(unsigned char *image, int width, int height, int channels, int colorLevels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height && x > 0 && y > 0 && x < width - 1 && y < height - 1)
  {
    int idx = (y * width + x) * channels;
    int step = 256 / colorLevels;

    for (int c = 0; c < channels; c++)
    {
      float oldPixel = image[idx + c];

      int newPixel = (int(oldPixel) / step) * step + step / 2;
      if (newPixel > 255)
        newPixel = 255;

      image[idx + c] = static_cast<unsigned char>(newPixel);

      float error = oldPixel - newPixel;

      if (x + 1 < width)
      {
        int rightIdx = (y * width + (x + 1)) * channels + c;
        float rightPixel = image[rightIdx] + error * (7.0f / 16.0f);
        image[rightIdx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, rightPixel)));
      }

      if (y + 1 < height)
      {
        if (x > 0)
        {
          int downLeftIdx = ((y + 1) * width + (x - 1)) * channels + c;
          float downLeftPixel = image[downLeftIdx] + error * (3.0f / 16.0f);
          image[downLeftIdx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, downLeftPixel)));
        }

        int downIdx = ((y + 1) * width + x) * channels + c;
        float downPixel = image[downIdx] + error * (5.0f / 16.0f);
        image[downIdx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, downPixel)));

        if (x + 1 < width)
        {
          int downRightIdx = ((y + 1) * width + (x + 1)) * channels + c;
          float downRightPixel = image[downRightIdx] + error * (1.0f / 16.0f);
          image[downRightIdx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, downRightPixel)));
        }
      }
    }
  }
}

__global__ void convertToGrayscaleKernel(unsigned char *image, int width, int height, int channels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height && channels == 3)
  {
    int idx = (y * width + x) * channels;

    // Fórmula estándar para conversión a escala de grises
    unsigned char b = image[idx];
    unsigned char g = image[idx + 1];
    unsigned char r = image[idx + 2];

    unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

    // Aplicar el mismo valor a los 3 canales
    image[idx] = gray;     // B
    image[idx + 1] = gray; // G
    image[idx + 2] = gray; // R
  }
}

__global__ void resizeKernel(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels, int pixelSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < outWidth && y < outHeight)
  {
    int startX = x * pixelSize;
    int startY = y * pixelSize;

    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

    // Calcular promedio del bloque
    for (int blockY = 0; blockY < pixelSize; blockY++)
    {
      int currentY = startY + blockY;
      if (currentY >= inHeight)
        break;

      for (int blockX = 0; blockX < pixelSize; blockX++)
      {
        int currentX = startX + blockX;
        if (currentX >= inWidth)
          break;

        int inputIdx = (currentY * inWidth + currentX) * channels;
        sum_b += input[inputIdx];
        sum_g += input[inputIdx + 1];
        sum_r += input[inputIdx + 2];
        count++;
      }
    }

    int outputIdx = (y * outWidth + x) * channels;
    if (count > 0)
    {
      output[outputIdx] = sum_b / count;     // B
      output[outputIdx + 1] = sum_g / count; // G
      output[outputIdx + 2] = sum_r / count; // R
    }
  }
}

__global__ void reduceColorsKernel(unsigned char *image, int width, int height, int channels, int colorLevels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = (y * width + x) * channels;
    int step = 256 / colorLevels;

    for (int c = 0; c < channels; c++)
    {
      unsigned char pixel = image[idx + c];
      // Reducir niveles de color
      int newPixel = (pixel / step) * step + step / 2;
      if (newPixel > 255)
        newPixel = 255;
      image[idx + c] = static_cast<unsigned char>(newPixel);
    }
  }
}

__global__ void scaleUpKernel(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels, int pixelSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < outWidth && y < outHeight)
  {
    int smallX = x / pixelSize;
    int smallY = y / pixelSize;

    if (smallX < inWidth && smallY < inHeight)
    {
      int inputIdx = (smallY * inWidth + smallX) * channels;

      int outputIdx = (y * outWidth + x) * channels;

      // Copiar el pixel de la imagen pequeña a la posición correspondiente
      // en la imagen grande (efecto de pixelación)
      for (int c = 0; c < channels; c++)
      {
        output[outputIdx + c] = input[inputIdx + c];
      }
    }
  }
}

extern "C" void resizeImageCUDA(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels)
{
  int pixelSize = inWidth / outWidth;

  dim3 blockSize(16, 16);
  dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x, (outHeight + blockSize.y - 1) / blockSize.y);

  unsigned char *d_input, *d_output;
  size_t inputSize = inWidth * inHeight * channels * sizeof(unsigned char);
  size_t outputSize = outWidth * outHeight * channels * sizeof(unsigned char);

  cudaMalloc(&d_input, inputSize);
  cudaMalloc(&d_output, outputSize);

  cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

  resizeKernel<<<gridSize, blockSize>>>(d_input, d_output, inWidth, inHeight, outWidth, outHeight, channels, pixelSize);

  cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

extern "C" void reduceColorsCUDA(unsigned char *image, int width, int height, int channels, int colorLevels)
{
  unsigned char *d_image;
  size_t imageSize = width * height * channels * sizeof(unsigned char);

  cudaMalloc(&d_image, imageSize);
  cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  reduceColorsKernel<<<gridSize, blockSize>>>(d_image, width, height, channels, colorLevels);

  cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);

  cudaFree(d_image);
}

extern "C" void scaleUpImageCUDA(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels, int pixelSize)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x, (outHeight + blockSize.y - 1) / blockSize.y);

  unsigned char *d_input, *d_output;
  size_t inputSize = inWidth * inHeight * channels * sizeof(unsigned char);
  size_t outputSize = outWidth * outHeight * channels * sizeof(unsigned char);

  cudaMalloc(&d_input, inputSize);
  cudaMalloc(&d_output, outputSize);

  cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

  scaleUpKernel<<<gridSize, blockSize>>>(d_input, d_output, inWidth, inHeight, outWidth, outHeight, channels, pixelSize);

  cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

extern "C" void applyDitheringCUDA(unsigned char *image, int width, int height, int channels, int colorLevels, int ditherType)
{
  unsigned char *d_image;
  size_t imageSize = width * height * channels * sizeof(unsigned char);

  cudaMalloc(&d_image, imageSize);
  cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  if (ditherType == 1) // Floyd-Steinberg
  {
    floydSteinbergKernel<<<gridSize, blockSize>>>(d_image, width, height, channels, colorLevels);
  }
  else if (ditherType == 2) // Ordered Dither
  {
    orderedDitherKernel<<<gridSize, blockSize>>>(d_image, width, height, channels, colorLevels);
  }

  cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);
  cudaFree(d_image);
}

extern "C" void convertToGrayscaleCUDA(unsigned char *image, int width, int height, int channels)
{
  if (channels != 3)
    return;

  unsigned char *d_image;
  size_t imageSize = width * height * channels * sizeof(unsigned char);

  cudaMalloc(&d_image, imageSize);
  cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  convertToGrayscaleKernel<<<gridSize, blockSize>>>(d_image, width, height, channels);

  cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);
  cudaFree(d_image);
}
