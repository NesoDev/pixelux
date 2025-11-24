#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cuda_runtime.h>
#include <filesystem>

// CUDA kernels
extern "C"
{
  void reduceColorsCUDA(unsigned char *image, int width, int height, int channels, int colorLevels);
  void resizeImageCUDA(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels);
  void scaleUpImageCUDA(unsigned char *input, unsigned char *output, int inWidth, int inHeight, int outWidth, int outHeight, int channels, int pixelSize);
  void applyDitheringCUDA(unsigned char *image, int width, int height, int channels, int colorLevels, int ditherType);
  void convertToGrayscaleCUDA(unsigned char *image, int width, int height, int channels);
}

class PixelArtConverter
{
private:
  int pixelSize;
  int colorLevels;
  int ditherType;
  bool grayScale;

public:
  PixelArtConverter(int pixelSize = 8, int colorLevels = 8, int ditherType = 0, bool grayScale = false) : pixelSize(pixelSize), colorLevels(colorLevels), ditherType(ditherType), grayScale(grayScale)
  {
    // Validar parámetros
    if (pixelSize < 1)
      pixelSize = 1;
    if (colorLevels < 2)
      colorLevels = 2;
    if (colorLevels > 256)
      colorLevels = 256;
    if (ditherType < 0 || ditherType > 2)
      ditherType = 0;
  }

  cv::Mat convertToPixelArtCUDA(const cv::Mat &inputImage)
  {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    cv::Mat processedImage;
    if (grayScale && channels == 3)
    {
      cv::cvtColor(inputImage, processedImage, cv::COLOR_BGR2GRAY);
      cv::cvtColor(processedImage, processedImage, cv::COLOR_GRAY2BGR);
    }
    else
    {
      processedImage = inputImage.clone();
    }

    int smallWidth = width / pixelSize;
    int smallHeight = height / pixelSize;

    cv::Mat smallImage(smallHeight, smallWidth, CV_8UC3);

    // Paso 1: Redimensionar la imagen a tamaño pequeño
    resizeImageCUDA(processedImage.data, smallImage.data, width, height, smallWidth, smallHeight, channels);

    // Paso 2: Reducir colores con o sin dithering
    if (ditherType > 0)
    {
      applyDitheringCUDA(smallImage.data, smallWidth, smallHeight, channels, colorLevels, ditherType);
    }
    else
    {
      reduceColorsCUDA(smallImage.data, smallWidth, smallHeight, channels, colorLevels);
    }

    // Paso 3: Escalar de vuelta
    cv::Mat result(height, width, CV_8UC3);
    scaleUpImageCUDA(smallImage.data, result.data, smallWidth, smallHeight, width, height, channels, pixelSize);

    if (grayScale)
    {
      cv::Mat grayResult;
      cv::cvtColor(result, grayResult, cv::COLOR_BGR2GRAY);
      cv::cvtColor(grayResult, result, cv::COLOR_GRAY2BGR);
    }

    return result;
  }
};

void printSystemInfo()
{
  std::cout << "=== Pixel Art Converter ===" << std::endl;
  std::cout << "OpenMP threads disponibles: " << omp_get_max_threads() << std::endl;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Dispositivos CUDA disponibles: " << deviceCount << std::endl;

  for (int i = 0; i < deviceCount; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "  GPU " << i << ": " << prop.name << std::endl;
  }
}

// Función para buscar archivos en un directorio
std::vector<std::string> findImageFiles(const std::string &directory, const std::vector<std::string> &extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
{
  std::vector<std::string> imageFiles;

  try
  {
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
      if (entry.is_regular_file())
      {
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end())
        {
          imageFiles.push_back(entry.path().string());
        }
      }
    }
  }
  catch (const std::filesystem::filesystem_error &e)
  {
    std::cerr << "Error al acceder al directorio: " << e.what() << std::endl;
  }

  return imageFiles;
}

// Función para generar nombres de salida
std::vector<std::string> generateOutputFiles(const std::vector<std::string> &inputFiles, const std::string &outputDir, const std::string &suffix = "_pixelart")
{
  std::vector<std::string> outputFiles;

  // Crear directorio de salida si no existe
  std::filesystem::create_directories(outputDir);

  for (const auto &inputFile : inputFiles)
  {
    std::filesystem::path inputPath(inputFile);
    std::string outputFile = outputDir + "/" + inputPath.stem().string() + suffix + ".png";
    outputFiles.push_back(outputFile);
  }

  return outputFiles;
}

// Función para procesar un lote de imágenes
void processBatchImages(const std::vector<std::string> &inputFiles, const std::vector<std::string> &outputFiles, int pixelSize, int colorBits, int ditherType, bool grayscale, int numThreads = 0)
{
  if (inputFiles.size() != outputFiles.size())
  {
    std::cerr << "Error: El número de archivos de entrada y salida no coincide" << std::endl;
    return;
  }

  if (numThreads > 0)
  {
    omp_set_num_threads(numThreads);
  }

  int colorLevels = 1 << colorBits;
  int totalImages = inputFiles.size();
  int processedImages = 0;
  int failedImages = 0;

  std::cout << "\nProcesando lote de " << totalImages << " imágenes..." << std::endl;
  std::cout << "Hilos OpenMP utilizados: " << omp_get_max_threads() << std::endl;
  std::cout << "Parámetros: pixelSize=" << pixelSize << ", colorBits=" << colorBits << ", ditherType=" << ditherType << ", grayscale=" << grayscale << std::endl;

  double startTime = omp_get_wtime();

#pragma omp parallel for schedule(dynamic) reduction(+ : processedImages, failedImages)
  for (int i = 0; i < totalImages; i++)
  {
    try
    {
      // Crear una instancia del converter para cada hilo
      PixelArtConverter converter(pixelSize, colorLevels, ditherType, grayscale);

      cv::Mat image = cv::imread(inputFiles[i]);
      if (image.empty())
      {
#pragma omp critical
        std::cerr << "Error: No se pudo cargar la imagen " << inputFiles[i] << std::endl;
        failedImages++;
        continue;
      }

      // Preprocesamiento de canales (igual que en main)
      if (image.channels() == 1)
      {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
      }
      else if (image.channels() == 4)
      {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
      }

      // Procesar la imagen
      cv::Mat pixelArt = converter.convertToPixelArtCUDA(image);

      // Guardar la imagen
      std::vector<int> compression_params;
      compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
      compression_params.push_back(9);

      if (cv::imwrite(outputFiles[i], pixelArt, compression_params))
      {
#pragma omp critical
        {
          processedImages++;
          std::cout << "Procesada: " << inputFiles[i] << " -> " << outputFiles[i] << " (" << processedImages << "/" << totalImages << ")" << std::endl;
        }
      }
      else
      {
#pragma omp critical
        {
          std::cerr << "Error: No se pudo guardar " << outputFiles[i] << std::endl;
          failedImages++;
        }
      }
    }
    catch (const std::exception &e)
    {
#pragma omp critical
      {
        std::cerr << "Excepción procesando " << inputFiles[i] << ": " << e.what() << std::endl;
        failedImages++;
      }
    }
  }

  double endTime = omp_get_wtime();
  double totalTime = (endTime - startTime) * 1000;

  std::cout << "\n=== Resumen del procesamiento por lotes ===" << std::endl;
  std::cout << "Imágenes procesadas exitosamente: " << processedImages << "/" << totalImages << std::endl;
  std::cout << "Imágenes fallidas: " << failedImages << std::endl;
  std::cout << "Tiempo total: " << totalTime << " ms" << std::endl;
  std::cout << "Tiempo promedio por imagen: " << totalTime / totalImages << " ms" << std::endl;
  std::cout << "Rendimiento: " << (totalImages / (totalTime / 1000)) << " imágenes/segundo" << std::endl;
}

void printUsage()
{
  std::cout << "Uso para imagen única:" << std::endl;
  std::cout << "  pixelart_converter <entrada> <salida> [pixel_size] [color_bits] [dither_type] [grayscale]" << std::endl;
  std::cout << std::endl;
  std::cout << "Uso para lote de imágenes:" << std::endl;
  std::cout << "  pixelart_converter --batch <directorio_entrada> <directorio_salida> [pixel_size] [color_bits] [dither_type] [grayscale] [threads]" << std::endl;
  std::cout << std::endl;
  std::cout << "Parámetros:" << std::endl;
  std::cout << "  pixel_size: tamaño del pixel (default: 8)" << std::endl;
  std::cout << "  color_bits: bits por canal (1-8, default: 6)" << std::endl;
  std::cout << "  dither_type: 0=sin, 1=Floyd-Steinberg, 2=Ordered Dither (default: 0)" << std::endl;
  std::cout << "  grayscale: 0=color, 1=escala grises (default: 0)" << std::endl;
  std::cout << "  threads: número de hilos OpenMP (default: automático)" << std::endl;
  std::cout << std::endl;
  std::cout << "Ejemplos:" << std::endl;
  std::cout << "  Procesar imagen única: ./pixelart_converter input.jpg output.png 8 6 1 0" << std::endl;
  std::cout << "  Procesar lote: ./pixelart_converter --batch ./input ./output 8 6 1 0 4" << std::endl;
}

int main(int argc, char *argv[])
{
  // Comandos de ejemplo rápidos:
  // ./pixelart_converter --batch ./entradas ./salidas 8 6 1 0 4
  // ./pixelart_converter input.jpg output.png 8 4 2 0

  // Modo procesamiento por lotes
  if (argc >= 4 && std::string(argv[1]) == "--batch")
  {
    std::string inputDir = argv[2];
    std::string outputDir = argv[3];

    // Parámetros con valores por defecto
    int pixelSize = 8;
    int colorBits = 6;
    int ditherType = 0;
    bool grayscale = false;
    int numThreads = 0; // 0 = automático

    if (argc >= 5)
      pixelSize = std::stoi(argv[4]);
    if (argc >= 6)
      colorBits = std::stoi(argv[5]);
    if (argc >= 7)
      ditherType = std::stoi(argv[6]);
    if (argc >= 8)
      grayscale = std::stoi(argv[7]);
    if (argc >= 9)
      numThreads = std::stoi(argv[8]);

    printSystemInfo();

    // Buscar archivos de imagen en el directorio de entrada
    std::vector<std::string> inputFiles = findImageFiles(inputDir);
    if (inputFiles.empty())
    {
      std::cerr << "No se encontraron archivos de imagen en " << inputDir << std::endl;
      return -1;
    }

    std::cout << "Encontradas " << inputFiles.size() << " imágenes en " << inputDir << std::endl;

    // Generar nombres de archivos de salida
    std::vector<std::string> outputFiles = generateOutputFiles(inputFiles, outputDir);

    // Procesar el lote
    processBatchImages(inputFiles, outputFiles, pixelSize, colorBits, ditherType, grayscale, numThreads);

    return 0;
  }

  // Modo imagen única (código original)
  if (argc < 3 || argc > 7)
  {
    printUsage();
    return -1;
  }

  std::string inputFile = argv[1];
  std::string outputFile = argv[2];

  // Parámetros con valores por defecto
  int pixelSize = 8;
  int colorBits = 6;
  int ditherType = 0;
  bool grayscale = false;

  if (argc >= 4)
    pixelSize = std::stoi(argv[3]);
  if (argc >= 5)
    colorBits = std::stoi(argv[4]);
  if (argc >= 6)
    ditherType = std::stoi(argv[5]);
  if (argc >= 7)
    grayscale = std::stoi(argv[6]);

  // Convertir bits a niveles de color
  int colorLevels = 1 << colorBits;
  int totalColors = colorLevels * colorLevels * colorLevels;

  if (grayscale)
  {
    totalColors = colorLevels;
  }

  if (outputFile.substr(outputFile.find_last_of(".")) != ".png")
  {
    std::cout << "Advertencia: El archivo de salida debería ser .png para mejor calidad" << std::endl;
  }

  cv::Mat image = cv::imread(inputFile);
  if (image.empty())
  {
    std::cerr << "Error: No se pudo cargar la imagen " << inputFile << std::endl;
    return -1;
  }

  printSystemInfo();

  if (image.channels() == 1)
  {
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
  }
  else if (image.channels() == 4)
  {
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
  }

  std::cout << "Parámetros: pixelSize=" << pixelSize << ", colorBits=" << colorBits << " (" << colorLevels << " niveles por canal, " << totalColors << " colores totales)" << ", ditherType=" << ditherType << std::endl;

  PixelArtConverter converter(pixelSize, colorLevels, ditherType, grayscale);

  cv::Mat pixelArt;
  double startTime, endTime;

  std::cout << "\nProcesando con CUDA:" << std::endl;
  startTime = omp_get_wtime();
  pixelArt = converter.convertToPixelArtCUDA(image);
  endTime = omp_get_wtime();
  std::cout << "Tiempo CUDA: " << (endTime - startTime) * 1000 << " ms" << std::endl;

  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  if (cv::imwrite(outputFile, pixelArt, compression_params))
  {
    std::cout << "Imagen guardada como: " << outputFile << std::endl;
    std::cout << "Tamaño original: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Tamaño pixel art: " << pixelArt.cols << "x" << pixelArt.rows << std::endl;
  }
  else
  {
    std::cerr << "Error: No se pudo guardar la imagen " << outputFile << std::endl;
    return -1;
  }

  return 0;
}