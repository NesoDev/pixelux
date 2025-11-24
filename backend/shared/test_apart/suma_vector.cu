#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void Vec_add(
    const float x[], /* in */
    const float y[], /* in */
    float z[],       /* out */
    const int n      /* in */
)
{
  int my_elt = blockDim.x * blockIdx.x + threadIdx.x;

  /* total threads = blk_ct*th_per_blk may be > n */
  if (my_elt < n)
    z[my_elt] = x[my_elt] + y[my_elt];
} /* Vec_add */

// Función auxiliar para asignar vectores
void Allocate_vectors(float **x, float **y, float **z, float **cz, int n)
{
  *x = (float *)malloc(n * sizeof(float));
  *y = (float *)malloc(n * sizeof(float));
  *z = (float *)malloc(n * sizeof(float));
  *cz = (float *)malloc(n * sizeof(float));
}

// Función auxiliar para inicializar vectores
void Init_vectors(float *x, float *y, int n, char i_g)
{
  for (int i = 0; i < n; i++)
  {
    if (i_g == 'y' || i_g == 'Y')
    {
      // Valores de ejemplo para prueba
      x[i] = (float)i;
      y[i] = (float)(2 * i);
    }
    else
    {
      // Valores aleatorios
      x[i] = (float)rand() / RAND_MAX;
      y[i] = (float)rand() / RAND_MAX;
    }
  }
}

// Función auxiliar para suma serial (verificación)
void Serial_vec_add(const float *x, const float *y, float *cz, int n)
{
  for (int i = 0; i < n; i++)
  {
    cz[i] = x[i] + y[i];
  }
}

// Función auxiliar para calcular la norma de la diferencia
double Two_norm_diff(const float *z, const float *cz, int n)
{
  double diff = 0.0;
  for (int i = 0; i < n; i++)
  {
    double error = z[i] - cz[i];
    diff += error * error;
  }
  return sqrt(diff);
}

// Función auxiliar para liberar memoria
void Free_vectors(float *x, float *y, float *z, float *cz)
{
  free(x);
  free(y);
  free(z);
  free(cz);
}

// Función auxiliar para obtener argumentos
void Get_args(int argc, char *argv[], int *n, int *blk_ct, int *th_per_blk, char *i_g)
{
  if (argc != 5)
  {
    printf("Uso: %s <n> <blk_ct> <th_per_blk> <i_g>\n", argv[0]);
    printf("  n: tamaño de los vectores\n");
    printf("  blk_ct: número de bloques\n");
    printf("  th_per_blk: hilos por bloque\n");
    printf("  i_g: 'y' para entrada fija, 'n' para aleatoria\n");
    exit(1);
  }

  *n = atoi(argv[1]);
  *blk_ct = atoi(argv[2]);
  *th_per_blk = atoi(argv[3]);
  *i_g = argv[4][0];
}

int main(int argc, char *argv[])
{
  int n, th_per_blk, blk_ct;
  char i_g; /* Are x and y user input or random? */
  float *x, *y, *z, *cz;
  float *d_x, *d_y, *d_z; // Dispositivo
  double diff_norm;

  /* Get the command line arguments, and set up vectors */
  Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &i_g);
  Allocate_vectors(&x, &y, &z, &cz, n);
  Init_vectors(x, y, n, i_g);

  // Asignar memoria en el dispositivo
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));
  cudaMalloc(&d_z, n * sizeof(float));

  // Copiar datos al dispositivo
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  /* Invoke kernel and wait for it to complete */
  Vec_add<<<blk_ct, th_per_blk>>>(d_x, d_y, d_z, n);
  cudaDeviceSynchronize();

  // Copiar resultado de vuelta al host
  cudaMemcpy(z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

  /* Check for correctness */
  Serial_vec_add(x, y, cz, n);
  diff_norm = Two_norm_diff(z, cz, n);
  printf("Two-norm of difference between host and ");
  printf("device = %e\n", diff_norm);

  /* Free storage and quit */
  Free_vectors(x, y, z, cz);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
} /* main */