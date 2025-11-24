#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000 // Tamaño por defecto

// Versión secuencial
void prodmv_secuencial(double a[N], double c[N], double B[N][N])
{
  int i, j;
  for (i = 0; i < N; i++)
  {
    double sum = 0.0;
    for (j = 0; j < N; j++)
    {
      sum += B[i][j] * c[j];
    }
    a[i] = sum;
  }
}

// Versión paralela con OpenMP
void prodmv_paralelo(double a[N], double c[N], double B[N][N])
{
  int i, j;

#pragma omp parallel for private(j)
  for (i = 0; i < N; i++)
  {
    double sum = 0.0;
    for (j = 0; j < N; j++)
    {
      sum += B[i][j] * c[j];
    }
    a[i] = sum;
  }
}

// Inicializar datos de prueba
void inicializar_datos(double c[N], double B[N][N])
{
  int i, j;
  for (i = 0; i < N; i++)
  {
    c[i] = i + 1.0;
    for (j = 0; j < N; j++)
    {
      B[i][j] = (i + 1.0) * (j + 1.0);
    }
  }
}

int main(int argc, char *argv[])
{
  double a_seq[N], a_par[N], c[N], B[N][N];
  int num_hilos = 4; // Valor por defecto

  // Leer número de hilos desde consola si se proporciona
  if (argc > 1)
  {
    num_hilos = atoi(argv[1]);
  }

  omp_set_num_threads(num_hilos);

  printf("=== Producto Matriz-Vector ===\n");
  printf("Tamaño: %dx%d\n", N, N);
  printf("Hilos: %d\n", num_hilos);

  // Inicializar
  printf("Inicializando datos...\n");
  inicializar_datos(c, B);

  // Ejecutar secuencial
  printf("Ejecutando version secuencial...\n");
  double start = omp_get_wtime();
  prodmv_secuencial(a_seq, c, B);
  double tiempo_seq = omp_get_wtime() - start;

  // Ejecutar paralelo
  printf("Ejecutando version paralela...\n");
  start = omp_get_wtime();
  prodmv_paralelo(a_par, c, B);
  double tiempo_par = omp_get_wtime() - start;

  // Verificar resultados
  int correcto = 1;
  for (int i = 0; i < N; i++)
  {
    if (fabs(a_seq[i] - a_par[i]) > 0.0001)
    {
      correcto = 0;
      break;
    }
  }

  // Resultados
  printf("\n--- RESULTADOS ---\n");
  printf("Tiempo secuencial: %.4f segundos\n", tiempo_seq);
  printf("Tiempo paralelo:   %.4f segundos\n", tiempo_par);

  double speedup = tiempo_seq / tiempo_par;
  double eficiencia = speedup / num_hilos;

  printf("Speedup: %.2f\n", speedup);
  printf("Eficiencia: %.2f%%\n", eficiencia * 100);

  // Cálculos teóricos
  printf("\n--- CALCULOS TEORICOS ---\n");
  printf("Operaciones secuencial: 2 * %d^2 = %d\n", N, 2 * N * N);
  printf("Operaciones por hilo: 2 * %d * %d = %d\n", N / num_hilos, N, 2 * (N / num_hilos) * N);
  printf("Speedup teorico: %d\n", num_hilos);
  printf("Eficiencia teorica: 100%%\n");

  if (correcto)
  {
    printf("\n✅ Resultados correctos\n");
  }
  else
  {
    printf("\n❌ Error en los resultados\n");
  }

  return 0;
}