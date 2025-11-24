#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 8 // Tamaño de la matriz y el vector

void leer(double A[N][N], double v[N])
{
  // Función para inicializar la matriz y el vector
  // En un caso real, aquí leerías los datos de un archivo o los generarías
  for (int i = 0; i < N; i++)
  {
    v[i] = i + 1; // Vector [1, 2, 3, ..., N]
    for (int j = 0; j < N; j++)
    {
      A[i][j] = (i == j) ? 2.0 : 0.5; // Matriz diagonal dominante
    }
  }
}

void escribir(double x[N])
{
  printf("Resultado x = [");
  for (int i = 0; i < N; i++)
  {
    printf("%.2f", x[i]);
    if (i < N - 1)
      printf(", ");
  }
  printf("]\n");
}

int main(int argc, char *argv[])
{
  int i, j, rank, p, k;
  double A[N][N], v[N], x[N];
  double *local_A, *local_x;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  k = N / p; // Número de filas por proceso

  // El proceso 0 lee los datos de entrada
  if (rank == 0)
  {
    leer(A, v);
    printf("Matriz A y vector v inicializados\n");
  }

  // Distribuir el vector v a todos los procesos
  MPI_Bcast(v, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Asignar memoria para la porción local de A y el resultado local
  local_A = (double *)malloc(k * N * sizeof(double));
  local_x = (double *)malloc(k * sizeof(double));

  // Distribuir la matriz A por bloques de filas consecutivas
  if (rank == 0)
  {
    // El proceso 0 envía bloques de filas a los demás procesos
    for (int dest = 0; dest < p; dest++)
    {
      if (dest == 0)
      {
        // Copia local para el proceso 0
        for (i = 0; i < k; i++)
        {
          for (j = 0; j < N; j++)
          {
            local_A[i * N + j] = A[i][j];
          }
        }
      }
      else
      {
        // Envía k filas al proceso dest
        MPI_Send(&A[dest * k], k * N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      }
    }
  }
  else
  {
    // Los demás procesos reciben su porción de A
    MPI_Recv(local_A, k * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Cada proceso calcula su parte del producto
  for (i = 0; i < k; i++)
  {
    local_x[i] = 0.0;
    for (j = 0; j < N; j++)
    {
      local_x[i] += local_A[i * N + j] * v[j];
    }
  }

  // Recolectar los resultados en el proceso 0
  if (rank == 0)
  {
    // Copia los resultados locales del proceso 0
    for (i = 0; i < k; i++)
    {
      x[i] = local_x[i];
    }
    // Recibe los resultados de los demás procesos
    for (int src = 1; src < p; src++)
    {
      MPI_Recv(&x[src * k], k, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  else
  {
    // Los demás procesos envían sus resultados al proceso 0
    MPI_Send(local_x, k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  // El proceso 0 escribe el resultado
  if (rank == 0)
  {
    escribir(x);
  }

  // Liberar memoria
  free(local_A);
  free(local_x);

  MPI_Finalize();
  return 0;
}