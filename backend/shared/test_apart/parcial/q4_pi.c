#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

double f(double x)
{
  return 4.0 / (1.0 + x * x);
}

double trapezoidal_rule(double a, double b, int n, int rank, int size)
{
  double h = (b - a) / n;
  double local_sum = 0.0;

  // Distribuir los trapecios entre procesos
  int local_n = n / size;
  int remainder = n % size;

  // Ajustar para procesos que reciben un trapecio extra
  int start, end;
  if (rank < remainder)
  {
    local_n++;
    start = rank * local_n;
  }
  else
  {
    start = rank * local_n + remainder;
  }
  end = start + local_n;

  // Calcular la suma local usando la regla trapezoidal
  for (int i = start; i < end; i++)
  {
    double x0 = a + i * h;
    double x1 = a + (i + 1) * h;
    local_sum += (f(x0) + f(x1)) * h / 2.0;
  }

  return local_sum;
}

int main(int argc, char *argv[])
{
  int rank, size;
  double a = 0.0, b = 1.0; // Valores por defecto para calcular π
  int n = 10;              // Valor por defecto

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // SOLUCIÓN: Solo el proceso 0 lee la entrada
  if (rank == 0)
  {
    printf("Calculo de PI usando la regla trapezoidal\n");
    printf("Integral de 4/(1+x^2) de 0 a 1\n\n");

    printf("Ingrese el numero de trapecios (n) [actual: %d]: ", n);
    scanf("%d", &n);

    // Validación
    if (n <= size)
    {
      printf("Error: El numero de trapecios (%d) debe ser mayor que el numero de procesos (%d)\n", n, size);
      printf("Usando n = %d en su lugar.\n", size * 100);
      n = size * 100;
    }

    printf("Usando %d trapecios con %d procesos\n\n", n, size);
  }

  // Broadcast de los parámetros a todos los procesos
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Cada proceso calcula su parte
  double local_result = trapezoidal_rule(a, b, n, rank, size);

  // Reducir todos los resultados al proceso 0
  double global_result;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Solo el proceso 0 imprime el resultado
  if (rank == 0)
  {
    printf("Resultados:\n");
    printf("Valor aproximado de PI: %.15f\n", global_result);
    printf("Valor real de PI:       %.15f\n", M_PI);
    printf("Error absoluto:         %.15f\n", fabs(M_PI - global_result));
    printf("Error relativo:         %.2e%%\n", fabs(M_PI - global_result) / M_PI * 100);

    // Mostrar distribución de trabajo
    printf("\nDistribucion de trabajo:\n");
    double h = (b - a) / n;
    for (int i = 0; i < size; i++)
    {
      int local_n = n / size;
      int remainder = n % size;
      int start, end;

      if (i < remainder)
      {
        local_n++;
        start = i * local_n;
      }
      else
      {
        start = i * local_n + remainder;
      }
      end = start + local_n;

      printf("Proceso %d: trapecios %d-%d (%d trapecios)\n",
             i, start, end - 1, local_n);
    }
  }

  MPI_Finalize();
  return 0;
}