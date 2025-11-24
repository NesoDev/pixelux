#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MAX_DATA 1000
#define NUM_BINS 10
#define MAX_VALUE 100

void generate_data(int *data, int n)
{
  srand(time(NULL));
  for (int i = 0; i < n; i++)
  {
    data[i] = rand() % MAX_VALUE;
  }
}

void print_histogram(int *histogram, int num_bins, int bin_size)
{
  printf("\nHISTOGRAMA\n");
  printf("----------\n");

  for (int i = 0; i < num_bins; i++)
  {
    int lower = i * bin_size;
    int upper = (i == num_bins - 1) ? MAX_VALUE : (i + 1) * bin_size - 1;
    printf("%3d - %3d: ", lower, upper);

    for (int j = 0; j < histogram[i]; j++)
    {
      printf("*");
    }
    printf(" (%d)\n", histogram[i]);
  }
}

int main(int argc, char *argv[])
{
  int rank, size;
  int *global_data = NULL;
  int *local_data = NULL;
  int *local_histogram = NULL;
  int *global_histogram = NULL;

  int data_count = 100; // Número total de datos
  int local_count;
  int bin_size = MAX_VALUE / NUM_BINS;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Proceso 0 genera los datos
  if (rank == 0)
  {
    global_data = (int *)malloc(data_count * sizeof(int));
    generate_data(global_data, data_count);

    printf("Datos generados:\n");
    for (int i = 0; i < data_count; i++)
    {
      printf("%d ", global_data[i]);
      if ((i + 1) % 20 == 0)
        printf("\n");
    }
    printf("\n");
  }

  // Distribuir el tamaño local a cada proceso
  local_count = data_count / size;
  int remainder = data_count % size;

  // Ajustar para procesos que reciben un dato extra
  if (rank < remainder)
  {
    local_count++;
  }

  local_data = (int *)malloc(local_count * sizeof(int));

  // Distribuir los datos usando MPI_Scatterv
  int *sendcounts = NULL;
  int *displs = NULL;

  if (rank == 0)
  {
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
      sendcounts[i] = data_count / size + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv(global_data, sendcounts, displs, MPI_INT,
               local_data, local_count, MPI_INT, 0, MPI_COMM_WORLD);

  // Cada proceso calcula su histograma local
  local_histogram = (int *)calloc(NUM_BINS, sizeof(int));

  for (int i = 0; i < local_count; i++)
  {
    int bin = local_data[i] / bin_size;
    if (bin >= NUM_BINS)
      bin = NUM_BINS - 1;
    local_histogram[bin]++;
  }

  // Reducir todos los histogramas locales al proceso 0
  if (rank == 0)
  {
    global_histogram = (int *)calloc(NUM_BINS, sizeof(int));
  }

  MPI_Reduce(local_histogram, global_histogram, NUM_BINS,
             MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Proceso 0 imprime el histograma
  if (rank == 0)
  {
    print_histogram(global_histogram, NUM_BINS, bin_size);

    free(global_data);
    free(global_histogram);
    free(sendcounts);
    free(displs);
  }

  free(local_data);
  free(local_histogram);

  MPI_Finalize();
  return 0;
}