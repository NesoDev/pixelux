#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MAX_DATA 1000
#define NUM_BINS 10
#define MAX_VALUE 100
#define NUM_THREADS 4

typedef struct
{
  int *data;
  int start;
  int end;
  int *histogram;
  int bin_size;
} thread_data_t;

int global_data[MAX_DATA];
int global_histogram[NUM_BINS];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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

void *calculate_histogram(void *arg)
{
  thread_data_t *data = (thread_data_t *)arg;
  int local_histogram[NUM_BINS] = {0};

  // Calcular histograma local
  for (int i = data->start; i < data->end; i++)
  {
    int value = data->data[i];
    int bin = value / data->bin_size;
    if (bin >= NUM_BINS)
      bin = NUM_BINS - 1;
    local_histogram[bin]++;
  }

  // Actualizar histograma global con mutex
  pthread_mutex_lock(&mutex);
  for (int i = 0; i < NUM_BINS; i++)
  {
    data->histogram[i] += local_histogram[i];
  }
  pthread_mutex_unlock(&mutex);

  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  thread_data_t thread_data[NUM_THREADS];

  int data_count = 100;
  int bin_size = MAX_VALUE / NUM_BINS;

  // Generar datos
  generate_data(global_data, data_count);

  printf("Datos generados:\n");
  for (int i = 0; i < data_count; i++)
  {
    printf("%d ", global_data[i]);
    if ((i + 1) % 20 == 0)
      printf("\n");
  }
  printf("\n");

  // Inicializar histograma global
  for (int i = 0; i < NUM_BINS; i++)
  {
    global_histogram[i] = 0;
  }

  // Crear threads
  int chunk_size = data_count / NUM_THREADS;

  for (int i = 0; i < NUM_THREADS; i++)
  {
    thread_data[i].data = global_data;
    thread_data[i].start = i * chunk_size;
    thread_data[i].end = (i == NUM_THREADS - 1) ? data_count : (i + 1) * chunk_size;
    thread_data[i].histogram = global_histogram;
    thread_data[i].bin_size = bin_size;

    pthread_create(&threads[i], NULL, calculate_histogram, (void *)&thread_data[i]);
  }

  // Esperar a que todos los threads terminen
  for (int i = 0; i < NUM_THREADS; i++)
  {
    pthread_join(threads[i], NULL);
  }

  // Imprimir histograma
  print_histogram(global_histogram, NUM_BINS, bin_size);

  pthread_mutex_destroy(&mutex);

  return 0;
}