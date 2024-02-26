#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
    #include <Windows.h>
#else
    #include <sys/time.h>
#endif

#define CHECK(func)\
{\
  const cudaError_t error = func;\
  if(error != cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
      exit(EXIT_FAILURE);\
  }\
}

#ifdef _WIN32
int gettimeofday(struct timeval *tv, void *tz)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tv->tv_sec = clock;
    tv->tv_usec = wtm.wMilliseconds * 1000;
    return 0;
}
#endif

double cpu_second()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;

}

void initial_data(float* data, int size)
{
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

void sum_arrays_cpu(float *h_A, float *h_B, float *h_res, int numElements)
{
    for(int i = 0; i < numElements; i++) {
        h_res[i] = h_A[i] + h_B[i];
    }
}

__global__ void sum_arrays_gpu(float *d_A, float *d_B, float *d_res, int numElements)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < numElements) {
        d_res[i] = d_A[i] + d_B[i];
    }
}

int main(int argc,char **argv)
{
    // set device
    int device = 0;
    cudaSetDevice(device);

    // sum arrays on CPU
    int numElements = 5000;
    int size = sizeof(float) * numElements;

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_res = (float *)malloc(size);
    memset(h_res, 0, size);

    if (h_A == nullptr || h_B == nullptr || h_res == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    initial_data(h_A, numElements);
    initial_data(h_B, numElements);

    double cpu_start_time = cpu_second();
    sum_arrays_cpu(h_A, h_B, h_res, numElements);
    double cpu_end_time = cpu_second();
    printf("Execution CPU time elapsed: %fsec\n", cpu_end_time - cpu_start_time);

    // verify cpu result
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_res[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("CPU result passed\n");

    // sum arrays on GPU    
    float *d_A, *d_B, *d_res;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_res, size));

    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    double gpu_start_time = cpu_second();
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    sum_arrays_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_res, numElements);
    double gpu_end_time = cpu_second();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch sum_arrays_gpu kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Execution GPU time elapsed: %fsec\n", gpu_end_time - gpu_start_time);

    CHECK(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));

    // verify GPU result
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_res[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("GPU result passed\n");

    // free
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_res));

    free(h_A);
    free(h_B);
    free(h_res);

    cudaDeviceReset();
    return 0;
}