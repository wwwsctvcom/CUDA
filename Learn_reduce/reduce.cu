#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
      printf("ERROR: %s: %d, ", __FILE__, __LINE__);\
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
      exit(EXIT_FAILURE);\
  }\
}

void init_device(int device) {
    int dev = device;
    int deviceCount = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaGetDeviceCount(&deviceCount));
    CHECK(cudaSetDevice(dev));
    printf("Set cuda device %d, device name: %s, device count: %d\n", dev, deviceProp.name, deviceCount);
}

void init_data(int *data, int size)
{
    // 随机产生-5到5之间的随机正负数
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++) {
        data[i] = rand() % 11 - 5;
    }
}

// histogram function kernel
// input: Nx1, output: count histogram
__global__ void histogram(int* input, int* output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(output[input[idx]]), 1);
}

// host code to invoke the kernel
void histogram_wrapper(int *h_input, int *h_output, int N)
{
    int *d_input, *d_output;
    size_t memSize = N * sizeof(int);

    CHECK(cudaMalloc((void **)&d_input, memSize));
    CHECK(cudaMalloc((void **)&d_output, memSize));

    // 将host的input copy到device的input
    CHECK(cudaMemcpy(d_input, h_input, memSize, cudaMemcpyHostToDevice));

    int blockSize = 128;
    dim3 block(blockSize, 1);
    dim3 grid((N - 1) / block.x + 1, 1);
    histogram<<<grid, block>>>(d_input, d_output, N);

    // 将device的output copy到host端
    CHECK(cudaMemcpy(h_output, d_output, memSize, cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) 
{
    // device
    init_device(0);

    // data
    int N = argc > 1 ? atoi(argv[1]) : 100;
    size_t memSize = N * sizeof(int);

    int *h_input = (int *)malloc(memSize);
    int *h_output = (int *)malloc(memSize);
    memset(h_input, 0, memSize);
    memset(h_output, 0, memSize);
    init_data(h_input, N);

    // kernel function 
    histogram_wrapper(h_input, h_output, N);

    printf("histogram count: %d\n", *h_output);

    // free
    free(h_input);
    free(h_output);
}