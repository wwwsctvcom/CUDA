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

void init_data(float *data, int size)
{
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++) {
        data[i] = i;
    }
}

// elementwise add function kernel
__global__ void elementwise_add(float *a, float *b, float *c, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) 
    {
        c[index] = a[index] + b[index];
    }
}

// host code to invoke the kernel
void elementwise_add_wrapper(float *h_a, float *h_b, float *h_c, int N)
{
    float *d_a, *d_b, *d_c;
    size_t memSize = N * sizeof(float);

    CHECK(cudaMalloc((void **)&d_a, memSize));
    CHECK(cudaMalloc((void **)&d_b, memSize));
    CHECK(cudaMalloc((void **)&d_c, memSize));

    // 将host的input copy到device的input
    CHECK(cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice));

    int blockSize = 256;
    dim3 block(blockSize, 1);
    dim3 grid((N - 1) / block.x + 1, 1);
    elementwise_add<<<grid, block>>>(d_a, d_b, d_c, N);

    // 将device的output copy到host端
    CHECK(cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char **argv) 
{
    // device
    init_device(0);

    // init host data
    int N = argc > 1 ? atoi(argv[1]) : 10;
    size_t memSize = N * sizeof(float);

    float *h_a = (float *)malloc(memSize);
    float *h_b = (float *)malloc(memSize);
    float *h_c = (float *)malloc(memSize);
    memset(h_a, 0, memSize);
    memset(h_b, 0, memSize);
    memset(h_c, 0, memSize);

    init_data(h_a, N);
    init_data(h_b, N);
    
    // kernel function 
    elementwise_add_wrapper(h_a, h_b, h_c, N);

    for (size_t i = 0; i < N; i++)
    {
        printf("h_a(%f) + h_b(%f) = h_c(%f)\n", h_a[i], h_b[i], h_c[i]);
    }

    // free
    free(h_a);
    free(h_b);
    free(h_c);
}