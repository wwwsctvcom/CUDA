#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

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

void initial_data(float* data, int size)
{
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

void inital_device(int device) {
    int dev = device;
    int device_count = 0;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaGetDeviceCount(&device_count));
    CHECK(cudaSetDevice(dev));
    printf("Set device %d, device name: %s, device count: %d\n", dev, deviceProp.name, device_count);
}

__global__ void print_global_index_1(float *A, const int nx, const int ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int threadId = iy * gridDim.x + ix;
    
    printf("Method1: thread_id(%d, %d) block_id(%d, %d) coordinate(%d, %d)"
          " global index %2d val %f\n", threadIdx.x, threadIdx.y,
          blockIdx.x, blockIdx.y, ix, iy, threadId, A[threadId]);
}

__global__ void print_global_index_2(float *A, const int nx, const int ny)
{
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("Method2: thread_id(%d, %d) block_id(%d, %d) global index %2d val %f\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, threadId, A[threadId]);
}


int main(int argc,char **argv)
{
    inital_device(0);

    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    memset(h_A, 0, nBytes);
    initial_data(h_A, nxy);

    // cuda
    float *d_A = nullptr;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMemcpy(d_A, h_A, nxy, cudaMemcpyHostToDevice);)

    dim3 block(4, 2);
    dim3 grid((nx-1) / block.x + 1, (ny - 1) / block.y + 1);

    print_global_index_1<<<grid, block>>>(d_A, nx, ny);
    print_global_index_2<<<grid, block>>>(d_A, nx, ny);

    // free
    cudaFree(d_A);

    free(h_A);

    cudaDeviceReset();
    return 0;
}