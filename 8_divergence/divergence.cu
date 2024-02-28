#include <stdio.h>
#include <stdlib.h>
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
      printf("ERROR: %s: %d, ", __FILE__, __LINE__);\
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
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

void inital_device(int device) {
    int dev = device;
    int deviceCount = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaGetDeviceCount(&deviceCount));
    CHECK(cudaSetDevice(dev));
    printf("Set cuda device %d, device name: %s, device count: %d\n", dev, deviceProp.name, deviceCount);
}

__global__ void warmup(float *data)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[tid] = a + b;
}


__global__ void kernel_no_warp(float *data)
{
    // 理论上效率会降低，但是GPU做了优化，所以实际跑出来和kernel_use_warp是一样的
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;

    if (tid % 2 == 0) {
        // tid为偶数的在第一个if中
        a = 100.0f;
    } else {
        // tid为奇数的在第二个if中
        b = 200.0f;
    }
    data[tid] = a + b;
}

__global__ void kernel_use_warp(float *data)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        // 如果64，0到31中tid/warpSize为0，可以保证0到31都在第一个if中，线程束内没有分支，这样可以速度更优
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[tid] = a + b;
}

int main(int argc, char **argv)
{
	inital_device(0);

    double iStart, iElaps;
	int size = 64;
	int blocksize = 64;
	if (argc > 1) blocksize = atoi(argv[1]);
	if (argc > 2) size = atoi(argv[2]);
	printf("Param size: %d, blocksize: %d\n", size, blocksize);

	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // malloc
	float * C_dev;
	size_t nBytes = size * sizeof(float);
	float * C_host = (float*)malloc(nBytes);
	CHECK(cudaMalloc((float**)&C_dev, nBytes));
	
    // run warmup
	cudaDeviceSynchronize();
	iStart = cpu_second();
	warmup<<<grid, block>>>(C_dev);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	printf("warmup	       <<<%d, %d>>>, elapsed %lf sec\n", grid.x, block.x, iElaps);
	
	//run kernel_no_warp
	iStart = cpu_second();
	kernel_no_warp<<<grid, block>>>(C_dev);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	printf("kernel_no_warp <<<%d, %d>>>, elapsed %lf sec\n", grid.x, block.x, iElaps);

	//run kernel_use_warp
	iStart = cpu_second();
	kernel_use_warp<<<grid, block>>>(C_dev);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	printf("kernel_use_warp<<<%d, %d>>>, elapsed %lf sec\n", grid.x, block.x, iElaps);

    // free
	cudaFree(C_dev);
	free(C_host);
    
	cudaDeviceReset();
	return 0;
}