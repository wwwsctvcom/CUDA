#include <stdio.h>
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

int check_result(float *A, float *B, int numElements)
{
    // compare A with B one by one
    for (int i = 0; i < numElements; i++) {
        if (fabs(A[i] - B[i]) > (double)(1.0E-8)) {
            return 0;
        }
    }
    return 1;
}

void sum_matrix_cpu(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    float *A = MatA;
    float *B = MatB;
    float *C = MatC;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            C[i] = A[i] + B[i];
        }
        // remove pointer to next row
        A += nx;
        B += nx;
        C += nx;
    }
}

__global__ void sum_matrix_gpu(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    // block的index * 每个block中的线程数量 + 计算到当前线程的位置
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * ny + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    // set cuda device, default 0;
    inital_device(0);

    int nx = 1<<12;
    int ny = 1<<12;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // cpu malloc
    float *A_host = (float *)malloc(nBytes);
    float *B_host = (float *)malloc(nBytes);
    float *C_host = (float *)malloc(nBytes);
    float *C_from_gpu = (float *)malloc(nBytes);
    initial_data(A_host, nxy);
    initial_data(B_host, nxy);

    // cudaMalloc
    float *A_dev=NULL;
    float *B_dev=NULL;
    float *C_dev=NULL;
    CHECK(cudaMalloc((void**)&A_dev, nBytes));
    CHECK(cudaMalloc((void**)&B_dev, nBytes));
    CHECK(cudaMalloc((void**)&C_dev, nBytes));

    CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    // cpu
    double iStart = cpu_second();
    sum_matrix_cpu(A_host, B_host, C_host, nx, ny);
    double iElaps = cpu_second() - iStart;
    printf("CPU execution time elapsed: %f sec.\n", iElaps);

    int dimx = 32;
    int dimy = 32;
    // 2d block and 2d grid
    dim3 block_0(dimx, dimy);
    dim3 grid_0((nx - 1) / block_0.x + 1, (ny - 1) / block_0.y + 1);

    iStart = cpu_second();
    sum_matrix_gpu<<<grid_0, block_0>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpu_second() - iStart;
    printf("CUDA configuration <<<(%d, %d), (%d, %d)>>>, Time elapsed: %f sec\n", grid_0.x, grid_0.y, block_0.x, block_0.y, iElaps);

    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    if (check_result(C_host, C_from_gpu, nxy) == 1) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }

    // 1d block and 1d grid
    dimx = 32;
    dim3 block_1(dimx);
    dim3 grid_1((nxy - 1) / block_1.x + 1);

    iStart = cpu_second();
    sum_matrix_gpu<<<grid_1, block_1>>>(A_dev, B_dev, C_dev, nxy, 1);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpu_second() - iStart;
    printf("CUDA configuration <<<(%d, %d), (%d, %d)>>>, Time elapsed: %f sec\n", grid_1.x, grid_1.y, block_1.x, block_1.y, iElaps);

    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    if (check_result(C_host, C_from_gpu, nxy) == 1) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }

    // 1d block and 2d grid
    dimx=32;
    dim3 block_2(dimx);
    dim3 grid_2((nx - 1) / block_2.x + 1, ny);

    iStart = cpu_second();
    sum_matrix_gpu<<<grid_2, block_2>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpu_second() - iStart;
    printf("CUDA configuration <<<(%d, %d), (%d, %d)>>>, Time elapsed: %f sec\n", grid_2.x, grid_2.y, block_2.x, block_2.y, iElaps);

    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    if (check_result(C_host, C_from_gpu, nxy) == 1) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }

    // free
    cudaFree(A_dev);
    cudaFree(A_dev);
    cudaFree(A_dev);
    
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}