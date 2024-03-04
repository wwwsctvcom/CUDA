#include <stdio.h>
#include <cuda_runtime.h>


__device__ float devData;  // 静态分配，和cudaMalloc动态分配不同，这里其实是一个指针，在host端定义为__device__，对host不可见；

__global__ void checkGlobalVariable()
{
    printf("Device: The value of the global variable is %f\n", devData);
    devData += 2.0;  // device端进行+2操作
}
int main()
{
    float value=3.14f;

    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: copy %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f \n", value);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}