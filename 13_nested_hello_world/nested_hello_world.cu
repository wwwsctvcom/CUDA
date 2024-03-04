#include <stdio.h>
#include <cuda_runtime.h>


__global__ void nested_hello_world(int iSize, int iDepth)
{
    // kernel中继续调用kernel
    unsigned int tid = threadIdx.x;
    printf("depth: %d blockIdx: %d, threadIdx: %d\n", iDepth, blockIdx.x, threadIdx.x);

    // 递归结束标志
    if (iSize == 1)
        return;
    
    int nthread = (iSize>>1);
    if (tid == 0 && nthread > 0)
    {
        nested_hello_world<<<1, nthread>>>(nthread, ++iDepth);
        printf("-----------> nested execution depth: %d\n", iDepth);
    }

}

int main(int argc, char **argv)
{
    int size=64;
    int blockSize=2;

    dim3 block(blockSize, 1);
    dim3 grid((blockSize - 1) / block.x + 1, 1);
    nested_hello_world<<<grid, block>>>(size, 0);
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}