#include <cuda_runtime.h>
#include <stdio.h>

__global__ void check_index(void)
{
    printf("threadId:(%d, %d ,%d) blockId:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim(%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc,char **argv)
{
    // init
    int n_elem=6;
    dim3 block(3);
    dim3 grid((n_elem + block.x-1) / block.x);
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // device
    check_index<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}