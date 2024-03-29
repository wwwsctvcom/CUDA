#include <cuda_runtime.h>
#include <stdio.h>

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

void inital_device(int device) {
    int dev = device;
    int deviceCount = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaGetDeviceCount(&deviceCount));
    CHECK(cudaSetDevice(dev));
    printf("Set cuda device %d, device name: %s, device count: %d\n", dev, deviceProp.name, deviceCount);
}

int main(int argc,char** argv)
{
    int driverVersion=0, runtimeVersion=0;
    
    inital_device(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n", 
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("CUDA Capability Major/Minor version number:   %d.%d\n", deviceProp.major, deviceProp.minor);

    printf("GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    printf("Memory Bus width:                             %d-bits\n", deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                            	%d bytes\n", deviceProp.l2CacheSize);
    }
    printf("Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    printf("Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
            deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory               %zu bytes\n",
            deviceProp.totalConstMem);
    printf("Total amount of shared memory per block:      %zu bytes\n",
            deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block:%d\n",
            deviceProp.regsPerBlock);
    printf("Wrap size:                                    %d\n", deviceProp.warpSize);
    printf("Maximun number of thread per multiprocesser:  %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximun number of thread per block:           %d\n",
            deviceProp.maxThreadsPerBlock);
    printf("Maximun size of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Maximun size of each dimension of a grid:     %d x %d x %d\n", 
            deviceProp.maxGridSize[0], 
            deviceProp.maxGridSize[1], 
            deviceProp.maxGridSize[2]);
    printf("Maximu memory pitch                           %zu bytes\n", deviceProp.memPitch);
    printf("----------------------------------------------------------\n");
    printf("Number of multiprocessors:                      %d\n", deviceProp.multiProcessorCount);
    printf("Total amount of constant memory:                %4.2f KB\n", deviceProp.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block:        %4.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
    printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
    printf("Warp size                                       %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:     %d\n", deviceProp.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}