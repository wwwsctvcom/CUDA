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
        data[i] = rand() / (float)RAND_MAX;
    }
}

// softmax function kernel
__global__ void softmax(float* input, float *output, int size)
{
    // 减去最大值，防止溢出
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) 
    {
        // 获取input数组里面最大的值，这里每个thread中都计算一次最大值这里可以进行优化
        float max_val = input[index];
        for (int i = 0; i < size; i++)
        {
            max_val = fmax(max_val, input[i]);
        }

        // 计算softmax中分母和
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            sum += exp(input[i] - max_val);
        }

        // 计算index位置的softmax值
        output[index] = exp(input[index] - max_val) / sum;
    }
}

// host code to invoke the kernel
void softmax_wrapper(float *h_input, float *h_output, int size)
{
    float *d_input, *d_output;
    CHECK(cudaMalloc((void **)&d_input, size * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_output, size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    dim3 block(blockSize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    softmax<<<grid, block>>>(d_input, d_output, size);

    CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) 
{
    // device
    init_device(0);

    // data
    int size = atoi(argv[1]);

    float *h_input = (float *)malloc(size * sizeof(float));
    float *h_output = (float *)malloc(size * sizeof(float));
    memset(h_input, 0, size * sizeof(float));
    memset(h_output, 0, size * sizeof(float));
    init_data(h_input, size);

    // kernel function 
    softmax_wrapper(h_input, h_output, size);

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        printf("probability: %f\n", h_output[i] * 100);
        sum += h_output[i];
    }
    printf("sum of all probability: %f\n", sum);
    

    // free
    free(h_input);
    free(h_output);
}