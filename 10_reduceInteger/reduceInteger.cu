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

void initial_data_int(int* data, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    
    for (int i = 0; i < size; i++) {
        data[i] = int(rand() & 0xff);
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

/*************** PART2 *****************/
int recursive_reduce(int *data, int const size)
{
	// 递归算法
	if (size == 1) return data[0];

	// 更新stride的值
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}

	return recursive_reduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
	// thread ID
	unsigned int tid = threadIdx.x;

	if (tid >= n) return;

	// blockIdx.x * blockDim.x，当前bloc的位置 * 每个block的dim，相当于去分block进行计算
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduce_neighbored(int *g_idata, int *g_odata, unsigned int n)
{
	// thread ID
	unsigned int tid = threadIdx.x;

	if (tid >= n) return;

	// blockIdx.x * blockDim.x，当前bloc的位置 * 每个block的dim，相当于去分block进行计算
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		// 这里相当于在一个blockDim.x内部进行index，计算比如tid为1时，blockDim.x内所有的都不为0，所以需要计算很多次
		// 当tid为2时，blockDim.x内都为0，依次类推，相当于效率为50%
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}

__global__ void reduce_neighbored_less(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;
	if (idx > n) return;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		// 这里相当于在一个blockDim.x内部进行index，计算比如tid为1，2，3，4，index为【2，6，10】和【4，12，20】和【6，18，40】
		// 此时的计算效率就比reduce_neighbored的计算效率更高，所以速度也就更快；
		int index = 2 * stride * tid;
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduce_interleaved(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

	// 这里相当于按block来切分数据，然后把idata获取出来然后后续在for循环中进行计算；
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;

	if (idx >= n) return;

	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		// 这里stride按照二分进行减少进行计算；
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}

	// write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv)
{
	inital_device(0);
	
	bool bResult = false;
	int size = 1 << 24;
	printf("with array size %d\n", size);

	// execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}

	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int *tmp = (int*)malloc(bytes);

	//initialize the array
	initial_data_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double iStart, iElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction
	int cpu_sum = 0;
	iStart = cpu_second();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++) {
		cpu_sum += tmp[i];
	}
	printf("cpu sum:%d \n", cpu_sum);
	iElaps = cpu_second() - iStart;
	printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

	//kernel 1:reduceNeighbored
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpu_second();
	warmup<<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 1:reduceNeighbored
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpu_second();
	reduce_neighbored<< <grid, block >> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 2:reduceNeighboredLess
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpu_second();
	reduce_neighbored_less<<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) {
		gpu_sum += odata_host[i];
	}
	printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 3:reduceInterleaved
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpu_second();
	reduce_interleaved<<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpu_second() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	
	// free host memory
	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return EXIT_SUCCESS;
}