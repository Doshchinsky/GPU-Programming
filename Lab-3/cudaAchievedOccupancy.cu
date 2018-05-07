#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>

#define SIZE 1024*1024*1000

#define CUDA_CHECK_RETURN(value) {\
        cudaError_t _m_cudaStat = value;\
        if (_m_cudaStat != cudaSuccess) {\
                fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat),__LINE__, __FILE__);\
                exit(1);\
        }\
}

__host__ int main()
{
	char dev;
	cudaSetDevice(dev); 
	cudaDeviceProp deviceProp; 
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("  Total amount of constant memory:  %lu bytes\n", deviceProp.totalConstMem); 
	printf("  Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock); 
	printf("  Warp size: %d\n", deviceProp.warpSize); 
	printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor); 
	printf("  Maximum number of threads per block:  %d\n", deviceProp.maxThreadsPerBlock);


	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float* vec1 = new float[SIZE];
	cudaEventRecord(start, 0);	
	for (int i = 0; i < SIZE; i++)
	{
		vec1[i] = i;
//		printf("#%d\t%f\t %f\n", i, vec1[i]);
	}

        cudaEventRecord(stop, 0);
//	float time = 0;

        cudaEvent_t syncEvent;
	printf("%g", elapsedTime);
	float* devVec1;
	cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
	cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	
	cudaFree(devVec1);
	delete[] vec1; vec1 = 0;
	return 0;
}
