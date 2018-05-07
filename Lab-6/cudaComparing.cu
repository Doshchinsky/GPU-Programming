#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value) {\
cudaError_t _m_cudaStat = value;\
if (_m_cudaStat != cudaSuccess) {\
	fprintf(stderr, "Error %s at line %d in file %s\n",\
	cudaGetErrorString(_m_cudaStat),__LINE__,__FILE__);\
	exit(1);\
}}

float cuda_host_alloc_test(int size, int niter, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsed_time;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	for (int i = 0; i < niter; i++) {
		if (up == true) {
			CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
		}
		else {
			CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
		}
	}

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, stop));

	CUDA_CHECK_RETURN(cudaFreeHost(a));
	CUDA_CHECK_RETURN(cudaFree(dev_a));

	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

	return elapsed_time;
}

float cuda_malloc_test(int size, int niter, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsed_time;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	a = (int*)malloc(size * sizeof(*a));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	for (int i = 0; i < niter; i++) {
		if (up == true) {
			CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
		}
		else
			CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, stop));

	free(a);
	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

	return elapsed_time;
}

int main(int argc, char const *argv[])
{
	const int size = (10 * pow(1024, 2));
	const int niter = 100;
	float elapsed_time;
	float MB = (float)niter * size * sizeof(int) / 1024 / 1024;

	elapsed_time = cuda_malloc_test(size, niter, true);
	printf("Time using cudaMalloc: %.6f\n", elapsed_time);
	printf("Speed CPU-->GPU: %.6f MB/s\n\n", MB / (elapsed_time / 1000));

	elapsed_time = cuda_malloc_test(size, niter, false);
	printf("Time using cudaMalloc: %.6f\n", elapsed_time);
	printf("Speed GPU-->CPU: %.6f MB/s\n\n", MB / (elapsed_time / 1000));

	elapsed_time = cuda_host_alloc_test(size, niter, true);
	printf("Time using cudaHostAlloc: %.6f\n", elapsed_time);
	printf("Speed CPU-->GPU: %.6f MB/s\n\n", MB / (elapsed_time / 1000));

	elapsed_time = cuda_host_alloc_test(size, niter, false);
	printf("Time using cudaHostAlloc: %.6f ms\n", elapsed_time);
	printf("Speed GPU-->CPU: %.6f MB/s\n\n", MB / (elapsed_time / 1000));

	return 0;
}
