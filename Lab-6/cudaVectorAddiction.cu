#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N (1024*1024)
#define FULL_DATA_SIZE  (N*20)

int getrand(int min, int max)
{
	return (double)rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

bool verification(float *host_a, float *host_b, float *host_c)
{
	for (int i = 0; i < FULL_DATA_SIZE; i++) {
		if (fabs(host_a[i] + host_b[i] - host_c[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			return false;
		}
	}
	return true;
}

__global__ void kernel(const float *a, const float *b, float *c)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx < N) {
		c[idx] = a[idx] + b[idx];
	}
}

int main(int argc, char const *argv[])
{
	cudaDeviceProp prop;
	int whichDevice;

	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap) {
		printf("Device does not support overlapping\n");
		return 0;
	}

	float *host_a, *host_b, *host_c;
	float *dev_a0, *dev_b0, *dev_c0;
	float *dev_a1, *dev_b1, *dev_c1;

	cudaMalloc((void**)&dev_a0, N * sizeof(float));
	cudaMalloc((void**)&dev_b0, N * sizeof(float));
	cudaMalloc((void**)&dev_c0, N * sizeof(float));
	cudaMalloc((void**)&dev_a1, N * sizeof(float));
	cudaMalloc((void**)&dev_b1, N * sizeof(float));
	cudaMalloc((void**)&dev_c1, N * sizeof(float));

	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(float), cudaHostAllocDefault);

	srand(time(NULL));
	for (int i = 0; i < FULL_DATA_SIZE; i++) {
		host_a[i] = getrand(1, 1000);
		host_b[i] = getrand(1, 1000);
	}

	float elapsedTime;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaEventRecord(start, 0);

	for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
		printf("i = %d, FULL_DATA_SIZE = %d\n", i, FULL_DATA_SIZE);
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(float), cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(float), cudaMemcpyHostToDevice, stream1);

		kernel << <N / 256, 256, 0, stream0 >> >(dev_a0, dev_b0, dev_c0);
		kernel << <N / 256, 256, 0, stream1 >> >(dev_a1, dev_b1, dev_c1);

		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(float), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	if (verification(host_a, host_b, host_c) == true)
		printf("Verification: OK\n");
	else
		printf("Verification: ERROR\n");

	printf("Elapsed time: %3.1f ms\n", elapsedTime);

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFree(dev_c1);

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	return 0;
}
