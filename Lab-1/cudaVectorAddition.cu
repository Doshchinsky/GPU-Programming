#include <stdio.h>
#include <stdlib.h>

#define SIZE 10

__global__ void addVector(float* left, float* right, float* result)
{
	int idx = threadIdx.x;
	result[idx] = left[idx] + right[idx];
}

__host__ int main()
{
	float* vec1 = new float[SIZE];
	float* vec2 = new float[SIZE];
	float* vec3 = new float[SIZE];

	for (int i = 0; i < SIZE; i++)
	{
		vec1[i] = i;
		vec2[i] = i;
	}

	float* devVec1;
	float* devVec2;
	float* devVec3;

	cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec2, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec3, sizeof(float) * SIZE);

	cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	dim3 gridSize = dim3(1, 1, 1);
	dim3 blockSize = dim3(SIZE, 1, 1);

	addVector<<<gridSize, blockSize>>>(devVec1, devVec2, devVec3);
	addVector<<<1, SIZE>>>(devVec1, devVec2, devVec3);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZE; i++) 
		printf("Element #%i: %.1f\n", i , vec3[i]);

	cudaEventDestroy(syncEvent);

	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);

	return EXIT_SUCCESS;
}
