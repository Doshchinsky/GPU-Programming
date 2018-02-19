#include <stdio.h>
#include <stdlib.h>

#define SIZE (1024*1024)

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

	cudaEvent_t start, stop;
	cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec2, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec3, sizeof(float) * SIZE);

	cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	int block = 512;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	addVector<<<SIZE/512, block>>>(devVec1, devVec2, devVec3);
	cudaEventRecord(stop);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

	float time = 0;
//	for (int i = 0; i < SIZE; i++) 
//		printf("Element #%i: %.1f\n", i , vec3[i]);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time: %f\n", time);

	FILE *f = fopen("time.txt", "a+");
	if (f == NULL) {
		fprintf(stderr, "FILE ERROR!\n");
	} else {
		fprintf(f, "%f 512\n", time);
	}
	fclose(f);
	cudaEventDestroy(syncEvent);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);

	return EXIT_SUCCESS;
}
