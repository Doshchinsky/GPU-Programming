#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

//#define BLOCK_SIZE 32
#define SIZE 1024*1024


__host__ void SaveMatrixToFile(char* fileName, int* matrix, int width, int height) {
	FILE* file = fopen(fileName, "wt");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			fprintf(file, "%d\t", matrix[y * width + x]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}


__global__ void transpose(int* inputMatrix, int* outputMatrix, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
			
}

__host__ int main() 
{

	int width;
	int height;
	printf("Input number of columns: ");
	scanf("%d", &width);
	printf("Input number of strings: ");
	scanf("%d", &height);
	int N = width*height;

	cudaEvent_t start, stop;
	float gpuTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int* A; 
	A = (int *)malloc(sizeof(int) * N);
	int* A_t;
	A_t = (int *)malloc(sizeof(int) * N);

	for (int i = 0; i < N; i++) 
	{
		A[i] = i + 1;
	}
	SaveMatrixToFile("matrix.txt", A, width, height);


	int* A_dev; 
	int* A_t_dev; 

	cudaMalloc((void**)&A_dev, sizeof(int) * N);
	cudaMalloc((void**)&A_t_dev, sizeof(int) * N);

	cudaMemcpy(A_dev, A, N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(512);
	cudaEventRecord(start, 0);
 
	transpose<<<SIZE/512, block>>>(A_dev, A_t_dev, width, height);
	cudaEvent_t syncEvent;
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("Time of transposing: %.2f milliseconds\n", gpuTime);  
//	getch();

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(A_t, A_t_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
	SaveMatrixToFile("matrix1.txt", A_t, height, width);

	cudaFree(A_dev);
	cudaFree(A_t_dev);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete[] A;
	delete[] A_t;

	return 0;
}
