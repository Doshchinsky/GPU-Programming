#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat),__LINE__, __FILE__);\
		exit(1);\
	}\
}

__global__ void addVector(float* left, float* right, float* result)
{
	int idx = threadIdx.x;

	result[idx] = left[idx] + right[idx];
}

#define SIZE 2048 
__host__ int main()
{
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//Âûäåëÿåì ïàìÿòü ïîä âåêòîðà
	float* vec1 = new float[SIZE];
	float* vec2 = new float[SIZE];
	float* vec3 = new float[SIZE];

	for (int i = 0; i < SIZE; i++)
	{
		vec1[i] = i;
		vec2[i] = i;
//		printf("#%d\t%f\t %f\n", i, vec1[i], vec2[i]);
	}

	float* devVec1;
	float* devVec2;
	float* devVec3;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&devVec1, sizeof(float) * SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&devVec2, sizeof(float) * SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&devVec3, sizeof(float) * SIZE));

	CUDA_CHECK_RETURN(cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice));

	
	dim3 block(512);
	cudaEventRecord(start,0);
	addVector <<<SIZE/512, block >>>(devVec1, devVec2, devVec3);
	
	cudaEventRecord(stop, 0);
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	cudaEvent_t syncEvent;

	CUDA_CHECK_RETURN(cudaEventCreate(&syncEvent));
	CUDA_CHECK_RETURN(cudaEventRecord(syncEvent, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(syncEvent));
	CUDA_CHECK_RETURN(cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < SIZE; i++)
	{
		//printf("Element #%i: %.1f\n", i, vec3[i]);
	}
	fprintf(stderr,"gTest took %g\n",elapsedTime);

	cudaEventDestroy(syncEvent);

	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);

	delete[] vec1; vec1 = 0;
	delete[] vec2; vec2 = 0;
	delete[] vec3; vec3 = 0;

	return 0;
}
