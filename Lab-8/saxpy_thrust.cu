#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const
    { 
        return a * x + y;
    }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

double test(const int N)
{
    float milliseconds = 0;
    float maxError     = 0.0f;

    /*
     * Create CUDA events for timing purposes
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*
     * Allocate host memory
     */
    thrust::host_vector<float> x(1 << 20);
    thrust::host_vector<float> y(1 << 20);

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Transfer to device
    thrust::device_vector<float> X = x;
    thrust::device_vector<float> Y = y;

    cudaEventRecord(start);
    // Perform SAXPY on 1M elements
    saxpy_fast(2.0, X, Y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    thrust::copy(Y.begin(), Y.end(), y.begin());

    cudaEventElapsedTime(&milliseconds, start, stop);

    for (int i = 0; i < N; i++) 
    {
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
/*
    printf("N = %d\n", N);
    //printf("Max error: %f\n", maxError);
    printf("Succesfully performed SAXPY on %d elements in %f milliseconds.\n", N, milliseconds);
    printf("Effective Bandwidth (GB/s): %f\n", (float)N * 4.0 / (float)milliseconds / (float)1e6);
*/
    return milliseconds;
}

int main(int argc, char const *argv[])
{
    int size = 0;
    const int n_tests = 10;
    double time = 0.0;

    size = 1 << 16;
    for (int i = 0; i < n_tests; i++)
    {
        time += test(size);
    }

    printf("[%s] N = %d, time %f msec\n", __FILE__, size, time / n_tests);

    time = 0.0;
    size = 1 << 18;
    for (int i = 0; i < n_tests; i++)
    {
        time += test(size);
    }

    printf("[%s] N = %d, time %f msec\n", __FILE__, size, time / n_tests);

    time = 0.0;
    size = 1 << 20;
    for (int i = 0; i < n_tests; i++)
    {
        time += test(size);
    }

    printf("[%s] N = %d, time %f msec\n", __FILE__, size, time / n_tests);

    return 0;
}
