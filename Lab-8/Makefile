CC       := nvcc
CXXFLAGS := -ccbin clang++ -Wno-deprecated-gpu-targets -arch=sm_21
CFLAGS   := -ccbin clang -Wno-deprecated-gpu-targets -arch=sm_21

all: thrust cublass

thrust: saxpy_thrust.cu
	mkdir -p build
	$(CC) $(CXXFLAGS) -g -o build/saxpy_thrust saxpy_thrust.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

cublass: saxpy_cublas.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/saxpy_cublas saxpy_cublas.cu -lcublas -lcudart

clean:
	rm -Rf build
