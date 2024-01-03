#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Size of the array
    int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float* h_A = (float*)malloc(size);

    // Initialize the host array
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
    }

    // Allocate memory on the device (GPU)
    float* d_A;
    cudaMalloc(&d_A, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Here you would typically perform some operations on the GPU
    
    std::cout << "~~~~ cudaFree ~~~~" << std::endl;
    // Free GPU memory
    cudaFree(d_A);

    // Free host memory
    free(h_A);

    return 0;
}

