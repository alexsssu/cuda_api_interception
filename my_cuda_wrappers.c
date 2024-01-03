// my_cuda_wrappers.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define MAX_ALLOCATIONS 1024  // Adjust this as necessary

typedef cudaError_t (*real_cudaGetDeviceCount_t)(int *);
typedef cudaError_t (*real_cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*real_cudaFree_t)(void *);

// A simple map to keep track of allocations
typedef struct {
    void *address;
    size_t size;
} AllocationEntry;

AllocationEntry allocationMap[MAX_ALLOCATIONS] = {0};

// Function to add an allocation to the map
void add_allocation(void *address, size_t size) {
    for (int i = 0; i < MAX_ALLOCATIONS; ++i) {
        if (allocationMap[i].address == NULL) {
            allocationMap[i].address = address;
            allocationMap[i].size = size;
            return;
        }
    }
    fprintf(stderr, "Allocation map is full!\n");
}

// Function to remove an allocation from the map
void remove_allocation(void *address) {
    for (int i = 0; i < MAX_ALLOCATIONS; ++i) {
        if (allocationMap[i].address == address) {
            allocationMap[i].address = NULL;
            allocationMap[i].size = 0;
            return;
        }
    }
    fprintf(stderr, "Address not found in allocation map!\n");
}


// Intercept cudaGetDeviceCount
cudaError_t cudaGetDeviceCount(int *count) {
    real_cudaGetDeviceCount_t real_cudaGetDeviceCount = (real_cudaGetDeviceCount_t)dlsym(RTLD_NEXT, "cudaGetDeviceCount");
    cudaError_t result = real_cudaGetDeviceCount(count);
    if (result == cudaSuccess) {
        printf("cudaGetDeviceCount - GPU Device Count: %d.\n", *count);
    }
    return result;
}

// Intercept cudaMalloc
// whey void **devPtr: (void **)&d_A: The first argument to cudaMalloc requires a pointer to a pointer. d_A is a pointer that is intended to point to the allocated GPU memory. Since cudaMalloc needs to set this pointer to point to the newly allocated memory, you pass the address of d_A (which is &d_A). However, cudaMalloc is defined to take a void** argument (a pointer to a pointer to void), which is a generic type in C that can point to any data type. That's why you cast the address of d_A to (void **). This tells the compiler to treat the address of d_A as a generic pointer to a pointer.
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    real_cudaMalloc_t real_cudaMalloc = (real_cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    cudaError_t result = real_cudaMalloc(devPtr, size);
    if (result == cudaSuccess) {
        printf("cudaMalloc - GPU Address: %p, Size: %zu\n", *devPtr, size);
        add_allocation(*devPtr, size);
    }
    return result;
}

// Intercept cudaFree
cudaError_t cudaFree(void *devPtr) {
    printf("~~~~~~~~ cudaFree ~~~~~~~~\n");
    real_cudaFree_t real_cudaFree = (real_cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    cudaError_t result = real_cudaFree(devPtr);
    if (result == cudaSuccess) {
        printf("cudaFree - GPU Address: %p\n", devPtr);
        remove_allocation(devPtr);
    } else {
    printf("Error freeing GPU Address %p: %s\n", devPtr, cudaGetErrorString(result));
    }
    
    return result;
}

// Copy data from gpu device into host memory
void copy_data_from_gpu(void *d_data, size_t size) {
    printf("~~~~~~~~ copy_data_from_gpu(GPU Address: %p, Size: %zu) ~~~~~~~~\n", d_data, size);
    // Allocate memory on the host
    void *h_data = malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(EXIT_FAILURE);
    }
    // Copy data from the device memory to host memory as a block of bytes
    cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Write data to disk
    // Create the filename using the address and size.
    char filename[256]; // Make sure this buffer is large enough.
    unsigned long long address_as_ull = (unsigned long long)(uintptr_t)d_data;
    snprintf(filename, sizeof(filename), "0x%llx_%zu.bin", address_as_ull, size);
    
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file for writing!\n");
        cudaFree(d_data);
        free(h_data);
        exit(EXIT_FAILURE);
    }
    size_t written = fwrite(h_data, 1, size, file);
    if (written != size) {
        fprintf(stderr, "Failed to write all data to file!\n");
        fclose(file);
        cudaFree(d_data);
        free(h_data);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    cudaFree(d_data);
    free(h_data);

    printf("Data from GPU was successfully saved to '%s'.\n", filename);
}


// Function to free all recorded allocations
void free_all_allocations() {
    printf("~~~~~~~~ free_all_allocations() ~~~~~~~~\n");
    //real_cudaFree_t real_cudaFree = (real_cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    for (int i = 0; i < MAX_ALLOCATIONS; ++i) {
        if (allocationMap[i].address != NULL) {
	    printf("~~~~ %d : GPU Address: %p, Size: %zu\n", i, allocationMap[i].address, allocationMap[i].size);
            copy_data_from_gpu(allocationMap[i].address, allocationMap[i].size);
	    //cudaFree(allocationMap[i].address);
            allocationMap[i].address = NULL;
            allocationMap[i].size = 0;
        }
    }
}

// Your self-defined function
void signal_sending() {
    // Your implementation goes here
    printf("Signal sending function called.\n");
    free_all_allocations();
}

/*
// Destructor to be called when the shared library is unloaded
__attribute__((destructor)) void cleanup() {
    free_all_allocations();
}
*/


