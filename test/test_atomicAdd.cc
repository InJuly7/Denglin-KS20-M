#include <stdio.h>
#include <cuda_runtime.h>
// dlcc -x cuda --offload-arch=dlgput64,dlgpux64 ./test_atomicAdd.cc
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void atomicAddKernel(int* result) {
    atomicAdd(result, 1);
}

int main() {

    const int NUM_THREADS = 128;
    const int NUM_BLOCKS = 128;
    int *d_result, h_result = 0;
    CHECK(cudaMalloc((void**)&d_result, sizeof(int)));
    CHECK(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));
    atomicAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_result);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Final count: %d\n", h_result);
    CHECK(cudaFree(d_result));

    return 0;
}

