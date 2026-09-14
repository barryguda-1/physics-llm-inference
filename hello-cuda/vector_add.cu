// vector_add.cu — the "hello world" of CUDA, complete and runnable.
//
//   compile:  nvcc -o vector_add vector_add.cu
//   run:      ./vector_add
//
// The chapter shows the kernel and the launch; this file adds everything a
// beginner actually needs around them: error checks on every CUDA call,
// both the one-thread-per-element and grid-stride kernels, verification
// against the CPU result, and CUDA-event timing.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Bail loudly on any CUDA error. Errors are asynchronous by nature: they
// surface at the next sync, far from their cause — always check.
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err_));                     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Version 1: one thread per element. Each thread works out which element
// is its own, guards the tail (n need not be a multiple of the block size),
// and does one addition.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Version 2: grid-stride. stride = total threads in the grid; each thread
// walks the array with that step, so ANY grid size is correct and fewer
// blocks often schedule better for large n.
__global__ void vector_add_stride(const float* a, const float* b, float* c,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1'000'000;
    const int block_size = 256;                            // multiple of 32
    const int num_blocks = (n + block_size - 1) / block_size;  // ceil div = 3907
    const size_t bytes = n * sizeof(float);

    printf("n=%d  launch<<<%d, %d>>> = %d threads (%d guarded off in the tail)\n",
           n, num_blocks, block_size, num_blocks * block_size,
           num_blocks * block_size - n);

    // ---- host memory (ordinary RAM the GPU cannot see) ----
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)(i % 100) * 0.01f;
        h_b[i] = (float)((i % 7)) * -0.5f;
    }

    // ---- device memory (GPU HBM; pointers are opaque on the host) ----
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // ---- copies over PCIe: these dominate the wall time ----
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms_h2d = 0;
    CHECK(cudaEventElapsedTime(&ms_h2d, t0, t1));

    // ---- the kernel itself ----
    CHECK(cudaEventRecord(t0));
    vector_add<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
    // grid-stride alternative with the same result, try both:
    // vector_add_stride<<<8 * 132, block_size>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError());          // catch launch-config errors
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms_kernel = 0;
    CHECK(cudaEventElapsedTime(&ms_kernel, t0, t1));

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ---- verify: spot-check plus a full pass on the CPU ----
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            if (errors < 5) {
                printf("MISMATCH at %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            }
            errors++;
        }
    }
    printf("copy H2D: %8.3f ms\nkernel:    %8.3f ms  (12 MB touched on HBM)\n"
           "verify:    %s (%d mismatches)\n",
           ms_h2d, ms_kernel, errors ? "FAILED" : "OK", errors);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    free(h_a); free(h_b); free(h_c);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
