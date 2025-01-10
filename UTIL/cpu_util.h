#define CPU_UTIL
#ifdef  CPU_UTIL


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
{                                                                                 \
    const cudaError_t error = call;                                               \
    if (error != cudaSuccess)                                                     \
    {                                                                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                             \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                  \
    }                                                                             \
}


#define CHECK_CUSOLVER(call)                              \
    {                                                     \
        cusolverStatus_t status = (call);                 \
        if (status != CUSOLVER_STATUS_SUCCESS) {          \
            std::cerr << "cuSOLVER error at "             \
                      << __FILE__ << ":" << __LINE__      \
                      << " with status: " << status       \
                      << std::endl;                       \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    }


#define CHECK_CUBLAS(call)                                \
    {                                                     \
        cublasStatus_t status = (call);                   \
        if (status != CUBLAS_STATUS_SUCCESS) {          \
        std::cerr << "cuBLAS error at "                   \
                      << __FILE__ << ":" << __LINE__      \
                      << " with status: " << status       \
                      << std::endl;                       \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    }




__global__ void simpleKernel() {
    printf("Simple kernel executed on device.\n");
}



void check_result(long int *host_ref, long int *gpu_ref, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("CPU %ld \nGPU %ld \nat current %d\n", host_ref[i], gpu_ref[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}



// Recursive Implementation of Interleaved Pair Approach
long int cpusum(int *data, int const size)
{
    if (size == 0) return 0; // Handle empty array

    long int *temp = (long int*) malloc( sizeof(long int) * size);
    if (!temp) return 0; // Allocation failed

    // Initialize temp array with input data
    for (int i = 0; i < size; i++) {
        temp[i] = data[i];
    }

    int isize = size;
    while (isize > 1) {
        int const stride = isize / 2;
        int remainder = isize % 2;

        for (int i = 0; i < stride; i++) {
            temp[i] = temp[i] + temp[i + stride];
        }

        // If the array size is odd, add the last element to the first element
        if (remainder != 0) {
            temp[0] += temp[isize - 1];
        }

        isize = stride + remainder;
    }

    long int result = temp[0]; // Final result
    free(temp); // Free the allocated memory
    return result;
}



int cpuRecursiveReduce(int *data, int const size)
{
    // stop condition
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return cpuRecursiveReduce(data, stride);
}


#endif