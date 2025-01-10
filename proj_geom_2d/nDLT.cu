#include <cstdlib>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "../UTIL/cpu_util.h"




__global__ void norm(const float* points, float* normed_points, float* T, const int n) {
    /*
     * Inputs: points        -> (1, 2 * n) original points in coordinates. n is the number of points.
               normed_points -> (1, 2 * n) scale * x - scale * centroid_x.
               T             -> (1, 3 * 3) Transformation Matrix.
               n             -> number of correspondence points. Equals to 4 by default.
    */

    if (n < 4) {
        if (threadIdx.x == 0) printf("Need at least 4 point correspondences!\n");
        return;
    }

    __shared__ float centroid[2]; // Shared memory for centroid
    __shared__ float avg_dist;    // Shared memory for average distance
    __shared__ float partial_sum[256]; // Temporary storage for reduction (assuming blockDim.x <= 256)

    // Initialize shared memory
    if (threadIdx.x == 0) {
        centroid[0] = 0.0f;
        centroid[1] = 0.0f;
        avg_dist = 0.0f;
    }
    __syncthreads();

    // Compute partial sums for centroid
    float local_sum_x = 0.0f, local_sum_y = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum_x += points[2 * i];
        local_sum_y += points[2 * i + 1];
    }

    // Reduce local sums to compute global centroid
    partial_sum[threadIdx.x] = local_sum_x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) centroid[0] = partial_sum[0] / n;

    partial_sum[threadIdx.x] = local_sum_y;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) centroid[1] = partial_sum[0] / n;
    __syncthreads();

    // Compute distances to centroid
    float local_dist = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dx = points[2 * i] - centroid[0];
        float dy = points[2 * i + 1] - centroid[1];
        local_dist += sqrtf(dx * dx + dy * dy);
    }

    // Reduce distances to compute average distance
    partial_sum[threadIdx.x] = local_dist;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) avg_dist = partial_sum[0] / n;
    __syncthreads();

    // Compute scale factor
    float scale = sqrtf(2.0f) / avg_dist;

    // Populate the transformation matrix T
    if (threadIdx.x == 0) {
        T[0] = scale; T[1] = 0; T[2] = -scale * centroid[0];
        T[3] = 0; T[4] = scale; T[5] = -scale * centroid[1];
        T[6] = 0; T[7] = 0; T[8] = 1;
    }
    __syncthreads();

    // Normalize the points
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        normed_points[2 * i] = scale * (points[2 * i] - centroid[0]);
        normed_points[2 * i + 1] = scale * (points[2 * i + 1] - centroid[1]);
    }
}


__global__ void norm_1(const float* points, float* normed_points, float* T, const int n) {
    if (n < 4) printf("Need at least 4 point correspondence! \n");
    
    /* initialize shared memory */
    __shared__ float centroid[2];
    __shared__ float avg_dist;

    if (threadIdx.x == 0)
    {
        centroid[0] = 0.0f;
        centroid[1] = 0.0f;
        avg_dist = 0.0f;
    }
    __syncthreads();
    /* initialize shared memory */

    float sum_x = 0, sum_y = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum_x += points[2 * i];     // 1d array. even positions x, odd positions y
        sum_y += points[2 * i + 1]; // 1d array. even positions x, odd positions y
    }
    atomicAdd(&centroid[0], sum_x);
    atomicAdd(&centroid[1], sum_y);
    __syncthreads();

    if (threadIdx.x == 0) {
        centroid[0] /= n; // Compute average for x
        centroid[1] /= n; // Compute average for y
    }
    __syncthreads();

    // compute distance
    float local_dist = 0.0;
    for (int i = threadIdx.x; i < n; i+= blockDim.x) {
        float dx = points[2 * i] - centroid[0];
        float dy = points[2 * i + 1] - centroid[1];
        local_dist += sqrtf(dx * dx + dy * dy);
    }
    atomicAdd(&avg_dist, local_dist);
    __syncthreads();

    // scale factor
    float scale = sqrtf(2.0f) / avg_dist;

    // populate T
    if (threadIdx.x == 0) {
        T[0] = scale; T[1] = 0; T[2] = -scale * centroid[0];
        T[3] = 0; T[4] = scale; T[5] = -scale * centroid[1];
        T[6] = 0; T[7] = 0; T[8] = 1;
    }
    __syncthreads();

    // normalize points
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        normed_points[2 * i] = scale * points[2 * i] - scale * centroid[0];
        normed_points[2 * i + 1] = scale * points[2 * i + 1] - scale * centroid[1];
    }
}


// Obtain A for SVD decomposition and Sigma hat (remove the minimal effect single value from Sigma)

__global__ void MatA(const float* points1, const float* points2, float* A, const int n) {
    /**
     * Constructs the matrix A for the DLT algorithm.
     * 
     * Inputs:
     *   @param points1 -> (1, 2 * n) array of points from the first image, flattened. 
     *              Each point is given as (x, y) and represented as consecutive elements in the array.
     *   @param points2 -> (1, 2 * n) array of points from the second image, flattened. 
     *              Each point is given as (x', y') and represented as consecutive elements in the array.
     *   @param n       -> Number of point correspondences. Must be at least 4.
     *   
     * Outputs:
     *   A       -> (2 * n, 9) matrix for solving Ah = 0 in the DLT algorithm, flattened.
     *              Each correspondence contributes two rows to A:                         (!!! 2 Rows Per Point Correspondence!!! )
     *              [ -x, -y, -1,  0,  0,  0, x * x', y * x', x' ]
     *              [  0,  0,  0, -x, -y, -1, x * y', y * y', y' ]
     * 
     * Construction:
     *   For each correspondence point (x, y) in the first image and (x', y') in the second image:
     *     Row 2 * i     = [ -x, -y, -1,  0,  0,  0, x * x', y * x', x' ]
     *     Row 2 * i + 1 = [  0,  0,  0, -x, -y, -1, x * y', y * y', y' ]
     *
     * Notes:
     *   - This kernel assumes that A is preallocated with sufficient memory for (2 * n * 9) elements.
     *   - Threads are responsible for processing individual rows of A, based on their index.
     *   - Ensure numerical stability by normalizing points before calling this kernel.
     */
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float x = points1[2 * idx];
        float y = points1[2 * idx + 1];
        float xp = points2[2 * idx];
        float yp = points2[2 * idx + 1];

        int row1 = 2 * idx;       // First row index for correspondence
        int row2 = 2 * idx + 1;   // Second row index for correspondence

        A[9 * row1 + 0] = -x;
        A[9 * row1 + 1] = -y;
        A[9 * row1 + 2] = -1;
        A[9 * row1 + 3] = 0;
        A[9 * row1 + 4] = 0;
        A[9 * row1 + 5] = 0;
        A[9 * row1 + 6] = xp * x;
        A[9 * row1 + 7] = xp * y;
        A[9 * row1 + 8] = xp;

        A[9 * row2 + 0] = 0;
        A[9 * row2 + 1] = 0;
        A[9 * row2 + 2] = 0;
        A[9 * row2 + 3] = -x;
        A[9 * row2 + 4] = -y;
        A[9 * row2 + 5] = -1;
        A[9 * row2 + 6] = yp * x;
        A[9 * row2 + 7] = yp * y;
        A[9 * row2 + 8] = yp;

    }
}




void matrix_inverse(cusolverDnHandle_t handle, float* d_A, float* d_Ainv, int n) {
    int* dev_info;
    CHECK(cudaMalloc((void**) &dev_info, sizeof(int)));

    // workspace query
    int lwork;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &lwork));
    float* d_work;
    CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));

    // pivot array for LU decomposition
    int* d_pivot;
    CHECK(cudaMalloc((void**)&d_pivot, n * sizeof(int)));

    // LU Decomposition (in-place on d_A)
    CHECK_CUSOLVER(cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_pivot, dev_info));

    // allocate identity matrix on device
    float* d_I;
    CHECK(cudaMalloc((void**)&d_I, n * n * sizeof(float)));
    std::vector<float> h_I(n * n, 0);
    for (int i = 0; i < n; ++i) h_I[i * n + i] = 1.0f; // Fill as identity matrix
    CHECK(cudaMemcpy(d_I, h_I.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Solve AX = I (result is stored in d_Ainv)
    CHECK_CUSOLVER(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n, d_A, n, d_pivot, d_I, n, dev_info));

    // Copy result to d_Ainv
    CHECK(cudaMemcpy(d_Ainv, d_I, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

    // Free temporary memory
    CHECK(cudaFree(d_work));
    CHECK(cudaFree(d_pivot));
    CHECK(cudaFree(d_I));
    CHECK(cudaFree(dev_info));
}




float* parse_points(int argc, char const * argv[], int offset, int n) {
    /**
      * Parses a set of 2D points from the command-line arguments.
      * 
      * @param argc       Total number of command-line arguments.
      * @param argv       Array of command-line arguments as strings.
      * @param offset     Index in argv where the points data starts.
      * @param num_points Number of points to parse.
      * 
      * @return           A vector of floats containing the parsed points in the 
      *                   form [x1, y1, x2, y2, ..., xn, yn].
      * 
      * Notes:
      * - Each point consists of two floating-point numbers (x, y).
      * - Points are provided sequentially as x1, y1, x2, y2, ..., xn, yn in argv.
      * - The function assumes that there are enough arguments in argv to parse all points.
      * - Points are returned in a flattened format: a single vector containing all
      *   x and y values consecutively.
      */
    float* points = (float*) malloc(2 * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        points[2 * i] = atof(argv[offset + 2 * i]);
        points[2 * i + 1] = atof(argv[offset + 2 * i + 1]);
    }
    return points;
}


template <typename T>
void transposeToColumnMajor(const T* h_A, T* h_A_col, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_A_col[j * rows + i] = h_A[i * cols + j];
            float val = h_A_col[j * rows + i];
            if (!std::isfinite(val)) {
                std::cerr << "Non-finite value at (" << i << ", " << j << "): " << val << std::endl;
                return;
            }
        }
    }
}





int main(int argc, char const *argv[]) {
    // device setup
    int dev = 0;
    cudaDeviceProp device_prop;
    CHECK(cudaGetDeviceProperties(&device_prop, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, device_prop.name);
    CHECK(cudaSetDevice(dev));



    // threads blocks configuration
    int n = atoi(argv[1]); // expect to be smaller than 128
    std::cout << "Number of points: n = " << n << std::endl;
    int maxThreadsPerBlock = 256; // A reasonable value for most GPUs
    int threads = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocks = (n + threads - 1) / threads; // Round up to cover all points

    dim3 block(threads);
    dim3 grids(blocks);





    // host parameter setup
    int a_row = 2 * n, a_col = 9, T_size = 9;
    float* h_A = new float[a_row * a_col], * h_T = new float[T_size], * h_T_prime = new float[T_size];
    float* h_points, * h_points_ref;

    // device param
    float * d_points, *d_points_norm, * d_points_ref, * d_points_ref_norm;
    float * d_A, * d_T, * d_T_prime; // prime is for ref points transformation

    // to cuda malloc
    h_points = parse_points(argc, argv, 2, n);
    h_points_ref = parse_points(argc, argv, 2 + 2 * n, n);




    // CUDA memory allocation

    CHECK(cudaMalloc((void**) &d_A, sizeof(float) * a_row * a_col));
    /** initialize output array with 0s **/
    cudaMemset(d_A, 0.0, sizeof(float) * a_row * a_col);

    CHECK(cudaMalloc((void**) &d_points, sizeof(float) * 2 * n));
    CHECK(cudaMalloc((void**) &d_points_norm, sizeof(float) * 2 * n));
    /** initialize output array with 0s **/
    cudaMemset(d_points_norm, 0.0, sizeof(float) * 2 * n);
    CHECK(cudaMalloc((void**) &d_points_ref, sizeof(float) * 2 * n));
    CHECK(cudaMalloc((void**) &d_points_ref_norm, sizeof(float) * 2 * n));
    /** initialize output array with 0s **/
    cudaMemset(d_points_ref_norm, 0.0, sizeof(float) * 2 * n);
    CHECK(cudaMalloc((void **) &d_T, T_size * sizeof(float)));
    /** initialize output array with 0s **/
    cudaMemset(d_T, 0.0, sizeof(float) * T_size);
    CHECK(cudaMalloc((void**) &d_T_prime, sizeof(float) * T_size));
    /** initialize output array with 0s **/
    cudaMemset(d_T_prime, 0.0, sizeof(float) * T_size);
    /* Assertion */
    assert(h_points != nullptr && h_points_ref != nullptr);
    
    // Memory transfer
    CHECK(cudaMemcpy(d_points, h_points, sizeof(float) * 2 * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_points_ref, h_points_ref, sizeof(float) * 2 * n, cudaMemcpyHostToDevice));

    // only once after alloc and transfer
    CHECK(cudaDeviceSynchronize());




    // event create
    cudaEvent_t i_start, i_elaps;
    float elapstime;

    CHECK(cudaEventCreate(&i_start));
    CHECK(cudaEventCreate(&i_elaps));

    // normalize (x, y)
    CHECK(cudaEventRecord(i_start, 0));
    norm<<< block, grids>>>(d_points, d_points_norm, d_T, n);
    CHECK(cudaEventRecord(i_elaps, 0));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CHECK(cudaMemcpy(h_T, d_T, sizeof(float) * T_size, cudaMemcpyDeviceToHost));

    // normalize (x', y')
    CHECK(cudaEventRecord(i_start, 0));
    norm<<< block, grids>>>(d_points_ref, d_points_ref_norm, d_T_prime, n);
    CHECK(cudaEventRecord(i_elaps, 0));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CHECK(cudaMemcpy(h_T_prime, d_T_prime, sizeof(float) * T_size, cudaMemcpyDeviceToHost));

    std::cout << "Transformation Matrix T:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_T[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Transformation Matrix T_prime:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_T_prime[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }



    // compute A for SVD decomposition
    MatA<<< block, grids>>>(d_points_norm, d_points_ref_norm, d_A, n);
    CHECK(cudaMemcpy(h_A, d_A, 2 * n * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    std::cout << "Matrix A on host (h_A):" << std::endl;
    for (int i = 0; i < 2 * n; ++i) {
        for (int j = 0; j < 9; ++j) {
            std::cout << h_A[i * 9 + j] << " ";
        }
        std::cout << std::endl;
    }





    
    /*      SVD     */



    float* h_A_col = new float[a_col * a_row];
    transposeToColumnMajor(h_A, h_A_col, a_row, a_col);

    float *d_U, *d_D, *d_VT, *d_A_col;
    int D_size = std::min(2 * n, 9);
    CHECK(cudaMalloc((void**) &d_U, sizeof(float) * (2 * n) * (2 * n)));   // U: (2n x 2n)
    CHECK(cudaMalloc((void**) &d_D, sizeof(float) * D_size));              // D: (min(2n = D_size, 9))
    CHECK(cudaMalloc((void**) &d_VT, sizeof(float) * 9 * 9));              // VT: (9 x 9)
    CHECK(cudaMalloc((void**) &d_A_col, sizeof(float) * a_col * a_row));

    // transfer h_A_col to d_A_col
    CHECK(cudaMemcpy(d_A_col, h_A_col, sizeof(float) * a_col * a_row, cudaMemcpyHostToDevice));

    int lwork = 0;
    float *d_work = nullptr, *d_rwork = nullptr;
    int *devInfo;

    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, 2 * n, 9, &lwork));
    CHECK(cudaMalloc((void**) &d_work, sizeof(float) * lwork));
    CHECK(cudaMalloc((void**) &devInfo, sizeof(int)));


    // svd debug
    std::cout << "Calling SVD with parameters:" << std::endl;
    std::cout << "m = " << 2*n << std::endl;
    std::cout << "n = " << 9 << std::endl;
    std::cout << "jobu = A" << std::endl;
    std::cout << "jobvt = A" << std::endl;
    std::cout << "ldA = " << (2*n) << std::endl;
    std::cout << "ldU = " << (2*n) << std::endl; // For S mode: U is 8x8 so ldU=8
    std::cout << "ldVT = " << 9 << std::endl;    // For S mode: VT is 8x9 so ldVT=8
    std::cout << "lwork = " << lwork << std::endl;



    CHECK_CUSOLVER(cusolverDnSgesvd( /* Note: m >= n, if not, use transpose */
        cusolverH,
        'A',                    // Compute all singular vectors
        'A',                    // Compute all singular vectors
        9,                      // Number of columns of A
        2 * n,                  // Number of rows of A
        d_A_col,                // Input matrix A
        a_col,                  // Leading dimension of A
        d_D,                    // Singular values (diagonal elements of D)
        d_U,                    // Left singular vectors (U)
        a_col,                  // Leading dimension of U
        d_VT,                   // Right singular vectors (VT)
        2 * n,                      // Leading dimension of VT
        d_work,                 // Workspace
        lwork,                  // Workspace size
        d_rwork,                // Not used for float
        devInfo                 // Info on success/failure
    ));
    CHECK(cudaDeviceSynchronize());


    std::cout << "Pointer checks:" << std::endl;
    std::cout << "d_A: " << d_A << std::endl;
    std::cout << "d_D: " << d_D << std::endl;
    std::cout << "d_U: " << d_U << std::endl;
    std::cout << "d_VT: " << d_VT << std::endl;
    std::cout << "d_work: " << d_work << std::endl;
    std::cout << "devInfo: " << devInfo << std::endl;




    int devInfo_h = 0;
    CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if (devInfo_h > 0) {
    std::cerr << "SVD failed to converge at superdiagonal " << devInfo_h << "." << std::endl;
    } else if (devInfo_h < 0) {
        std::cerr << "SVD encountered an illegal parameter at position " << -devInfo_h << "." << std::endl;
    } else {
        std::cout << "SVD succeeded!" << std::endl;
    }

    float *h_D = new float[D_size];
    CHECK(cudaMemcpy(h_D, d_D, sizeof(float) * D_size, cudaMemcpyDeviceToHost));
    std::cout << "Singular values (h_D):" << std::endl;
    for (int i = 0; i < D_size; ++i) {
        std::cout << h_D[i] << " ";
    }
    std::cout << std::endl;




    /** The LAST ROW of V^T (1, 9) corresponds to the singular vector associated with 
     *  the smallest singular value. This row is used to compute h, the solution to Ah = 0.
    */
    float* h_H_hat = (float*) malloc(9 * sizeof(float));
    float* d_H_hat = nullptr;
    CHECK(cudaMalloc((void**) &d_H_hat, 9 * sizeof(float)));
    // Assuming leading dimension (ld) is 9:
    // h_H_hat[i] = element of last row in the i-th column
    // last row: i = 8, column = j, element = d_VT[8 + j*ld]
    for (int j = 0; j < 9; ++j) {
        CHECK(cudaMemcpy(h_H_hat + j, d_VT + (8 + j*9), sizeof(float), cudaMemcpyDeviceToHost));
    }

    

    /*      SVD     */







    int devInfo_host = 0;
    CHECK(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "devInfo after SVD: " << devInfo_host << std::endl;








    /***        denormalization: H = T'^-1 H_hat T          ***/

    float *h_Tinv_prime = new float[9];
    // Device memory allocation
    float *d_Tinv_prime;
    CHECK(cudaMalloc((void**)&d_Tinv_prime, T_size * sizeof(float)));

    // Copy matrix to device
    CHECK(cudaMemcpy(d_T_prime, h_T_prime, T_size * sizeof(float), cudaMemcpyHostToDevice));

    // cuSOLVER handle
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    // Compute inverse
    matrix_inverse(handle, d_T_prime, d_Tinv_prime, std::sqrt(T_size));

    // Copy result back to host
    CHECK(cudaMemcpy(h_Tinv_prime, d_Tinv_prime, T_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Inverse Matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_Tinv_prime[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }


    // Multiplication T'^-1 H_hat T
    float *d_H_tmp, *d_H_final;
    CHECK(cudaMalloc((void**)&d_H_tmp, T_size * sizeof(float)));    // Temp storage for T'^{-1} H_hat
    CHECK(cudaMalloc((void**)&d_H_final, T_size * sizeof(float)));  // Final H matrix
    
    // Multiply T'^{-1} and H_hat
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS(cublasSgemm(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 3, 3,  // Matrix dimensions
        &alpha,
        d_Tinv_prime, 3,  // Left matrix
        d_H_hat, 3,       // Right matrix
        &beta,
        d_H_tmp, 3        // Output matrix
    ));
    
    // Multiply the result with T
    CHECK_CUBLAS(cublasSgemm(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 3, 3,
        &alpha,
        d_H_tmp, 3,
        d_T, 3,
        &beta,
        d_H_final, 3
    ));



    /***        denormalization: H = T'^-1 H_hat T          ***/








    // free memory
    CHECK(cudaFree(d_points));
    CHECK(cudaFree(d_points_ref));
    CHECK(cudaFree(d_T));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_Tinv_prime));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK(cudaFree(d_H_tmp));
    CHECK(cudaFree(d_H_hat));
    CHECK(cudaFree(d_H_final));

    
    free(h_points);
    free(h_points_ref);
    free(h_A);
    free(h_T);
    free(h_T_prime);
    free(h_Tinv_prime);
    return 0;
}
