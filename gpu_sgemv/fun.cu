/* Author: Shailesh Tripathi
   Descrition: 
   Wrapper for matrix-vector multiplication using cuBlas library
   This function can be used just like the mkl library with the same function signature. No need to take explicit care of CUDA variables.*/

#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Matrix size */
#define N (4)
#define K (3)
#define M (2)

int gpu_sgemv(char transa,
		int* m, int* n,
		float* alpha,
		float* A, int* lda,
		float* X, int* incx,
		float* beta,
		float* Y, int* incy)
{
    float* d_A = NULL;
    float* d_X = NULL;
    float* d_Y = NULL;
    int kY, kX;
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasOperation_t cu_transa;
    
    kX = ( transa == 'N' ) ? *n : *m ;
    kY = ( transa == 'N' ) ? *m : *n ;
    
    cu_transa = ( transa == 'N' ) ? CUBLAS_OP_N : CUBLAS_OP_T ;
      
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, (*m) * (*n) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X, kX * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_Y, kY * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector( (*m) *(*n), sizeof(A[0]), A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector( kX, sizeof(X[0]), X, 1, d_X, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector( kY, sizeof(Y[0]), Y, 1, d_Y, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using cublas */
    status = cublasSgemv(handle, cu_transa, *m, *n, alpha, d_A, *lda, d_X, *incx, beta, d_Y, *incy);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

        /* Read the result back */
    status = cublasGetVector(kY, sizeof(Y[0]), d_Y, 1, Y, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
      
    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_X) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_Y) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;

}
