/* Author: Shailesh Tripathi
   Descrition: 
   Wrapper for matrix multiplication using cuBlas library
   This function can be used just like the mkl library with the same function signature. No need to take explicit care of CUDA variables.*/

#include <stdio.h>
#include <stdlib.h>
#include "fun.h"

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

int gpu_sgemm(char transa, char transb,
		 		int m, int n, int k,
		 		float alpha,
		 		float* A, int lda,
		 		float* B, int ldb,
		 		float beta,
		 		float* C, int ldc)
{


	float* d_A = NULL;
	float* d_B = NULL;
	float* d_C = NULL;
	int kA, kB;
	cublasStatus_t status;
	cublasHandle_t handle;
	cublasOperation_t cu_transa, cu_transb;

	kA = ( transa == 'N' ) ? k : m ;
	kB = ( transb == 'N' ) ? n : k ;

	cu_transa = ( transa == 'N' ) ? CUBLAS_OP_N : CUBLAS_OP_T ;
	cu_transb = ( transb == 'N' ) ? CUBLAS_OP_N : CUBLAS_OP_T ;

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A,  lda * kA * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B, ldb * kB * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C, m * n * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(lda * kA , sizeof(A[0]), A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(ldb * kB, sizeof(B[0]), B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(m* n, sizeof(C[0]), C, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using cublas */
    status = cublasSgemm(handle, cu_transa, cu_transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

        /* Read the result back */
    status = cublasGetVector(m * n, sizeof(C[0]), d_C, 1, C, 1);

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

    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess)
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
