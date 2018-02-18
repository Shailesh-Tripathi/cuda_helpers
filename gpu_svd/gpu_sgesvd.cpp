/*
 * Author: Shailesh Tripathi
 * Desciption:  Singular Value Decomposition
 *  	[A] = U S V'
 * The gpu_sgesvd is a wrapper function for cusolverDnSgesvd. 
 * The wrapper has a function signature exactly similar to standard sgesvd function so that the user can a simply plug-in the 
 * function without being concerned about the device variable. 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)
#include <iostream>

using namespace std;

void printMatrix(int m, int n, const float *A, int lda, const char* name) 
{
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			float Areg= A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row+1,col+1, Areg);
		}
	}
}

int gpu_sgesvd(const char* jobu,const char* jobvt, int* m, int* n, float* A,
                int* lda, float* S, float* U, int* ldu, float* VT, int* ldvt,
                float* work, int* lwork, int* info )
{
	printf("using gpu\n");
	cusolverDnHandle_t cusolverH = NULL;
	
	cublasHandle_t cublasH = NULL;

	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	cudaError_t cudaStat = cudaSuccess;  // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L
	
	float *d_A = NULL;
	float *d_S = NULL;
	float *d_U = NULL; 
	float *d_VT = NULL; 
	int *devInfo = NULL;
	float *d_work = NULL;
	float *d_rwork = NULL;
	
// step 1: create cusolverDn/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
	{
		fprintf(stderr, "!!!! cusolverDnCreate error\n");
		return EXIT_FAILURE;
	}
	
	cublas_status = cublasCreate(&cublasH);
	if( cublas_status != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "!!!! cublasCreate error\n");
		return EXIT_FAILURE;
	}
	
// step 2: copy A and B to device
	cudaMalloc((void**)&d_A , sizeof(float)*(*lda)*(*n));
	cudaMalloc((void**)&d_S , sizeof(float)*(*n));
	cudaMalloc((void**)&d_U , sizeof(float)*(*ldu)*(*m));
	cudaMalloc((void**)&d_VT , sizeof(float)*(*ldvt)*(*n));
	cudaMalloc((void**)&devInfo, sizeof(int));
	
	cudaMemcpy(d_A, A, sizeof(float)*(*lda)*(*n), cudaMemcpyHostToDevice);
	
// step 3: query working space of SVD 
	cusolver_status = cusolverDnSgesvd_bufferSize(
		cusolverH,
		*m, *n,
		lwork );
	
	if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
	{
		fprintf(stderr, "!!!! cusolverDnDgesvd_bufferSize error\n");
		return EXIT_FAILURE;
	}

	cudaMalloc((void**)&d_work , sizeof(float)*(*lwork));
	
// step 4: compute SVD 
	cusolver_status = cusolverDnSgesvd(
		cusolverH,
		jobu[0], jobvt[0],
		*m, *n,
		d_A, *lda,
		d_S,
		d_U,*ldu, 	// ldu
		d_VT, *ldvt, 	// ldvt,
		d_work,
		*lwork,
		d_rwork,
		devInfo);
	cudaStat = cudaDeviceSynchronize();   //is this required?
	if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
	{
		fprintf(stderr, "!!!! cusolverDnDgesvd error\n");
		return EXIT_FAILURE;
	}

	cudaStat = cudaMemcpy(U , d_U , sizeof(float)*(*ldu)*(*m), cudaMemcpyDeviceToHost);
	cudaStat = cudaMemcpy(VT , d_VT , sizeof(float)*(*ldvt)*(*n), cudaMemcpyDeviceToHost);
	cudaStat = cudaMemcpy(S , d_S , sizeof(float)*(*n), cudaMemcpyDeviceToHost);
	cudaStat = cudaMemcpy(info , devInfo , sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("after gesvd: info_gpu = %d\n", *info);
	printf("=====\n");

	printf("S = (matlab base-1)\n");
	printMatrix(*n, 1, S, *lda, "S");
	printf("=====\n");
	
	printf("U = (matlab base-1)\n");
	printMatrix(*m, *m, U, *ldu, "U");
	printf("=====\n");
	
	printf("VT = (matlab base-1)\n");
	printMatrix(*n, *n, VT, *ldvt, "VT");
	printf("=====\n");

	return EXIT_SUCCESS;

}


int main(int argc, char* argv[]) {

	int m = 3;
	int n = 2;
	int lda = m;

	/*		| 1 2 	| 
	 *          A = | 3 4 	| 
	 * 		| 5 4 	| 
	 * */
	 
	float A[]={1.0,2.0,3.0,4.0,5.0,4.0};//,7.0,8.0,8.0};
	float U[9]; // m-by-m unitary matrix
	float VT[9]; // n-by-n unitary matrix

	float S[3];//]; 	// singular value
	float S_exact[2] = {7.065283497082729, 1.040081297712078};
	float *work;	
	int lwork = 0;
	int info_gpu = 0;
	const float h_one = 1;
	const float h_minus_one = -1; 
		
	printf("A = (matlab base-1)\n");
	printf("=====\n");
	printMatrix(m, n, A, m, "A");
	char jobu[]="All";
	char jobvt[]="All";
	gpu_sgesvd(
		jobu,
		jobvt,
		&m,
		&n,
		A,
		&lda,
		S,
		U,
		&lda, 	// ldu
		VT,
		&n, 	// ldvt,
		work,
		&lwork,
		&info_gpu);

	cudaDeviceReset();
	return 0;
}
