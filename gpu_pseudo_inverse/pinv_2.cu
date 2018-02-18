/*
    Author: Shailesh Tripathi
    Description: 
	The given function computes the pseudo-inverse of a rectangular matrix using Moore-Penrose Inverse formula. This function in turn uses the Singular Value Decomposition. 
	SVD of matrix A =     U * S * V'
	Pseudo inverse of A = V * inv(S) * U'
	Since S is just a diagonal matrix its inverse can simply be computed by taking inverse of each diagonal element
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

using namespace std;

static const char *_cudaGetErrorEnum(cusolverStatus_t error)
{
   switch(error)
   {
       case CUSOLVER_STATUS_SUCCESS:
           return "CUSOLVER_STATUS_SUCCESS";
       case CUSOLVER_STATUS_NOT_INITIALIZED:
           return "CUSOLVER_STATUS_NOT_INITIALIZED";
       case CUSOLVER_STATUS_ALLOC_FAILED:
           return "CUSOLVER_STATUS_ALLOC_FAILED";
       case CUSOLVER_STATUS_INVALID_VALUE:
           return "CUSOLVER_STATUS_INVALID_VALUE";
       case CUSOLVER_STATUS_ARCH_MISMATCH:
           return "CUSOLVER_STATUS_ARCH_MISMATCH";
       case CUSOLVER_STATUS_MAPPING_ERROR:
           return "CUSOLVER_STATUS_MAPPING_ERROR";
       case CUSOLVER_STATUS_EXECUTION_FAILED:
           return "CUSOLVER_STATUS_EXECUTION_FAILED";
       case CUSOLVER_STATUS_INTERNAL_ERROR:
           return "CUSOLVER_STATUS_INTERNAL_ERROR";
       case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
           return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
       case CUSOLVER_STATUS_NOT_SUPPORTED :
           return "CUSOLVER_STATUS_NOT_SUPPORTED ";
       case CUSOLVER_STATUS_ZERO_PIVOT:
           return "CUSOLVER_STATUS_ZERO_PIVOT";
       case CUSOLVER_STATUS_INVALID_LICENSE:
           return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";

}

//kernel to take transpose of matrix
__global__ void transpose(float *A, float *B, int m , int n)
{
	int t_id = blockIdx.x * blockDim.x + threadIdx.x;
	int i , j;
	i = t_id % m;
	j = t_id / m;
	if ( j < n )
		B[ i*n + j] = A[ j * m + i];

}


//kernel to compute V * Sinv; include the steps to inverse S, take VT transpose and multiply V* Sinv
__global__ void compute_V_into_Sinv ( int m , int n , int S_size , float *VT , float *S , float *V_into_Sinv) 
{
	int t_id = blockIdx.x * blockDim.x + threadIdx.x;
	int i , j;
	if(t_id < m*n)
	{
		for( ; t_id <m*n ; t_id+=(blockDim.x * gridDim.x))
		{
			i = t_id / n;
			j = t_id % n;
			if( i < S_size )
				if ( n == 1 )
					V_into_Sinv[ i * n + j] = 1  / S[i];
				else
					V_into_Sinv[ i * n + j] = VT[ j * n + i]  / S[i];
			else
				V_into_Sinv[ i * n + j] = 0;
		} 
	}

} //end kernel compute_V_into_Sinv 

void gpu_pseudo_inverse( float *A, int& m_original, int& n_original )
{

    // cuda setup variables
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t cu_transa, cu_transb;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;  // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L
	
    int m = max(m_original, n_original);
    int n = min(m_original, n_original);
 
    // device variables for SVD of A to get U, Vt and S
    float *d_A = NULL;
    float *d_S = NULL;
    float *d_U = NULL; 
    float *d_VT = NULL; 
    float * d_B = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    float *d_V_into_Sinv = NULL;

    int S_size = min(m, n);

    //host variables for U
    float *U = NULL;
    int *h_devInfo = NULL;

    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT


    int ldu = m, lwork = 0;



    float alpha = 1.0, beta = 0.0;
   
    cusolver_status = cusolverDnCreate(&cusolverH);
    if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
    {
            fprintf(stderr, "!!!! cusolverDnCreate error\n");
            return;
    }

    cublas_status = cublasCreate(&cublasH);
    if( cublas_status != CUBLAS_STATUS_SUCCESS )
    {
            fprintf(stderr, "!!!! cublasCreate error\n");
            return;
    }

    cudaMalloc((void**)&d_A , sizeof(float) * m * n);
    cudaMalloc((void**)&d_S , sizeof(float) * S_size);
    cudaMalloc((void**)&d_U , sizeof(float) * m * m);
    cudaMalloc((void**)&d_VT , sizeof(float) * n * n);
    cudaMalloc((void**)&devInfo, sizeof(int));
    h_devInfo = (int *)malloc(sizeof(int));
 
    if ( m_original < n_original )
    {
	cudaMalloc((void**)&d_B , sizeof(float) * m * n);
	cudaMemcpy(d_B, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	transpose<<<5,5>>> (d_B, d_A, n , m);		//make it more generic by using variable number of threads and blocks
	cudaStat = cudaDeviceSynchronize();   //is this required?
    }
    else
	cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    cusolver_status = cusolverDnSgesvd_bufferSize(
                cusolverH,
                m, n,
                &lwork );

    if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
    {
            fprintf(stderr, "!!!! cusolverDnDgesvd_bufferSize error\n");
            return;
    }
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);

    cusolver_status = cusolverDnSgesvd(
            cusolverH,
            jobu, jobvt,
            m, n,
            d_A, m,
            d_S,
            d_U, m,
            d_VT, n,
            d_work,
            lwork,
            d_rwork,
            devInfo);

    cudaStat = cudaDeviceSynchronize();   //is this required?

    if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
    {
        printf("error code : %d\n", cusolver_status);
        fprintf(stderr, "!!!! cusolverDnSgesvd error\n");
        return;
    }


    //check for solution of SVD
    cudaMemcpy( h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if( *h_devInfo < 0 ) 
	printf("Wrong parameter %d \n", *h_devInfo);
    else
	if(*h_devInfo > 0)
	{
      	  	printf( "The algorithm computing SVD failed to converge.\n" );
       		exit(1);
    	}
	
    U = (float*) malloc(ldu * m * sizeof(float));
    cudaMemcpy(U , d_U , sizeof(float)*ldu*m, cudaMemcpyDeviceToHost);
    cudaFree(d_U);

	
    cudaMalloc((void**)&d_V_into_Sinv , sizeof(float)* n * m);
	

    compute_V_into_Sinv<<< m,n >>> ( m , n, S_size , d_VT , d_S , d_V_into_Sinv );
    
    cudaStat = cudaDeviceSynchronize();  //wait for the kernel to complete


    float *V_into_Sinv = NULL;
    V_into_Sinv= (float*) malloc(sizeof(float) * n * m);



    //allocate d_U again, this was freed to reduce GPU memory
    cudaMalloc((void**)&d_U , sizeof(float) * ldu * m);
	
    cudaMemcpy(d_U , U , sizeof(float)*ldu*m, cudaMemcpyHostToDevice);

    cu_transa = CUBLAS_OP_N; 
    cu_transb = CUBLAS_OP_T;
    cublas_status = cublasSgemm(cublasH , cu_transa , cu_transb ,
				n , m , m ,
				&alpha , d_V_into_Sinv , n ,
				d_U , m ,
				&beta , d_A , n); 
	
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return;
    }

    if ( m_original < n_original )
    {
  	transpose<<<5,5>>> (d_A, d_B, n , m);
	cudaMemcpy(A , d_B , sizeof(float) * n * m, cudaMemcpyDeviceToHost);
    }
    else
 	cudaMemcpy(A , d_A , sizeof(float) * n * m, cudaMemcpyDeviceToHost);


   cudaFree(d_A);
   cudaFree(d_V_into_Sinv);

}  //gpu_pseudo_inverse

int main()
{
	float A[]={1.0,2.0,3.0,5.0};
	int m =2, n =2;
	int j;
	printf("Matrix A:\n");
	for(int i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			cout<<A[j*m + i] << ' ';

		cout<<endl;
	}
	
	gpu_pseudo_inverse(A,m,n);
	printf("Pseudo Inverse of matrix A:\n");
	for(int i=0;i<n;i++)
	{
		for(j=0;j<m;j++)
			cout<<A[j*n + i] << ' ';

		cout<<endl;
	}
}
