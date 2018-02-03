/* Author : Shailesh Tripathi
 * Sample code for invoking the gpu_sgemm wrapper function
 *  command to compile : nvcc -o res main.cpp fun.cu -lcudart -lcublas
*/

#include <stdio.h>
#include <stdlib.h>
#include "fun.h"

/* Matrix size */
#define N (4)
#define K (3)
#define M (2)

/* Main */
int main(int argc, char **argv)
{
    float *h_A;
    float *h_B;
    float *h_C;
    float alpha = 1.0f;
    float beta = 0.0f;
    int i,j;
    float error_norm;
    float ref_norm;
    float diff;

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(M*K * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(K*N * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(M*N * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < M*N; i++)
    {
        h_C[i] =i ;
    }

    for (i = 0; i < M*K; i++)
    {
        h_A[i] = i;
    }

   for (i = 0; i < K*N; i++)
    {
        h_B[i] = i;
    }

    //invoke wrapper function for mm
    gpu_sgemm('N', 'N', M, N, K, alpha, h_A, M, h_B, K, beta, h_C, M);    
    
	printf("A: \n"); 
	for( i = 0 ; i < M ; i++ )
	{
		for( j = 0 ; j < K ; j++ )
			printf("%f ", h_A[i + M*j]);
		printf("\n");
	}

	
	printf("\nB: \n"); 
	for( i = 0 ; i < K ; i++ )
	{
		for( j = 0 ; j < N ; j++ )
			printf("%f ", h_B[i + K*j]);
		printf("\n");
	}

	printf("\nC: \n"); 
	for( i = 0 ; i < M ; i++ )
	{
		for( j = 0 ; j < N ; j++ )
			printf("%f ", h_C[i + M*j]);
		printf("\n");
	}

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;


    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
}
