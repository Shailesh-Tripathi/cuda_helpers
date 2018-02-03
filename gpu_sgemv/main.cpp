/* Author : Shailesh Tripathi
 * Sample code for invoking the gpu_sgemv wrapper function
 *  command to compile : nvcc -o res main.cpp fun.cu -lcudart -lcublas
*/

#include <stdio.h>
#include <stdlib.h>
#include "fun.h"


/* Main */
int main(int argc, char **argv)
{
    int M=4;
    int N=3, X_size=M;
    int Y_size=N;
    float *h_A;
    float *h_X;
    float *h_Y;
    float alpha = 1.0f;
    float beta = 0.0f;
    int i,j;
    float error_norm;
    float ref_norm;
    float diff;

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(M*N * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_X = (float *)malloc( X_size * sizeof(h_X[0]));

    if (h_X == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_Y = (float *)malloc(Y_size * sizeof(h_Y[0]));

    if (h_Y == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }
    
    /* Fill the matrices with test data */
    for (i = 0; i < M*N; i++)
    {
        h_A[i] =i ;
    }

    for (i = 0; i < X_size; i++)
    {
        h_X[i] = i;
    }

    for (i = 0; i < Y_size; i++)
    {
        h_Y[i] = i;
    }


    int incx=1,incy=1;
    /* Y = alpha * A' * X + beta * Y */
    gpu_sgemv('T', &M, &N, &alpha, h_A, &M, h_X, &incx, &beta, h_Y, &incy );    
     

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;
	
    
    printf("A:\n");
    for( i = 0 ; i < M ; i++ )
    {
	for( j = 0 ; j < N ; j++ )
	     printf("%f ",h_A[ i + j * M]);
	printf("\n");
    }    

    printf("\nX:\n");
    for( i = 0 ; i< X_size ; i++ )
    {
	printf("%f ",h_X[i]);
    }    

    printf("\nY:\n");
    for( i = 0 ; i < Y_size ; i++ )
    {
	printf("%f ",h_Y[i]);
    }    

    /* Memory clean up */
    free(h_A);
    free(h_X);
    free(h_Y);
}
