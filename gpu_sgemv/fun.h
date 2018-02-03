#ifndef FUN_H
#define FUN_H

int gpu_sgemv(char transa,
                int* m, int* n,
                float* alpha,
                float* A, int* lda,
                float* X, int* incx,
                float* beta,
                float* Y, int* incy);
#endif  //FUN_H
