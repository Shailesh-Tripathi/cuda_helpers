#ifndef FUN_H
#define FUN_H

int gpu_sgemm(char transa, char transb,
                                int m, int n, int k,
                                float alpha,
                                float* A, int lda,
                                float* B, int ldb,
                                float beta,
                                float* C, int ldc);

#endif  //FUN_H
