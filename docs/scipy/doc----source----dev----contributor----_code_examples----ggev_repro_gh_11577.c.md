# `D:\src\scipysrc\scipy\doc\source\dev\contributor\_code_examples\ggev_repro_gh_11577.c`

```
#include <stdio.h>
#include "lapacke.h"

#define n 4  // 定义矩阵维度为4

int main()
{
    int lda=n, ldb=n, ldvr=n, ldvl=n, info;  // 定义矩阵和向量的 leading dimensions 和 LAPACK 函数返回的信息变量
    char jobvl='V', jobvr='V';  // 指定 LAPACK 函数计算左特征向量和右特征向量

    double alphar[n], alphai[n], beta[n];  // 存储特征值和广义特征向量的实部、虚部和模

    double vl[n*n], vr[n*n];  // 存储计算得到的左特征向量和右特征向量

    double a[n*n] = {12.0, 28.0, 76.0, 220.0,
                     16.0, 32.0, 80.0, 224.0,
                     24.0, 40.0, 88.0, 232.0,
                     40.0, 56.0, 104.0, 248.0};  // 输入矩阵 A

    double b[n*n] = {2.0, 4.0, 10.0, 28.0,
                     3.0, 5.0, 11.0, 29.0,
                     5.0, 7.0, 13.0, 31.0,
                     9.0, 11.0, 17.0, 35.0};  // 输入矩阵 B

    info = LAPACKE_dggev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, a, lda, b,
                         ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr);  // 调用 LAPACK 函数 dggev 求解广义特征值问题

    printf("info = %d\n", info);  // 打印 LAPACK 函数的返回信息

    printf("Re(eigv) = ");
    for(int i=0; i < n; i++){
        printf("%f , ", alphar[i] / beta[i] );  // 打印实部特征值的计算结果
    }
    printf("\nIm(eigv = ");
    for(int i=0; i < n; i++){
        printf("%f , ", alphai[i] / beta[i] );  // 打印虚部特征值的计算结果
    }
    printf("\n");
}
```