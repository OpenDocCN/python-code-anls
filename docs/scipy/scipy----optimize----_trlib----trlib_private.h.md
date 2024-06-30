# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_private.h`

```
/*
 * MIT License
 *
 * Copyright (c) 2016--2017 Felix Lenders
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef TRLIB_PRIVATE_H
#define TRLIB_PRIVATE_H

/* #undef TRLIB_MEASURE_TIME */
/* #undef TRLIB_MEASURE_SUBTIME */

#include "numpy/arrayobject.h"
#include "npy_cblas.h"

#include "trlib.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// blas
// 定义了一系列基本的 BLAS 线性代数函数接口
void BLAS_FUNC(daxpy)(CBLAS_INT *n, double *alpha, double *x, CBLAS_INT *incx, double *y, CBLAS_INT *incy);
void BLAS_FUNC(dscal)(CBLAS_INT *n, double *alpha, double *x, CBLAS_INT *incx);
void BLAS_FUNC(dcopy)(CBLAS_INT *n, double *x, CBLAS_INT *incx, double *y, CBLAS_INT *incy);
double BLAS_FUNC(dnrm2)(CBLAS_INT *n, double *x, CBLAS_INT *incx);
double BLAS_FUNC(ddot)(CBLAS_INT *n, double *x, CBLAS_INT *incx, double *y, CBLAS_INT *incy);

// lapack
// 定义了一系列 LAPACK 线性代数函数接口
void BLAS_FUNC(dpttrf)(CBLAS_INT *n, double *d, double *e, CBLAS_INT *info);
void BLAS_FUNC(dpttrs)(CBLAS_INT *n, CBLAS_INT *nrhs, double *d, double *e, double *b, CBLAS_INT *ldb, CBLAS_INT *info);
void BLAS_FUNC(dptrfs)(CBLAS_INT *n, CBLAS_INT *nrhs, double *d, double *e, double *df, double *ef, double *b, CBLAS_INT *ldb, double *x, CBLAS_INT *ldx, double *ferr, double *berr, double *work, CBLAS_INT *info);
void BLAS_FUNC(dlagtm)(char *trans, CBLAS_INT *n, CBLAS_INT *nrhs, double *alpha, double *dl, double *d, double *du, double *x, CBLAS_INT *ldx, double *beta, double *b, CBLAS_INT *ldb, int);


static void trlib_daxpy(trlib_int_t *n, double *alpha, double *x, trlib_int_t *incx, double *y, trlib_int_t *incy)
{
    CBLAS_INT n_ = *n, incx_ = *incx, incy_ = *incy;
    BLAS_FUNC(daxpy)(&n_, alpha, x, &incx_, y, &incy_);
}

static void trlib_dscal(trlib_int_t *n, double *alpha, double *x, trlib_int_t *incx)
{
    CBLAS_INT n_ = *n, incx_ = *incx;
    BLAS_FUNC(dscal)(&n_, alpha, x, &incx_);
}

static void trlib_dcopy(trlib_int_t *n, double *x, trlib_int_t *incx, double *y, trlib_int_t *incy)
{
    CBLAS_INT n_ = *n, incx_ = *incx, incy_ = *incy;
    BLAS_FUNC(dcopy)(&n_, x, &incx_, y, &incy_);
}
{
    CBLAS_INT n_ = *n, incx_ = *incx, incy_ = *incy;
    BLAS_FUNC(dcopy)(&n_, x, &incx_, y, &incy_);
}

# 调用 BLAS 库中的 dcopy 函数，用于向量的拷贝操作
static double trlib_dnrm2(trlib_int_t *n, double *x, trlib_int_t *incx)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, incx_ = *incx;
    # 调用 BLAS 库中的 dnrm2 函数，计算向量的二范数
    return BLAS_FUNC(dnrm2)(&n_, x, &incx_);
}

# 调用 BLAS 库中的 ddot 函数，用于向量的内积计算
static double trlib_ddot(trlib_int_t *n, double *x, trlib_int_t *incx, double *y, trlib_int_t *incy)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, incx_ = *incx, incy_ = *incy;
    # 调用 BLAS 库中的 ddot 函数，计算两个向量的内积
    return BLAS_FUNC(ddot)(&n_, x, &incx_, y, &incy_);
}

# 调用 BLAS 库中的 dpttrf 函数，用于实对称正定三对角矩阵的分解
static void trlib_dpttrf(trlib_int_t *n, double *d, double *e, trlib_int_t *info)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, info_ = *info;
    # 调用 BLAS 库中的 dpttrf 函数，对三对角矩阵进行分解
    BLAS_FUNC(dpttrf)(&n_, d, e, &info_);
    # 将结果返回给原始指针
    *info = (trlib_int_t)info_;
}

# 调用 BLAS 库中的 dpttrs 函数，用于解实对称正定三对角矩阵的线性方程组
static void trlib_dpttrs(trlib_int_t *n, trlib_int_t *nrhs, double *d, double *e, double *b, trlib_int_t *ldb, trlib_int_t *info)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, nrhs_ = *nrhs, ldb_ = *ldb, info_ = *info;
    # 调用 BLAS 库中的 dpttrs 函数，解实对称正定三对角矩阵的线性方程组
    BLAS_FUNC(dpttrs)(&n_, &nrhs_, d, e, b, &ldb_, &info_);
    # 将结果返回给原始指针
    *info = (trlib_int_t)info_;
}

# 调用 BLAS 库中的 dptrfs 函数，用于求解实对称正定三对角矩阵线性方程组的扰动分析
static void trlib_dptrfs(trlib_int_t *n, trlib_int_t *nrhs, double *d, double *e, double *df, double *ef, double *b, trlib_int_t *ldb,
                         double *x, trlib_int_t *ldx, double *ferr, double *berr, double *work, trlib_int_t *info)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, nrhs_ = *nrhs, ldb_ = *ldb, ldx_ = *ldx, info_ = *info;
    # 调用 BLAS 库中的 dptrfs 函数，进行实对称正定三对角矩阵线性方程组的扰动分析
    BLAS_FUNC(dptrfs)(&n_, &nrhs_, d, e, df, ef, b, &ldb_, x, &ldx_, ferr, berr, work, &info_);
    # 将结果返回给原始指针
    *info = (trlib_int_t)info_;
}

# 调用 BLAS 库中的 dlagtm 函数，用于实对称矩阵与多个向量的乘法
static void trlib_dlagtm(char *trans, trlib_int_t *n, trlib_int_t *nrhs, double *alpha, double *dl, double *d, double *du, double *x,
                         trlib_int_t *ldx, double *beta, double *b, trlib_int_t *ldb)
{
    # 将传入的指针解引用，并赋值给本地变量
    CBLAS_INT n_ = *n, nrhs_ = *nrhs, ldb_ = *ldb, ldx_ = *ldx;
    # 调用 BLAS 库中的 dlagtm 函数，进行实对称矩阵与多个向量的乘法
    BLAS_FUNC(dlagtm)(trans, &n_, &nrhs_, alpha, dl, d, du, x, &ldx_, beta, b, &ldb_, 1);
}

# 如果定义了 TRLIB_MEASURE_TIME 宏，则使用基于时钟的计时宏和返回宏
# 否则定义为空操作
#if TRLIB_MEASURE_TIME
    #define TRLIB_TIC(X) { clock_gettime(CLOCK_MONOTONIC, &X); }
    #define TRLIB_DURATION(X, Y, Z) { clock_gettime(CLOCK_MONOTONIC, &Y); Z += 1000000000L*(Y.tv_sec-X.tv_sec)+Y.tv_nsec-X.tv_nsec; }
    #define TRLIB_SIZE_TIMING_LINALG (9)
    #if TRLIB_MEASURE_SUBTIME
        #define TRLIB_DURATION_SUB(X, Y, Z) { clock_gettime(CLOCK_MONOTONIC, &Y); Z += 1000000000L*(Y.tv_sec-X.tv_sec)+Y.tv_nsec-X.tv_nsec; }
    #else
        #define TRLIB_DURATION_SUB(X, Y, Z)
    #endif
#else
    #define TRLIB_TIC(X)
    #define TRLIB_DURATION(X, Y, Z)
    #define TRLIB_DURATION_SUB(X, Y, Z)
#endif

# 定义宏 TRLIB_RETURN(X)，用于返回函数执行结果并计时
#define TRLIB_RETURN(X) { TRLIB_DURATION(verystart, end, timing[0]) return X; }

# 定义宏 TRLIB_DCOPY(...)，用于调用 trlib_dcopy 函数，并计时
#define TRLIB_DCOPY(...) { TRLIB_TIC(start) trlib_dcopy(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[1]) }
# 定义宏 TRLIB_DAXPY(...)，用于调用 trlib_daxpy 函数，并计时
#define TRLIB_DAXPY(...) { TRLIB_TIC(start) trlib_daxpy(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[2]) }
# 定义宏 TRLIB_DSCAL(...)，用于调用 trlib_dscal 函数，并计时
#define TRLIB_DSCAL(...) { TRLIB_TIC(start) trlib_dscal(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[3]) }
# 定义宏 TRLIB_DNRM2(A, X, Y, Z)，用于调用 trlib_dnrm2 函数，并计时
#define TRLIB_DNRM2(A, X, Y, Z) { TRLIB_TIC(start) A = trlib_dnrm2(X, Y, Z); TRLIB_DURATION_SUB(start, end, timing[4]) }
// 定义宏 TRLIB_DDOT，用于计算向量点积并记录执行时间
#define TRLIB_DDOT(A, N, X, IX, Y, IY) { TRLIB_TIC(start) A = trlib_ddot(N, X, IX, Y, IY); TRLIB_DURATION_SUB(start, end, timing[5]) }

// 定义宏 TRLIB_DPTTRF，用于执行平对角矩阵的 LU 分解并记录执行时间
#define TRLIB_DPTTRF(...) { TRLIB_TIC(start) trlib_dpttrf(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[6]) }

// 定义宏 TRLIB_DPTTRS，用于解平对角矩阵的线性方程组并记录执行时间
#define TRLIB_DPTTRS(...) { TRLIB_TIC(start) trlib_dpttrs(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[7]) }

// 定义宏 TRLIB_DPTRFS，用于求解带对角线线性方程组并记录执行时间
#define TRLIB_DPTRFS(...) { TRLIB_TIC(start) trlib_dptrfs(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[8]) }

// 定义宏 TRLIB_DLAGTM，用于执行矩阵-向量乘法并记录执行时间
#define TRLIB_DLAGTM(...) { TRLIB_TIC(start) trlib_dlagtm(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[9]) }

// 定义宏 TRLIB_PRINTLN_1，根据 verbose 级别打印带有前缀的信息到文件或标准输出
#define TRLIB_PRINTLN_1(...) if (verbose > 0) { if (fout) { fprintf(fout, "%s", prefix); fprintf(fout, __VA_ARGS__); fprintf(fout, "\n"); } else { printf("%s", prefix); printf(__VA_ARGS__); printf("\n"); } }

// 定义宏 TRLIB_PRINTLN_2，根据 verbose 级别打印更详细的带有前缀的信息到文件或标准输出
#define TRLIB_PRINTLN_2(...) if (verbose > 1) { if (fout) { fprintf(fout, "%s", prefix); fprintf(fout, __VA_ARGS__); fprintf(fout, "\n"); } else { printf("%s", prefix); printf(__VA_ARGS__); printf("\n"); } }

// 定义宏 TRLIB_PRINT_VEC，用于打印向量的内容和索引
#define TRLIB_PRINT_VEC(P, N, X) { for(int vc = 0; vc < N; ++vc) { printf("%s %ld: %e\n", P, vc, *(X+vc)); } }

#endif


这些宏定义包括了对各种数值计算和输出的封装，以及根据条件进行输出的功能。
```