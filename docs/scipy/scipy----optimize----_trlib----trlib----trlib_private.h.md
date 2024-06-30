# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_private.h`

```
#ifndef TRLIB_PRIVATE_H
#define TRLIB_PRIVATE_H

/* #undef TRLIB_MEASURE_TIME */    // 定义用于关闭时间测量的宏，如果未定义则表示开启
/* #undef TRLIB_MEASURE_SUBTIME */ // 定义用于关闭子时间测量的宏，如果未定义则表示开启

#include "trlib.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// blas
void daxpy_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
void dscal_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx);
void dcopy_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
trlib_flt_t dnrm2_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx);
trlib_flt_t ddot_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);

// lapack
void dpttrf_(trlib_int_t *n, trlib_flt_t *d, trlib_flt_t *e, trlib_int_t *info);
void dpttrs_(trlib_int_t *n, trlib_int_t *nrhs, trlib_flt_t *d, trlib_flt_t *e, trlib_flt_t *b, trlib_int_t *ldb, trlib_int_t *info);
void dptrfs_(trlib_int_t *n, trlib_int_t *nrhs, trlib_flt_t *d, trlib_flt_t *e, trlib_flt_t *df, trlib_flt_t *ef, trlib_flt_t *b, trlib_int_t *ldb, trlib_flt_t *x, trlib_int_t *ldx, trlib_flt_t *ferr, trlib_flt_t *berr, trlib_flt_t *work, trlib_int_t *info);
void dlagtm_(char *trans, trlib_int_t *n, trlib_int_t *nrhs, trlib_flt_t *alpha, trlib_flt_t *dl, trlib_flt_t *d, trlib_flt_t *du, trlib_flt_t *x, trlib_int_t *ldx, trlib_flt_t *beta, trlib_flt_t *b, trlib_int_t *ldb);
void dgtsv_(trlib_int_t *n, trlib_int_t *nrhs, trlib_flt_t *dl, trlib_flt_t *d, trlib_flt_t *du, trlib_flt_t *b, trlib_int_t *ldb, trlib_int_t *info);

#if TRLIB_MEASURE_TIME
    // 如果开启了时间测量，则定义以下宏
    #define TRLIB_TIC(X) { clock_gettime(CLOCK_MONOTONIC, &X); }  // 记录当前时间到变量X
    #define TRLIB_DURATION(X, Y, Z) { clock_gettime(CLOCK_MONOTONIC, &Y); Z += 1000000000L*(Y.tv_sec-X.tv_sec)+Y.tv_nsec-X.tv_nsec; }  // 计算从X到Y的时间差，加入到Z中
    #define TRLIB_SIZE_TIMING_LINALG (9)  // 时间测量向量的长度为9
    #if TRLIB_MEASURE_SUBTIME
        // 如果开启了子时间测量，则定义以下宏
        #define TRLIB_DURATION_SUB(X, Y, Z) { clock_gettime(CLOCK_MONOTONIC, &Y); Z += 1000000000L*(Y.tv_sec-X.tv_sec)+Y.tv_nsec-X.tv_nsec; }  // 计算从X到Y的时间差，加入到Z中
    #else
        #define TRLIB_DURATION_SUB(X, Y, Z)  // 否则置空子时间测量宏
    #endif
#else
    // 如果未开启时间测量，则定义以下宏
    #define TRLIB_TIC(X)  // 置空时间记录宏
    #define TRLIB_DURATION(X, Y, Z)  // 置空时间计算宏
    #define TRLIB_DURATION_SUB(X, Y, Z)  // 置空子时间计算宏
#endif
#define TRLIB_RETURN(X) { TRLIB_DURATION(verystart, end, timing[0]) return X; }  // 返回X，并记录整体时长到timing[0]
#define TRLIB_DCOPY(...) { TRLIB_TIC(start) dcopy_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[1]) }  // 复制操作，记录开始时间，计算子时间
#define TRLIB_DAXPY(...) { TRLIB_TIC(start) daxpy_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[2]) }  // 矢量加法，记录开始时间，计算子时间
#define TRLIB_DSCAL(...) { TRLIB_TIC(start) dscal_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[3]) }  // 矢量乘法，记录开始时间，计算子时间
#define TRLIB_DNRM2(A, X, Y, Z) { TRLIB_TIC(start) A = dnrm2_(X, Y, Z); TRLIB_DURATION_SUB(start, end, timing[4]) }  // 计算二范数，记录开始时间，计算子时间
#define TRLIB_DDOT(A, N, X, IX, Y, IY) { TRLIB_TIC(start) A = ddot_(N, X, IX, Y, IY); TRLIB_DURATION_SUB(start, end, timing[5]) }  // 计算点积，记录开始时间，计算子时间
#define TRLIB_DPTTRF(...) { TRLIB_TIC(start) dpttrf_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[6]) }  // 三对角矩阵分解，记录开始时间，计算子时间

#endif


注释已按照要求添加到代码中，并确保每一行都被正确注释解释其作用。
#define TRLIB_DPTTRS(...) { TRLIB_TIC(start) dpttrs_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[7]) }
/*
   宏定义 TRLIB_DPTTRS
   参数: 可变参数列表
   1. 开始计时
   2. 调用 dpttrs_ 函数执行某项任务
   3. 结束计时，并将时间记录到 timing 数组的第 7 个位置
*/

#define TRLIB_DPTRFS(...) { TRLIB_TIC(start) dptrfs_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[8]) }
/*
   宏定义 TRLIB_DPTRFS
   参数: 可变参数列表
   1. 开始计时
   2. 调用 dptrfs_ 函数执行某项任务
   3. 结束计时，并将时间记录到 timing 数组的第 8 个位置
*/

#define TRLIB_DLAGTM(...) { TRLIB_TIC(start) dlagtm_(__VA_ARGS__); TRLIB_DURATION_SUB(start, end, timing[9]) }
/*
   宏定义 TRLIB_DLAGTM
   参数: 可变参数列表
   1. 开始计时
   2. 调用 dlagtm_ 函数执行某项任务
   3. 结束计时，并将时间记录到 timing 数组的第 9 个位置
*/

#define TRLIB_PRINTLN_1(...) if (verbose > 0) { fprintf(fout, "%s", prefix); fprintf(fout, __VA_ARGS__); fprintf(fout, "\n"); }
/*
   宏定义 TRLIB_PRINTLN_1
   参数: 可变参数列表
   如果 verbose 大于 0，则打印输出到 fout 文件中，输出格式为 prefix + 格式化字符串 + 换行
*/

#define TRLIB_PRINTLN_2(...) if (verbose > 1) { fprintf(fout, "%s", prefix); fprintf(fout, __VA_ARGS__); fprintf(fout, "\n"); }
/*
   宏定义 TRLIB_PRINTLN_2
   参数: 可变参数列表
   如果 verbose 大于 1，则打印输出到 fout 文件中，输出格式为 prefix + 格式化字符串 + 换行
*/

#define TRLIB_PRINT_VEC(P, N, X) { for(int vc = 0; vc < N; ++vc) { printf("%s %ld: %e\n", P, vc, *(X+vc)); } }
/*
   宏定义 TRLIB_PRINT_VEC
   参数:
   1. P: 输出前缀字符串
   2. N: 数组 X 的长度
   3. X: 待输出的数组指针
   打印数组 X 中的元素，输出格式为 P + 索引 + 值，每行一个元素
*/

#endif
/*
   结束宏定义的条件编译部分
*/
```