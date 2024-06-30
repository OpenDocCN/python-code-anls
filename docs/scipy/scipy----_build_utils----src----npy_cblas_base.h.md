# `D:\src\scipysrc\scipy\scipy\_build_utils\src\npy_cblas_base.h`

```
/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
#ifndef _NPY_CBLAS_BASE_H_
#define _NPY_CBLAS_BASE_H_

/*
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 */

/*
 * Functions for single precision real numbers (S)
 */
float  BLASNAME(cblas_sdsdot)(const BLASINT N, const float alpha, const float *X,
                              const BLASINT incX, const float *Y, const BLASINT incY);
double BLASNAME(cblas_dsdot)(const BLASINT N, const float *X, const BLASINT incX, const float *Y,
                             const BLASINT incY);
float  BLASNAME(cblas_sdot)(const BLASINT N, const float  *X, const BLASINT incX,
                            const float  *Y, const BLASINT incY);
double BLASNAME(cblas_ddot)(const BLASINT N, const double *X, const BLASINT incX,
                            const double *Y, const BLASINT incY);

/*
 * Functions for complex numbers (C and Z)
 */
void   BLASNAME(cblas_cdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotu);
void   BLASNAME(cblas_cdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotc);

void   BLASNAME(cblas_zdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotu);
void   BLASNAME(cblas_zdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotc);

/*
 * Functions for computing norms and sums
 */
float  BLASNAME(cblas_snrm2)(const BLASINT N, const float *X, const BLASINT incX);
float  BLASNAME(cblas_sasum)(const BLASINT N, const float *X, const BLASINT incX);

double BLASNAME(cblas_dnrm2)(const BLASINT N, const double *X, const BLASINT incX);
double BLASNAME(cblas_dasum)(const BLASINT N, const double *X, const BLASINT incX);

float  BLASNAME(cblas_scnrm2)(const BLASINT N, const void *X, const BLASINT incX);
float  BLASNAME(cblas_scasum)(const BLASINT N, const void *X, const BLASINT incX);

double BLASNAME(cblas_dznrm2)(const BLASINT N, const void *X, const BLASINT incX);
double BLASNAME(cblas_dzasum)(const BLASINT N, const void *X, const BLASINT incX);

/*
 * Functions for finding index of element with maximum absolute value
 */
CBLAS_INDEX BLASNAME(cblas_isamax)(const BLASINT N, const float  *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_idamax)(const BLASINT N, const double *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_icamax)(const BLASINT N, const void   *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_izamax)(const BLASINT N, const void   *X, const BLASINT incX);

#endif  /* _NPY_CBLAS_BASE_H_ */


注释：
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */

// Define prototype for swapping elements in float arrays X and Y
void BLASNAME(cblas_sswap)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);

// Define prototype for copying elements from float array X to float array Y
void BLASNAME(cblas_scopy)(const BLASINT N, const float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);

// Define prototype for axpy operation: Y = alpha*X + Y, where alpha is float
void BLASNAME(cblas_saxpy)(const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, float *Y, const BLASINT incY);

// Define prototype for swapping elements in double arrays X and Y
void BLASNAME(cblas_dswap)(const BLASINT N, double *X, const BLASINT incX,
                           double *Y, const BLASINT incY);

// Define prototype for copying elements from double array X to double array Y
void BLASNAME(cblas_dcopy)(const BLASINT N, const double *X, const BLASINT incX,
                           double *Y, const BLASINT incY);

// Define prototype for axpy operation: Y = alpha*X + Y, where alpha is double
void BLASNAME(cblas_daxpy)(const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, double *Y, const BLASINT incY);

// Define prototype for swapping elements in complex float arrays X and Y
void BLASNAME(cblas_cswap)(const BLASINT N, void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);

// Define prototype for copying elements from complex float array X to complex float array Y
void BLASNAME(cblas_ccopy)(const BLASINT N, const void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);

// Define prototype for axpy operation: Y = alpha*X + Y, where alpha is complex float
void BLASNAME(cblas_caxpy)(const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, void *Y, const BLASINT incY);

// Define prototype for swapping elements in complex double arrays X and Y
void BLASNAME(cblas_zswap)(const BLASINT N, void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);

// Define prototype for copying elements from complex double array X to complex double array Y
void BLASNAME(cblas_zcopy)(const BLASINT N, const void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);

// Define prototype for axpy operation: Y = alpha*X + Y, where alpha is complex double
void BLASNAME(cblas_zaxpy)(const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, void *Y, const BLASINT incY);


/*
 * Routines with S and D prefix only
 */

// Define prototype for generating Givens rotation for float elements
void BLASNAME(cblas_srotg)(float *a, float *b, float *c, float *s);

// Define prototype for modified Givens rotation for float elements
void BLASNAME(cblas_srotmg)(float *d1, float *d2, float *b1, const float b2, float *P);

// Define prototype for applying Givens rotation to float arrays X and Y
void BLASNAME(cblas_srot)(const BLASINT N, float *X, const BLASINT incX,
                          float *Y, const BLASINT incY, const float c, const float s);

// Define prototype for modified Givens rotation to float arrays X and Y
void BLASNAME(cblas_srotm)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY, const float *P);

// Define prototype for generating Givens rotation for double elements
void BLASNAME(cblas_drotg)(double *a, double *b, double *c, double *s);

// Define prototype for modified Givens rotation for double elements
void BLASNAME(cblas_drotmg)(double *d1, double *d2, double *b1, const double b2, double *P);

// Define prototype for applying Givens rotation to double arrays X and Y
void BLASNAME(cblas_drot)(const BLASINT N, double *X, const BLASINT incX,
                          double *Y, const BLASINT incY, const double c, const double  s);

// Define prototype for modified Givens rotation to double arrays X and Y
void BLASNAME(cblas_drotm)(const BLASINT N, double *X, const BLASINT incX,
                           double *Y, const BLASINT incY, const double *P);
/*
 * Routines with S D C Z CS and ZD prefixes
 */
// 定义一组函数声明，涵盖了针对不同数据类型和复数类型的标量乘法操作
void BLASNAME(cblas_sscal)(const BLASINT N, const float alpha, float *X, const BLASINT incX);
void BLASNAME(cblas_dscal)(const BLASINT N, const double alpha, double *X, const BLASINT incX);
void BLASNAME(cblas_cscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_zscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_csscal)(const BLASINT N, const float alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_zdscal)(const BLASINT N, const double alpha, void *X, const BLASINT incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
// 矩阵-向量乘法和相关操作的函数声明，涵盖了不同数据类型和复数类型的变体
void BLASNAME(cblas_sgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *X, const BLASINT incX, const float beta,
                           float *Y, const BLASINT incY);
void BLASNAME(cblas_sgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const float alpha,
                           const float *A, const BLASINT lda, const float *X,
                           const BLASINT incX, const float beta, float *Y, const BLASINT incY);
void BLASNAME(cblas_strmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
void BLASNAME(cblas_stbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
void BLASNAME(cblas_stpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *Ap, float *X, const BLASINT incX);
void BLASNAME(cblas_strsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda, float *X,
                           const BLASINT incX);


这段代码是一个头文件或者接口声明，定义了一系列的数学函数，这些函数实现了基本线性代数子程序（BLAS）的一部分功能，包括标量乘法、矩阵向量乘法、三角矩阵向量操作等。每个函数都根据其前缀（S, D, C, Z等）和数据类型（float, double, complex等）来命名，以区分不同的变体。
// 解释了 BLAS 库中一些特定函数的声明，这些函数用于线性代数计算

void BLASNAME(cblas_stbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
// 声明了解方程组的函数 cblas_stbsv，使用单精度浮点数
// order: 矩阵顺序（行优先或列优先）
// Uplo: 指定 A 的存储方式（上三角或下三角）
// TransA: 指定 A 是否被转置
// Diag: 指定 A 是否为单位对角阵
// N: 矩阵和向量的大小
// K: 矩阵 A 的带宽
// A: 矩阵 A 的数据指针
// lda: A 的第一个维度大小
// X: 输出向量的数据指针
// incX: X 中元素的增量

void BLASNAME(cblas_stpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *Ap, float *X, const BLASINT incX);
// 声明了解三角矩阵方程组的函数 cblas_stpsv，使用单精度浮点数
// 参数详细说明同上，但 A 以紧缩列存储的方式提供（仅存储非零元素）

void BLASNAME(cblas_dgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *X, const BLASINT incX, const double beta,
                           double *Y, const BLASINT incY);
// 声明了矩阵向量乘法函数 cblas_dgemv，使用双精度浮点数
// TransA: 指定 A 是否被转置
// M: 矩阵 A 的行数
// N: 矩阵 A 的列数
// alpha: 乘法的比例因子
// A: 矩阵 A 的数据指针
// lda: A 的第一个维度大小
// X: 输入向量的数据指针
// incX: X 中元素的增量
// beta: Y 的比例因子
// Y: 输出向量的数据指针
// incY: Y 中元素的增量

void BLASNAME(cblas_dgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const double alpha,
                           const double *A, const BLASINT lda, const double *X,
                           const BLASINT incX, const double beta, double *Y, const BLASINT incY);
// 声明了一般带状矩阵向量乘法函数 cblas_dgbmv，使用双精度浮点数
// KL: 矩阵 A 的下带宽
// KU: 矩阵 A 的上带宽
// 参数详细说明同上

void BLASNAME(cblas_dtrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
// 声明了矩阵-向量乘法函数 cblas_dtrmv，使用双精度浮点数
// 参数详细说明同前面的函数

void BLASNAME(cblas_dtbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
// 声明了带状矩阵-向量乘法函数 cblas_dtbmv，使用双精度浮点数
// 参数详细说明同前面的函数

void BLASNAME(cblas_dtpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);
// 声明了紧缩列存储的三角矩阵-向量乘法函数 cblas_dtpmv，使用双精度浮点数
// 参数详细说明同前面的函数

void BLASNAME(cblas_dtrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda, double *X,
                           const BLASINT incX);
// 声明了解三角矩阵方程组的函数 cblas_dtrsv，使用双精度浮点数
// 参数详细说明同前面的函数

void BLASNAME(cblas_dtbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
// 声明了带状矩阵方程组的函数 cblas_dtbsv，使用双精度浮点数
// 参数详细说明同前面的函数
// 调用 BLAS 库中的 cblas_dtpsv 函数，对双精度浮点数进行三角求解
void BLASNAME(cblas_dtpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);

// 调用 BLAS 库中的 cblas_cgemv 函数，执行复数矩阵向量乘法
void BLASNAME(cblas_cgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);

// 调用 BLAS 库中的 cblas_cgbmv 函数，执行一般带状复数矩阵向量乘法
void BLASNAME(cblas_cgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);

// 调用 BLAS 库中的 cblas_ctrmv 函数，对复数三角矩阵进行向量乘法
void BLASNAME(cblas_ctrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 调用 BLAS 库中的 cblas_ctbmv 函数，对带状复数三角矩阵进行向量乘法
void BLASNAME(cblas_ctbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 调用 BLAS 库中的 cblas_ctpmv 函数，对带压缩列的复数三角矩阵进行向量乘法
void BLASNAME(cblas_ctpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

// 调用 BLAS 库中的 cblas_ctrsv 函数，解复数三角线性方程
void BLASNAME(cblas_ctrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);

// 调用 BLAS 库中的 cblas_ctbsv 函数，解带状复数三角线性方程
void BLASNAME(cblas_ctbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 调用 BLAS 库中的 cblas_ctpsv 函数，解带压缩列的复数三角线性方程
void BLASNAME(cblas_ctpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);
/*
 * Level 2 BLAS (Basic Linear Algebra Subprograms) routines for complex numbers
 */

// 一般情况下，使用 C 标准库的枚举和类型定义
void BLASNAME(cblas_zgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);

// 稀疏矩阵-向量乘法
void BLASNAME(cblas_zgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);

// 三角矩阵-向量乘法
void BLASNAME(cblas_ztrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 三角带状矩阵-向量乘法
void BLASNAME(cblas_ztbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 三角带状矩阵-向量乘法（压缩存储）
void BLASNAME(cblas_ztpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

// 三角矩阵-向量求解
void BLASNAME(cblas_ztrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);

// 三角带状矩阵-向量求解
void BLASNAME(cblas_ztbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// 三角带状矩阵-向量求解（压缩存储）
void BLASNAME(cblas_ztpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);


/*
 * Routines with S and D prefixes only
 */

// 对称矩阵-向量乘法（单精度浮点数）
void BLASNAME(cblas_ssymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
# 对称矩阵向量乘法（单精度浮点数）
void BLASNAME(cblas_ssymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);

# 对称矩阵带偏移向量乘法（单精度浮点数）
void BLASNAME(cblas_ssbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);

# 对称矩阵压缩列存储向量乘法（单精度浮点数）
void BLASNAME(cblas_sspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *Ap,
                           const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);

# 一般矩阵向量乘法（单精度浮点数）
void BLASNAME(cblas_sger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const float alpha, const float *X, const BLASINT incX,
                          const float *Y, const BLASINT incY, float *A, const BLASINT lda);

# 对称矩阵向量乘法（单精度浮点数）
void BLASNAME(cblas_ssyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *A, const BLASINT lda);

# 对称矩阵压缩列存储向量乘法（单精度浮点数）
void BLASNAME(cblas_sspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *Ap);

# 两个向量的对称矩阵乘法（单精度浮点数）
void BLASNAME(cblas_ssyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A,
                           const BLASINT lda);

# 两个向量的对称矩阵压缩列存储乘法（单精度浮点数）
void BLASNAME(cblas_sspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A);

# 对称矩阵向量乘法（双精度浮点数）
void BLASNAME(cblas_dsymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);

# 对称矩阵带偏移向量乘法（双精度浮点数）
void BLASNAME(cblas_dsbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);

# 对称矩阵压缩列存储向量乘法（双精度浮点数）
void BLASNAME(cblas_dspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *Ap,
                           const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
// 定义函数 cblas_dger，执行双精度一般矩阵的外积操作
void BLASNAME(cblas_dger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const double alpha, const double *X, const BLASINT incX,
                          const double *Y, const BLASINT incY, double *A, const BLASINT lda);

// 定义函数 cblas_dsyr，执行双精度对称矩阵的乘积运算
void BLASNAME(cblas_dsyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *A, const BLASINT lda);

// 定义函数 cblas_dspr，执行双精度对称矩阵的外积运算
void BLASNAME(cblas_dspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *Ap);

// 定义函数 cblas_dsyr2，执行双精度对称矩阵乘积的二次运算
void BLASNAME(cblas_dsyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A,
                           const BLASINT lda);

// 定义函数 cblas_dspr2，执行双精度对称矩阵的二次外积运算
void BLASNAME(cblas_dspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A);

/*
 * 以下是仅带有 C 和 Z 前缀的例程
 */

// 定义函数 cblas_chemv，执行复数 Hermite 矩阵向量乘积运算
void BLASNAME(cblas_chemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 定义函数 cblas_chbmv，执行复数 Hermite 矩阵带带状存储格式向量乘积运算
void BLASNAME(cblas_chbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 定义函数 cblas_chpmv，执行复数 Hermite 矩阵带压缩存储格式向量乘积运算
void BLASNAME(cblas_chpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 定义函数 cblas_cgeru，执行复数一般矩阵的外积运算
void BLASNAME(cblas_cgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 定义函数 cblas_cgerc，执行复数一般矩阵的共轭外积运算
void BLASNAME(cblas_cgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
// 计算单精度复数埃尔米特矩阵与向量的乘积并加到另一个复数埃尔米特矩阵中
void BLASNAME(cblas_cher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);

// 计算单精度复数埃尔米特矩阵的外积并加到另一个复数埃尔米特矩阵中
void BLASNAME(cblas_chpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X,
                          const BLASINT incX, void *A);

// 计算两个单精度复数向量的埃尔米特外积并加到复数埃尔米特矩阵中
void BLASNAME(cblas_cher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 计算两个单精度复数向量的埃尔米特外积并加到复数埃尔米特矩阵中
void BLASNAME(cblas_chpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *Ap);

// 计算双精度复数埃尔米特矩阵与向量的乘积并加到另一个复数向量中
void BLASNAME(cblas_zhemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 计算双精度复数带状埃尔米特矩阵与向量的乘积并加到另一个复数向量中
void BLASNAME(cblas_zhbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 计算双精度复数带状埃尔米特矩阵与向量的乘积并加到另一个复数向量中
void BLASNAME(cblas_zhpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 计算两个双精度复数向量的外积并加到双精度复数矩阵中
void BLASNAME(cblas_zgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 计算两个双精度复数向量的外积（共轭）并加到双精度复数矩阵中
void BLASNAME(cblas_zgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 计算双精度复数埃尔米特矩阵与向量的乘积并加到另一个双精度复数埃尔米特矩阵中
void BLASNAME(cblas_zher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);

// 计算双精度复数埃尔米特矩阵的外积并加到另一个双精度复数埃尔米特矩阵中
void BLASNAME(cblas_zhpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X,
                          const BLASINT incX, void *A);
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with complex type double complex (Z) for Hermitian rank 2 operation
 */
void BLASNAME(cblas_zher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, const void *Y, const BLASINT incY,
                           void *A, const BLASINT lda);

/*
 * Routines with complex type double complex (Z) for Hermitian packed rank 2 operation
 */
void BLASNAME(cblas_zhpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, const void *Y, const BLASINT incY,
                           void *Ap);

/*
 * Routines with standard 4 prefixes (S, D, C, Z) for general matrix multiplication
 */
void BLASNAME(cblas_sgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *B, const BLASINT ldb,
                           const float beta, float *C, const BLASINT ldc);

/*
 * Routines with standard 4 prefixes (S, D, C, Z) for symmetric matrix multiplication
 */
void BLASNAME(cblas_ssymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *B, const BLASINT ldb, const float beta,
                           float *C, const BLASINT ldc);

/*
 * Routines with standard 4 prefixes (S, D, C, Z) for symmetric rank K update
 */
void BLASNAME(cblas_ssyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const float alpha, const float *A, const BLASINT lda,
                           const float beta, float *C, const BLASINT ldc);

/*
 * Routines with standard 4 prefixes (S, D, C, Z) for symmetric rank 2K update
 */
void BLASNAME(cblas_ssyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const float alpha, const float *A, const BLASINT lda,
                            const float *B, const BLASINT ldb, const float beta,
                            float *C, const BLASINT ldc);

/*
 * Routines with standard 4 prefixes (S, D, C, Z) for triangular matrix multiplication
 */
void BLASNAME(cblas_strmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);
# 定义调用 BLAS 库中的 cblas_strsm 函数，该函数执行单精度矩阵乘法的特定形式
void BLASNAME(cblas_strsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);

# 定义调用 BLAS 库中的 cblas_dgemm 函数，该函数执行双精度矩阵乘法
void BLASNAME(cblas_dgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *B, const BLASINT ldb,
                           const double beta, double *C, const BLASINT ldc);

# 定义调用 BLAS 库中的 cblas_dsymm 函数，该函数执行双精度对称矩阵乘法
void BLASNAME(cblas_dsymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *B, const BLASINT ldb, const double beta,
                           double *C, const BLASINT ldc);

# 定义调用 BLAS 库中的 cblas_dsyrk 函数，该函数执行双精度对称矩阵乘法
void BLASNAME(cblas_dsyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const double *A, const BLASINT lda,
                           const double beta, double *C, const BLASINT ldc);

# 定义调用 BLAS 库中的 cblas_dsyr2k 函数，该函数执行双精度对称矩阵乘法的扩展形式
void BLASNAME(cblas_dsyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const double alpha, const double *A, const BLASINT lda,
                            const double *B, const BLASINT ldb, const double beta,
                            double *C, const BLASINT ldc);

# 定义调用 BLAS 库中的 cblas_dtrmm 函数，该函数执行双精度三角矩阵乘法
void BLASNAME(cblas_dtrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);

# 定义调用 BLAS 库中的 cblas_dtrsm 函数，该函数执行双精度三角矩阵求解
void BLASNAME(cblas_dtrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);
// 声明一个用于复数矩阵乘法的函数，使用单精度浮点数。
void BLASNAME(cblas_cgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);

// 声明一个用于复数对称矩阵乘法的函数，使用单精度浮点数。
void BLASNAME(cblas_csymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);

// 声明一个用于复数对称矩阵乘法的函数，使用单精度浮点数。
void BLASNAME(cblas_csyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *beta, void *C, const BLASINT ldc);

// 声明一个用于复数对称矩阵乘法的函数，使用单精度浮点数。
void BLASNAME(cblas_csyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const void *beta,
                            void *C, const BLASINT ldc);

// 声明一个用于复数三角矩阵与矩阵的乘法的函数，使用单精度浮点数。
void BLASNAME(cblas_ctrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

// 声明一个用于复数三角矩阵的解的函数，使用单精度浮点数。
void BLASNAME(cblas_ctrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

// 声明一个用于复数矩阵乘法的函数，使用双精度浮点数。
void BLASNAME(cblas_zgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);
/*
 * Perform complex matrix multiplication C = alpha * A * B + beta * C
 */
void BLASNAME(cblas_zsymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);

/*
 * Perform complex symmetric rank-K update C = alpha * A * A^T + beta * C or
 * C = alpha * A^T * A + beta * C depending on 'Trans' parameter.
 */
void BLASNAME(cblas_zsyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *beta, void *C, const BLASINT ldc);

/*
 * Perform complex symmetric rank-2k update C = alpha * A * B^T + alpha * B * A^T + beta * C
 * or C = alpha * A^T * B + alpha * B^T * A + beta * C depending on 'Trans' parameter.
 */
void BLASNAME(cblas_zsyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const void *beta,
                            void *C, const BLASINT ldc);

/*
 * Perform complex triangular matrix multiplication B = alpha * A * B or B = alpha * B * A
 * depending on 'Side' and 'TransA' parameters.
 */
void BLASNAME(cblas_ztrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

/*
 * Solve complex triangular system of equations B = alpha * A^(-1) * B or B = alpha * B * A^(-1)
 * depending on 'Side' and 'TransA' parameters.
 */
void BLASNAME(cblas_ztrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

/*
 * Perform complex Hermitian matrix multiplication C = alpha * A * B + beta * C
 * where A is complex Hermitian matrix.
 */
void BLASNAME(cblas_chemm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);

/*
 * Perform complex Hermitian rank-K update C = alpha * A * A^H + beta * C or
 * C = alpha * A^H * A + beta * C depending on 'Trans' parameter.
 */
void BLASNAME(cblas_cherk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const float alpha, const void *A, const BLASINT lda,
                           const float beta, void *C, const BLASINT ldc);
# 定义了一系列使用不同数据类型和操作的 BLAS 函数原型，用于矩阵运算

void BLASNAME(cblas_cher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const float beta,
                            void *C, const BLASINT ldc);
# cblas_cher2k 函数原型：执行 Hermitean rank-2k operation on complex matrix

void BLASNAME(cblas_zhemm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);
# cblas_zhemm 函数原型：执行 Hermitian matrix-matrix multiplication

void BLASNAME(cblas_zherk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const void *A, const BLASINT lda,
                           const double beta, void *C, const BLASINT ldc);
# cblas_zherk 函数原型：执行 Hermitian rank-K update on complex matrix

void BLASNAME(cblas_zher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const double beta,
                            void *C, const BLASINT ldc);
# cblas_zher2k 函数原型：执行 Hermitian rank-2k update on complex matrix

void BLASNAME(cblas_xerbla)(BLASINT p, const char *rout, const char *form, ...);
# cblas_xerbla 函数原型：处理 BLAS 函数中的错误，如无效参数等

#endif  /* _NPY_CBLAS_BASE_H_ */
# 结束条件编译指令，用于保护头文件不被重复包含
```