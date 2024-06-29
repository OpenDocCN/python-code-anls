# `.\numpy\numpy\_core\src\common\npy_cblas_base.h`

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
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_

// Single precision dot product with extended precision accumulation
float BLASNAME(cblas_sdsdot)(const BLASINT N, const float alpha, const float *X,
                             const BLASINT incX, const float *Y, const BLASINT incY);
// Double precision dot product
double BLASNAME(cblas_dsdot)(const BLASINT N, const float *X, const BLASINT incX, const float *Y,
                             const BLASINT incY);
// Single precision dot product
float BLASNAME(cblas_sdot)(const BLASINT N, const float *X, const BLASINT incX,
                           const float *Y, const BLASINT incY);
// Double precision dot product
double BLASNAME(cblas_ddot)(const BLASINT N, const double *X, const BLASINT incX,
                            const double *Y, const BLASINT incY);

// Complex dot product (unconjugated)
void BLASNAME(cblas_cdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                               const void *Y, const BLASINT incY, void *dotu);
// Complex dot product (conjugated)
void BLASNAME(cblas_cdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                               const void *Y, const BLASINT incY, void *dotc);

// Double complex dot product (unconjugated)
void BLASNAME(cblas_zdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                               const void *Y, const BLASINT incY, void *dotu);
// Double complex dot product (conjugated)
void BLASNAME(cblas_zdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                               const void *Y, const BLASINT incY, void *dotc);

// Euclidean norm of a single precision vector
float BLASNAME(cblas_snrm2)(const BLASINT N, const float *X, const BLASINT incX);
// Sum of absolute values of a single precision vector
float BLASNAME(cblas_sasum)(const BLASINT N, const float *X, const BLASINT incX);

// Euclidean norm of a double precision vector
double BLASNAME(cblas_dnrm2)(const BLASINT N, const double *X, const BLASINT incX);
// Sum of absolute values of a double precision vector
double BLASNAME(cblas_dasum)(const BLASINT N, const double *X, const BLASINT incX);

// Euclidean norm of a single precision complex vector
float BLASNAME(cblas_scnrm2)(const BLASINT N, const void *X, const BLASINT incX);
// Sum of absolute values of a single precision complex vector
float BLASNAME(cblas_scasum)(const BLASINT N, const void *X, const BLASINT incX);

// Euclidean norm of a double precision complex vector
double BLASNAME(cblas_dznrm2)(const BLASINT N, const void *X, const BLASINT incX);
// Sum of absolute values of a double precision complex vector
double BLASNAME(cblas_dzasum)(const BLASINT N, const void *X, const BLASINT incX);

// Index of maximum absolute value of a single precision vector
CBLAS_INDEX BLASNAME(cblas_isamax)(const BLASINT N, const float *X, const BLASINT incX);
// Index of maximum absolute value of a double precision vector
CBLAS_INDEX BLASNAME(cblas_idamax)(const BLASINT N, const double *X, const BLASINT incX);
// Index of maximum absolute value of a single precision complex vector
CBLAS_INDEX BLASNAME(cblas_icamax)(const BLASINT N, const void *X, const BLASINT incX);

#endif  // NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_


注释：
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */

/*
 * Function prototype for cblas_izamax:
 * Returns the index of the first element with maximum absolute value in X.
 * Parameters:
 *   - N: Number of elements in X
 *   - X: Pointer to the array of elements (void pointer)
 *   - incX: Increment for indexing into X
 * Returns:
 *   - CBLAS_INDEX: Index of the element with maximum absolute value
 */
CBLAS_INDEX BLASNAME(cblas_izamax)(const BLASINT N, const void *X, const BLASINT incX);

/*
 * Function prototypes for standard BLAS level 1 routines:
 * sswap, scopy, saxpy, dswap, dcopy, daxpy, cswap, ccopy, caxpy, zswap, zcopy, zaxpy
 */

/*
 * Function prototype for cblas_sswap:
 * Swaps elements between two arrays X and Y.
 * Parameters:
 *   - N: Number of elements in X and Y
 *   - X: Pointer to the array X
 *   - incX: Increment for indexing into X
 *   - Y: Pointer to the array Y
 *   - incY: Increment for indexing into Y
 */
void BLASNAME(cblas_sswap)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);

/*
 * Function prototype for cblas_scopy:
 * Copies elements from array X to array Y.
 * Parameters:
 *   - N: Number of elements in X and Y
 *   - X: Pointer to the source array X
 *   - incX: Increment for indexing into X
 *   - Y: Pointer to the destination array Y
 *   - incY: Increment for indexing into Y
 */
void BLASNAME(cblas_scopy)(const BLASINT N, const float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);

/*
 * Function prototype for cblas_saxpy:
 * Computes Y = alpha*X + Y.
 * Parameters:
 *   - N: Number of elements in X and Y
 *   - alpha: Scalar alpha
 *   - X: Pointer to the array X
 *   - incX: Increment for indexing into X
 *   - Y: Pointer to the array Y
 *   - incY: Increment for indexing into Y
 */
void BLASNAME(cblas_saxpy)(const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, float *Y, const BLASINT incY);

/*
 * Similar function prototypes for double precision (d prefix), complex
 * single precision (c prefix), and complex double precision (z prefix) routines
 * cblas_dswap, cblas_dcopy, cblas_daxpy, cblas_cswap, cblas_ccopy, cblas_caxpy,
 * cblas_zswap, cblas_zcopy, cblas_zaxpy.
 */

/*
 * Function prototype for cblas_srotg:
 * Constructs a Givens plane rotation matrix.
 * Parameters:
 *   - a: Input/output parameter (see BLAS documentation)
 *   - b: Input/output parameter (see BLAS documentation)
 *   - c: Output parameter (see BLAS documentation)
 *   - s: Output parameter (see BLAS documentation)
 */
void BLASNAME(cblas_srotg)(float *a, float *b, float *c, float *s);

/*
 * Function prototype for cblas_srotmg:
 * Constructs modified Givens plane rotation matrix.
 * Parameters:
 *   - d1: Input/output parameter (see BLAS documentation)
 *   - d2: Input/output parameter (see BLAS documentation)
 *   - b1: Input/output parameter (see BLAS documentation)
 *   - b2: Input parameter (see BLAS documentation)
 *   - P: Output parameter (see BLAS documentation)
 */
void BLASNAME(cblas_srotmg)(float *d1, float *d2, float *b1, const float b2, float *P);

/*
 * Function prototype for cblas_srot:
 * Applies a Givens rotation to vectors X and Y.
 * Parameters:
 *   - N: Number of elements in X and Y
 *   - X: Pointer to the array X
 *   - incX: Increment for indexing into X
 *   - Y: Pointer to the array Y
 *   - incY: Increment for indexing into Y
 *   - c: Cosine of the angle of rotation
 *   - s: Sine of the angle of rotation
 */
void BLASNAME(cblas_srot)(const BLASINT N, float *X, const BLASINT incX,
                          float *Y, const BLASINT incY, const float c, const float s);

/*
 * Function prototype for cblas_srotm:
 * Applies modified Givens rotation to vectors X and Y.
 * Parameters:
 *   - N: Number of elements in X and Y
 *   - X: Pointer to the array X
 *   - incX: Increment for indexing into X
 *   - Y: Pointer to the array Y
 *   - incY: Increment for indexing into Y
 *   - P: Pointer to the P array
 */
void BLASNAME(cblas_srotm)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY, const float *P);

/*
 * Similar function prototypes for double precision (d prefix) routines:
 * cblas_drotg, cblas_drotmg, cblas_drot.
 */
/*
 * BLASNAME(cblas_drotm) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - X: 双精度浮点数数组
 * - incX: 数组 X 中元素的增量
 * - Y: 双精度浮点数数组
 * - incY: 数组 Y 中元素的增量
 * - P: 双精度浮点数数组，包含参数 P
 */
void BLASNAME(cblas_drotm)(const BLASINT N, double *X, const BLASINT incX,
                           double *Y, const BLASINT incY, const double *P);


/*
 * 带有 S D C Z CS 和 ZD 前缀的例程
 */

/*
 * BLASNAME(cblas_sscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 浮点数倍乘因子
 * - X: 单精度浮点数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_sscal)(const BLASINT N, const float alpha, float *X, const BLASINT incX);

/*
 * BLASNAME(cblas_dscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 双精度浮点数倍乘因子
 * - X: 双精度浮点数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_dscal)(const BLASINT N, const double alpha, double *X, const BLASINT incX);

/*
 * BLASNAME(cblas_cscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 复数倍乘因子的指针
 * - X: 复数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_cscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);

/*
 * BLASNAME(cblas_zscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 双复数倍乘因子的指针
 * - X: 双复数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_zscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);

/*
 * BLASNAME(cblas_csscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 实部为浮点数倍乘因子，虚部为零的复数
 * - X: 复数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_csscal)(const BLASINT N, const float alpha, void *X, const BLASINT incX);

/*
 * BLASNAME(cblas_zdscal) 函数的声明
 * 
 * 参数:
 * - N: 数组中元素的数量
 * - alpha: 实部为双精度浮点数倍乘因子，虚部为零的双复数
 * - X: 双复数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_zdscal)(const BLASINT N, const double alpha, void *X, const BLASINT incX);


/*
 * ===========================================================================
 * level 2 BLAS 的原型
 * ===========================================================================
 */

/*
 * 带有标准 4 个前缀 (S, D, C, Z) 的例程
 */

/*
 * BLASNAME(cblas_sgemv) 函数的声明
 * 
 * 参数:
 * - order: 矩阵的存储顺序
 * - TransA: 矩阵 A 的转置方式
 * - M: 矩阵 A 的行数
 * - N: 矩阵 A 的列数
 * - alpha: 浮点数倍乘因子
 * - A: 单精度浮点数矩阵
 * - lda: A 矩阵的行跨度
 * - X: 单精度浮点数数组
 * - incX: 数组 X 中元素的增量
 * - beta: 浮点数倍乘因子
 * - Y: 单精度浮点数数组
 * - incY: 数组 Y 中元素的增量
 */
void BLASNAME(cblas_sgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *X, const BLASINT incX, const float beta,
                           float *Y, const BLASINT incY);

/*
 * BLASNAME(cblas_sgbmv) 函数的声明
 * 
 * 参数:
 * - order: 矩阵的存储顺序
 * - TransA: 矩阵 A 的转置方式
 * - M: 矩阵 A 的行数
 * - N: 矩阵 A 的列数
 * - KL: 矩阵 A 下三角带的宽度
 * - KU: 矩阵 A 上三角带的宽度
 * - alpha: 浮点数倍乘因子
 * - A: 单精度浮点数矩阵
 * - lda: A 矩阵的行跨度
 * - X: 单精度浮点数数组
 * - incX: 数组 X 中元素的增量
 * - beta: 浮点数倍乘因子
 * - Y: 单精度浮点数数组
 * - incY: 数组 Y 中元素的增量
 */
void BLASNAME(cblas_sgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const float alpha,
                           const float *A, const BLASINT lda, const float *X,
                           const BLASINT incX, const float beta, float *Y, const BLASINT incY);

/*
 * BLASNAME(cblas_strmv) 函数的声明
 * 
 * 参数:
 * - order: 矩阵的存储顺序
 * - Uplo: 矩阵 A 的上/下三角部分
 * - TransA: 矩阵 A 的转置方式
 * - Diag: 矩阵 A 的对角元是否为单位矩阵
 * - N: 矩阵 A 的阶数
 * - A: 单精度浮点数矩阵
 * - lda: A 矩阵的行跨度
 * - X: 单精度浮点数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_strmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);

/*
 * BLASNAME(cblas_stbmv) 函数的声明
 * 
 * 参数:
 * - order: 矩阵的存储顺序
 * - Uplo: 矩阵 A 的上/下三角部分
 * - TransA: 矩阵 A 的转置方式
 * - Diag: 矩阵 A 的对角元是否为单位矩阵
 * - N: 矩阵 A 的阶数
 * - K: 矩阵 A 的带宽
 * - A: 单精度浮点数矩阵
 * - lda: A 矩阵的行跨度
 * - X: 单精度浮点数数组
 * - incX: 数组 X 中元素的增量
 */
void BLASNAME(cblas_stbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BL
# 解决线性方程组 A*x = b，其中 A 是一个上/下三角矩阵，返回解 x
void BLASNAME(cblas_strsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda, float *X,
                           const BLASINT incX);

# 解决带有带状矩阵 A 的线性方程组 A*x = b，其中 A 是一个带宽为 K 的上/下三角矩阵，返回解 x
void BLASNAME(cblas_stbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);

# 解决带有带状矩阵 Ap 的线性方程组 Ap*x = b，其中 Ap 是一个带宽为 1 的上/下三角矩阵，返回解 x
void BLASNAME(cblas_stpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *Ap, float *X, const BLASINT incX);

# 计算矩阵 A 与向量 X 之间的乘积，并加上另一个向量 Y 的倍数，结果存入 Y
void BLASNAME(cblas_dgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *X, const BLASINT incX, const double beta,
                           double *Y, const BLASINT incY);

# 计算带有带状矩阵 A 的矩阵乘向量运算，并加上另一个向量 Y 的倍数，结果存入 Y
void BLASNAME(cblas_dgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const double alpha,
                           const double *A, const BLASINT lda, const double *X,
                           const BLASINT incX, const double beta, double *Y, const BLASINT incY);

# 解决矩阵 A 与向量 X 的乘积，其中 A 是一个上/下三角矩阵，返回解 X
void BLASNAME(cblas_dtrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);

# 解决带有带状矩阵 A 的矩阵乘向量运算，其中 A 是一个带宽为 K 的上/下三角矩阵，返回解 X
void BLASNAME(cblas_dtbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);

# 解决带有带状矩阵 Ap 的矩阵乘向量运算，其中 Ap 是一个带宽为 1 的上/下三角矩阵，返回解 X
void BLASNAME(cblas_dtpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);

# 解决带有带状矩阵 A 的线性方程组 A*x = b，其中 A 是一个上/下三角矩阵，返回解 x
void BLASNAME(cblas_dtrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda, double *X,
                           const BLASINT incX);
// Solve triangular banded system of equations with double precision
void BLASNAME(cblas_dtbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);

// Solve triangular packed system of equations with double precision
void BLASNAME(cblas_dtpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);

// Matrix-vector multiplication for complex numbers with single precision
void BLASNAME(cblas_cgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);

// General banded matrix-vector multiplication for complex numbers with single precision
void BLASNAME(cblas_cgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);

// Triangular matrix-vector multiplication for complex numbers with single precision
void BLASNAME(cblas_ctrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// Triangular banded matrix-vector multiplication for complex numbers with single precision
void BLASNAME(cblas_ctbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// Triangular packed matrix-vector multiplication for complex numbers with single precision
void BLASNAME(cblas_ctpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

// Solve triangular system of equations for complex numbers with single precision
void BLASNAME(cblas_ctrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);

// Solve triangular banded system of equations for complex numbers with single precision
void BLASNAME(cblas_ctbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
// cblas_ctpsv: 解决复数三角矩阵的向量方程，使用 CBLAS 库函数
void BLASNAME(cblas_ctpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

// cblas_zgemv: 执行复数一般矩阵-向量乘法，使用 CBLAS 库函数
void BLASNAME(cblas_zgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);

// cblas_zgbmv: 执行复数带状矩阵-向量乘法，使用 CBLAS 库函数
void BLASNAME(cblas_zgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);

// cblas_ztrmv: 执行复数三角矩阵-向量乘法，使用 CBLAS 库函数
void BLASNAME(cblas_ztrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// cblas_ztbmv: 执行复数带状三角矩阵-向量乘法，使用 CBLAS 库函数
void BLASNAME(cblas_ztbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// cblas_ztpmv: 执行复数带状压缩三角矩阵-向量乘法，使用 CBLAS 库函数
void BLASNAME(cblas_ztpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

// cblas_ztrsv: 解决复数三角矩阵的线性方程组，使用 CBLAS 库函数
void BLASNAME(cblas_ztrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);

// cblas_ztbsv: 解决复数带状三角矩阵的线性方程组，使用 CBLAS 库函数
void BLASNAME(cblas_ztbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);

// cblas_ztpsv: 解决复数带状压缩三角矩阵的线性方程组，使用 CBLAS 库函数
void BLASNAME(cblas_ztpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

/*
 * Routines with S and D prefixes only
 */
# 定义了一系列针对对称矩阵的 BLAS （基础线性代数子程序）函数接口，用于浮点数操作

void BLASNAME(cblas_ssymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
# cblas_ssymv 函数：对称矩阵向量乘法，计算 Y := alpha * A * X + beta * Y

void BLASNAME(cblas_ssbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
# cblas_ssbmv 函数：对称带状矩阵向量乘法，计算 Y := alpha * A * X + beta * Y

void BLASNAME(cblas_sspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *Ap,
                           const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
# cblas_sspmv 函数：对称矩阵向量乘法（打包存储格式），计算 Y := alpha * A * X + beta * Y

void BLASNAME(cblas_sger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const float alpha, const float *X, const BLASINT incX,
                          const float *Y, const BLASINT incY, float *A, const BLASINT lda);
# cblas_sger 函数：一般矩阵-向量乘法，计算 A := alpha * X * Y^T + A

void BLASNAME(cblas_ssyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *A, const BLASINT lda);
# cblas_ssyr 函数：对称矩阵更新，计算 A := alpha * X * X^T + A  或  A := alpha * X^T * X + A

void BLASNAME(cblas_sspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *Ap);
# cblas_sspr 函数：对称矩阵更新（打包存储格式），计算 A := alpha * X * X^T + A  或  A := alpha * X^T * X + A

void BLASNAME(cblas_ssyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A,
                           const BLASINT lda);
# cblas_ssyr2 函数：两个向量的对称矩阵更新，计算 A := alpha * X * Y^T + alpha * Y * X^T + A

void BLASNAME(cblas_sspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A);
# cblas_sspr2 函数：两个向量的对称矩阵更新（打包存储格式），计算 A := alpha * X * Y^T + alpha * Y * X^T + A

void BLASNAME(cblas_dsymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
# cblas_dsymv 函数：对称矩阵向量乘法，计算 Y := alpha * A * X + beta * Y

void BLASNAME(cblas_dsbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
# cblas_dsbmv 函数：对称带状矩阵向量乘法，计算 Y := alpha * A * X + beta * Y
/*
 * BLASNAME(cblas_dspmv) 函数
 */
void BLASNAME(cblas_dspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *Ap,
                           const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
/*
 * BLASNAME(cblas_dger) 函数
 */
void BLASNAME(cblas_dger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const double alpha, const double *X, const BLASINT incX,
                          const double *Y, const BLASINT incY, double *A, const BLASINT lda);
/*
 * BLASNAME(cblas_dsyr) 函数
 */
void BLASNAME(cblas_dsyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *A, const BLASINT lda);
/*
 * BLASNAME(cblas_dspr) 函数
 */
void BLASNAME(cblas_dspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *Ap);
/*
 * BLASNAME(cblas_dsyr2) 函数
 */
void BLASNAME(cblas_dsyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A,
                           const BLASINT lda);
/*
 * BLASNAME(cblas_dspr2) 函数
 */
void BLASNAME(cblas_dspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A);


/*
 * 以下是只有 C 和 Z 前缀的例程
 */

/*
 * BLASNAME(cblas_chemv) 函数
 */
void BLASNAME(cblas_chemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
/*
 * BLASNAME(cblas_chbmv) 函数
 */
void BLASNAME(cblas_chbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
/*
 * BLASNAME(cblas_chpmv) 函数
 */
void BLASNAME(cblas_chpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
/*
 * BLASNAME(cblas_cgeru) 函数
 */
void BLASNAME(cblas_cgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
// 执行复杂数单精度通用矩阵-向量乘法：A = alpha * X * Y^H + A，其中 A 是复数矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_cgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 执行复数单精度埃尔米特矩阵乘法：A = alpha * X * X^H + A，其中 A 是复数埃尔米特矩阵，X 是复数向量
void BLASNAME(cblas_cher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);

// 执行复数单精度埃尔米特矩阵乘法（packed 格式）：A = alpha * X * X^H + A，其中 A 是复数埃尔米特矩阵，X 是复数向量
void BLASNAME(cblas_chpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X,
                          const BLASINT incX, void *A);

// 执行复数单精度埃尔米特矩阵乘法（level 2）：A = alpha * X * Y^H + conj(alpha) * Y * X^H + A，
// 其中 A 是复数埃尔米特矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_cher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 执行复数单精度埃尔米特矩阵乘法（packed 格式，level 2）：A = alpha * X * Y^H + conj(alpha) * Y * X^H + A，
// 其中 A 是复数埃尔米特矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_chpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *Ap);

// 执行复数双精度埃尔米特矩阵-向量乘法：Y = alpha * A * X + beta * Y，其中 A 是复数埃尔米特矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_zhemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 执行复数双精度埃尔米特带状矩阵-向量乘法：Y = alpha * A * X + beta * Y，其中 A 是复数埃尔米特带状矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_zhbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 执行复数双精度埃尔米特矩阵-向量乘法（packed 格式）：Y = alpha * A * X + beta * Y，其中 A 是复数埃尔米特矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_zhpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);

// 执行复数双精度通用矩阵-向量乘法：A = alpha * X * Y^H + A，其中 A 是复数矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_zgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 执行复数双精度通用矩阵-向量乘法（conjugate transposed 格式）：A = alpha * X * Y^H + A，其中 A 是复数矩阵，X 和 Y 是复数向量
void BLASNAME(cblas_zgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);

// 执行复数双精度埃尔米特矩阵乘法：A = alpha * X * X^H + A，其中 A 是复数埃尔米特矩阵，X 是复数向量
void BLASNAME(cblas_zher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * 原型定义了一些 Level 3 BLAS 函数，用于高效的矩阵运算，如矩阵乘法、矩阵向量乘法等。
 */

void BLASNAME(cblas_zhpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X,
                          const BLASINT incX, void *A);
/*
 * 执行 Hermitian rank-1 update 操作，对复数 Hermitian 矩阵 A 进行更新，使用向量 X。
 * Hermitian 矩阵 A 存储在 A 中，更新过程由 alpha 和 X 控制。
 */

void BLASNAME(cblas_zher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
/*
 * 执行 Hermitian rank-2 update 操作，对复数 Hermitian 矩阵 A 进行更新，使用向量 X 和 Y。
 * Hermitian 矩阵 A 存储在 A 中，更新过程由 alpha、X 和 Y 控制。
 */

void BLASNAME(cblas_zhpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *Ap);
/*
 * 执行 Hermitian rank-2 update 操作，对复数 Hermitian 矩阵 A 进行更新，使用向量 X 和 Y。
 * 更新后的结果存储在 Ap 中，更新过程由 alpha、X 和 Y 控制。
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

void BLASNAME(cblas_sgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *B, const BLASINT ldb,
                           const float beta, float *C, const BLASINT ldc);
/*
 * 执行矩阵乘法运算 C = alpha * A * B + beta * C。
 * A、B、C 分别是输入和输出矩阵，alpha 和 beta 是标量系数。
 */

void BLASNAME(cblas_ssymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *B, const BLASINT ldb, const float beta,
                           float *C, const BLASINT ldc);
/*
 * 执行对称矩阵乘法运算 C = alpha * A * B + beta * C 或者 C = alpha * B * A + beta * C，
 * 具体操作取决于 Side 参数。A 是对称矩阵，B 和 C 是输入和输出矩阵。
 */

void BLASNAME(cblas_ssyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const float alpha, const float *A, const BLASINT lda,
                           const float beta, float *C, const BLASINT ldc);
/*
 * 执行对称矩阵乘积运算 C = alpha * A * A^T + beta * C 或者 C = alpha * A^T * A + beta * C，
 * 具体操作取决于 Trans 参数。A 是输入矩阵，C 是输出矩阵。
 */

void BLASNAME(cblas_ssyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const float alpha, const float *A, const BLASINT lda,
                            const float *B, const BLASINT ldb, const float beta,
                            float *C, const BLASINT ldc);
/*
 * 执行对称矩阵乘积运算 C = alpha * A * B^T + alpha * B * A^T + beta * C 或者
 * C = alpha * A^T * B + alpha * B^T * A + beta * C，具体操作取决于 Trans 参数。
 * A 和 B 是输入矩阵，C 是输出矩阵。
 */

void BLASNAME(cblas_strmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);
/*
 * 执行三角矩阵乘法运算 B = alpha * A * B 或者 B = alpha * B * A，
 * 具体操作取决于 Side 参数。A 是三角矩阵，B 是输入和输出矩阵。
 */
// 调用 Level 3 BLAS 库中的 cblas_strsm 函数，用于解决形如 B = alpha * op(A) * B 或 B = alpha * B * op(A) 的矩阵方程
void BLASNAME(cblas_strsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);

// 调用 Level 3 BLAS 库中的 cblas_dgemm 函数，执行一般矩阵乘法 C = alpha * op(A) * op(B) + beta * C
void BLASNAME(cblas_dgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *B, const BLASINT ldb,
                           const double beta, double *C, const BLASINT ldc);

// 调用 Level 3 BLAS 库中的 cblas_dsymm 函数，执行对称矩阵乘法 C = alpha * A * B + beta * C 或 C = alpha * B * A + beta * C
void BLASNAME(cblas_dsymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *B, const BLASINT ldb, const double beta,
                           double *C, const BLASINT ldc);

// 调用 Level 3 BLAS 库中的 cblas_dsyrk 函数，执行对称矩阵乘法 C = alpha * op(A) * op(A)^T + beta * C 或 C = alpha * op(A)^T * op(A) + beta * C
void BLASNAME(cblas_dsyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const double *A, const BLASINT lda,
                           const double beta, double *C, const BLASINT ldc);

// 调用 Level 3 BLAS 库中的 cblas_dsyr2k 函数，执行两个对称矩阵的乘法 C = alpha * op(A) * op(B)^T + alpha * op(B) * op(A)^T + beta * C
void BLASNAME(cblas_dsyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const double alpha, const double *A, const BLASINT lda,
                            const double *B, const BLASINT ldb, const double beta,
                            double *C, const BLASINT ldc);

// 调用 Level 3 BLAS 库中的 cblas_dtrmm 函数，用于解决形如 B = alpha * op(A) * B 或 B = alpha * B * op(A) 的三角矩阵方程
void BLASNAME(cblas_dtrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);

// 调用 Level 3 BLAS 库中的 cblas_dtrsm 函数，用于解决形如 B = alpha * op(A) * B 或 B = alpha * B * op(A) 的三角矩阵方程
void BLASNAME(cblas_dtrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);
# 调用 BLAS 库中的 cblas_cgemm 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_cgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);

# 调用 BLAS 库中的 cblas_csymm 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_csymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);

# 调用 BLAS 库中的 cblas_csyrk 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_csyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *beta, void *C, const BLASINT ldc);

# 调用 BLAS 库中的 cblas_csyr2k 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_csyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const void *beta,
                            void *C, const BLASINT ldc);

# 调用 BLAS 库中的 cblas_ctrmm 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_ctrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

# 调用 BLAS 库中的 cblas_ctrsm 函数，进行复数矩阵的三角求解
void BLASNAME(cblas_ctrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

# 调用 BLAS 库中的 cblas_zgemm 函数，进行复数矩阵乘法运算
void BLASNAME(cblas_zgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);
/*
 * BLASNAME(cblas_zsymm)函数：
 * 实现复数对称矩阵乘法，计算 C := alpha * A * B + beta * C 或者 C := alpha * B * A + beta * C，依赖于 Side 参数
 * Order：矩阵存储顺序
 * Side：指定 A 出现在 B 的左侧还是右侧
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * M：矩阵 C 的行数
 * N：矩阵 C 的列数
 * alpha：复数标量，用于乘法操作
 * A：复数对称矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * B：复数矩阵 B
 * ldb：B 矩阵的列数
 * beta：复数标量，用于乘法操作
 * C：结果矩阵 C
 * ldc：C 矩阵的列数
 */

/*
 * BLASNAME(cblas_zsyrk)函数：
 * 实现复数对称矩阵乘法，计算 C := alpha * A * A^T + beta * C 或者 C := alpha * A^T * A + beta * C，依赖于 Trans 参数
 * Order：矩阵存储顺序
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * Trans：指定 A 是否进行转置操作
 * N：矩阵 C 的阶数
 * K：矩阵 A 的列数或行数，依赖于 Trans 参数
 * alpha：复数标量，用于乘法操作
 * A：复数矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * beta：复数标量，用于乘法操作
 * C：结果矩阵 C
 * ldc：C 矩阵的列数
 */

/*
 * BLASNAME(cblas_zsyr2k)函数：
 * 实现复数对称矩阵乘法，计算 C := alpha * A * B^T + alpha * B * A^T + beta * C 或者 C := alpha * A^T * B + alpha * B^T * A + beta * C，依赖于 Trans 参数
 * Order：矩阵存储顺序
 * Uplo：指定矩阵 A 和 B 的存储类型（上三角或下三角）
 * Trans：指定 A 和 B 是否进行转置操作
 * N：矩阵 C 的阶数
 * K：矩阵 A 和 B 的列数或行数，依赖于 Trans 参数
 * alpha：复数标量，用于乘法操作
 * A：复数矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * B：复数矩阵 B
 * ldb：B 矩阵的列数
 * beta：复数标量，用于乘法操作
 * C：结果矩阵 C
 * ldc：C 矩阵的列数
 */

/*
 * BLASNAME(cblas_ztrmm)函数：
 * 实现复数矩阵的三角矩阵乘法，计算 B := alpha * op(A) * B 或者 B := alpha * B * op(A)，依赖于 Side 和 TransA 参数
 * Order：矩阵存储顺序
 * Side：指定 op(A) 出现在 B 的左侧还是右侧
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * TransA：指定 A 是否进行转置操作
 * Diag：指定是否使用 A 的对角线元素
 * M：矩阵 B 的行数
 * N：矩阵 B 的列数
 * alpha：复数标量，用于乘法操作
 * A：复数矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * B：结果矩阵 B
 * ldb：B 矩阵的列数
 */

/*
 * BLASNAME(cblas_ztrsm)函数：
 * 实现复数矩阵的三角矩阵解方程，计算 B := alpha * op(A)^{-1} * B 或者 B := alpha * B * op(A)^{-1}，依赖于 Side 和 TransA 参数
 * Order：矩阵存储顺序
 * Side：指定 op(A) 出现在 B 的左侧还是右侧
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * TransA：指定 A 是否进行转置操作
 * Diag：指定是否使用 A 的对角线元素
 * M：矩阵 B 的行数
 * N：矩阵 B 的列数
 * alpha：复数标量，用于乘法操作
 * A：复数矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * B：结果矩阵 B
 * ldb：B 矩阵的列数
 */

/*
 * BLASNAME(cblas_chemm)函数：
 * 实现复数 Hermite 矩阵的矩阵乘法，计算 C := alpha * A * B + beta * C 或者 C := alpha * B * A + beta * C，依赖于 Side 参数
 * Order：矩阵存储顺序
 * Side：指定 A 出现在 B 的左侧还是右侧
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * M：矩阵 C 的行数
 * N：矩阵 C 的列数
 * alpha：复数标量，用于乘法操作
 * A：复数 Hermite 矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * B：复数矩阵 B
 * ldb：B 矩阵的列数
 * beta：复数标量，用于乘法操作
 * C：结果矩阵 C
 * ldc：C 矩阵的列数
 */

/*
 * BLASNAME(cblas_cherk)函数：
 * 实现复数 Hermite 矩阵的乘积与其转置的乘积的厄米矩阵，计算 C := alpha * A * A^H + beta * C 或者 C := alpha * A^H * A + beta * C，依赖于 Trans 参数
 * Order：矩阵存储顺序
 * Uplo：指定矩阵 A 的存储类型（上三角或下三角）
 * Trans：指定 A 是否进行转置操作
 * N：矩阵 C 的阶数
 * K：矩阵 A 的列数或行数，依赖于 Trans 参数
 * alpha：实数标量，用于乘法操作
 * A：复数矩阵 A
 * lda：A 矩阵的列数（对于 CUBLAS，通常为 A 矩阵的行数）
 * beta：实数标量，用于乘法操作
 * C：结果矩阵 C
 * ldc：C 矩阵的列数
 */
# 定义了一系列使用不同数据类型和参数组合的 BLAS 函数声明

void BLASNAME(cblas_cher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const float beta,
                            void *C, const BLASINT ldc);

# 定义了一系列使用不同数据类型和参数组合的 BLAS 函数声明

void BLASNAME(cblas_zhemm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);

# 定义了一系列使用不同数据类型和参数组合的 BLAS 函数声明

void BLASNAME(cblas_zherk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const void *A, const BLASINT lda,
                           const double beta, void *C, const BLASINT ldc);

# 定义了一系列使用不同数据类型和参数组合的 BLAS 函数声明

void BLASNAME(cblas_zher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const double beta,
                            void *C, const BLASINT ldc);

# 定义了一个函数声明，用于在错误情况下产生错误消息
void BLASNAME(cblas_xerbla)(BLASINT p, const char *rout, const char *form, ...);

# 结束条件编译指令，用于确保在包含该头文件时只包含一次
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_ */
```