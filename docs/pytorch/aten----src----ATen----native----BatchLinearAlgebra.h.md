# `.\pytorch\aten\src\ATen\native\BatchLinearAlgebra.h`

```
#pragma once

#include <c10/util/Optional.h>         // 包含 c10 库中的 Optional 类型定义
#include <c10/util/string_view.h>      // 包含 c10 库中的 string_view 类型定义
#include <ATen/Config.h>               // 包含 ATen 配置信息
#include <ATen/native/DispatchStub.h>  // 包含 ATen 中的 DispatchStub 前向声明

// 前向声明 TI
namespace at {
class Tensor;
struct TensorIterator;

namespace native {
enum class TransposeType;            // 声明枚举类型 TransposeType
}

}

namespace at::native {

enum class LapackLstsqDriverType : int64_t { Gels, Gelsd, Gelsy, Gelss};  // 声明 LapackLstsqDriverType 枚举类型，包含几种 LAPACK 求解方法

#if AT_BUILD_WITH_LAPACK()
// Define per-batch functions to be used in the implementation of batched
// linear algebra operations

template <class scalar_t>
void lapackCholesky(char uplo, int n, scalar_t *a, int lda, int *info);  // 声明 LAPACK 中的 Cholesky 分解函数模板

template <class scalar_t>
void lapackCholeskyInverse(char uplo, int n, scalar_t *a, int lda, int *info);  // 声明 LAPACK 中的 Cholesky 逆函数模板

template <class scalar_t, class value_t=scalar_t>
void lapackEig(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *w, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, value_t *rwork, int *info);  // 声明 LAPACK 中的特征值计算函数模板

template <class scalar_t>
void lapackGeqrf(int m, int n, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);  // 声明 LAPACK 中的 QR 分解函数模板

template <class scalar_t>
void lapackOrgqr(int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);  // 声明 LAPACK 中的 Q 矩阵构造函数模板

template <class scalar_t>
void lapackOrmqr(char side, char trans, int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc, scalar_t *work, int lwork, int *info);  // 声明 LAPACK 中的 Q 矩阵乘法函数模板

template <class scalar_t, class value_t = scalar_t>
void lapackSyevd(char jobz, char uplo, int n, scalar_t* a, int lda, value_t* w, scalar_t* work, int lwork, value_t* rwork, int lrwork, int* iwork, int liwork, int* info);  // 声明 LAPACK 中的对称特征值计算函数模板

template <class scalar_t>
void lapackGels(char trans, int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    scalar_t *work, int lwork, int *info);  // 声明 LAPACK 中的最小二乘解函数模板

template <class scalar_t, class value_t = scalar_t>
void lapackGelsd(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    value_t *s, value_t rcond, int *rank,
    scalar_t* work, int lwork,
    value_t *rwork, int* iwork, int *info);  // 声明 LAPACK 中的最小二乘解函数模板（带 SVD）

template <class scalar_t, class value_t = scalar_t>
void lapackGelsy(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    int *jpvt, value_t rcond, int *rank,
    scalar_t *work, int lwork, value_t* rwork, int *info);  // 声明 LAPACK 中的最小二乘解函数模板（带选主元）

template <class scalar_t, class value_t = scalar_t>
void lapackGelss(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    value_t *s, value_t rcond, int *rank,
    scalar_t *work, int lwork,
    value_t *rwork, int *info);  // 声明 LAPACK 中的最小二乘解函数模板（带奇异值分解）

template <LapackLstsqDriverType, class scalar_t, class value_t = scalar_t>
struct lapackLstsq_impl;  // 声明 LAPACK 最小二乘解实现结构模板

template <class scalar_t, class value_t>
// 实现 lapackLstsq_impl 结构体模板，使用 LapackLstsqDriverType::Gels 的具体实现
template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gels, scalar_t, value_t> {
  // 定义静态方法 call，用于调用 lapackGels 函数执行最小二乘解法
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    // 调用 lapackGels 函数，传递参数执行最小二乘解法
    lapackGels<scalar_t>(
        trans, m, n, nrhs,
        a, lda, b, ldb,
        work, lwork, info);
  }
};

// 实现 lapackLstsq_impl 结构体模板，使用 LapackLstsqDriverType::Gelsy 的具体实现
template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelsy, scalar_t, value_t> {
  // 定义静态方法 call，用于调用 lapackGelsy 函数执行最小二乘解法
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    // 调用 lapackGelsy 函数，传递参数执行最小二乘解法
    lapackGelsy<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        jpvt, rcond, rank,
        work, lwork, rwork, info);
  }
};

// 实现 lapackLstsq_impl 结构体模板，使用 LapackLstsqDriverType::Gelsd 的具体实现
template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelsd, scalar_t, value_t> {
  // 定义静态方法 call，用于调用 lapackGelsd 函数执行最小二乘解法
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    // 调用 lapackGelsd 函数，传递参数执行最小二乘解法
    lapackGelsd<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        s, rcond, rank,
        work, lwork,
        rwork, iwork, info);
  }
};

// 实现 lapackLstsq_impl 结构体模板，使用 LapackLstsqDriverType::Gelss 的具体实现
template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelss, scalar_t, value_t> {
  // 定义静态方法 call，用于调用 lapackGelss 函数执行最小二乘解法
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    // 调用 lapackGelss 函数，传递参数执行最小二乘解法
    lapackGelss<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        s, rcond, rank,
        work, lwork,
        rwork, info);
  }
};

// 实现 lapackLstsq 函数模板，根据 driver_type 调用相应的 lapackLstsq_impl::call 方法
template <LapackLstsqDriverType driver_type, class scalar_t, class value_t = scalar_t>
void lapackLstsq(
    char trans, int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    scalar_t *work, int lwork, int *info, // Gels flavor
    int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
    value_t *s, // Gelss flavor
    int *iwork // Gelsd flavor
    ) {
  // 调用 lapackLstsq_impl 结构体模板中的 call 方法，实现最小二乘解法
  lapackLstsq_impl<driver_type, scalar_t, value_t>::call(
      trans, m, n, nrhs,
      a, lda, b, ldb,
      work, lwork, info,
      jpvt, rcond, rank, rwork,
      s,
      iwork);
}
// Solve a linear system using LAPACK LU decomposition and forward/backward substitution
void lapackLuSolve(char trans, int n, int nrhs, scalar_t *a, int lda, int *ipiv, scalar_t *b, int ldb, int *info);

// Perform LU factorization of a matrix using LAPACK
template <class scalar_t>
void lapackLu(int m, int n, scalar_t *a, int lda, int *ipiv, int *info);

// Perform LDL^H (Hermitian) factorization of a matrix using LAPACK
template <class scalar_t>
void lapackLdlHermitian(
    char uplo,
    int n,
    scalar_t* a,
    int lda,
    int* ipiv,
    scalar_t* work,
    int lwork,
    int* info);

// Perform LDL (Symmetric) factorization of a matrix using LAPACK
template <class scalar_t>
void lapackLdlSymmetric(
    char uplo,
    int n,
    scalar_t* a,
    int lda,
    int* ipiv,
    scalar_t* work,
    int lwork,
    int* info);

// Solve a linear system using LDL^H (Hermitian) factorization using LAPACK
template <class scalar_t>
void lapackLdlSolveHermitian(
    char uplo,
    int n,
    int nrhs,
    scalar_t* a,
    int lda,
    int* ipiv,
    scalar_t* b,
    int ldb,
    int* info);

// Solve a linear system using LDL (Symmetric) factorization using LAPACK
template <class scalar_t>
void lapackLdlSolveSymmetric(
    char uplo,
    int n,
    int nrhs,
    scalar_t* a,
    int lda,
    int* ipiv,
    scalar_t* b,
    int ldb,
    int* info);

// Perform Singular Value Decomposition (SVD) using LAPACK
template<class scalar_t, class value_t=scalar_t>
void lapackSvd(char jobz, int m, int n, scalar_t *a, int lda, value_t *s, scalar_t *u, int ldu, scalar_t *vt, int ldvt, scalar_t *work, int lwork, value_t *rwork, int *iwork, int *info);

#if AT_BUILD_WITH_BLAS()
// Solve a triangular linear system using BLAS
template <class scalar_t>
void blasTriangularSolve(char side, char uplo, char trans, char diag, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb);
#endif

// Function pointer type for Cholesky decomposition operation
using cholesky_fn = void (*)(const Tensor& /*input*/, const Tensor& /*info*/, bool /*upper*/);
DECLARE_DISPATCH(cholesky_fn, cholesky_stub);

// Function pointer type for Cholesky inverse operation
using cholesky_inverse_fn = Tensor& (*)(Tensor& /*result*/, Tensor& /*infos*/, bool /*upper*/);
DECLARE_DISPATCH(cholesky_inverse_fn, cholesky_inverse_stub);

// Function pointer type for Eigenvalue decomposition operation
using linalg_eig_fn = void (*)(Tensor& /*eigenvalues*/, Tensor& /*eigenvectors*/, Tensor& /*infos*/, const Tensor& /*input*/, bool /*compute_eigenvectors*/);
DECLARE_DISPATCH(linalg_eig_fn, linalg_eig_stub);

// Function pointer type for QR factorization using LAPACK
using geqrf_fn = void (*)(const Tensor& /*input*/, const Tensor& /*tau*/);
DECLARE_DISPATCH(geqrf_fn, geqrf_stub);

// Function pointer type for constructing Q from elementary reflectors using LAPACK
using orgqr_fn = Tensor& (*)(Tensor& /*result*/, const Tensor& /*tau*/);
DECLARE_DISPATCH(orgqr_fn, orgqr_stub);

// Function pointer type for applying Q to a matrix using LAPACK
using ormqr_fn = void (*)(const Tensor& /*input*/, const Tensor& /*tau*/, const Tensor& /*other*/, bool /*left*/, bool /*transpose*/);
DECLARE_DISPATCH(ormqr_fn, ormqr_stub);

// Function pointer type for Eigenvalue decomposition of a Hermitian matrix using LAPACK
using linalg_eigh_fn = void (*)(
    const Tensor& /*eigenvalues*/,
    const Tensor& /*eigenvectors*/,
    const Tensor& /*infos*/,
    bool /*upper*/,
    bool /*compute_eigenvectors*/);
DECLARE_DISPATCH(linalg_eigh_fn, linalg_eigh_stub);

// Function pointer type for solving least squares problem using LAPACK
using lstsq_fn = void (*)(
    const Tensor& /*a*/,
    Tensor& /*b*/,
    Tensor& /*rank*/,
    Tensor& /*singular_values*/,
    Tensor& /*infos*/,
    double /*rcond*/,
    std::string /*driver_name*/);
DECLARE_DISPATCH(lstsq_fn, lstsq_stub);

// Function pointer type for solving triangular systems of equations
using triangular_solve_fn = void (*)(
    const Tensor& /*A*/,
    const Tensor& /*B*/,
    bool /*left*/,
    bool /*upper*/,
    TransposeType /*transpose*/,
    bool /*unitriangular*/);
// 声明一个用于求解三角线性方程组的函数指针 triangular_solve_fn，由具体的实现函数 triangular_solve_stub 提供
DECLARE_DISPATCH(triangular_solve_fn, triangular_solve_stub);

// 使用 lu_factor_fn 别名声明一个函数指针，该函数用于对输入张量进行 LU 分解，需要额外的置换矩阵和信息张量来存储中间结果
DECLARE_DISPATCH(lu_factor_fn, lu_factor_stub);

// 使用 unpack_pivots_fn 别名声明一个函数指针，该函数用于解包置换矩阵，并在张量迭代器上执行操作
DECLARE_DISPATCH(unpack_pivots_fn, unpack_pivots_stub);

// 使用 lu_solve_fn 别名声明一个函数指针，该函数用于解线性方程组 LUx = B，其中 LU 是 LU 分解后的结果，pivots 是置换矩阵，B 是右侧向量，trans 是转置类型
DECLARE_DISPATCH(lu_solve_fn, lu_solve_stub);

// 使用 ldl_factor_fn 别名声明一个函数指针，该函数用于对输入张量进行 LDL 分解，需要额外的置换矩阵和信息张量来存储中间结果，upper 和 hermitian 控制分解类型
DECLARE_DISPATCH(ldl_factor_fn, ldl_factor_stub);

// 使用 svd_fn 别名声明一个函数指针，该函数用于计算奇异值分解 (SVD)，full_matrices 控制是否计算完整的 U 和 V 矩阵，compute_uv 控制是否计算 U 和 V，driver 是驱动类型，U、S、Vh 和 info 是存储结果的张量
DECLARE_DISPATCH(svd_fn, svd_stub);

// 使用 ldl_solve_fn 别名声明一个函数指针，该函数用于解线性方程组 LDLx = B，其中 LDL 是 LDL 分解后的结果，pivots 是置换矩阵，result 是输出向量，upper 和 hermitian 控制分解类型
DECLARE_DISPATCH(ldl_solve_fn, ldl_solve_stub);
```