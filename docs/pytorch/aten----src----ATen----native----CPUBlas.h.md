# `.\pytorch\aten\src\ATen\native\CPUBlas.h`

```py
#pragma once
// 声明了一个预处理指令，确保头文件只被编译一次

#include <ATen/OpMathType.h>
// 包含了 ATen 库中的 OpMathType 头文件，用于定义操作的数学类型

#include <ATen/native/DispatchStub.h>
// 包含了 ATen 库中的 DispatchStub 头文件，用于分发函数调用的机制

#include <ATen/native/TransposeType.h>
// 包含了 ATen 库中的 TransposeType 头文件，用于定义矩阵转置的类型

#include <c10/util/complex.h>
// 包含了 c10 库中的 complex.h 头文件，用于复数的处理

#include <c10/core/ScalarType.h>
// 包含了 c10 库中的 ScalarType 头文件，定义了标量的数据类型

#include <c10/core/Scalar.h>
// 包含了 c10 库中的 Scalar 头文件，定义了标量类型

namespace at::native::cpublas {

namespace internal {
// 在 at::native::cpublas 命名空间中声明了一个内部命名空间 internal

void normalize_last_dims(
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  int64_t *lda, int64_t *ldb, int64_t *ldc);
// 声明了一个 normalize_last_dims 函数，用于规范化最后的维度信息
}  // namespace internal

using gemm_fn = void(*)(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);
// 定义了一个 gemm_fn 类型别名，代表一个函数指针类型，用于执行矩阵乘法

DECLARE_DISPATCH(gemm_fn, gemm_stub);
// 使用宏 DECLARE_DISPATCH 声明了一个 gemm_stub 函数，用于分发 gemm_fn 类型的函数

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    at::opmath_type<scalar_t> beta,
    scalar_t *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 调用 internal 命名空间中的 normalize_last_dims 函数，规范化维度信息

  gemm_stub(
    kCPU, c10::CppTypeToScalarType<scalar_t>::value,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  // 调用 gemm_stub 函数指针，执行 gemm 矩阵乘法的分发操作
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    double beta,
    double *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行双精度浮点数的矩阵乘法

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行单精度浮点数的矩阵乘法

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行 BF16 浮点数的矩阵乘法

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行 BF16 浮点数的矩阵乘法，并将结果存储为单精度浮点数

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行 FP16 浮点数的矩阵乘法

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行 FP16 浮点数的矩阵乘法，并将结果存储为单精度浮点数

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc);
// 声明了一个 gemm 函数重载，用于执行双精度复数的矩阵乘法
// 声明一个函数 gemm，用于执行矩阵乘法操作，支持复数类型 c10::complex<float>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc);

// 声明一个函数 gemm，用于执行矩阵乘法操作，支持整数类型 int64_t
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    int64_t beta,
    int64_t *c, int64_t ldc);

// 声明一个模板函数 gemm_batched，用于执行批量矩阵乘法操作，支持不同类型的 scalar_t
template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t * const *a, int64_t lda,
    const scalar_t * const *b, int64_t ldb,
    const scalar_t beta,
    scalar_t * const *c, int64_t ldc);

// 声明一个模板函数 gemm_batched_with_stride，用于执行带步幅的批量矩阵乘法操作，支持不同类型的 scalar_t
template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c);

// 声明一个类型别名 axpy_fn，表示函数指针，用于定义 AXPY 操作的接口
using axpy_fn = void(*)(at::ScalarType type, int64_t n, const Scalar& a, const void *x, int64_t incx, void *y, int64_t incy);

// 声明一个 DISPATCH 宏，将指定的函数类型 axpy_fn 映射到具体的 axpy_stub 函数
DECLARE_DISPATCH(axpy_fn, axpy_stub);

// 声明一个模板函数 axpy，用于执行 AXPY 操作，支持不同类型的 scalar_t
template<typename scalar_t>
void axpy(int64_t n, scalar_t a, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy){
  // 如果 n 为 1，强制设置增量 incx 和 incy 为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 调用 axpy_stub 函数执行具体的 AXPY 操作
  axpy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, a, x, incx, y, incy);
}

// 声明多个重载函数 axpy，支持不同类型的 scalar_t
void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy);
void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy);
void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

// 声明一个类型别名 copy_fn，表示函数指针，用于定义复制操作的接口
using copy_fn = void(*)(at::ScalarType type, int64_t n, const void *x, int64_t incx, void *y, int64_t incy);

// 声明一个 DISPATCH 宏，将指定的函数类型 copy_fn 映射到具体的 copy_stub 函数
DECLARE_DISPATCH(copy_fn, copy_stub);

// 声明一个模板函数 copy，用于执行数据复制操作，支持不同类型的 scalar_t
template<typename scalar_t>
void copy(int64_t n, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy) {
  // 如果 n 为 1，强制设置增量 incx 和 incy 为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 调用 copy_stub 函数执行具体的复制操作
  copy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, x, incx, y, incy);
}

// 声明多个重载函数 copy，支持不同类型的 scalar_t
void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy);
void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy);
void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

}  // namespace at::native::cpublas
```