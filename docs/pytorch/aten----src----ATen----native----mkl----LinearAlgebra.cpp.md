# `.\pytorch\aten\src\ATen\native\mkl\LinearAlgebra.cpp`

```
// 定义宏 TORCH_ASSERT_NO_OPERATORS，并包含 MKL 线性代数头文件
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/mkl/LinearAlgebra.h>
// 包含 ATen 配置头文件
#include <ATen/Config.h>

// 如果未启用 MKL 支持，定义相关函数在 ATen 命名空间内
#if !AT_MKL_ENABLED()

namespace at { namespace native {

// 实现针对 float 类型的 mkl_gemm_batched 函数
void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha,
    const float** A, const MKL_INT lda, const float** B, const MKL_INT ldb, const float beta,
    float** C, const MKL_INT ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

// 实现针对 double 类型的 mkl_gemm_batched 函数
void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha,
    const double** A, const MKL_INT lda, const double** B, const MKL_INT ldb, const double beta,
    double** C, const MKL_INT ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

// 实现针对复数 float 类型的 mkl_gemm_batched 函数
void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const MKL_INT lda, const c10::complex<float>** B, const MKL_INT ldb,
    const c10::complex<float> beta, c10::complex<float>** C, const MKL_INT ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

// 实现针对复数 double 类型的 mkl_gemm_batched 函数
void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const MKL_INT lda, const c10::complex<double>** B, const MKL_INT ldb,
    const c10::complex<double> beta, c10::complex<double>** C, const MKL_INT ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

// 实现针对 BFloat16 与 float 类型的 mkl_gemm_bf16bf16f32 函数
void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K, const float alpha,
    const c10::BFloat16* A, MKL_INT lda, const c10::BFloat16* B, MKL_INT ldb,
    const float beta, float* C, MKL_INT ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_bf16bf16f32: ATen not compiled with MKL support");
}

// 实现针对 Half 与 float 类型的 mkl_gemm_f16f16f32 函数
void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::Half* A, int lda, const c10::Half* B, int ldb,
    const float beta, float* C, int ldc) {
  // 引发断言错误，指示 ATen 未使用 MKL 支持
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_f16f16f32: ATen not compiled with MKL support");
}

}}

// 结束 AT_MKL_ENABLED 的条件编译块

#else // AT_MKL_ENABLED

// 启用 MKL 支持，包含 MKL 头文件和 c10 工具中的 irange.h
#include <mkl.h>
#include <c10/util/irange.h>

namespace at { namespace native {

// 将 TransposeType 转换为 CBLAS_TRANSPOSE 类型的静态函数
static CBLAS_TRANSPOSE to_cblas(TransposeType x) {
  switch (x) {
    // 如果 x 为 TransposeType::NoTranspose，返回 CblasNoTrans
    case TransposeType::NoTranspose: return CblasNoTrans;
    // 如果 x 为 TransposeType::Transpose，返回 CblasTrans
    case TransposeType::Transpose: return CblasTrans;
    // 默认情况下返回 CblasNoTrans
    default: return CblasNoTrans;
  }
}

// 结束 to_cblas 的定义

// 结束 AT_MKL_ENABLED 的条件编译块
    case TransposeType::ConjTranspose: return CblasConjTrans;


    // 如果 TransposeType 是 ConjTranspose，则返回 CblasConjTrans
    // 这里是一个 switch 语句的分支，根据 TransposeType 的不同选择不同的返回值
    return CblasConjTrans;
  }
  // 如果程序运行到这里，说明 TransposeType 不是预期的任何已知值
  TORCH_INTERNAL_ASSERT(false, "Unknown TransposeType");


    // 如果程序执行到这里，表示出现了未知的 TransposeType 类型
    // 通过 TORCH_INTERNAL_ASSERT 断言来报告错误，输出错误信息 "Unknown TransposeType"
void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha,
    const float** A, const MKL_INT lda, const float** B, const MKL_INT ldb, const float beta,
    float** C, const MKL_INT ldc) {
  // 将 TransposeType 转换为 CBLAS 需要的格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 SGEMM 批处理函数，执行矩阵乘法操作
  cblas_sgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha,
    const double** A, const MKL_INT lda, const double** B, const MKL_INT ldb, const double beta,
    double** C, const MKL_INT ldc) {
  // 将 TransposeType 转换为 CBLAS 需要的格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 DGEMM 批处理函数，执行矩阵乘法操作
  cblas_dgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const MKL_INT lda, const c10::complex<float>** B, const MKL_INT ldb,
    const c10::complex<float> beta, c10::complex<float>** C, const MKL_INT ldc) {
  // 将 TransposeType 转换为 CBLAS 需要的格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 CGEMM 批处理函数，执行复数矩阵乘法操作
  cblas_cgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const MKL_INT lda, const c10::complex<double>** B, const MKL_INT ldb,
    const c10::complex<double> beta, c10::complex<double>** C, const MKL_INT ldc) {
  // 将 TransposeType 转换为 CBLAS 需要的格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 ZGEMM 批处理函数，执行双精度复数矩阵乘法操作
  cblas_zgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K, const float alpha,
    const c10::BFloat16* A, MKL_INT lda, const c10::BFloat16* B, MKL_INT ldb,
    // 调用 BF16 乘法的函数
    MKL_INT ldc, float* C, const float beta) {
  // 调用 MKL 的 BF16 乘法函数，执行 BF16 格式的矩阵乘法操作
  mkl_bf16_gemm(trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, C, ldc, beta);
}
    const float beta, float* C, MKL_INT ldc) {
#ifdef MKL_HAS_SBGEMM
  // 将转置类型转换为 CBLAS 格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 MKL 提供的 bf16bf16f32 GEMM 函数，执行矩阵乘法运算
  cblas_gemm_bf16bf16f32(CblasColMajor, transa_cblas, transb_cblas, M, N, K, alpha,
                         (const MKL_BF16*)A, lda, (const MKL_BF16*)B, ldb, beta, C, ldc);
#else
  // 如果不支持 MKL 的 bf16bf16f32 GEMM 函数，则抛出错误
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_bf16bf16f32 requires mkl version > 2021.0");
#endif
}

void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::Half* A, int lda, const c10::Half* B, int ldb,
    const float beta, float* C, int ldc) {
#ifdef MKL_HAS_SHGEMM
  // 将转置类型转换为 CBLAS 格式
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  // 调用 MKL 提供的 f16f16f32 GEMM 函数，执行矩阵乘法运算
  cblas_gemm_f16f16f32(CblasColMajor, transa_cblas, transb_cblas, M, N, K, alpha,
                         (const MKL_F16*)A, lda, (const MKL_F16*)B, ldb, beta, C, ldc);
#else
  // 如果不支持 MKL 的 f16f16f32 GEMM 函数，则抛出错误
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_f16f16f32 requires mkl version >= 2024.0");
#endif
}

}} // namespace at::native

#endif
```