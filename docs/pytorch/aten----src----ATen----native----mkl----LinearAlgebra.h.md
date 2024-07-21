# `.\pytorch\aten\src\ATen\native\mkl\LinearAlgebra.h`

```
#pragma once
#include <ATen/Config.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>

// 如果未启用 MKL，则定义 MKL_INT 为 int 类型
#if !AT_MKL_ENABLED()
#define MKL_INT int
#else
#include <mkl.h>
#endif

namespace at {
namespace native {

// 声明用于批量矩阵乘法的函数，支持不同数据类型和复数类型
void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,  // 矩阵 A 和 B 的转置类型
    MKL_INT batch_size,  // 批次大小
    MKL_INT M, MKL_INT N, MKL_INT K,  // 矩阵维度
    float alpha,  // 缩放因子 alpha
    const float** A, MKL_INT lda,  // 矩阵 A 数据和其列偏移
    const float** B, MKL_INT ldb,  // 矩阵 B 数据和其列偏移
    float beta,  // 缩放因子 beta
    float** C, MKL_INT ldc);  // 输出矩阵 C 数据和其列偏移

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K,  // 参数同上，但是用于 double 类型
    double alpha,
    const double** A, MKL_INT lda,
    const double** B, MKL_INT ldb,
    double beta,
    double** C, MKL_INT ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K,  // 参数同上，但是用于 c10::complex<float> 类型
    c10::complex<float> alpha,
    const c10::complex<float>** A, MKL_INT lda,
    const c10::complex<float>** B, MKL_INT ldb,
    c10::complex<float> beta,
    c10::complex<float>** C, MKL_INT ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K,  // 参数同上，但是用于 c10::complex<double> 类型
    c10::complex<double> alpha,
    const c10::complex<double>** A, MKL_INT lda,
    const c10::complex<double>** B, MKL_INT ldb,
    c10::complex<double> beta,
    c10::complex<double>** C, MKL_INT ldc);

// 以下两个函数用于不同精度的混合矩阵乘法
void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K,  // 矩阵维度
    const float alpha,  // 缩放因子 alpha
    const c10::BFloat16* A, MKL_INT lda,  // 矩阵 A 数据和其列偏移
    const c10::BFloat16* B, MKL_INT ldb,  // 矩阵 B 数据和其列偏移
    const float beta,  // 缩放因子 beta
    float* C, MKL_INT ldc);  // 输出矩阵 C 数据和其列偏移

void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K,  // 矩阵维度，但是用于 c10::Half 类型
    const float alpha,
    const c10::Half* A, int lda,
    const c10::Half* B, int ldb,
    const float beta,
    float* C, int ldc);
}}  // namespace at::native
```