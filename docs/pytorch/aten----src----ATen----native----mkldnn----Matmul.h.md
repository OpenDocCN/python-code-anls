# `.\pytorch\aten\src\ATen\native\mkldnn\Matmul.h`

```
#pragma once
// 预处理指令，确保本头文件只被包含一次

#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 类定义

#include <ATen/Config.h>
// 引入 ATen 库的配置信息

#include <ATen/native/LinearAlgebraUtils.h>
// 引入 ATen 库中的线性代数工具，用于 TransposeType

namespace at { namespace native {

// 声明 mkldnn_matmul 函数，用于执行 MKL-DNN 优化的矩阵乘法
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);
// 参数 mat1, mat2: 输入张量
// 参数 result: 输出张量
// 参数 beta: 结果张量的缩放因子，默认为 1
// 参数 alpha: 输入张量的缩放因子，默认为 1

// 检查是否可以使用 MKL-DNN 优化的 BF16 矩阵乘法
bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);
// 参数 mat1, mat2: 输入张量
// 参数 result_opt: 输出张量的优化版本

// 检查是否可以使用 MKL-DNN 优化的 FP16 矩阵乘法
bool use_mkldnn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);
// 参数 mat1, mat2: 输入张量
// 参数 result_opt: 输出张量的优化版本

// 检查是否可以使用 MKL-DNN 优化的 BF32 矩阵乘法
bool use_mkldnn_bf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);
// 参数 mat1, mat2: 输入张量
// 参数 result_opt: 输出张量的优化版本

// 尝试运行 MKL-DNN 优化的 GEMM，或者如果原始 GEMM 更快则返回 false
bool mkldnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc);
// 参数 transa, transb: 矩阵 A 和 B 的转置类型
// 参数 m, n, k: 矩阵尺寸
// 参数 alpha: 输入张量的缩放因子
// 参数 a, b, c: 输入和输出张量的指针
// 参数 lda, ldb, ldc: 矩阵 A, B, C 的 leading dimension

// 尝试运行 MKL-DNN 优化的 FP16 GEMM
bool mkldnn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc);
// 参数和注释同上，不同之处在于操作的数据类型为 FP16

// 尝试运行 MKL-DNN 优化的 BF32 GEMM
bool mkldnn_bf32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);
// 参数和注释同上，不同之处在于操作的数据类型为 BF32

// 检查是否可以使用 MKL-DNN 优化的矩阵乘法
bool use_mkldnn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result);
// 参数和注释同 use_mkldnn_bf16_matmul，用于通用的 MKL-DNN 矩阵乘法检查

// x:s8 * w:s8 -> y:s32
TORCH_API void mkldnn_matmul_i8i8i32(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result);
// 参数 mat1, mat2: 输入张量，数据类型为 int8
// 参数 result: 输出张量，数据类型为 int32

}

}
```