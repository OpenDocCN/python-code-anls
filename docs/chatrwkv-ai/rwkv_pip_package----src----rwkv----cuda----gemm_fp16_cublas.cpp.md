# `ChatRWKV\rwkv_pip_package\src\rwkv\cuda\gemm_fp16_cublas.cpp`

```
// 包含 CUDA 相关的头文件
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

// 定义宏，用于检查 cuBLAS 函数调用是否成功
#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _cublas_check_status = (condition);                      \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                         \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_cublas_check_status) + " at " +   \
                             std::to_string(__LINE__));

// 定义宏，用于检查 CUDA 函数调用是否成功
#define CUDA_CHECK(condition)                                                  \
  for (cudaError_t _cuda_check_status = (condition);                           \
       _cuda_check_status != cudaSuccess;)                                     \
    throw std::runtime_error(                                                  \
        "CUDA error " + std::string(cudaGetErrorString(_cuda_check_status)) +  \
        " at " + std::to_string(__LINE__));

/*
  注意：blas gemm 默认是列主序的，但我们需要行主序的输出。
  行主序、转置矩阵的数据与列主序、非转置矩阵的数据完全相同，且 C = A * B ---> C^T = B^T * A^T
 */
void gemm_fp16_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // 设置当前设备为张量 a 所在的设备
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  // 定义 CUDA 数据类型为半精度浮点数
  const auto cuda_data_type = CUDA_R_16F;
  // 如果张量 c 的数据类型为 32 位浮点数，则设置 CUDA 计算数据类型为 32 位浮点数，否则为半精度浮点数
  const auto cuda_c_data_type =
      c.dtype() == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
  // 定义计算类型为 32 位浮点数
  const auto compute_type = CUDA_R_32F;
  // 设置 alpha 值为 1.0
  const float sp_alpha = 1.f;
  // 交换张量 a 和 b，并使用 CUBLAS_OP_N 进行操作
  std::swap(a, b);
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  // 根据张量的维度计算矩阵乘法的参数
  const int m = a.size(-1);
  const int k = a.size(-2);
  const int n = b.size(-2);
  const int cublas_lda = m;
  const int cublas_ldb = k;
  const int cublas_ldc = m;
  // 获取当前 CUDA 上下文的 CUBLAS 句柄
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

  // 根据 CUDA 版本选择不同的算法
#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
  // 设置 beta 值为 0.0
  const float sp_beta = 0.f;
  // 如果张量 a 和 b 的维度均为 2，则执行矩阵乘法
  if (a.sizes().size() == 2 && b.sizes().size() == 2) {
    // 调用 CUBLAS 进行矩阵乘法运算
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, m, n, k, &sp_alpha,
        a.data_ptr(), cuda_data_type, cublas_lda, b.data_ptr(), cuda_data_type,
        cublas_ldb, &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc,
        compute_type, algo));
  } else {
    // 如果张量 a 和 b 的维度均为 3，则执行批量矩阵乘法
    assert(a.sizes().size() == 3 && b.sizes().size() == 3);

    // 计算批量矩阵乘法的步长
    const long long int cublas_stride_a = m * k;
    const long long int cublas_stride_b = k * n;
    const long long int cublas_stride_c = m * n;
    # 使用CUBLAS库执行批量矩阵乘法运算，并检查运行结果
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        # cublas_handle: CUBLAS库的句柄
        cublas_handle, 
        # cublas_trans_a: 矩阵a的转置选项
        cublas_trans_a, 
        # cublas_trans_b: 矩阵b的转置选项
        cublas_trans_b, 
        # m: 矩阵c的行数
        m,
        # n: 矩阵c的列数
        n, 
        # k: 矩阵a的列数/矩阵b的行数
        k, 
        # &sp_alpha: 矩阵a的缩放因子
        &sp_alpha, 
        # a.data_ptr(): 矩阵a的数据指针
        a.data_ptr(), 
        # cuda_data_type: 数据类型
        cuda_data_type, 
        # cublas_lda: 矩阵a的leading dimension
        cublas_lda,
        # cublas_stride_a: 矩阵a的步长
        cublas_stride_a, 
        # b.data_ptr(): 矩阵b的数据指针
        b.data_ptr(), 
        # cuda_data_type: 数据类型
        cuda_data_type, 
        # cublas_ldb: 矩阵b的leading dimension
        cublas_ldb, 
        # cublas_stride_b: 矩阵b的步长
        cublas_stride_b,
        # &sp_beta: 矩阵c的缩放因子
        &sp_beta, 
        # c.data_ptr(): 矩阵c的数据指针
        c.data_ptr(), 
        # cuda_c_data_type: 数据类型
        cuda_c_data_type, 
        # cublas_ldc: 矩阵c的leading dimension
        cublas_ldc, 
        # cublas_stride_c: 矩阵c的步长
        cublas_stride_c,
        # a.size(0): 批量矩阵乘法的批大小
        a.size(0), 
        # compute_type: 计算类型
        compute_type, 
        # algo: 算法选项
        algo));
  }
# 闭合前面的函数定义
```