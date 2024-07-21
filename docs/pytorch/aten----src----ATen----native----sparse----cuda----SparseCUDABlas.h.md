# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseCUDABlas.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/cuda/ATenCUDAGeneral.h>
// 包含 ATen CUDA 模块通用头文件

namespace at::native::sparse::cuda{

TORCH_CUDA_CU_API void Xcoo2csr(
    const int* coorowind,
    int64_t nnz,
    int64_t m,
    int* csrrowptr);
// 定义一个 CUDA API 函数 Xcoo2csr，用于将 COO 格式的稀疏矩阵转换为 CSR 格式

/* Level 3 */
template <typename T>
TORCH_CUDA_CU_API void csrmm2(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t nnz,
    T alpha,
    T* csrvala,
    int* csrrowptra,
    int* csrcolinda,
    T* b,
    int64_t ldb,
    T beta,
    T* c,
    int64_t ldc);
// 定义一个 CUDA API 模板函数 csrmm2，执行 CSR 格式的稀疏矩阵乘法运算

/* format conversion */
TORCH_CUDA_CU_API void CreateIdentityPermutation(int64_t nnz, int* P);
// 定义一个 CUDA API 函数 CreateIdentityPermutation，用于创建长度为 nnz 的单位置换数组

TORCH_CUDA_CU_API void Xcsrsort_bufferSizeExt(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* csrRowPtr,
    const int* csrColInd,
    size_t* pBufferSizeInBytes);
// 定义一个 CUDA API 函数 Xcsrsort_bufferSizeExt，计算执行 CSR 格式排序所需的缓冲区大小

TORCH_CUDA_CU_API void Xcsrsort(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* csrRowPtr,
    int* csrColInd,
    int* P,
    void* pBuffer);
// 定义一个 CUDA API 函数 Xcsrsort，执行 CSR 格式排序

TORCH_CUDA_CU_API void Xcoosort_bufferSizeExt(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* cooRows,
    const int* cooCols,
    size_t* pBufferSizeInBytes);
// 定义一个 CUDA API 函数 Xcoosort_bufferSizeExt，计算执行 COO 格式排序所需的缓冲区大小

TORCH_CUDA_CU_API void XcoosortByRow(
    int64_t m,
    int64_t n,
    int64_t nnz,
    int* cooRows,
    int* cooCols,
    int* P,
    void* pBuffer);
// 定义一个 CUDA API 函数 XcoosortByRow，按行执行 COO 格式排序

} // namespace at::native::sparse::cuda
// 结束 at::native::sparse::cuda 命名空间
```