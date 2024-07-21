# `.\pytorch\aten\src\ATen\cuda\CUDASparseBlas.h`

```
#pragma once

/*
  提供 cuSPARSE 函数的子集作为模板:

    csrgeam2<scalar_t>(...)

  其中 scalar_t 可以是 double、float、c10::complex<double> 或 c10::complex<float>。
  这些函数位于 at::cuda::sparse 命名空间中。
*/

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

namespace at::cuda::sparse {

// 定义 csrgeam2_bufferSizeExt 函数模板，用于计算缓冲区大小
#define CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)             \
  cusparseHandle_t handle, int m, int n, const scalar_t *alpha,     \
      const cusparseMatDescr_t descrA, int nnzA,                    \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,   \
      const int *csrSortedColIndA, const scalar_t *beta,            \
      const cusparseMatDescr_t descrB, int nnzB,                    \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,   \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC, \
      const scalar_t *csrSortedValC, const int *csrSortedRowPtrC,   \
      const int *csrSortedColIndC, size_t *pBufferSizeInBytes

// 实现 csrgeam2_bufferSizeExt 函数模板
template <typename scalar_t>
inline void csrgeam2_bufferSizeExt(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)) {
  // 抛出错误，说明该函数未实现对特定 scalar_t 类型的处理
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::csrgeam2_bufferSizeExt: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化 csrgeam2_bufferSizeExt 函数模板，针对不同的 scalar_t 类型
template <>
void csrgeam2_bufferSizeExt<float>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float));
template <>
void csrgeam2_bufferSizeExt<double>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double));
template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>));

// 定义 csrgeam2Nnz 函数模板，用于计算 CSR 格式矩阵的非零元素个数
#define CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()                                      \
  cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,     \
      int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,     \
      const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB, \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC,           \
      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace

// 实现 csrgeam2Nnz 函数模板
template <typename scalar_t>
inline void csrgeam2Nnz(CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()) {
  // 调用 cuSPARSE 库的 csrgeam2Nnz 函数
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgeam2Nnz(
      handle,
      m,
      n,
      descrA,
      nnzA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      descrB,
      nnzB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedRowPtrC,
      nnzTotalDevHostPtr,
      workspace));
}
#define CUSPARSE_CSRGEAM2_ARGTYPES(scalar_t)                                 \
  cusparseHandle_t handle, int m, int n, const scalar_t *alpha,              \
      const cusparseMatDescr_t descrA, int nnzA,                             \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,            \
      const int *csrSortedColIndA, const scalar_t *beta,                     \
      const cusparseMatDescr_t descrB, int nnzB,                             \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,            \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC,          \
      scalar_t *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, \
      void *pBuffer


// 定义一个宏 CUSPARSE_CSRGEAM2_ARGTYPES，用于生成不同类型 scalar_t 的 csrgeam2 函数参数列表

template <typename scalar_t>
inline void csrgeam2(CUSPARSE_CSRGEAM2_ARGTYPES(scalar_t)) {
  // 断言，如果执行到这里表示未实现 csrgeam2 函数对当前类型 scalar_t 的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::csrgeam2: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数 csrgeam2，支持 float, double, c10::complex<float>, c10::complex<double> 四种类型
template <>
void csrgeam2<float>(CUSPARSE_CSRGEAM2_ARGTYPES(float));

template <>
void csrgeam2<double>(CUSPARSE_CSRGEAM2_ARGTYPES(double));

template <>
void csrgeam2<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>));

template <>
void csrgeam2<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRMM_ARGTYPES(scalar_t)                                    \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, \
      int kb, int nnzb, const scalar_t *alpha,                               \
      const cusparseMatDescr_t descrA, const scalar_t *bsrValA,              \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      const scalar_t *B, int ldb, const scalar_t *beta, scalar_t *C, int ldc


// 定义一个宏 CUSPARSE_BSRMM_ARGTYPES，用于生成不同类型 scalar_t 的 bsrmm 函数参数列表

template <typename scalar_t>
inline void bsrmm(CUSPARSE_BSRMM_ARGTYPES(scalar_t)) {
  // 断言，如果执行到这里表示未实现 bsrmm 函数对当前类型 scalar_t 的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrmm: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数 bsrmm，支持 float, double, c10::complex<float>, c10::complex<double> 四种类型
template <>
void bsrmm<float>(CUSPARSE_BSRMM_ARGTYPES(float));

template <>
void bsrmm<double>(CUSPARSE_BSRMM_ARGTYPES(double));

template <>
void bsrmm<c10::complex<float>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<float>));

template <>
void bsrmm<c10::complex<double>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRMV_ARGTYPES(scalar_t)                                    \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, int mb, int nb, int nnzb,                  \
      const scalar_t *alpha, const cusparseMatDescr_t descrA,                \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, const scalar_t *x, const scalar_t *beta, scalar_t *y


// 定义一个宏 CUSPARSE_BSRMV_ARGTYPES，用于生成不同类型 scalar_t 的 bsrvm 函数参数列表
# 当前函数声明了一个内联函数 bsrmv，用于执行稀疏矩阵向量乘法。
# 该函数使用了 CUDA 的 CUSPARSE 库，但是目前只是抛出一个错误，指示该函数对特定的数据类型尚未实现。
inline void bsrmv(CUSPARSE_BSRMV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrmv: not implemented for ",
      typeid(scalar_t).name());
}

# 以下是特化模板函数 bsrmv<float> 的实现，用于 float 类型的稀疏矩阵向量乘法。
template <>
void bsrmv<float>(CUSPARSE_BSRMV_ARGTYPES(float));

# 以下是特化模板函数 bsrmv<double> 的实现，用于 double 类型的稀疏矩阵向量乘法。
template <>
void bsrmv<double>(CUSPARSE_BSRMV_ARGTYPES(double));

# 以下是特化模板函数 bsrmv<c10::complex<float>> 的实现，用于复数类型 c10::complex<float> 的稀疏矩阵向量乘法。
template <>
void bsrmv<c10::complex<float>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<float>));

# 以下是特化模板函数 bsrmv<c10::complex<double>> 的实现，用于复数类型 c10::complex<double> 的稀疏矩阵向量乘法。
template <>
void bsrmv<c10::complex<double>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<double>));

# 如果使用 HIPSPARSE 的求解功能（AT_USE_HIPSPARSE_TRIANGULAR_SOLVE 宏定义为真）
#define CUSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)                 \
  cusparseHandle_t handle, cusparseDirection_t dirA,              \
      cusparseOperation_t transA, int mb, int nnzb,               \
      const cusparseMatDescr_t descrA, scalar_t *bsrValA,         \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim, \
      bsrsv2Info_t info, int *pBufferSizeInBytes

# 内联函数 bsrsv2_bufferSize 是用于计算 BSR 矩阵求解器所需的缓冲区大小。
# 同样地，它当前只是抛出一个错误，指示该函数对特定的数据类型尚未实现。
template <typename scalar_t>
inline void bsrsv2_bufferSize(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

# 以下是特化模板函数 bsrsv2_bufferSize<float> 的实现，用于 float 类型的 BSR 矩阵求解器缓冲区大小计算。
template <>
void bsrsv2_bufferSize<float>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(float));

# 以下是特化模板函数 bsrsv2_bufferSize<double> 的实现，用于 double 类型的 BSR 矩阵求解器缓冲区大小计算。
template <>
void bsrsv2_bufferSize<double>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(double));

# 以下是特化模板函数 bsrsv2_bufferSize<c10::complex<float>> 的实现，用于复数类型 c10::complex<float> 的 BSR 矩阵求解器缓冲区大小计算。
template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>));

# 以下是特化模板函数 bsrsv2_bufferSize<c10::complex<double>> 的实现，用于复数类型 c10::complex<double> 的 BSR 矩阵求解器缓冲区大小计算。
template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>));

# 如果使用 HIPSPARSE 的求解功能，进入以下预处理代码块

# 定义了模板函数 bsrsv2_analysis，用于执行 BSR 矩阵的解析步骤。
# 目前该函数也只是抛出一个错误，指示该函数对特定的数据类型尚未实现。
template <typename scalar_t>
inline void bsrsv2_analysis(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_analysis: not implemented for ",
      typeid(scalar_t).name());
}

# 以下是特化模板函数 bsrsv2_analysis<float> 的实现，用于 float 类型的 BSR 矩阵解析。
template <>
void bsrsv2_analysis<float>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float));

# 以下是特化模板函数 bsrsv2_analysis<double> 的实现，用于 double 类型的 BSR 矩阵解析。
template <>
void bsrsv2_analysis<double>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double));

# 以下是特化模板函数 bsrsv2_analysis<c10::complex<float>> 的实现，用于复数类型 c10::complex<float> 的 BSR 矩阵解析。
template <>
void bsrsv2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>));

# 以下是特化模板函数 bsrsv2_analysis<c10::complex<double>> 的实现，用于复数类型 c10::complex<double> 的 BSR 矩阵解析。
template <>
void bsrsv2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>));
#define CUSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)                           \
  cusparseHandle_t handle, cusparseDirection_t dirA,                       \
      cusparseOperation_t transA, int mb, int nnzb, const scalar_t *alpha, \
      const cusparseMatDescr_t descrA, const scalar_t *bsrValA,            \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,          \
      bsrsv2Info_t info, const scalar_t *x, scalar_t *y,                   \
      cusparseSolvePolicy_t policy, void *pBuffer


// 定义了用于 bsrsV2_solve 函数的参数类型宏，其中 scalar_t 是模板类型参数
template <typename scalar_t>
inline void bsrsv2_solve(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)) {
  // 抛出内部断言错误，说明函数未实现对指定 scalar_t 类型的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_solve: not implemented for ",
      typeid(scalar_t).name());
}

// 下面四个特化模板函数是为了实现对不同类型（float、double、复数类型等）的支持
template <>
void bsrsv2_solve<float>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(float));

template <>
void bsrsv2_solve<double>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(double));

template <>
void bsrsv2_solve<c10::complex<float>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>));

template <>
void bsrsv2_solve<c10::complex<double>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>));


#define CUSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)                            \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const cusparseMatDescr_t descrA, scalar_t *bsrValA,          \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, int *pBufferSizeInBytes


// 定义了用于 bsrsm2_bufferSize 函数的参数类型宏，其中 scalar_t 是模板类型参数
template <typename scalar_t>
inline void bsrsm2_bufferSize(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)) {
  // 抛出内部断言错误，说明函数未实现对指定 scalar_t 类型的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

// 下面四个特化模板函数是为了实现对不同类型（float、double、复数类型等）的支持
template <>
void bsrsm2_bufferSize<float>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(float));

template <>
void bsrsm2_bufferSize<double>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(double));

template <>
void bsrsm2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<float>));

template <>
void bsrsm2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>));


#define CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)                          \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const cusparseMatDescr_t descrA, const scalar_t *bsrValA,    \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer


// 定义了用于 bsrsm2_analysis 函数的参数类型宏，其中 scalar_t 是模板类型参数
template <typename scalar_t>
inline void bsrsm2_analysis(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)) {
  // 抛出内部断言错误，说明函数未实现对指定 scalar_t 类型的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_analysis: not implemented for ",
      typeid(scalar_t).name());
}
// 声明 CUDA 稀疏矩阵 BSRSM2 分析函数模板，处理 float 类型参数
void bsrsm2_analysis<float>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float));

// 声明 CUDA 稀疏矩阵 BSRSM2 分析函数模板，处理 double 类型参数
template <>
void bsrsm2_analysis<double>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double));

// 声明 CUDA 稀疏矩阵 BSRSM2 分析函数模板，处理复数 float 类型参数
template <>
void bsrsm2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>));

// 声明 CUDA 稀疏矩阵 BSRSM2 分析函数模板，处理复数 double 类型参数
template <>
void bsrsm2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>));

// 定义 CUDA 稀疏矩阵 BSRSM2 求解函数模板的参数宏，处理不同 scalar_t 类型参数
#define CUSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)                             \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const scalar_t *alpha, const cusparseMatDescr_t descrA,      \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, bsrsm2Info_t info, const scalar_t *B, int ldb,           \
      scalar_t *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer

// 定义 CUDA 稀疏矩阵 BSRSM2 求解函数模板，处理 scalar_t 类型参数
template <typename scalar_t>
inline void bsrsm2_solve(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)) {
  // 抛出断言异常，表示函数未实现对当前类型的支持
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_solve: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化 CUDA 稀疏矩阵 BSRSM2 求解函数模板，处理 float 类型参数
template <>
void bsrsm2_solve<float>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(float));

// 显式实例化 CUDA 稀疏矩阵 BSRSM2 求解函数模板，处理 double 类型参数
template <>
void bsrsm2_solve<double>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(double));

// 显式实例化 CUDA 稀疏矩阵 BSRSM2 求解函数模板，处理复数 float 类型参数
template <>
void bsrsm2_solve<c10::complex<float>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>));

// 显式实例化 CUDA 稀疏矩阵 BSRSM2 求解函数模板，处理复数 double 类型参数
template <>
void bsrsm2_solve<c10::complex<double>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>));

// 结束 AT_USE_HIPSPARSE_TRIANGULAR_SOLVE 宏定义区域
#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

// 结束命名空间 at::cuda::sparse
} // namespace at::cuda::sparse
```