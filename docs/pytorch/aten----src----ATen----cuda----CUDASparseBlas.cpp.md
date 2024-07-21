# `.\pytorch\aten\src\ATen\cuda\CUDASparseBlas.cpp`

```
/*
  Provides the implementations of cuSPARSE function templates.
*/

#include <ATen/cuda/CUDASparseBlas.h>

// 定义命名空间：at::cuda::sparse
namespace at::cuda::sparse {

// 模板特化：float 类型的 csrgeam2_bufferSizeExt 函数
template <>
void csrgeam2_bufferSizeExt<float>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float)) {
  // 调用 cuSPARSE 库函数 cusparseScsrgeam2_bufferSizeExt 来计算所需的缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseScsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

// 模板特化：double 类型的 csrgeam2_bufferSizeExt 函数
template <>
void csrgeam2_bufferSizeExt<double>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double)) {
  // 调用 cuSPARSE 库函数 cusparseDcsrgeam2_bufferSizeExt 来计算所需的缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

// 模板特化：c10::complex<float> 类型的 csrgeam2_bufferSizeExt 函数
template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSPARSE 库函数 cusparseCcsrgeam2_bufferSizeExt 来计算所需的缓冲区大小，
  // 复数类型需要使用 cuComplex 数据类型进行转换
  TORCH_CUDASPARSE_CHECK(cusparseCcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const cuComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

// 模板特化：c10::complex<double> 类型的 csrgeam2_bufferSizeExt 函数
template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSPARSE 库函数 cusparseZcsrgeam2_bufferSizeExt 来计算所需的缓冲区大小，
  // 复数类型需要使用 cuDoubleComplex 数据类型进行转换
  TORCH_CUDASPARSE_CHECK(cusparseZcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

// 模板特化结束
// 定义模板函数 csrgeam2，处理 float 类型的稀疏矩阵加法
template <>
void csrgeam2<float>(CUSPARSE_CSRGEAM2_ARGTYPES(float)) {
  // 调用 CUDA 的 cusparseScsrgeam2 函数，执行 CSR 格式的稀疏矩阵加法
  TORCH_CUDASPARSE_CHECK(cusparseScsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

// 定义模板函数 csrgeam2，处理 double 类型的稀疏矩阵加法
template <>
void csrgeam2<double>(CUSPARSE_CSRGEAM2_ARGTYPES(double)) {
  // 调用 CUDA 的 cusparseDcsrgeam2 函数，执行 CSR 格式的稀疏矩阵加法
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

// 定义模板函数 csrgeam2，处理复数类型 c10::complex<float> 的稀疏矩阵加法
template <>
void csrgeam2<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>)) {
  // 调用 CUDA 的 cusparseCcsrgeam2 函数，执行 CSR 格式的复数稀疏矩阵加法
  TORCH_CUDASPARSE_CHECK(cusparseCcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<cuComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

// 定义模板函数 csrgeam2，处理复数类型 c10::complex<double> 的稀疏矩阵加法
template <>
void csrgeam2<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>)) {
  // 调用 CUDA 的 cusparseZcsrgeam2 函数，执行 CSR 格式的双精度复数稀疏矩阵加法
  TORCH_CUDASPARSE_CHECK(cusparseZcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<cuDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

// 定义模板函数 bsrmm，处理 float 类型的块矩阵乘法
template <>
void bsrmm<float>(CUSPARSE_BSRMM_ARGTYPES(float)) {
  // 调用 CUDA 的 cusparseSbsrmm 函数，执行 BSR 格式的块矩阵乘法
  TORCH_CUDASPARSE_CHECK(cusparseSbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}

// 定义模板函数 bsrmm，处理 double 类型的块矩阵乘法
template <>
void bsrmm<double>(CUSPARSE_BSRMM_ARGTYPES(double)) {
  // 调用 CUDA 的 cusparseDbsrmm 函数，执行 BSR 格式的块矩阵乘法
  TORCH_CUDASPARSE_CHECK(cusparseDbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}
template <>
void bsrmm<c10::complex<float>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSPARSE 库执行复杂浮点数的 BSR 稀疏矩阵乘法
  TORCH_CUDASPARSE_CHECK(cusparseCbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuComplex*>(B),
      ldb,
      reinterpret_cast<const cuComplex*>(beta),
      reinterpret_cast<cuComplex*>(C),
      ldc));
}

template <>
void bsrmm<c10::complex<double>>(
    CUSPARSE_BSRMM_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSPARSE 库执行双精度复杂数的 BSR 稀疏矩阵乘法
  TORCH_CUDASPARSE_CHECK(cusparseZbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuDoubleComplex*>(B),
      ldb,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      reinterpret_cast<cuDoubleComplex*>(C),
      ldc));
}

template <>
void bsrmv<float>(CUSPARSE_BSRMV_ARGTYPES(float)) {
  // 调用 cuSPARSE 库执行单精度浮点数的 BSR 稀疏矩阵向量乘法
  TORCH_CUDASPARSE_CHECK(cusparseSbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<double>(CUSPARSE_BSRMV_ARGTYPES(double)) {
  // 调用 cuSPARSE 库执行双精度浮点数的 BSR 稀疏矩阵向量乘法
  TORCH_CUDASPARSE_CHECK(cusparseDbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<c10::complex<float>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSPARSE 库执行复杂浮点数的 BSR 稀疏矩阵向量乘法
  TORCH_CUDASPARSE_CHECK(cusparseCbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuComplex*>(x),
      reinterpret_cast<const cuComplex*>(beta),
      reinterpret_cast<cuComplex*>(y)));
}

template <>
void bsrmv<c10::complex<double>>(
    CUSPARSE_BSRMV_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSPARSE 库执行双精度复杂数的 BSR 稀疏矩阵向量乘法
  TORCH_CUDASPARSE_CHECK(cusparseZbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuDoubleComplex*>(x),
      reinterpret_cast<const cuDoubleComplex*>(beta),
      reinterpret_cast<cuDoubleComplex*>(y)));
}

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

template <>
void bsrsv2_bufferSize<float>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(float)) {
  // 调用 cuSPARSE 库函数 cusparseSbsrsv2_bufferSize，计算 BSR 存储格式求解过程中所需缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_bufferSize(
      handle,               // cuSPARSE 上下文句柄
      dirA,                 // BSR 存储格式中的方向参数
      transA,               // 稀疏矩阵 A 的转置标记
      mb,                   // BSR 矩阵的行数
      nnzb,                 // 非零块数目
      descrA,               // BSR 矩阵的描述符
      bsrValA,              // BSR 块值数组
      bsrRowPtrA,           // BSR 行指针数组
      bsrColIndA,           // BSR 列索引数组
      blockDim,             // BSR 块大小
      info,                 // cuSPARSE 操作信息结构体
      pBufferSizeInBytes)); // 返回所需缓冲区大小的指针
}

template <>
void bsrsv2_bufferSize<double>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(double)) {
  // 同上，使用 double 类型的精度
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_bufferSize(
      handle, dirA, transA, mb, nnzb, descrA, bsrValA, bsrRowPtrA,
      bsrColIndA, blockDim, info, pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>)) {
  // 同上，使用 c10::complex<float> 类型的精度，需要将 bsrvValA 转换为 cuComplex*
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_bufferSize(
      handle, dirA, transA, mb, nnzb, descrA,
      reinterpret_cast<cuComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>)) {
  // 同上，使用 c10::complex<double> 类型的精度，需要将 bsrvValA 转换为 cuDoubleComplex*
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_bufferSize(
      handle, dirA, transA, mb, nnzb, descrA,
      reinterpret_cast<cuDoubleComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, pBufferSizeInBytes));
}

template <>
void bsrsv2_analysis<float>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float)) {
  // 调用 cuSPARSE 库函数 cusparseSbsrsv2_analysis，执行 BSR 矩阵的解析分析
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_analysis(
      handle,               // cuSPARSE 上下文句柄
      dirA,                 // BSR 存储格式中的方向参数
      transA,               // 稀疏矩阵 A 的转置标记
      mb,                   // BSR 矩阵的行数
      nnzb,                 // 非零块数目
      descrA,               // BSR 矩阵的描述符
      bsrValA,              // BSR 块值数组
      bsrRowPtrA,           // BSR 行指针数组
      bsrColIndA,           // BSR 列索引数组
      blockDim,             // BSR 块大小
      info,                 // cuSPARSE 操作信息结构体
      policy,               // 解析策略
      pBuffer));            // 用于存储中间结果的缓冲区
}

template <>
void bsrsv2_analysis<double>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double)) {
  // 同上，使用 double 类型的精度
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_analysis(
      handle, dirA, transA, mb, nnzb, descrA, bsrValA, bsrRowPtrA,
      bsrColIndA, blockDim, info, policy, pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  // 同上，使用 c10::complex<float> 类型的精度，需要将 bsrvValA 转换为 cuComplex*
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_analysis(
      handle, dirA, transA, mb, nnzb, descrA,
      reinterpret_cast<const cuComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, policy, pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  // 同上，使用 c10::complex<double> 类型的精度，需要将 bsrvValA 转换为 cuDoubleComplex*
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_analysis(
      handle, dirA, transA, mb, nnzb, descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, policy, pBuffer));
}
void bsrsv2_solve<float>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(float)) {
  // 调用 cuSPARSE 库中的单精度 BSRSV2 求解函数，解决方程组 A * x = y
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_solve(
      handle,             // cuSPARSE 上下文句柄
      dirA,               // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,             // A 矩阵的转置操作
      mb,                 // BSR 子矩阵的数量
      nnzb,               // A 矩阵中非零块的数量
      alpha,              // A 矩阵的缩放因子
      descrA,             // A 矩阵的描述符
      bsrValA,            // A 矩阵的数值部分
      bsrRowPtrA,         // A 矩阵的行指针数组
      bsrColIndA,         // A 矩阵的列索引数组
      blockDim,           // BSR 块的大小
      info,               // cuSPARSE 操作状态信息
      x,                  // 输入向量 x
      y,                  // 输出向量 y
      policy,             // cuSPARSE 操作策略
      pBuffer));          // 临时缓冲区
}

template <>
void bsrsv2_solve<double>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(double)) {
  // 调用 cuSPARSE 库中的双精度 BSRSV2 求解函数，解决方程组 A * x = y
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_solve(
      handle,             // cuSPARSE 上下文句柄
      dirA,               // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,             // A 矩阵的转置操作
      mb,                 // BSR 子矩阵的数量
      nnzb,               // A 矩阵中非零块的数量
      alpha,              // A 矩阵的缩放因子
      descrA,             // A 矩阵的描述符
      bsrValA,            // A 矩阵的数值部分
      bsrRowPtrA,         // A 矩阵的行指针数组
      bsrColIndA,         // A 矩阵的列索引数组
      blockDim,           // BSR 块的大小
      info,               // cuSPARSE 操作状态信息
      x,                  // 输入向量 x
      y,                  // 输出向量 y
      policy,             // cuSPARSE 操作策略
      pBuffer));          // 临时缓冲区
}

template <>
void bsrsv2_solve<c10::complex<float>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSPARSE 库中的单精度复数 BSRSV2 求解函数，解决方程组 A * x = y
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_solve(
      handle,                         // cuSPARSE 上下文句柄
      dirA,                           // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,                         // A 矩阵的转置操作
      mb,                             // BSR 子矩阵的数量
      nnzb,                           // A 矩阵中非零块的数量
      reinterpret_cast<const cuComplex*>(alpha),  // A 矩阵的缩放因子
      descrA,                         // A 矩阵的描述符
      reinterpret_cast<const cuComplex*>(bsrValA),  // A 矩阵的数值部分
      bsrRowPtrA,                     // A 矩阵的行指针数组
      bsrColIndA,                     // A 矩阵的列索引数组
      blockDim,                       // BSR 块的大小
      info,                           // cuSPARSE 操作状态信息
      reinterpret_cast<const cuComplex*>(x),        // 输入向量 x
      reinterpret_cast<cuComplex*>(y),              // 输出向量 y
      policy,                         // cuSPARSE 操作策略
      pBuffer));                      // 临时缓冲区
}

template <>
void bsrsv2_solve<c10::complex<double>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSPARSE 库中的双精度复数 BSRSV2 求解函数，解决方程组 A * x = y
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_solve(
      handle,                                 // cuSPARSE 上下文句柄
      dirA,                                   // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,                                 // A 矩阵的转置操作
      mb,                                     // BSR 子矩阵的数量
      nnzb,                                   // A 矩阵中非零块的数量
      reinterpret_cast<const cuDoubleComplex*>(alpha),  // A 矩阵的缩放因子
      descrA,                                 // A 矩阵的描述符
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),  // A 矩阵的数值部分
      bsrRowPtrA,                             // A 矩阵的行指针数组
      bsrColIndA,                             // A 矩阵的列索引数组
      blockDim,                               // BSR 块的大小
      info,                                   // cuSPARSE 操作状态信息
      reinterpret_cast<const cuDoubleComplex*>(x),        // 输入向量 x
      reinterpret_cast<cuDoubleComplex*>(y),              // 输出向量 y
      policy,                                 // cuSPARSE 操作策略
      pBuffer));                              // 临时缓冲区
}

template <>
void bsrsm2_bufferSize<float>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(float)) {
  // 计算 cuSPARSE 库中单精度 BSRSM2 函数所需的缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_bufferSize(
      handle,             // cuSPARSE 上下文句柄
      dirA,               // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,             // A 矩阵的转置操作
      transX,             // 输入向量 X 的转置操作
      mb,                 // BSR 子矩阵的数量
      n,                  // 输入向量 X 的长度
      nnzb,               // A 矩阵中非零块的数量
      descrA,             // A 矩阵的描述符
      bsrValA,            // A 矩阵的数值部分
      bsrRowPtrA,         // A 矩阵的行指针数组
      bsrColIndA,         // A 矩阵的列索引数组
      blockDim,           // BSR 块的大小
      info,               // cuSPARSE 操作状态信息
      pBufferSizeInBytes));// 返回的缓冲区大小（字节数）
}

template <>
void bsrsm2_bufferSize<double>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(double)) {
  // 计算 cuSPARSE 库中双精度 BSRSM2 函数所需的缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_bufferSize(
      handle,             // cuSPARSE 上下文句柄
      dirA,               // BSR 存储格式的 A 矩阵方向（一般是行主序或列主序）
      transA,             // A 矩阵的转置操作
      transX,             // 输入向量 X 的转置操作
      mb,                 // BSR 子矩阵的数量
      n,                  // 输入向量 X 的长度
      nnzb,               // A 矩
    # 定义一个宏 CUSPARSE_BSRSM2_BUFFER_ARGTYPES，它接受类型 c10::complex<double> 作为参数
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>)) {
  # 调用 cusparseZbsrsm2_bufferSize 函数，用于计算执行 BSR 稀疏矩阵-矩阵求解操作所需的缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_bufferSize(
      handle,                             # cuSPARSE 库的句柄
      dirA,                               # 稀疏矩阵的存储格式
      transA,                             # 稀疏矩阵 A 的转置选项
      transX,                             # 稀疏矩阵 X 的转置选项
      mb,                                 # 稀疏矩阵 A 的行块数
      n,                                  # 稀疏矩阵 A 的列数
      nnzb,                               # 稀疏矩阵 A 的非零块数
      descrA,                             # 稀疏矩阵 A 的描述符
      reinterpret_cast<cuDoubleComplex*>(bsrValA),  # 稀疏矩阵 A 的块值数组
      bsrRowPtrA,                         # 稀疏矩阵 A 的行指针数组
      bsrColIndA,                         # 稀疏矩阵 A 的列索引数组
      blockDim,                           # 稀疏矩阵 A 的块大小
      info,                               # cuSPARSE 操作信息
      pBufferSizeInBytes));               # 输出参数，返回所需的缓冲区大小
template <>
void bsrsm2_analysis<float>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float)) {
  // 调用 cuSPARSE 库进行 BSRSM2 解析操作，用于分析 BSR 稀疏矩阵的解法
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_analysis(
      handle,          // cuSPARSE 上下文句柄
      dirA,            // 稀疏矩阵 A 的存储格式（行主/列主）
      transA,          // 矩阵 A 的转置选项
      transX,          // 向量 X 的转置选项
      mb,              // 矩阵 A 的行块数
      n,               // 矩阵 A 的列数
      nnzb,            // 矩阵 A 的非零块数
      descrA,          // 矩阵 A 的描述符
      bsrValA,         // BSR 格式的矩阵 A 的值数组
      bsrRowPtrA,      // BSR 格式的矩阵 A 的行指针数组
      bsrColIndA,      // BSR 格式的矩阵 A 的列索引数组
      blockDim,        // BSR 块的维度
      info,            // cuSPARSE 操作的信息结构体
      policy,          // 执行策略选项
      pBuffer));       // 临时缓冲区
}

template <>
void bsrsm2_analysis<double>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double)) {
  // 类似上述，但是针对双精度浮点数类型
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_analysis(
      handle, dirA, transA, transX, mb, n, nnzb, descrA, bsrValA, bsrRowPtrA,
      bsrColIndA, blockDim, info, policy, pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  // 类似上述，但是针对单精度复数类型
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsm2_analysis(
      handle, dirA, transA, transX, mb, n, nnzb, descrA,
      reinterpret_cast<const cuComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, policy, pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  // 类似上述，但是针对双精度复数类型
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_analysis(
      handle, dirA, transA, transX, mb, n, nnzb, descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, policy, pBuffer));
}

template <>
void bsrsm2_solve<float>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(float)) {
  // 调用 cuSPARSE 库进行 BSRSM2 解法操作，解决 BSR 稀疏矩阵的线性系统
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_solve(
      handle, dirA, transA, transX, mb, n, nnzb, alpha, descrA, bsrValA,
      bsrRowPtrA, bsrColIndA, blockDim, info, B, ldb, X, ldx, policy, pBuffer));
}

template <>
void bsrsm2_solve<double>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(double)) {
  // 类似上述，但是针对双精度浮点数类型
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_solve(
      handle, dirA, transA, transX, mb, n, nnzb, alpha, descrA, bsrValA,
      bsrRowPtrA, bsrColIndA, blockDim, info, B, ldb, X, ldx, policy, pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<float>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>)) {
  // 类似上述，但是针对单精度复数类型
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsm2_solve(
      handle, dirA, transA, transX, mb, n, nnzb,
      reinterpret_cast<const cuComplex*>(alpha), descrA,
      reinterpret_cast<const cuComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, reinterpret_cast<const cuComplex*>(B), ldb,
      reinterpret_cast<cuComplex*>(X), ldx, policy, pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<double>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>)) {
  // 类似上述，但是针对双精度复数类型
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_solve(
      handle, dirA, transA, transX, mb, n, nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha), descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA), bsrRowPtrA, bsrColIndA,
      blockDim, info, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
      reinterpret_cast<cuDoubleComplex*>(X), ldx, policy, pBuffer));
}
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>)) {
  // 调用 CUSPARSE 库中的 cusparseZbsrsm2_solve 函数进行稀疏矩阵求解，参数如下：
  // - handle: CUSPARSE 库的句柄，用于管理与 CUDA 稀疏操作相关的状态
  // - dirA: 矩阵 A 的方向（行主序或列主序）
  // - transA: 矩阵 A 是否需要转置
  // - transX: 矩阵 X 是否需要转置
  // - mb: 矩阵 A 的分块行数
  // - n: 矩阵 A 的列数
  // - nnzb: 矩阵 A 的非零分块数量
  // - alpha: 一个复数，用于矩阵运算中的缩放
  // - descrA: 矩阵 A 的描述符，定义了 A 的存储格式等信息
  // - bsrValA: 矩阵 A 的非零分块值数组
  // - bsrRowPtrA: 矩阵 A 的行偏移指针数组
  // - bsrColIndA: 矩阵 A 的列索引数组
  // - blockDim: 矩阵 A 的分块大小
  // - info: 存储操作状态和诊断信息的结构体
  // - B: 输入矩阵 B 的数据
  // - ldb: 输入矩阵 B 的行偏移
  // - X: 输出矩阵 X 的数据
  // - ldx: 输出矩阵 X 的行偏移
  // - policy: 操作策略，如何管理内存和执行顺序
  // - pBuffer: 用于临时存储的缓冲区
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const cuDoubleComplex*>(B),
      ldb,
      reinterpret_cast<cuDoubleComplex*>(X),
      ldx,
      policy,
      pBuffer));
}

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE
```