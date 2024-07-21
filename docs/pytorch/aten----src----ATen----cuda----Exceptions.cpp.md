# `.\pytorch\aten\src\ATen\cuda\Exceptions.cpp`

```py
// 包含 CUDA Caching Allocator 头文件，以确保在获取 CUDART_VERSION 之前已定义 CUDACachingAllocator
#include <c10/cuda/CUDACachingAllocator.h>

// 包含 CUDA 异常处理的头文件
#include <ATen/cuda/Exceptions.h>

// 定义在命名空间 at::cuda::blas 中
namespace at::cuda {
namespace blas {

// 导出函数，根据 cublasStatus_t 的错误码返回对应的错误字符串
C10_EXPORT const char* _cublasGetErrorEnum(cublasStatus_t error) {
  // 检查 cublasStatus_t 的错误码，返回对应的字符串
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  // 如果定义了 CUBLAS_STATUS_LICENSE_ERROR，则处理此错误码
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  // 未知错误码时返回 "<unknown>"
  return "<unknown>";
}

} // namespace blas

// 如果定义了 CUDART_VERSION，则进入 solver 命名空间
#ifdef CUDART_VERSION
namespace solver {

// 导出函数，根据 cusolverStatus_t 的错误码返回对应的错误字符串
C10_EXPORT const char* cusolverGetErrorMessage(cusolverStatus_t status) {
  // 使用 switch-case 结构根据 cusolverStatus_t 的错误码返回对应的字符串
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:                                          return "Unknown cusolver error number";
  }
}

} // namespace solver
#endif

} // namespace at::cuda
```