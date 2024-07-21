# `.\pytorch\aten\src\ATen\mkl\Exceptions.h`

```py
#pragma once

#include <string>                               // 包含标准字符串库
#include <stdexcept>                            // 包含标准异常库
#include <sstream>                              // 包含字符串流库
#include <mkl_dfti.h>                           // 包含 MKL DFTI 头文件
#include <mkl_spblas.h>                         // 包含 MKL Sparse BLAS 头文件

namespace at::native {

static inline void MKL_DFTI_CHECK(MKL_INT status)
{
  // 检查 MKL DFTI 函数调用返回状态，若出错则抛出异常
  if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    std::ostringstream ss;
    ss << "MKL FFT error: " << DftiErrorMessage(status);
    throw std::runtime_error(ss.str());
  }
}

}  // namespace at::native

namespace at::mkl::sparse {

static inline const char* _mklGetErrorString(sparse_status_t status) {
  // 返回对应稀疏矩阵库状态的字符串表示
  if (status == SPARSE_STATUS_SUCCESS) {
    return "SPARSE_STATUS_SUCCESS";
  }
  if (status == SPARSE_STATUS_NOT_INITIALIZED) {
    return "SPARSE_STATUS_NOT_INITIALIZED";
  }
  if (status == SPARSE_STATUS_ALLOC_FAILED) {
    return "SPARSE_STATUS_ALLOC_FAILED";
  }
  if (status == SPARSE_STATUS_INVALID_VALUE) {
    return "SPARSE_STATUS_INVALID_VALUE";
  }
  if (status == SPARSE_STATUS_EXECUTION_FAILED) {
    return "SPARSE_STATUS_EXECUTION_FAILED";
  }
  if (status == SPARSE_STATUS_INTERNAL_ERROR) {
    return "SPARSE_STATUS_INTERNAL_ERROR";
  }
  if (status == SPARSE_STATUS_NOT_SUPPORTED) {
    return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  // 若状态未知则返回 "<unknown>"
  return "<unknown>";
}
} // namespace at::mkl::sparse

// 定义宏，用于检查 MKL Sparse 函数调用的返回状态
#define TORCH_MKLSPARSE_CHECK(EXPR)                                 \
  do {                                                              \
    sparse_status_t __err = EXPR;                                   \
    TORCH_CHECK(                                                    \
        __err == SPARSE_STATUS_SUCCESS,                             \
        "MKL error: ",                                              \
        at::mkl::sparse::_mklGetErrorString(__err),                 \
        " when calling `" #EXPR "`");                               \
  } while (0)
```