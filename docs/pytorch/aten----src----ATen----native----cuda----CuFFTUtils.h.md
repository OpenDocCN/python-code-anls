# `.\pytorch\aten\src\ATen\native\cuda\CuFFTUtils.h`

```py
#pragma once
// 预处理指令，确保本头文件只被包含一次

#include <ATen/Config.h>
// 包含 ATen 库的配置文件

#include <string>
// 包含处理字符串的标准库

#include <stdexcept>
// 包含异常处理的标准库

#include <sstream>
// 包含字符串流处理的标准库

#include <cufft.h>
#include <cufftXt.h>
// 包含 CUDA FFT 库的头文件

namespace at { namespace native {

// 定义最大维度为 3 + 2 = 5，支持批处理和复杂维度
constexpr int max_rank = 3;

static inline std::string _cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
#if !defined(USE_ROCM)
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
#endif
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error)
{
  // 检查 cuFFT 操作返回的错误码，如果不是成功，则抛出异常
  if (error != CUFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "cuFFT error: " << _cudaGetErrorEnum(error);
    AT_ERROR(ss.str());
  }
}

}} // at::native
```