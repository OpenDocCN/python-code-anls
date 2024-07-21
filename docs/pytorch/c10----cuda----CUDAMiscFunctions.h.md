# `.\pytorch\c10\cuda\CUDAMiscFunctions.h`

```
#pragma once
// 防止 CUDAFunctions.h 和 CUDAExceptions.h 之间的循环依赖

#include <c10/cuda/CUDAMacros.h>  // 包含 CUDA 宏定义

#include <mutex>  // 包含互斥量的标准头文件

namespace c10::cuda {
// 声明一个 CUDA API，返回一个指向 CUDA 检查后缀字符串的常量指针，并且不抛出异常
C10_CUDA_API const char* get_cuda_check_suffix() noexcept;

// 声明一个函数，返回一个互斥量的指针，用于管理 CUDA 相关的资源
C10_CUDA_API std::mutex* getFreeMutex();
} // namespace c10::cuda
```