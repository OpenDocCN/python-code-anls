# `.\pytorch\aten\src\ATen\native\cuda\thread_constants.h`

```
#pragma once
// 声明标记一个 lambda 函数在主机和设备上都可执行。__host__ 属性非常重要，以便我们可以从主机访问静态类型信息，
// 即使该函数通常只在设备上执行。
#include <c10/macros/Macros.h>

// 如果未定义 GPU_LAMBDA 宏，则定义为 __host__ __device__
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

// 如果编译器使用 ROCm（AMD ROCm 平台），定义一个返回固定线程数 256 的函数
#if defined(USE_ROCM)
constexpr int num_threads() {
  return 256;
}
// 否则，定义一个返回 C10_WARP_SIZE（线程簇大小）乘以 4 的无符号整数函数
#else
constexpr uint32_t num_threads() {
  return C10_WARP_SIZE * 4;
}
#endif

// 定义一个返回线程工作大小 4 的函数
constexpr int thread_work_size() { return 4; }

// 定义一个返回块工作大小的函数，其值为线程工作大小乘以线程数
constexpr int block_work_size() { return thread_work_size() * num_threads(); }
```