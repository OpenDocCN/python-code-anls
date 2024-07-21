# `.\pytorch\aten\src\ATen\native\sparse\Macros.h`

```
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器是 CUDA 或者 HIP，则定义 GPUCC 宏
#define GPUCC
// 定义 FUNCAPI 宏为 __host__ __device__
#define FUNCAPI __host__ __device__
// 定义 INLINE 宏为 __forceinline__
#define INLINE __forceinline__
#else
// 如果不是 CUDA 或者 HIP 编译器，则清空 GPUCC 宏
#define FUNCAPI
// 定义 INLINE 宏为 inline
#define INLINE inline
#endif

#if defined(_WIN32) || defined(_WIN64)
// 如果是 Windows 系统，则临时禁用 __restrict，
// 因为并非所有的 MSVC 版本都支持它。
#define RESTRICT
#else
// 如果不是 Windows 系统，则定义 RESTRICT 宏为 __restrict__
#define RESTRICT __restrict__
#endif
```