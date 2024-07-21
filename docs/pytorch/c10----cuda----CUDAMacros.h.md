# `.\pytorch\c10\cuda\CUDAMacros.h`

```
#pragma once
// 如果尚未定义 C10_USING_CUSTOM_GENERATED_MACROS 宏，则进入此条件编译
#ifndef C10_USING_CUSTOM_GENERATED_MACROS

// 如果未定义 C10_CUDA_NO_CMAKE_CONFIGURE_FILE 宏，则包含 CUDA CMake 宏文件
// 这是因为我们尚未修改 AMD HIP 构建来生成此文件，所以添加了一个额外选项以明确忽略它。
#ifndef C10_CUDA_NO_CMAKE_CONFIGURE_FILE
#include <c10/cuda/impl/cuda_cmake_macros.h>
#endif // C10_CUDA_NO_CMAKE_CONFIGURE_FILE

#endif

// 查看 c10/macros/Export.h 以获取这些宏的详细解释。
// 我们为每个单独的库构建需要一个宏集。
#ifdef _WIN32
// 在 Windows 下根据是否构建为共享库定义导出和导入宏
#if defined(C10_CUDA_BUILD_SHARED_LIBS)
#define C10_CUDA_EXPORT __declspec(dllexport)
#define C10_CUDA_IMPORT __declspec(dllimport)
#else
#define C10_CUDA_EXPORT
#define C10_CUDA_IMPORT
#endif
#else // _WIN32
// 在非 Windows 平台下，根据是否使用 GNU 编译器定义默认可见性宏
#if defined(__GNUC__)
#define C10_CUDA_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_CUDA_EXPORT
#endif // defined(__GNUC__)
// 在非 Windows 平台下，导入宏与导出宏一致
#define C10_CUDA_IMPORT C10_CUDA_EXPORT
#endif // _WIN32

// 这个宏被 libc10_cuda.so 使用
#ifdef C10_CUDA_BUILD_MAIN_LIB
// 如果正在构建主要库，则使用导出宏
#define C10_CUDA_API C10_CUDA_EXPORT
#else
// 否则使用导入宏
#define C10_CUDA_API C10_CUDA_IMPORT
#endif

/**
 * 我们识别的最大 GPU 数量。将其增加到初始限制 16 以上会导致 Caffe2 测试失败，
 * 因此有 ifdef 保护。这个值不能超过 128，因为我们的 DeviceIndex 是 uint8_t 类型。
 */
#ifdef FBCODE_CAFFE2
// 如果是在 FBCODE_CAFFE2 下编译，则使用值 16
#define C10_COMPILE_TIME_MAX_GPUS 16
#else
// 否则默认使用 120
#define C10_COMPILE_TIME_MAX_GPUS 120
#endif
```