# `.\pytorch\test\cpp\c10d\CUDATest.hpp`

```py
#pragma once

// 使用 `#pragma once` 指令确保头文件只被编译一次，避免重复包含


#include <c10/cuda/CUDAStream.h>

// 包含 CUDAStream 头文件，用于支持 CUDA 流操作


namespace c10d {
namespace test {

// 声明命名空间 `c10d` 和 `test`，用于封装测试相关的功能


#ifdef _WIN32
#define EXPORT_TEST_API __declspec(dllexport)
#else
#define EXPORT_TEST_API
#endif

// 根据操作系统判断，定义导出符号的方式：
// - 在 Windows 上，使用 `__declspec(dllexport)` 导出符号
// - 在其他平台上，不做任何导出操作


EXPORT_TEST_API void cudaSleep(at::cuda::CUDAStream& stream, uint64_t clocks);

// 声明导出函数 `cudaSleep`，接受 CUDA 流对象和时钟数作为参数，用于在 CUDA 流上进行休眠操作


EXPORT_TEST_API int cudaNumDevices();

// 声明导出函数 `cudaNumDevices`，返回当前系统上 CUDA 设备的数量


} // namespace test
} // namespace c10d

// 命名空间闭合，结束 `test` 和 `c10d` 命名空间的声明
```