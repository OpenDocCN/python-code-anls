# `.\pytorch\aten\src\ATen\cuda\CUDADevice.h`

```py
// 包含预处理指令，确保头文件只被包含一次
#pragma once

// 引入 ATen 库中 CUDA 异常处理的头文件
#include <ATen/cuda/Exceptions.h>

// 引入 CUDA 运行时和驱动程序的头文件
#include <cuda.h>
#include <cuda_runtime.h>

// 定义了 at::cuda 命名空间
namespace at::cuda {

// 从指针获取设备信息的内联函数
inline Device getDeviceFromPtr(void* ptr) {
  // CUDA 指针属性结构体
  cudaPointerAttributes attr{};

  // 获取指针的 CUDA 属性
  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

  // 如果不是 ROCm 平台，进行条件检查
#if !defined(USE_ROCM)
  // 如果指针类型为未注册的主机内存，抛出异常
  TORCH_CHECK(attr.type != cudaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  // 返回由指针对应的设备构成的 Device 对象
  return {c10::DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::cuda
```