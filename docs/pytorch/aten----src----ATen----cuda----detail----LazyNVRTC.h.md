# `.\pytorch\aten\src\ATen\cuda\detail\LazyNVRTC.h`

```
#pragma once
// 包含 CUDA 的相关头文件：CUDA 核心函数接口定义
#include <ATen/detail/CUDAHooksInterface.h>

// 声明 at::cuda 命名空间
namespace at::cuda {

// 前置声明 at::cuda::NVRTC
// 表示 NVRTC 是一个结构体，提前声明，后续会定义其详细内容
struct NVRTC;

// 声明 at::cuda::detail 命名空间
namespace detail {

// 声明 extern 变量 lazyNVRTC，类型为 NVRTC
extern NVRTC lazyNVRTC;

} // namespace detail

}  // namespace at::cuda
```