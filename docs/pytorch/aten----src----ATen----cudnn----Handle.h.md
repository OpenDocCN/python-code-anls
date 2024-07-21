# `.\pytorch\aten\src\ATen\cudnn\Handle.h`

```py
#pragma once
// 使用 #pragma once 预处理指令，确保头文件只被编译一次，避免重复包含

#include <ATen/cudnn/cudnn-wrapper.h>
// 包含 ATen 框架中提供的 cudnn-wrapper.h 头文件，用于 CUDA 的深度学习库 cuDNN 的包装

#include <ATen/cuda/ATenCUDAGeneral.h>
// 包含 ATen 框架中的 ATenCUDAGeneral.h 头文件，提供 CUDA 相关的一般性功能

namespace at { namespace native {

TORCH_CUDA_CPP_API cudnnHandle_t getCudnnHandle();
// 声明一个函数 getCudnnHandle()，返回类型为 cudnnHandle_t，这是一个 cudnn 库的句柄类型

}} // namespace at::native
// 命名空间声明结束，分别为 at 和 native 命名空间
```