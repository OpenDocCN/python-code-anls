# `.\pytorch\aten\src\ATen\cudnn\Types.h`

```
#pragma once
// 指令，确保此头文件只被编译一次

#include <ATen/cudnn/cudnn-wrapper.h>
// 包含 ATen 库中的 cudnn-wrapper.h 头文件

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor.h 头文件

namespace at { namespace native {
// 命名空间声明开始，命名空间为 at::native

TORCH_CUDA_CPP_API cudnnDataType_t
getCudnnDataTypeFromScalarType(const at::ScalarType dtype);
// 声明一个名为 getCudnnDataTypeFromScalarType 的函数，接受一个 ScalarType 参数并返回 cudnnDataType_t 类型

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);
// 声明一个名为 getCudnnDataType 的函数，接受一个 Tensor 引用并返回 cudnnDataType_t 类型

int64_t cudnn_version();
// 声明一个返回 int64_t 类型的函数 cudnn_version

}}  // namespace at::cudnn
// 命名空间声明结束，命名空间为 at::cudnn
```