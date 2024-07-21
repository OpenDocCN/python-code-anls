# `.\pytorch\aten\src\ATen\cudnn\Utils.h`

```py
#pragma once
// 使用 #pragma once 指令，确保头文件只被编译一次，避免重复包含

#include <ATen/core/Tensor.h>
// 包含 ATen 核心库中的 Tensor 头文件

#include <ATen/cuda/Exceptions.h>
// 包含 ATen CUDA 异常处理的头文件

#include <ATen/cudnn/cudnn-wrapper.h>
// 包含 ATen cuDNN 封装的头文件

#include <ATen/cudnn/Handle.h>
// 包含 ATen cuDNN 句柄处理的头文件

namespace at { namespace native {

// 在 at::native 命名空间中声明函数 contiguousIfZeroInStrides
// cuDNN 对于张量的连续性检查有缺陷（即对于维度为0的情况，不会忽略步幅）。
// 此函数用于将步幅为0的张量变成连续的，通过将步幅设置为1来符合 cuDNN 的要求。
inline Tensor contiguousIfZeroInStrides(const Tensor& t) {
  // 遍历张量 t 的所有步幅
  for (auto s : t.strides()) {
    // 如果发现步幅 s 等于 0，则返回 t 的连续版本
    if (s == 0) return t.contiguous();
  }
  // 如果没有步幅为 0 的情况，则返回原始张量 t
  return t;
}

}}  // namespace at::native
```