# `.\pytorch\aten\src\ATen\miopen\Utils.h`

```
#pragma once
// 使用 #pragma once 防止头文件重复包含

#include <ATen/core/Tensor.h>
// 包含 ATen 核心的 Tensor 头文件

#include <ATen/miopen/miopen-wrapper.h>
// 包含 ATen 的 MIOpen 封装头文件

#include <ATen/miopen/Handle.h>
// 包含 ATen 的 MIOpen 处理句柄头文件

namespace at { namespace native {

// 在 at::native 命名空间内声明函数

// This function makes tensors which have zero stride contiguous, by
// setting the strides to 1.
// 此函数将具有零步长的张量设置为连续的，将步长设置为 1。
inline Tensor contiguousIfZeroInStrides(const Tensor& t) {
  // 遍历张量 t 的步长
  for (auto s : t.strides()) {
    // 如果步长为 0，则返回张量的连续版本
    if (s == 0) return t.contiguous();
  }
  // 如果没有步长为 0 的情况，则返回原始张量
  return t;
}

}} // namespace at::native
```