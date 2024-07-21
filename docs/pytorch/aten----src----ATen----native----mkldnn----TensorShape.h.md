# `.\pytorch\aten\src\ATen\native\mkldnn\TensorShape.h`

```
#pragma once


// 指令，用于告诉编译器只包含此头文件一次，避免重复包含
#include <ATen/ATen.h>
// 引入 ATen 库中的 Tensor 类型和相关函数声明

#include <c10/core/SymIntArrayRef.h>
// 引入 c10 库中的 SymIntArrayRef 类型声明

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size);
// 声明函数 mkldnn_view，接受一个 Tensor 引用和一个 IntArrayRef 对象作为参数，返回一个 Tensor

Tensor mkldnn_view_symint(const Tensor& self, c10::SymIntArrayRef size);
// 声明函数 mkldnn_view_symint，接受一个 Tensor 引用和一个 SymIntArrayRef 对象作为参数，返回一个 Tensor

Tensor mkldnn_clone(const Tensor& self);
// 声明函数 mkldnn_clone，接受一个 Tensor 引用作为参数，返回一个 Tensor

} // namespace native
} // namespace at
// 结束命名空间声明
```