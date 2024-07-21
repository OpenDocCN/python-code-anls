# `.\pytorch\torch\csrc\lazy\ts_backend\tensor_aten_ops.h`

```
#pragma once
// 预处理指令，确保本头文件在编译过程中只被包含一次

#include <torch/csrc/lazy/core/tensor.h>
// 包含 Torch 框架中懒计算相关的核心张量头文件

namespace torch {
namespace lazy {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
// ATEN 操作符按字母顺序列在此处。
//////////////////////////////////////////////////////////////////////////////

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);
// 函数原型声明：用源张量的值填充目标张量

// Fills the input with the given value.
// 使用给定的标量值填充输入张量
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

} // namespace lazy
} // namespace torch
// 命名空间定义结束
```