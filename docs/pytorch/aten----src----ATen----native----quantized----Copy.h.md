# `.\pytorch\aten\src\ATen\native\quantized\Copy.h`

```
#pragma once
// 使用 pragma once 预处理指令，确保头文件只被包含一次，防止多重包含问题

#include <ATen/core/Tensor.h>
// 包含 ATen 核心库中的 Tensor 头文件，提供 Tensor 类型的定义和操作接口

namespace at {
namespace native {

// 声明 quantized_copy_from_float_ 函数，该函数用于从浮点数张量复制到量化张量
Tensor& quantized_copy_from_float_(Tensor& self, const Tensor& src);

} // namespace native
} // namespace at
// 结束 at 和 native 命名空间的定义
```