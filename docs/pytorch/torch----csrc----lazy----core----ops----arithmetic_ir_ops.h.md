# `.\pytorch\torch\csrc\lazy\core\ops\arithmetic_ir_ops.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/lazy/core/ir.h>
// 包含 torch 框架中 lazy 模块的核心 IR 头文件

namespace torch {
namespace lazy {

TORCH_API NodePtr operator+(const Value& node1, const Value& node2);
// 定义 torch::lazy 命名空间中的加法运算符重载，接受两个 Value 类型参数，返回 NodePtr 类型对象

TORCH_API NodePtr operator-(const Value& node1, const Value& node2);
// 定义 torch::lazy 命名空间中的减法运算符重载，接受两个 Value 类型参数，返回 NodePtr 类型对象

TORCH_API NodePtr operator*(const Value& node1, const Value& node2);
// 定义 torch::lazy 命名空间中的乘法运算符重载，接受两个 Value 类型参数，返回 NodePtr 类型对象

TORCH_API NodePtr operator/(const Value& node1, const Value& node2);
// 定义 torch::lazy 命名空间中的除法运算符重载，接受两个 Value 类型参数，返回 NodePtr 类型对象

} // namespace lazy
} // namespace torch
// 结束 torch 和 lazy 命名空间的定义
```