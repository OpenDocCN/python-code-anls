# `.\pytorch\torch\csrc\jit\passes\lower_grad_of.h`

```
#pragma once
// 使用预处理指令 #pragma once 确保头文件只被包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块中的 IR 头文件

namespace torch {
namespace jit {

// Torch JIT 模块的命名空间开始

// 这个函数移除 'grad_of' 节点，并用如下形式的条件替换它们：
// 如果输入中有任何一个定义了:
//  则输出 = <原始计算>
// 否则:
//  输出 = undefineds
TORCH_API void LowerGradOf(Graph& g);
// 声明 LowerGradOf 函数，接受一个 Graph 对象的引用作为参数

} // namespace jit
} // namespace torch
// Torch JIT 模块的命名空间结束
```