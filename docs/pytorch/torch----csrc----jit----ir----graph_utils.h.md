# `.\pytorch\torch\csrc\jit\ir\graph_utils.h`

```py
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 Torch 库的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含标准库向量容器的头文件
#include <vector>

// Torch 的命名空间开始
namespace torch {
// Torch JIT 的命名空间开始
namespace jit {

// 定义在 TORCH_API 下的函数，返回输入张量 t 的类型指针，根据 complete 参数是否完全推断类型
TORCH_API TypePtr getTensorType(const at::Tensor& t, bool complete);

// 定义在 TORCH_API 下的函数，推断输入类型 input_type 的形状和类型，根据输入栈的迭代器和 complete 参数
TORCH_API TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete);

// 定义在 TORCH_API 下的函数，设置图 g 的输入张量类型，根据栈 stack、complete 参数以及参数计数列表 param_count_list
TORCH_API void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {});

} // namespace jit
} // namespace torch
```