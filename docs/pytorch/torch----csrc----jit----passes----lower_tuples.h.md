# `.\pytorch\torch\csrc\jit\passes\lower_tuples.h`

```py
#pragma once

// 指令，确保此头文件在编译过程中只包含一次，防止重复定义


#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 库中的 IR 头文件，用于处理图形结构


namespace torch {
namespace jit {

// 命名空间声明，定义了 torch::jit 命名空间


// removes tuples where TupleConstruct and TupleUnpack are matched
// but leaves tuples in place across if statements, loops, and as inputs/outputs
TORCH_API void LowerSimpleTuples(const std::shared_ptr<Graph>& graph);

// 函数声明，用于在图中移除匹配的 TupleConstruct 和 TupleUnpack 对，但保留在 if 语句、循环以及输入/输出中的元组


// removes _all_ tuples and raises an error if some cannot be removed
// this is used by ONNX to ensure there are not tuples before conversion,
// but will not work on graphs whose inputs contain tuples.
TORCH_API void LowerAllTuples(const std::shared_ptr<Graph>& graph);

// 函数声明，用于移除所有的元组，并在无法移除某些元组时引发错误。在转换之前，ONNX 使用此函数确保图中没有元组，但不适用于输入包含元组的图形结构。


TORCH_API void LowerSimpleTuples(Block* block);

// 函数声明，用于在基本块中移除元组。


} // namespace jit
} // namespace torch

// 命名空间结束标记，关闭 torch::jit 命名空间声明
```