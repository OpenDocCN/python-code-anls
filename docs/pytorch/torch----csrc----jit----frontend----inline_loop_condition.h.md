# `.\pytorch\torch\csrc\jit\frontend\inline_loop_condition.h`

```py
#pragma once
#include <functional>  // 包含标准库中的 functional 头文件
#include <memory>       // 包含标准库中的 memory 头文件
#include <string>       // 包含标准库中的 string 头文件

#include <torch/csrc/Export.h>   // 包含 Torch 库中的 Export.h 头文件
#include <torch/csrc/jit/ir/ir.h>   // 包含 Torch 库中的 ir.h 头文件

namespace torch {
namespace jit {

TORCH_API void InlineLoopCondition(std::shared_ptr<Graph>& graph);
    // 声明 InlineLoopCondition 函数，用于内联循环条件，接受一个指向图的共享指针作为参数
TORCH_API void InlineBlockBeforeNode(Node* before_node, Block* block);
    // 声明 InlineBlockBeforeNode 函数，用于在指定节点之前内联块，接受一个指向节点的指针和一个指向块的指针作为参数

} // namespace jit
} // namespace torch
```