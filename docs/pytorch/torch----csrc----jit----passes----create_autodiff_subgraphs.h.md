# `.\pytorch\torch\csrc\jit\passes\create_autodiff_subgraphs.h`

```
// 一旦
#pragma once

// 包含 Torch 库的导出文件，用于导出接口
#include <torch/csrc/Export.h>

// 包含 Torch JIT IR 的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含标准库的大小定义
#include <cstddef>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 创建自动微分子图，将组合不同 JIT 自动微分通道中可微分的子图
// threshold - 子图中最小节点数的阈值
// 返回所有找到的可微分子图节点
TORCH_API std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold = 2);

} // namespace jit
} // namespace torch
```