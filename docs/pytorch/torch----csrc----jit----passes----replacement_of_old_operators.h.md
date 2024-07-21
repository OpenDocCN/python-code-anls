# `.\pytorch\torch\csrc\jit\passes\replacement_of_old_operators.h`

```py
#pragma once
// 防止头文件被多次包含

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR 操作相关头文件

namespace torch {
namespace jit {

// 在给定的升级器名称上查找有效的升级器图，并缓存结果以供后续查找使用。
// 如果没有为升级器名称提供有效的升级器图，则会引发错误。
std::shared_ptr<Graph> getUpgraderGraph(const std::string& upgrader_name);
// 函数声明：根据升级器名称获取升级器图的共享指针

TORCH_API void ReplaceOldOperatorsWithUpgraders(std::shared_ptr<Graph> graph);
// 函数声明：用升级器替换图中的旧操作

} // namespace jit
} // namespace torch
// 命名空间结束
```