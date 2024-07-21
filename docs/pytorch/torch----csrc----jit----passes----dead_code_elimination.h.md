# `.\pytorch\torch\csrc\jit\passes\dead_code_elimination.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 如果给定顶层图形（graph），DCE 将构建别名分析，允许“更智能”地进行死代码消除
// （如果可以证明变异的值未被使用，则会消除可变操作）。否则，我们将不允许 DCE 消除可变操作。
//
// 因此，如果可能，请优先使用图形版本。
enum class DCESideEffectPolicy : uint8_t {
  // 默认行为：死代码消除将检查节点是否具有副作用，并在具有副作用时不删除它。
  DONT_DELETE_NODES_WITH_SIDE_EFFECTS,
  // 使用此标志，死代码消除将不检查节点是否具有副作用，并将具有副作用的节点视为任何其他节点，
  // 即如果它们的输出在任何地方都没有使用，则删除它们。
  ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS
};

// 消除死代码，适用于给定的图形对象
TORCH_API void EliminateDeadCode(
    const std::shared_ptr<Graph>& graph,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);

// 消除死代码，适用于给定的块对象
TORCH_API void EliminateDeadCode(
    Block* block,
    bool recurse = true,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);

// 在删除任何内容之前，调用用户提供的回调函数处理所有活跃值
TORCH_API void EliminateDeadCode(
    Block* block,
    std::function<void(const std::unordered_set<const Value*>&)> cb,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);

} // namespace jit
} // namespace torch
```