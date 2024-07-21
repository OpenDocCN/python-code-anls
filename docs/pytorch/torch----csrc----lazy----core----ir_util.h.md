# `.\pytorch\torch\csrc\lazy\core\ir_util.h`

```py
#pragma once

#include <unordered_map>
#include <vector>

#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

class TORCH_API Util {
 public:
  // 跟踪节点在后序生成过程中的发射状态。
  // 它有助于跟踪计算图中的循环。
  enum EmitStatus {
    kNotEmitted, // 未发射状态
    kEmitting,   // 正在发射状态
    kEmitted,    // 已发射状态
  };

  using EmissionMap = std::unordered_map<const Node*, EmitStatus>;

  // 从给定节点计算后序顺序，避免使用递归。发射映射可以作为状态保存，用于多次独立调用此 API。
  // 如果节点已在发射映射中被发射，则返回的后序顺序可能为空。
  // 如果检测到循环，则生成错误。
  static std::vector<const Node*> ComputePostOrder(
      const Node* node,
      EmissionMap* emap);

  // 在指定的节点集合上计算后序顺序，不使用递归。
  static std::vector<const Node*> ComputePostOrder(
      c10::ArrayRef<const Node*> nodes,
      EmissionMap* emap);

  // 与上述相同，但在作为参数指定的节点集上计算后序顺序。
  static std::vector<const Node*> ComputePostOrder(
      c10::ArrayRef<const Node*> nodes);

  // 获取图中以节点集合为终点的节点数。
  static size_t GetGraphSize(c10::ArrayRef<const Node*> nodes);
};

} // namespace lazy
} // namespace torch


这些注释解释了C++代码中每个函数、枚举和数据结构的作用和功能，包括如何处理循环、状态跟踪以及计算图节点的处理方式。
```