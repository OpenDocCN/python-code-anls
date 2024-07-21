# `.\pytorch\torch\csrc\lazy\core\dynamic_ir.h`

```py
#pragma once

#include <ATen/core/symbol.h>  // 包含 ATen 核心库中的符号定义

#include <functional>  // 包含函数对象的标准库
#include <memory>  // 包含智能指针和内存管理的标准库
#include <set>  // 包含集合容器的标准库
#include <string>  // 包含字符串操作的标准库
#include <unordered_map>  // 包含无序映射容器的标准库
#include <unordered_set>  // 包含无序集合容器的标准库
#include <utility>  // 包含实用工具的标准库
#include <vector>  // 包含向量容器的标准库

#include <c10/core/ScalarType.h>  // 包含 Tensor 标量类型的定义
#include <c10/util/Flags.h>  // 包含 C10 库中的标志定义
#include <torch/csrc/lazy/core/hash.h>  // 包含 Torch 惰性计算核心模块的哈希函数定义
#include <torch/csrc/lazy/core/ir.h>  // 包含 Torch 惰性计算核心模块的 IR 定义
#include <torch/csrc/lazy/core/ir_metadata.h>  // 包含 Torch 惰性计算核心模块的 IR 元数据定义
#include <torch/csrc/lazy/ts_backend/ts_node.h>  // 包含 Torch 惰性计算后端的节点定义

namespace torch {
namespace lazy {

/**
 * "dynamic" Nodes 的目标是填补我们追踪中的空白。
 * 以前，如果用户在张量上调用 `sizes`，它会泄漏出我们的追踪系统，因为
 * `sizes` 返回 torch.Size 或 int。为了防止这种情况发生，
 * 我们引入 DimensionNode，一个新的节点类型，抽象出获取张量维度的操作。
 *
 * 考虑以下示例：
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```py
 * 这里，`x.shape()[i]` 将是 SizeNode（DimensionNode 的子类），
 * 两个 SizeNode 的乘积将由 SizeMul（也是 DimensionNode 的子类）表示。
 * 通过这种方式，我们可以防止 `numel` 被表示为 Python int，从而烧入图中。
 */
class TORCH_API DimensionNode {
 public:
  // 虚函数，用于判断节点是否是符号化的
  virtual bool isSymbolic() const {
    return false;
  };
  // 虚函数，获取动态值（未实现）
  virtual int64_t getDynamicValue() const {
    TORCH_CHECK(false, "NYI");
  };
  // 虚函数，获取静态值（未实现）
  virtual int64_t getStaticValue() const {
    TORCH_CHECK(false, "NYI");
  };
  // 默认析构函数
  virtual ~DimensionNode() = default;
};

} // namespace lazy
} // namespace torch
```