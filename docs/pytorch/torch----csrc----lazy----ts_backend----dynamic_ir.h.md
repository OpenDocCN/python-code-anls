# `.\pytorch\torch\csrc\lazy\ts_backend\dynamic_ir.h`

```py
#pragma once

#include <ATen/core/symbol.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/core/ScalarType.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

// 声明一个全局标志，用于控制是否启用动态形状
C10_DECLARE_bool(ltc_enable_dynamic_shapes);

namespace torch {
namespace lazy {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce DimensionNode, a new type
 * of Node that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```py
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

// 表示在 Tensor 上调用 `size` 方法的结果
class TORCH_API SizeNode : public TsNode, public DimensionNode {
 public:
  // 构造函数，初始化 SizeNode，指定输入值和维度
  SizeNode(Value input, size_t dim);
  // 覆盖父类方法，返回静态值
  int64_t getStaticValue() const override;
  // 覆盖父类方法，判断是否为符号值
  bool isSymbolic() const override;
  // 覆盖父类方法，返回对象的字符串表示
  std::string ToString() const override;
  // 成员变量，表示维度
  size_t dim_ = 0;
  // 覆盖父类方法，执行下层操作，降低到 GraphFunction 中
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;
};

// 表示两个 SizeNode 相加的结果
class TORCH_API SizeAdd : public TsNode, public DimensionNode {
 public:
  // 构造函数，初始化 SizeAdd，指定两个输入值
  SizeAdd(Value a, Value b);
  // 覆盖父类方法，返回静态值
  int64_t getStaticValue() const override;
  // 覆盖父类方法，判断是否为符号值
  bool isSymbolic() const override;
  // 覆盖父类方法，返回对象的字符串表示
  std::string ToString() const override;
};

// 表示两个 SizeNode 相乘的结果
class TORCH_API SizeMul : public TsNode, public DimensionNode {
 public:
  // 构造函数，初始化 SizeMul，指定两个输入值
  SizeMul(Value a, Value b);
  // 覆盖父类方法，返回静态值
  int64_t getStaticValue() const override;
  // 覆盖父类方法，判断是否为符号值
  bool isSymbolic() const override;
  // 覆盖父类方法，返回对象的字符串表示
  std::string ToString() const override;
};

// 表示两个 SizeNode 相除的结果
class TORCH_API SizeDiv : public TsNode, public DimensionNode {
 public:
  // 构造函数，初始化 SizeDiv，指定两个输入值
  SizeDiv(Value a, Value b);
  // 覆盖父类方法，返回静态值
  int64_t getStaticValue() const override;
  // 覆盖父类方法，判断是否为符号值
  bool isSymbolic() const override;
  // 覆盖父类方法，返回对象的字符串表示
  std::string ToString() const override;
};

} // namespace lazy
} // namespace torch
```