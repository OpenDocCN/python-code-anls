# `.\pytorch\torch\csrc\jit\mobile\train\sequential.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 库的导出功能

#include <torch/data/samplers/base.h>
// 引入 Torch 数据采样器基类

#include <torch/types.h>
// 引入 Torch 类型定义

#include <cstddef>
// C++ 标准库头文件，定义了 size_t 类型

#include <vector>
// 引入标准库向量容器

namespace torch {
namespace serialize {
// Torch 序列化命名空间

class OutputArchive;
class InputArchive;
// 声明输出和输入归档类

} // namespace serialize

} // namespace torch

namespace torch {
namespace jit {
namespace mobile {
// Torch 移动 JIT 命名空间

/// A lighter `Sampler` that returns indices sequentially and cannot be
/// serialized.
// 一个轻量级的采样器，按顺序返回索引，不支持序列化
class TORCH_API SequentialSampler : public torch::data::samplers::Sampler<> {
// 继承自 Torch 数据采样器基类 Sampler

 public:
  /// Creates a `SequentialSampler` that will return indices in the range
  /// `0...size - 1`.
  // 构造函数，创建一个会返回在 `0...size - 1` 范围内索引的 `SequentialSampler`
  explicit SequentialSampler(size_t size);

  /// Resets the `SequentialSampler` to zero.
  // 重置 `SequentialSampler`，将当前索引重置为零
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns the next batch of indices.
  // 返回下一个批次的索引
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Not supported for mobile SequentialSampler
  // 移动设备不支持的序列化函数，不实现
  void save(serialize::OutputArchive& archive) const override;

  /// Not supported for mobile SequentialSampler
  // 移动设备不支持的反序列化函数，不实现
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `SequentialSampler`.
  // 返回当前 `SequentialSampler` 的索引
  size_t index() const noexcept;

 private:
  size_t size_;   // 采样器的大小
  size_t index_{0};  // 当前索引，初始为零
};

} // namespace mobile
} // namespace jit
} // namespace torch
```