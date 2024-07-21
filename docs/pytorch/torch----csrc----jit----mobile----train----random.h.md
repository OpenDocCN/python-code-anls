# `.\pytorch\torch\csrc\jit\mobile\train\random.h`

```py
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 的导出相关定义

#include <torch/data/samplers/base.h>
// 引入 Torch 数据采样器的基类定义

#include <torch/types.h>
// 引入 Torch 的数据类型定义

#include <cstddef>
// 引入标准库中的 size_t 类型定义

#include <vector>
// 引入标准库中的 vector 容器定义

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch
// 命名空间 torch 中嵌套命名空间 serialize，声明 OutputArchive 和 InputArchive 类

namespace torch {
namespace jit {
namespace mobile {

/// A lighter `Sampler` that returns indices randomly and cannot be
/// serialized.
// 类 RandomSampler 的注释说明，它是一个轻量级的采样器，返回随机索引，并且无法被序列化。
class TORCH_API RandomSampler : public torch::data::samplers::Sampler<> {
 public:
  /// Constructs a `RandomSampler` with a size and dtype for the stored indices.
  ///
  /// The constructor will eagerly allocate all required indices, which is the
  /// sequence `0 ... size - 1`. `index_dtype` is the data type of the stored
  /// indices. You can change it to influence memory usage.
  // 构造函数，使用给定大小和存储索引的数据类型 `index_dtype` 构造 RandomSampler。
  // 构造函数会立即分配所有必需的索引，即序列 `0 ... size - 1`。
  explicit RandomSampler(int64_t size, Dtype index_dtype = torch::kInt64);

  ~RandomSampler() override;
  // 析构函数，用于清理资源

  /// Resets the `RandomSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;
  // 重置 RandomSampler 到一个新的索引集合

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;
  // 返回下一个批次的索引集合

  /// Serializes the `RandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;
  // 将 RandomSampler 序列化到给定的 OutputArchive 中

  /// Deserializes the `RandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;
  // 从给定的 InputArchive 中反序列化 RandomSampler

  /// Returns the current index of the `RandomSampler`.
  size_t index() const noexcept;
  // 返回 RandomSampler 的当前索引值

 private:
  at::Tensor indices_;
  // 存储索引的 Tensor 对象

  int64_t index_ = 0;
  // 当前索引值
};

} // namespace mobile
} // namespace jit
} // namespace torch
// 命名空间 torch 中嵌套命名空间 jit，然后嵌套命名空间 mobile，最终声明 RandomSampler 类
```