# `.\pytorch\torch\csrc\api\include\torch\data\samplers\base.h`

```py
#pragma once

#include <torch/csrc/Export.h>
#include <torch/types.h>

#include <cstddef>
#include <mutex>
#include <vector>

namespace torch {
namespace serialize {
// 前置声明，命名空间 torch::serialize 包含 OutputArchive 和 InputArchive 类
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace samplers {
/// A `Sampler` is an object that yields an index with which to access a
/// dataset.
// 模板类 Sampler，模板参数为 BatchRequest，默认为 std::vector<size_t>
template <typename BatchRequest = std::vector<size_t>>
class Sampler {
 public:
  using BatchRequestType = BatchRequest;

  virtual ~Sampler() = default;

  /// Resets the `Sampler`'s internal state.
  /// Typically called before a new epoch.
  /// Optionally, accepts a new size when reseting the sampler.
  // 纯虚函数，重置 Sampler 内部状态，可选地接受一个新的大小参数
  virtual void reset(optional<size_t> new_size) = 0;

  /// Returns the next index if possible, or an empty optional if the
  /// sampler is exhausted for this epoch.
  // 纯虚函数，返回下一个索引（如果可能），否则返回空的 optional，表示该 epoch 的采样器已经耗尽
  virtual optional<BatchRequest> next(size_t batch_size) = 0;

  /// Serializes the `Sampler` to the `archive`.
  // 纯虚函数，将 Sampler 序列化到 OutputArchive 中
  virtual void save(serialize::OutputArchive& archive) const = 0;

  /// Deserializes the `Sampler` from the `archive`.
  // 纯虚函数，从 InputArchive 中反序列化 Sampler
  virtual void load(serialize::InputArchive& archive) = 0;
};

} // namespace samplers
} // namespace data
} // namespace torch
```