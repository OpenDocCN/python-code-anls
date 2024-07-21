# `.\pytorch\torch\csrc\api\include\torch\data\datasets\stateful.h`

```
#pragma once
// 预处理指令，确保此头文件仅被编译一次

#include <torch/data/datasets/base.h>
// 包含 Torch 数据集基类的头文件

#include <torch/data/example.h>
// 包含 Torch 数据示例的头文件

#include <cstddef>
// 包含标准库的头文件，定义了 size_t 类型等

#include <vector>
// 包含标准库的头文件，定义了 vector 容器

namespace torch {
namespace serialize {
// 命名空间 torch 下的 serialize 命名空间

class OutputArchive;
class InputArchive;
// 声明 OutputArchive 和 InputArchive 类

} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace datasets {

/// A stateful dataset is a dataset that maintains some internal state, which
/// will be `reset()` at the beginning of each epoch. Subclasses can override
/// the `reset()` method to configure this behavior. Further, the return type of
/// a stateful dataset's `get_batch()` method is always an `optional`. When the
/// stateful dataset wants to indicate to the dataloader that its epoch has
/// ended, it should return an empty optional. The dataloader knows to modify
/// its implementation based on whether the dataset is stateless or stateful.
///
/// Note that when subclassing a from `StatefulDataset<Self, T>`, the return
/// type of `get_batch()`, which the subclass must override, will be
/// `optional<T>` (i.e. the type specified in the `StatefulDataset`
/// specialization is automatically boxed into an `optional` for the dataset's
/// `BatchType`).
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = size_t>
class StatefulDataset
    : public BatchDataset<Self, optional<Batch>, BatchRequest> {
    // StatefulDataset 类模板，继承自 BatchDataset，使用 optional 包装批量数据

 public:
  /// Resets internal state of the dataset.
  virtual void reset() = 0;
  // 纯虚函数，重置数据集的内部状态

  /// Saves the statefulDataset's state to OutputArchive.
  virtual void save(serialize::OutputArchive& archive) const = 0;
  // 纯虚函数，将 StatefulDataset 的状态保存到 OutputArchive

  /// Deserializes the statefulDataset's state from the `archive`.
  virtual void load(serialize::InputArchive& archive) = 0;
  // 纯虚函数，从 InputArchive 反序列化 StatefulDataset 的状态
};

/// Serializes a statefulDataset to `OutputArchive`.
template <typename... Args>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const StatefulDataset<Args...>& statefulDataset) {
  statefulDataset.save(archive);
  return archive;
}
// 重载操作符 << ，用于将 StatefulDataset 序列化到 OutputArchive

/// Deserializes a statefulDataset from an `InputArchive`.
template <typename... Args>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    StatefulDataset<Args...>& statefulDataset) {
  statefulDataset.load(archive);
  return archive;
}
// 重载操作符 >> ，用于从 InputArchive 反序列化 StatefulDataset

} // namespace datasets
} // namespace data
} // namespace torch
```