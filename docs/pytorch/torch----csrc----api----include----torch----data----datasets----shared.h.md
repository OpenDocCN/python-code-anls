# `.\pytorch\torch\csrc\api\include\torch\data\datasets\shared.h`

```py
#pragma once

#include <torch/data/datasets/base.h>  // 引入 Torch 数据集基础库

#include <memory>  // 引入内存管理相关头文件
#include <utility>  // 引入实用程序组件

namespace torch {
namespace data {
namespace datasets {

/// 将另一个数据集包装在共享指针中，并实现 `BatchDataset` API，将所有调用委托给共享实例。
/// 当希望数据加载器中的所有工作线程访问同一数据集实例时，这将非常有用。
/// 数据集必须自行处理同步和线程安全访问。
///
/// 使用 `torch::data::datasets::make_shared_dataset()` 创建新的 `SharedBatchDataset`，
/// 就像创建 `std::shared_ptr` 一样。
template <typename UnderlyingDataset>
class SharedBatchDataset : public BatchDataset<
                               SharedBatchDataset<UnderlyingDataset>,
                               typename UnderlyingDataset::BatchType,
                               typename UnderlyingDataset::BatchRequestType> {
 public:
  using BatchType = typename UnderlyingDataset::BatchType;
  using BatchRequestType = typename UnderlyingDataset::BatchRequestType;

  /// 从 `shared_ptr` 到 `UnderlyingDataset` 构造一个新的 `SharedBatchDataset`。
  /* implicit */ SharedBatchDataset(
      std::shared_ptr<UnderlyingDataset> shared_dataset)
      : dataset_(std::move(shared_dataset)) {}

  /// 调用底层数据集的 `get_batch` 方法。
  BatchType get_batch(BatchRequestType request) override {
    return dataset_->get_batch(std::move(request));
  }

  /// 返回底层数据集的 `size`。
  optional<size_t> size() const override {
    return dataset_->size();
  }

  /// 访问底层数据集。
  UnderlyingDataset& operator*() {
    return *dataset_;
  }

  /// 访问底层数据集。
  const UnderlyingDataset& operator*() const {
    return *dataset_;
  }

  /// 访问底层数据集。
  UnderlyingDataset* operator->() {
    return dataset_.get();
  }

  /// 访问底层数据集。
  const UnderlyingDataset* operator->() const {
    return dataset_.get();
  }

  /// 调用底层数据集的 `reset()` 方法。
  void reset() {
    dataset_->reset();
  }

 private:
  std::shared_ptr<UnderlyingDataset> dataset_;  // 存储底层数据集的共享指针
};

/// 通过创建 `shared_ptr<UnderlyingDataset>` 构造一个新的 `SharedBatchDataset`。
/// 所有参数都转发到 `make_shared<UnderlyingDataset>`。
template <typename UnderlyingDataset, typename... Args>
SharedBatchDataset<UnderlyingDataset> make_shared_dataset(Args&&... args) {
  return std::make_shared<UnderlyingDataset>(std::forward<Args>(args)...);
}

} // namespace datasets
} // namespace data
} // namespace torch
```