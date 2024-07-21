# `.\pytorch\torch\csrc\api\include\torch\data\dataloader.h`

```py
#pragma once

#include <torch/data/dataloader/stateful.h>
#include <torch/data/dataloader/stateless.h>

#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace torch {
namespace data {

/// 创建一个基于无状态 `dataset`、`sampler` 和 `options` 的 `DataLoader` 实例。
template <typename Dataset, typename Sampler>
std::enable_if_t<
    !Dataset::is_stateful,
    std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
make_data_loader(Dataset dataset, Sampler sampler, DataLoaderOptions options) {
  return std::make_unique<StatelessDataLoader<Dataset, Sampler>>(
      std::move(dataset), std::move(sampler), std::move(options));
}

/// 创建一个基于无状态 `dataset` 和 `options` 的 `DataLoader` 实例。
/// 如果未提供 `sampler`，将使用默认的 `RandomSampler`，其大小将根据数据集大小自动构建。
template <typename Sampler = samplers::RandomSampler, typename Dataset>
std::enable_if_t<
    !Dataset::is_stateful && std::is_constructible_v<Sampler, size_t>,
    std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  // 获取数据集的大小
  const optional<size_t> size = dataset.size();
  // 检查数据集是否具有大小，以便构建 Sampler
  TORCH_CHECK(
      size.has_value(),
      "Expected the dataset to be sized in "
      "order to construct the Sampler");
  // 调用上面定义的带有 sampler 参数的 make_data_loader 函数
  return make_data_loader(
      std::move(dataset), Sampler(*size), std::move(options));
}

/// 创建一个基于有状态 `dataset` 和 `options` 的 `DataLoader` 实例。
template <typename Dataset, typename = std::enable_if_t<Dataset::is_stateful>>
std::unique_ptr<StatefulDataLoader<Dataset>> make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  return std::make_unique<StatefulDataLoader<Dataset>>(
      std::move(dataset), std::move(options));
}
} // namespace data
} // namespace torch
```