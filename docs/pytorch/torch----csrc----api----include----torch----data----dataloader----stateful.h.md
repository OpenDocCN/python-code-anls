# `.\pytorch\torch\csrc\api\include\torch\data\dataloader\stateful.h`

```
#pragma once

#pragma once 指令：确保头文件只被编译一次，防止多重包含。


#include <c10/util/irange.h>
#include <torch/data/dataloader/base.h>

#include <cstddef>
#include <thread>
#include <utility>

包含必要的头文件：引入依赖的C++头文件，包括C10的工具库、PyTorch数据加载器基类以及标准库头文件。


namespace torch {
namespace data {

命名空间定义：进入 torch::data 命名空间。


/// A dataloader for stateful datasets.
///
/// A dataloader for stateful datatasets differs from one for stateless
/// datasets one in that the dataset is shared among worker threads, and that
/// this dataset is itself responsible for producing batches rather than
/// depending on a sampler. The statefulness here actually refers to the
/// dataset. The StatefulDataLoader simply alters the data loading algorithm to
/// accommodate the stateful, shared nature of the dataset. Note that the
/// dataset must be thread safe if more than one worker thread is used.
///
/// A stateful dataloader is created by calling `make_data_loader` with a
/// stateful dataset.
template <typename Dataset>
class StatefulDataLoader : public DataLoaderBase<
                               Dataset,
                               typename Dataset::BatchType::value_type,
                               typename Dataset::BatchRequestType> {
 public:
  using super = DataLoaderBase<
      Dataset,
      typename Dataset::BatchType::value_type,
      typename Dataset::BatchRequestType>;
  using typename super::BatchRequestType;

StatefulDataLoader 类定义：定义了一个用于状态化数据集的数据加载器类。它继承自 DataLoaderBase 类，负责管理数据加载相关的操作。


  /// Constructs the `StatefulDataLoader` from a `dataset` and some `options`.
  StatefulDataLoader(Dataset dataset, DataLoaderOptions options)
      : super(options, std::make_unique<Dataset>(std::move(dataset))) {
    for ([[maybe_unused]] const auto _ : c10::irange(this->options_.workers)) {
      // As opposed to the stateless case, here all worker threads access the
      // same underlying dataset.
      this->workers_.emplace_back(
          [this] { this->worker_thread(*this->main_thread_dataset_); });
    }
  }

StatefulDataLoader 构造函数：从给定的 dataset 和 options 构造一个 StatefulDataLoader 对象。使用了 workers 数量的线程来访问共享的 dataset，并创建对应的 worker 线程。


 private:
  /// Resets the internal state of the dataloader and the dataset.
  void reset() override {
    this->main_thread_dataset_->reset();
    // Call the base class method last because it calls `prefetch()`
    super::reset();
  }

reset 方法：重置数据加载器和数据集的内部状态。首先调用主线程数据集的 reset 方法，然后调用基类的 reset 方法，后者会调用 prefetch 方法。


  /// For stateful datasets, the batch request is always the batch size. The
  /// dataset is responsible for determining what goes into the batch next.
  optional<BatchRequestType> get_batch_request() override {
    return this->options_.batch_size;
  }
};
} // namespace data
} // namespace torch

get_batch_request 方法：对于状态化数据集，批次请求总是等于批次大小，由数据集确定下一个批次的内容。
```