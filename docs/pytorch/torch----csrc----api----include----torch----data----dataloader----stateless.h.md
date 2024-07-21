# `.\pytorch\torch\csrc\api\include\torch\data\dataloader\stateless.h`

```
#pragma once

#include <torch/data/dataloader/base.h>  // 引入 Torch 数据加载器基类的头文件
#include <torch/data/worker_exception.h>  // 引入 Torch 数据加载器工作异常的头文件

#include <c10/util/Exception.h>  // 引入 C10 异常处理工具的头文件
#include <c10/util/irange.h>  // 引入 C10 中的整数范围工具

#include <cstddef>  // 引入标准库的 cstddef 头文件，定义了各种标准大小的类型
#include <thread>  // 引入标准库的 thread 头文件，支持多线程编程
#include <utility>  // 引入标准库的 utility 头文件，提供了一些通用工具和函数

namespace torch {
namespace data {

/// A dataloader for stateless datasets.
///
/// This dataloader follows the traditional PyTorch dataloader design, whereby a
/// (posssibly) stateful sampler produces *batch requests* for a stateless
/// dataset, which acts as a simple batch request to batch mapping. The batch
/// request will often be an array of indices, and if the dataset is a simple
/// image dataset, the dataset would produce the images at those indices.
template <typename Dataset, typename Sampler>
class StatelessDataLoader : public DataLoaderBase<
                                Dataset,
                                typename Dataset::BatchType,
                                typename Sampler::BatchRequestType> {
 public:
  using super = DataLoaderBase<
      Dataset,
      typename Dataset::BatchType,
      typename Sampler::BatchRequestType>;
  using typename super::BatchRequestType;

  /// Constructs the `StatelessDataLoader` from a `dataset`, a `sampler` and
  /// some `options`.
  ///
  /// \param dataset The dataset to load from.
  /// \param sampler The sampler used to generate batch requests.
  /// \param options Configuration options for the dataloader.
  StatelessDataLoader(
      Dataset dataset,
      Sampler sampler,
      DataLoaderOptions options)
      : super(std::move(options)), sampler_(std::move(sampler)) {
    for (const auto w : c10::irange(this->options_.workers)) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      (void)w; // Suppress unused variable warning
      this->workers_.emplace_back(
          [this, dataset]() mutable { this->worker_thread(dataset); });
    }
    if (this->options_.workers == 0) {
      this->main_thread_dataset_ =
          std::make_unique<Dataset>(std::move(dataset));
    }
  }

 private:
  /// Resets the internal state of the dataloader and the sampler.
  ///
  /// This function resets the sampler's state and then calls the base class
  /// `reset()` method to perform additional reset operations, which may
  /// include prefetching.
  void reset() override {
    sampler_.reset();  // 重置采样器的内部状态
    // Call the base class method last because it calls `prefetch()`
    super::reset();  // 调用基类的 reset() 方法以完成额外的重置操作
  }

  /// Queries the sampler for the next batch request.
  ///
  /// This function queries the sampler to obtain the next batch request,
  /// which is a collection of indices representing the data batch to load.
  ///
  /// \return An optional batch request, or `nullopt` if no valid batch can be
  ///         generated.
  optional<BatchRequestType> get_batch_request() override {
    auto indices = sampler_.next(this->options_.batch_size);  // 从采样器获取下一个批次请求的索引集合
    if (!indices ||
        (indices->size() < this->options_.batch_size &&
         this->options_.drop_last)) {
      return nullopt;  // 如果索引为空或者索引数小于批次大小且设置了丢弃最后一批次，则返回空
    }
    AT_ASSERT(indices->size() > 0);  // 断言索引集合的大小大于 0
    return indices;  // 返回获取到的索引集合作为批次请求
  }

  /// The `Sampler` used to produce batch requests.
  Sampler sampler_;  // 用于生成批次请求的采样器对象
};

} // namespace data
} // namespace torch
```