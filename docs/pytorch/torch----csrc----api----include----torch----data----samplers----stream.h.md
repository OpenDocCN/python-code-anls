# `.\pytorch\torch\csrc\api\include\torch\data\samplers\stream.h`

```
#pragma once

#include <torch/csrc/Export.h>  // 导出符号定义
#include <torch/data/samplers/base.h>  // 数据采样器基类
#include <torch/data/samplers/custom_batch_request.h>  // 自定义批量请求
#include <torch/types.h>  // Torch 类型定义

#include <cstddef>  // 标准库定义

namespace torch {
namespace serialize {
class InputArchive;
class OutputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace samplers {

/// A wrapper around a batch size value, which implements the
/// `CustomBatchRequest` interface.
struct TORCH_API BatchSize : public CustomBatchRequest {
  explicit BatchSize(size_t size);
  size_t size() const noexcept override;
  operator size_t() const noexcept;
  size_t size_;  // 批量大小成员变量
};

/// A sampler for (potentially infinite) streams of data.
///
/// The major feature of the `StreamSampler` is that it does not return
/// particular indices, but instead only the number of elements to fetch from
/// the dataset. The dataset has to decide how to produce those elements.
class TORCH_API StreamSampler : public Sampler<BatchSize> {
 public:
  /// Constructs the `StreamSampler` with the number of individual examples that
  /// should be fetched until the sampler is exhausted.
  explicit StreamSampler(size_t epoch_size);

  /// Resets the internal state of the sampler.
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns a `BatchSize` object with the number of elements to fetch in the
  /// next batch. This number is the minimum of the supplied `batch_size` and
  /// the difference between the `epoch_size` and the current index. If the
  /// `epoch_size` has been reached, returns an empty optional.
  optional<BatchSize> next(size_t batch_size) override;

  /// Serializes the `StreamSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `StreamSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

 private:
  size_t examples_retrieved_so_far_ = 0;  // 已获取的样本数
  size_t epoch_size_;  // Epoch 大小
};

} // namespace samplers
} // namespace data
} // namespace torch
```