# `.\pytorch\torch\csrc\api\include\torch\data\samplers\distributed.h`

```
/// 防止头文件被多次包含
#pragma once

/// 导入 Torch 库中的 Export.h 文件
#include <torch/csrc/Export.h>
/// 导入 Torch 数据模块中的基础采样器
#include <torch/data/samplers/base.h>

/// 导入标准库头文件
#include <cstddef>
#include <vector>

/// 定义 Torch 命名空间
namespace torch {
/// 定义数据序列化命名空间
namespace serialize {
/// 前向声明序列化输出存档和输入存档类
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

/// 定义 Torch 数据模块命名空间下的采样器命名空间
namespace torch {
namespace data {
namespace samplers {

/// 分布式采样器类，用于选择要采样的索引子集并定义采样行为。
/// 在分布式设置中，根据提供的 num_replicas 和 rank 参数选择索引的子集。
/// 根据 allow_duplicates 参数，采样器执行基于舍入操作来决定本地采样数量。
template <typename BatchRequest = std::vector<size_t>>
class DistributedSampler : public Sampler<BatchRequest> {
 public:
  /// 构造函数，初始化分布式采样器对象。
  DistributedSampler(
      size_t size,                      // 数据集大小
      size_t num_replicas = 1,          // 副本数，默认为1
      size_t rank = 0,                  // 当前进程的排名，默认为0
      bool allow_duplicates = true)     // 是否允许重复采样，默认允许
      : size_(size),                    // 初始化数据集大小
        num_replicas_(num_replicas),    // 初始化副本数
        rank_(rank),                    // 初始化当前进程的排名
        epoch_(0),                      // 初始化当前 epoch
        allow_duplicates_(allow_duplicates) {}  // 初始化是否允许重复采样

  /// 设置当前枚举的 epoch 值，用于改变采样选择和打乱行为。
  void set_epoch(size_t epoch) {
    epoch_ = epoch;
  }

  /// 返回当前 epoch 值。
  size_t epoch() const {
    return epoch_;
  }

 protected:
  /// 计算本地样本数量，根据 allow_duplicates 参数执行不同的计算。
  size_t local_sample_count() {
    if (allow_duplicates_) {
      return (size_ + num_replicas_ - 1) / num_replicas_;
    } else {
      return size_ / num_replicas_;
    }
  }

  // 下面的成员变量不进行私有成员变量的类内声明检查，以允许在类定义中声明成员变量。
  // 用 NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes) 注释来禁止检查。

  // 数据集大小
  size_t size_;
  // 副本数
  size_t num_replicas_;
  // 当前进程的排名
  size_t rank_;
  // 当前 epoch
  size_t epoch_;
  // 是否允许重复采样
  bool allow_duplicates_;
};

/// 随机选择样本的采样器。每次调用 reset() 时，采样顺序都会被打乱。
/// `TORCH_API`宏指示该类是供 Torch 库外部使用的公共 API。
class TORCH_API DistributedRandomSampler : public DistributedSampler<> {
 public:
  /// 构造函数，初始化分布式随机采样器。
  /// 参数：
  ///   - size: 数据集的大小
  ///   - num_replicas: 参与采样的副本数，默认为1
  ///   - rank: 当前副本的排名，默认为0
  ///   - allow_duplicates: 是否允许重复采样，默认为true
  DistributedRandomSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// 重置 `DistributedRandomSampler` 到一个新的索引集合。
  void reset(optional<size_t> new_size = nullopt) override;

  /// 返回下一个批次的索引集合。
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// 将 `DistributedRandomSampler` 序列化到 `archive` 中。
  void save(serialize::OutputArchive& archive) const override;

  /// 从 `archive` 中反序列化 `DistributedRandomSampler`。
  void load(serialize::InputArchive& archive) override;

  /// 返回当前 `DistributedRandomSampler` 的索引。
  size_t index() const noexcept;

 private:
  /// 填充索引集合。
  void populate_indices();

  size_t begin_index_;     ///< 起始索引
  size_t end_index_;       ///< 结束索引
  size_t sample_index_;    ///< 当前样本索引
  std::vector<size_t> all_indices_;  ///< 所有索引集合
};

/// `TORCH_API`宏指示该类是供 Torch 库外部使用的公共 API。
/// 顺序选择样本。
class TORCH_API DistributedSequentialSampler : public DistributedSampler<> {
 public:
  /// 构造函数，初始化分布式顺序采样器。
  /// 参数：
  ///   - size: 数据集的大小
  ///   - num_replicas: 参与采样的副本数，默认为1
  ///   - rank: 当前副本的排名，默认为0
  ///   - allow_duplicates: 是否允许重复采样，默认为true
  DistributedSequentialSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// 重置 `DistributedSequentialSampler` 到一个新的索引集合。
  void reset(optional<size_t> new_size = nullopt) override;

  /// 返回下一个批次的索引集合。
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// 将 `DistributedSequentialSampler` 序列化到 `archive` 中。
  void save(serialize::OutputArchive& archive) const override;

  /// 从 `archive` 中反序列化 `DistributedSequentialSampler`。
  void load(serialize::InputArchive& archive) override;

  /// 返回当前 `DistributedSequentialSampler` 的索引。
  size_t index() const noexcept;

 private:
  /// 填充索引集合。
  void populate_indices();

  size_t begin_index_;     ///< 起始索引
  size_t end_index_;       ///< 结束索引
  size_t sample_index_;    ///< 当前样本索引
  std::vector<size_t> all_indices_;  ///< 所有索引集合
};

} // namespace samplers
} // namespace data
} // namespace torch
```