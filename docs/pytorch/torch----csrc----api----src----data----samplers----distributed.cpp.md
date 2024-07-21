# `.\pytorch\torch\csrc\api\src\data\samplers\distributed.cpp`

```
// 引入必要的头文件
#include <c10/util/irange.h>
#include <torch/data/samplers/distributed.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

// 定义命名空间 torch::data::samplers

namespace torch {
namespace data {
namespace samplers {

// DistributedRandomSampler 类的构造函数，继承自 DistributedSampler
DistributedRandomSampler::DistributedRandomSampler(
    size_t size,                     // 数据集总大小
    size_t num_replicas,             // 副本数
    size_t rank,                     // 当前副本的排名
    bool allow_duplicates)           // 是否允许重复样本
    : DistributedSampler(size, num_replicas, rank, allow_duplicates),  // 调用基类构造函数初始化列表
      begin_index_(0),               // 起始索引
      end_index_(0),                 // 结束索引
      sample_index_(0) {             // 当前样本索引
  // 第一次进行洗牌
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset(size_);
}

// 获取下一个批次的样本索引
optional<std::vector<size_t>> DistributedRandomSampler::next(
    size_t batch_size) {
  if (sample_index_ == end_index_) {
    return nullopt;                 // 如果样本索引已达结束索引，返回空值
  }

  size_t end = sample_index_ + batch_size;
  if (end > end_index_) {
    end = end_index_;
  }

  auto iter = all_indices_.begin();  // 迭代器指向所有索引的起始位置
  std::vector<size_t> res(iter + sample_index_, iter + end);  // 提取当前批次的样本索引
  sample_index_ = end;               // 更新当前样本索引为批次结束位置
  return res;                        // 返回批次的样本索引
}

// 重置样本索引
void DistributedRandomSampler::reset(optional<size_t> new_size) {
  size_ = new_size.value_or(size_);  // 更新数据集大小
  populate_indices();                // 填充索引数组

  std::mt19937 rand(epoch_);         // 随机数生成器，使用 epoch 作为种子
  std::shuffle(all_indices_.begin(), all_indices_.end(), rand);  // 洗牌所有索引
  sample_index_ = begin_index_;       // 重置当前样本索引为起始位置
}

// 填充所有可能的索引
void DistributedRandomSampler::populate_indices() {
  size_t num_local_samples = local_sample_count();  // 获取本地样本数量
  size_t sample_count =
      num_replicas_ == 1 ? size_ : num_local_samples * num_replicas_;  // 计算总样本数量
  all_indices_.resize(sample_count);    // 调整索引数组大小
  std::iota(std::begin(all_indices_), std::end(all_indices_), 0);  // 使用顺序数填充索引数组
  for (const auto i : c10::irange(size_, sample_count)) {
    // 可能会添加重复样本以确保所有副本具有相同数量的样本
    all_indices_[i] = i - size_;       // 为确保所有副本样本数相等，可能会添加重复样本
  }
  begin_index_ = rank_ * num_local_samples;  // 计算起始索引
  end_index_ = begin_index_ + num_local_samples;  // 计算结束索引
  sample_index_ = begin_index_;         // 设置当前样本索引为起始位置
}

// 将当前状态保存到存档
void DistributedRandomSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "sample_index_",                 // 存储样本索引
      torch::tensor(static_cast<int64_t>(sample_index_)),
      /*is_buffer=*/true);
  archive.write(
      "epoch_",                        // 存储 epoch
      torch::tensor(static_cast<int64_t>(epoch_)),
      /*is_buffer=*/true);
}

// 从存档中加载状态
void DistributedRandomSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);  // 创建一个空的 int64 张量
  archive.read("epoch_", tensor, /*is_buffer=*/true);  // 从存档中读取 epoch
  epoch_ = tensor.item<int64_t>();     // 从张量中获取 epoch 值
  // 加载 epoch 后调用 reset() 方法来填充索引
  reset(size_);

  tensor = torch::empty(1, torch::kInt64);  // 再次创建一个空的 int64 张量
  archive.read("sample_index_", tensor, /*is_buffer=*/true);  // 从存档中读取样本索引
  sample_index_ = tensor.item<int64_t>();  // 从张量中获取样本索引值
}

// 返回当前样本索引
size_t DistributedRandomSampler::index() const noexcept {
  return sample_index_;
}

// DistributedSequentialSampler 类的构造函数
DistributedSequentialSampler::DistributedSequentialSampler(
    size_t size,                     // 数据集总大小
    size_t num_replicas,             // 副本数
    size_t rank,                     // 当前副本的排名
    bool allow_duplicates)           // 是否允许重复样本
    // 创建 DistributedSampler 对象，传入参数 size, num_replicas, rank, allow_duplicates
    // 初始化 begin_index_ 为 0
    // 初始化 end_index_ 为 0
    // 初始化 sample_index_ 为 0
    : DistributedSampler(size, num_replicas, rank, allow_duplicates),
      // 调用 populate_indices() 方法，填充索引数据
      begin_index_(0),
      // 将 begin_index_ 初始化为 0
      end_index_(0),
      // 将 end_index_ 初始化为 0
      sample_index_(0) {
    // 初始化 sample_index_ 为 0，并调用 populate_indices() 方法，填充索引数据
    populate_indices();
}

optional<std::vector<size_t>> DistributedSequentialSampler::next(
    size_t batch_size) {
  // 如果当前采样索引等于结束索引，返回空optional
  if (sample_index_ == end_index_) {
    return nullopt;
  }

  // 计算本次采样结束的索引位置
  size_t end = sample_index_ + batch_size;
  if (end > end_index_) {
    end = end_index_;
  }

  // 创建一个包含采样索引范围内所有索引的vector
  std::vector<size_t> res(end - sample_index_);
  std::iota(std::begin(res), std::end(res), sample_index_);

  // 如果结束索引超过数据集大小，将索引进行循环（模）操作
  if (end >= size_) {
    for (size_t& index : res) {
      index = index % size_;
    }
  }

  // 更新采样索引为本次结束的索引位置
  sample_index_ = end;
  return res;
}

void DistributedSequentialSampler::reset(optional<size_t> new_size) {
  // 获取新的数据集大小，如果未提供则使用当前大小
  size_t size = new_size.value_or(size_);
  // 如果数据集大小改变，则更新大小并重新生成索引
  if (size != size_) {
    size_ = size;
    populate_indices();
  } else {
    // 否则重置采样索引为起始索引
    sample_index_ = begin_index_;
  }
}

void DistributedSequentialSampler::populate_indices() {
  // 根据进程编号和本地采样数量计算起始和结束索引
  begin_index_ = rank_ * local_sample_count();
  end_index_ = begin_index_ + local_sample_count();
  // 设置当前采样索引为起始索引
  sample_index_ = begin_index_;
}

void DistributedSequentialSampler::save(
    serialize::OutputArchive& archive) const {
  // 将采样索引以缓冲形式写入存档
  archive.write(
      "sample_index_",
      torch::tensor(static_cast<int64_t>(sample_index_)),
      /*is_buffer=*/true);
}

void DistributedSequentialSampler::load(serialize::InputArchive& archive) {
  // 从存档中读取采样索引，并将其转换为int64_t类型
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read("sample_index_", tensor, /*is_buffer=*/true);
  sample_index_ = tensor.item<int64_t>();
}

size_t DistributedSequentialSampler::index() const noexcept {
  // 返回当前采样索引
  return sample_index_;
}

} // namespace samplers
} // namespace data
} // namespace torch
```