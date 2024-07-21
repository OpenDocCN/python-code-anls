# `.\pytorch\torch\csrc\api\src\data\samplers\sequential.cpp`

```py
// 包含 Torch 库中的必要头文件
#include <torch/data/samplers/sequential.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

// 包含标准库头文件
#include <algorithm>
#include <cstddef>
#include <vector>

// Torch 命名空间
namespace torch {
namespace data {
namespace samplers {

// 构造函数，初始化顺序采样器对象并设置大小
SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

// 重置采样器，可选地设置新的大小
void SequentialSampler::reset(optional<size_t> new_size) {
  if (new_size.has_value()) {
    size_ = *new_size;
  }
  index_ = 0;  // 重置索引为起始位置
}

// 获取下一个批次的索引集合，如果没有剩余索引则返回空
optional<std::vector<size_t>> SequentialSampler::next(size_t batch_size) {
  const auto remaining_indices = size_ - index_;
  if (remaining_indices == 0) {
    return nullopt;  // 如果没有剩余索引，则返回空
  }
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));  // 创建索引批次向量
  for (auto& i : index_batch) {
    i = index_++;  // 填充批次向量并更新索引
  }
  return index_batch;  // 返回批次索引向量
}

// 将当前对象状态保存到存档中
void SequentialSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "index",
      torch::tensor(static_cast<int64_t>(index_), torch::kInt64),
      /*is_buffer=*/true);  // 将当前索引写入存档
}

// 从存档中加载对象状态
void SequentialSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);  // 创建一个空张量
  archive.read(
      "index",
      tensor,
      /*is_buffer=*/true);  // 从存档中读取索引值
  index_ = tensor.item<int64_t>();  // 将读取的张量转换为整数，并设置为当前索引
}

// 返回当前索引值
size_t SequentialSampler::index() const noexcept {
  return index_;  // 返回当前对象的索引
}

} // namespace samplers
} // namespace data
} // namespace torch
```