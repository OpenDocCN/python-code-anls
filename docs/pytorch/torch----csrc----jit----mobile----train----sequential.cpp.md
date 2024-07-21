# `.\pytorch\torch\csrc\jit\mobile\train\sequential.cpp`

```py
// 包含 Torch 库中相关的头文件
#include <torch/csrc/jit/mobile/train/sequential.h>
#include <torch/types.h>

// 包含标准库的头文件
#include <algorithm>
#include <cstddef>
#include <vector>

// Torch 命名空间
namespace torch {
namespace jit {
namespace mobile {

// SequentialSampler 类的构造函数，初始化采样器的大小
SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

// 重置采样器状态，可选地更新大小
void SequentialSampler::reset(optional<size_t> new_size) {
  if (new_size.has_value()) {
    size_ = *new_size;
  }
  index_ = 0;  // 重置索引为 0
}

// 获取下一个批次的索引集合，返回一个可选的索引向量
optional<std::vector<size_t>> SequentialSampler::next(size_t batch_size) {
  const auto remaining_indices = size_ - index_;  // 计算剩余的索引数量
  if (remaining_indices == 0) {
    return nullopt;  // 如果没有剩余索引，返回空
  }
  // 创建包含批次大小和剩余索引数量较小值的索引批次向量
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  for (auto& i : index_batch) {
    i = index_++;  // 填充索引批次向量并递增全局索引
  }
  return index_batch;  // 返回索引批次向量
}

// 在移动端环境下不支持序列化操作，抛出异常
void SequentialSampler::save(serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false, "Serialization of SequentialSampler not supported on mobile.");
}

// 在移动端环境下不支持序列化操作，抛出异常
void SequentialSampler::load(serialize::InputArchive& archive) {
  TORCH_CHECK(
      false, "Serialization of SequentialSampler not supported on mobile.");
}

// 返回当前全局索引的值
size_t SequentialSampler::index() const noexcept {
  return index_;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```