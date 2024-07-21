# `.\pytorch\torch\csrc\jit\mobile\train\random.cpp`

```
// 包含 Torch 的头文件，用于移动端训练中的随机采样器
#include <torch/csrc/jit/mobile/train/random.h>
// 包含 Torch 的基本类型定义
#include <torch/types.h>

// 包含标准库头文件
#include <algorithm>
#include <cstddef>
#include <vector>

// Torch 命名空间开始
namespace torch {
// JIT 命名空间开始
namespace jit {
// 移动端命名空间开始
namespace mobile {

// RandomSampler 类的构造函数，用于初始化随机采样器
RandomSampler::RandomSampler(int64_t size, Dtype index_dtype)
    : indices_(torch::randperm(size, index_dtype)) {}

// RandomSampler 类的析构函数，默认实现
RandomSampler::~RandomSampler() = default;

// 重置随机采样器的状态，可选地更改采样器大小
void RandomSampler::reset(optional<size_t> new_size) {
  // 提示：每次都分配一个新的内存块。希望这在整个 epoch 中是摊销的。
  const auto size = new_size.value_or(static_cast<size_t>(indices_.numel()));
  // 重新生成大小为 `size` 的随机排列索引
  indices_ = torch::randperm(size, indices_.options());
  // 重置当前索引为起始位置
  index_ = 0;
}

// 获取下一个批次的随机索引
optional<std::vector<size_t>> RandomSampler::next(size_t batch_size) {
  // 断言：当前索引不应超过总索引数
  AT_ASSERT(index_ <= indices_.numel());
  // 计算剩余的索引数量
  const size_t remaining_indices = indices_.numel() - index_;
  // 如果没有剩余索引，返回空值
  if (remaining_indices == 0) {
    return nullopt;
  }
  // 创建一个大小为批次大小和剩余索引数量较小值的索引批次向量
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  // 从当前索引开始，复制到批次向量中
  auto slice = indices_.slice(/*dim=*/0, index_, index_ + index_batch.size());
  // 强制将切片转换为 64 位整数类型
  slice = slice.to(torch::kInt64);
  // 获取切片数据指针
  const auto* data = slice.const_data_ptr<int64_t>();
  // 将数据复制到 index_batch 向量中
  std::copy(data, data + index_batch.size(), index_batch.begin());
  // 更新当前索引位置
  index_ += index_batch.size();
  // 返回生成的索引批次向量
  return index_batch;
}

// 保存随机采样器的状态（不支持移动端的序列化）
void RandomSampler::save(serialize::OutputArchive& archive) const {
  TORCH_CHECK(false, "Serialization of RandomSampler not supported on mobile.");
}

// 加载随机采样器的状态（不支持移动端的序列化）
void RandomSampler::load(serialize::InputArchive& archive) {
  TORCH_CHECK(false, "Serialization of RandomSampler not supported on mobile.");
}

// 获取当前索引值的大小
size_t RandomSampler::index() const noexcept {
  return index_;
}

// 移动端命名空间结束
} // namespace mobile
// JIT 命名空间结束
} // namespace jit
// Torch 命名空间结束
} // namespace torch
```