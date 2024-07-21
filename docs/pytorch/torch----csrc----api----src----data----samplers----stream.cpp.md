# `.\pytorch\torch\csrc\api\src\data\samplers\stream.cpp`

```py
// 包含必要的头文件以及命名空间声明
#include <torch/data/samplers/stream.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>

// 定义 torch 命名空间下的 data 子命名空间和 samplers 子命名空间
namespace torch {
namespace data {
namespace samplers {

// BatchSize 类的构造函数，初始化批次大小
BatchSize::BatchSize(size_t size) : size_(size) {}

// 返回批次大小的成员函数
size_t BatchSize::size() const noexcept {
  return size_;
}

// BatchSize 类型转换为 size_t 类型的操作符重载
BatchSize::operator size_t() const noexcept {
  return size_;
}

// StreamSampler 类的构造函数，初始化 epoch 大小
StreamSampler::StreamSampler(size_t epoch_size) : epoch_size_(epoch_size) {}

// 重置抽样器状态的函数，根据需要更新 epoch 大小
void StreamSampler::reset(optional<size_t> new_size) {
  if (new_size.has_value()) {
    epoch_size_ = *new_size;
  }
  examples_retrieved_so_far_ = 0;
}

// 获取下一个批次大小的函数，返回一个 optional<BatchSize> 对象
optional<BatchSize> StreamSampler::next(size_t batch_size) {
  // 断言已检索的示例数量不超过 epoch 大小
  AT_ASSERT(examples_retrieved_so_far_ <= epoch_size_);

  // 如果已经达到 epoch 大小，返回空的 optional 对象
  if (examples_retrieved_so_far_ == epoch_size_) {
    return nullopt;
  }

  // 如果剩余的示例数量不足一个批次大小，将批次大小调整为剩余数量
  if (examples_retrieved_so_far_ + batch_size > epoch_size_) {
    batch_size = epoch_size_ - examples_retrieved_so_far_;
  }

  // 更新已检索的示例数量并返回相应的 BatchSize 对象
  examples_retrieved_so_far_ += batch_size;
  return BatchSize(batch_size);
}

// 将对象状态保存到序列化输出存档的函数
void StreamSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "examples_retrieved_so_far", // 存档中的字段名
      torch::tensor(
          static_cast<int64_t>(examples_retrieved_so_far_), torch::kInt64), // 将状态转换为张量并写入存档
      /*is_buffer=*/true); // 指示写入的数据是一个缓冲区
}

// 从序列化输入存档中加载对象状态的函数
void StreamSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64); // 创建一个大小为 1 的 int64 的张量
  archive.read(
      "examples_retrieved_so_far", // 从存档中读取的字段名
      tensor, // 将读取的数据存储到张量中
      /*is_buffer=*/true); // 指示读取的数据是一个缓冲区
  examples_retrieved_so_far_ = tensor.item<int64_t>(); // 从张量中提取 int64 值并更新对象状态
}

} // namespace samplers
} // namespace data
} // namespace torch
```