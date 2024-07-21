# `.\pytorch\torch\csrc\api\src\data\samplers\random.cpp`

```py
namespace torch {
namespace data {
namespace samplers {
// RandomSampler 类的构造函数，接受一个大小和索引数据类型作为参数
RandomSampler::RandomSampler(int64_t size, Dtype index_dtype)
    // 使用 torch::randperm 函数生成一个大小为 size 的随机排列的索引张量
    : indices_(torch::randperm(size, index_dtype)) {}

// RandomSampler 类的析构函数，默认实现
RandomSampler::~RandomSampler() = default;

// 重置采样器状态，可选参数 new_size 指定新的大小
void RandomSampler::reset(optional<size_t> new_size) {
  // 如果未指定新大小，则默认使用当前 indices_ 张量的大小
  const auto size = new_size.value_or(static_cast<size_t>(indices_.numel()));
  // 使用 torch::randperm 重新生成大小为 size 的随机排列的索引张量
  indices_ = torch::randperm(size, indices_.options());
  // 重置索引位置为 0
  index_ = 0;
}

// 获取下一个批次的索引，返回一个可选的大小为 batch_size 的索引向量
optional<std::vector<size_t>> RandomSampler::next(size_t batch_size) {
  // 断言当前索引位置不超过 indices_ 张量的元素数量
  AT_ASSERT(index_ <= indices_.numel());
  // 计算剩余的索引数量
  const size_t remaining_indices = indices_.numel() - index_;
  // 如果剩余索引数为 0，则返回空的可选值
  if (remaining_indices == 0) {
    return nullopt;
  }
  // 创建一个大小为 batch_size 的索引向量，最大不超过剩余索引数量
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  // 从 indices_ 张量中切片获取当前批次的索引
  auto slice = indices_.slice(/*dim=*/0, index_, index_ + index_batch.size());
  // 将切片的数据类型提升为 int64_t 类型
  slice = slice.to(torch::kInt64);
  // 获取切片数据的指针，并将数据复制到 index_batch 向量中
  const auto* data = slice.const_data_ptr<int64_t>();
  std::copy(data, data + index_batch.size(), index_batch.begin());
  // 更新当前索引位置
  index_ += index_batch.size();
  // 返回批次索引向量
  return index_batch;
}

// 将 RandomSampler 的状态保存到序列化输出档案中
void RandomSampler::save(serialize::OutputArchive& archive) const {
  // 将 index_ 变量保存为 int64_t 类型的张量
  archive.write(
      "index",
      torch::tensor(static_cast<int64_t>(index_), torch::kInt64),
      /*is_buffer=*/true);
  // 将 indices_ 张量保存到序列化输出档案中
  archive.write(
      "indices",
      indices_,
      /*is_buffer=*/true);
}

// 从序列化输入档案中加载 RandomSampler 的状态
void RandomSampler::load(serialize::InputArchive& archive) {
  // 创建一个空的 int64_t 类型的张量
  auto tensor = torch::empty(1, torch::kInt64);
  // 从序列化输入档案中读取 index_ 变量的值
  archive.read(
      "index",
      tensor,
      /*is_buffer=*/true);
  // 将读取的值赋给 index_ 变量
  index_ = tensor.item<int64_t>();
  // 从序列化输入档案中读取 indices_ 张量的值
  archive.read(
      "indices",
      indices_,
      /*is_buffer=*/true);
}

// 返回当前索引的值
size_t RandomSampler::index() const noexcept {
  return index_;
}

} // namespace samplers
} // namespace data
} // namespace torch
```