# `.\pytorch\aten\src\ATen\test\stride_properties_test.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 的头文件

#include <ATen/ATen.h> // 包含 PyTorch 的 ATen 头文件

using namespace at; // 使用 ATen 命名空间

// TODO: failing sizes {4, 1, 4, 1}
std::vector<std::vector<int64_t>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};

// 检查给定张量的步幅索引是否符合指定的内存格式
inline bool CheckStrideIndices(const Tensor& t, at::MemoryFormat format) {
  size_t n_dim = t.dim(); // 获取张量的维度数
  std::vector<size_t> stride_indices(n_dim); // 创建一个存储步幅索引的向量

  if (format == at::MemoryFormat::ChannelsLast) {
    // 当内存格式为 ChannelsLast 时，步幅索引应为 {1, n-1, n-2, ..., 2, 0}
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2); // 生成逆序排列的索引
    stride_indices[0] = 1; // 第一个索引为 1
    stride_indices[n_dim - 1] = 0; // 最后一个索引为 0
  } else if (format == at::MemoryFormat::Contiguous) {
    // 当内存格式为 Contiguous 时，步幅索引应为 {n-1, n-2, n-3, ..., 0}
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0); // 生成逆序排列的索引
  } else {
    TORCH_INTERNAL_ASSERT(false, "not recognized memory format"); // 如果格式不被识别，抛出错误
  }

  // 使用 TensorType::create 创建张量类型，并检查步幅属性是否完整
  auto tt = TensorType::create(c10::nullopt, c10::nullopt, t.sizes(), t.strides(), c10::nullopt);
  TORCH_INTERNAL_ASSERT(tt->stride_properties().isComplete(), "complete stride properties is needed for the test");

  auto index_iter = stride_indices.begin(); // 迭代步幅索引的迭代器
  for (const auto& opt_stride : *tt->stride_properties().sizes()) {
    // 检查计算的步幅索引是否与期望的索引相符
    if (*index_iter++ != opt_stride->stride_index_.value()) {
      return false; // 不符合则返回 false
    }
  }

  return true; // 如果全部符合则返回 true
}

// 测试步幅索引是否正确的 Google Test 测试用例
TEST(StridePropertiesTest, StrideIndicesTest) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (const auto& size : sizes) {
    Tensor t = at::rand(size); // 使用随机值创建张量
    for (auto memory_format : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::Contiguous}) {
      t.resize_(size, memory_format); // 调整张量大小和内存格式
      EXPECT_TRUE(CheckStrideIndices(t, memory_format)); // 断言步幅索引检查通过
    }
  }
}

// 测试零步幅索引情况下的一致性的 Google Test 测试用例
TEST(StridePropertiesTest, ZeroStrideIndicesEagerConsistencyTest) {
  auto permuted_tensor = at::rand({6, 3, 1, 5, 2}).permute({0, 3, 2, 1, 4}); // 维度 1 和 3 进行置换
  auto tensor = permuted_tensor.expand({6, 5, 4, 3, 2}); // 扩展维度 2

  auto temp = TensorType::create(c10::nullopt, c10::nullopt, tensor.sizes(), tensor.strides(), c10::nullopt);

  // TensorIterator 会保持步幅顺序，这是急切引用
  auto eager_tensor = tensor.relu(); // 计算张量的 ReLU
  auto ref_type = TensorType::create(c10::nullopt, c10::nullopt, eager_tensor.sizes(), eager_tensor.strides(), c10::nullopt);

  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test"); // 断言步幅属性完整性

  auto ref_iter = (*(ref_type->stride_properties().sizes())).begin(); // 引用迭代器
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    EXPECT_TRUE(opt_stride->stride_index_.value() == (*ref_iter)->stride_index_.value()); // 断言计算的步幅索引与参考相符
    ref_iter++;
  }
}
TEST(StridePropertiesTest, ExpandedStrideIndicesTest) {
  // 创建一个形状为 [1] 的随机张量 t
  Tensor t = at::rand({1});
  // 注意：当维度大小为 1 时，使用 expand 可能会导致步幅顺序不同
  // 取决于展开维度的顺序。
  t = t.expand({4, 4, 4});
  // 断言在内存格式为连续时，检查张量 t 的步幅顺序是否正确
  EXPECT_TRUE(CheckStrideIndices(t, at::MemoryFormat::Contiguous));
}

TEST(StridePropertiesTest, SlicedStrideIndicesTest) {
  // 对切片后的张量，步幅顺序不应该改变
  Tensor t = at::rand({16, 4}).slice(1, 0, 4, 4);

  // 创建临时张量对象，复制 t 的形状和步幅
  auto temp = TensorType::create(c10::nullopt, c10::nullopt, t.sizes(), t.strides(), c10::nullopt);
  // 断言临时张量的步幅属性是否完整
  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test");

  // 创建一个逆序的步幅索引向量
  std::vector<size_t> stride_indices(2);
  std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

  // 迭代步幅属性中的每个可选步幅
  auto index_iter = stride_indices.begin();
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    // 断言当前迭代的步幅索引是否等于期望的逆序索引
    EXPECT_TRUE(*index_iter++ == opt_stride->stride_index_.value());
  }
}
```