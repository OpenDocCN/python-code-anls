# `.\pytorch\aten\src\ATen\test\memory_overlapping_test.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 PyTorch 的 ATen 头文件

using namespace at;  // 使用 ATen 命名空间

std::vector<std::vector<int64_t>> sizes = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {3, 1, 2}, {3, 2, 1}, {2, 3, 1}};

TEST(MemoryOverlapTest, TensorExpanded) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::ones({1}).expand(size);  // 创建一个张量 t，内容为 1，并按给定大小 size 进行扩展
    EXPECT_FALSE(t.is_contiguous());  // 断言张量 t 不是连续的
    EXPECT_FALSE(t.is_non_overlapping_and_dense());  // 断言张量 t 不是非重叠且稠密的
  }
}

TEST(MemoryOverlapTest, ScalarExpanded) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::tensor(1).expand(size);  // 创建一个标量张量 t，值为 1，并按给定大小 size 进行扩展
    EXPECT_FALSE(t.is_contiguous());  // 断言张量 t 不是连续的
    EXPECT_FALSE(t.is_non_overlapping_and_dense());  // 断言张量 t 不是非重叠且稠密的
  }
}

TEST(MemoryOverlapTest, NonContiguousTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);  // 创建一个随机张量 t，并对其进行两次转置操作
    if (!t.is_contiguous()) {  // 如果张量 t 不是连续的
      EXPECT_TRUE(t.is_non_overlapping_and_dense());  // 断言张量 t 是非重叠且稠密的
    }
  }
}

TEST(MemoryOverlapTest, NonContiguousExpandedTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);  // 创建一个随机张量 t，并对其进行两次转置操作
    if (!t.is_contiguous()) {  // 如果张量 t 不是连续的
      for (auto size_to_add : {1, 2, 3, 4}) {  // 对每个可能添加的大小进行迭代
        auto transpose_size = t.sizes().vec();  // 获取张量 t 的大小并转为向量
        std::vector<int64_t> expanded_size(transpose_size);  // 创建一个扩展后大小的向量，并复制 transpose_size 的内容
        expanded_size.insert(expanded_size.begin(), size_to_add);  // 在向量开头插入 size_to_add 大小
        auto expanded = t.expand(expanded_size);  // 对张量 t 进行扩展，得到新的张量 expanded
        EXPECT_FALSE(t.is_contiguous());  // 断言张量 t 不是连续的
        if (size_to_add == 1) {  // 如果添加的大小为 1
          EXPECT_TRUE(expanded.is_non_overlapping_and_dense());  // 断言扩展后的张量 expanded 是非重叠且稠密的
        } else {
          EXPECT_FALSE(expanded.is_non_overlapping_and_dense());  // 否则断言扩展后的张量 expanded 不是非重叠且稠密的
        }
      }
    }
  }
}

TEST(MemoryOverlapTest, ContiguousTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::rand(size);  // 创建一个随机张量 t，大小为 size
    EXPECT_TRUE(t.is_contiguous());  // 断言张量 t 是连续的
    EXPECT_TRUE(t.is_non_overlapping_and_dense());  // 断言张量 t 是非重叠且稠密的
  }
}

TEST(MemoryOverlapTest, ContiguousExpandedTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {  // 对 sizes 中的每个大小进行迭代
    Tensor t = at::rand(size);  // 创建一个随机张量 t，大小为 size
    for (auto size_to_add : {1, 2, 3, 4}) {  // 对每个可能添加的大小进行迭代
      std::vector<int64_t> expanded_size(size);  // 创建一个扩展后大小的向量，初始化为 size
      expanded_size.insert(expanded_size.begin(), size_to_add);  // 在向量开头插入 size_to_add 大小
      auto expanded = t.expand(expanded_size);  // 对张量 t 进行扩展，得到新的张量 expanded
      EXPECT_TRUE(t.is_contiguous());  // 断言张量 t 是连续的
      EXPECT_TRUE(t.is_non_overlapping_and_dense());  // 断言张量 t 是非重叠且稠密的
    }
  }
}
```