# `.\pytorch\aten\src\ATen\test\memory_format_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

// 定义测试用例中的各种张量尺寸
std::vector<std::vector<int64_t>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 4, 1}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};

// 内存格式设置测试
TEST(MemoryFormatTest, SetMemoryFormat) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  // 遍历不同尺寸的张量
  for (auto size : sizes) {
    // 创建随机数据张量
    Tensor t = at::rand(size);
    // 测试两种内存格式：ChannelsLast 和 Contiguous
    for (auto memory_format : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::Contiguous}) {
      // 调整张量的尺寸和内存格式
      t.resize_(size, memory_format);
      // 断言当前张量的推荐内存格式是否符合设置的内存格式
      EXPECT_TRUE(t.suggest_memory_format() == memory_format);
    }
  }

  // 创建特定尺寸的随机数据张量
  Tensor t = at::rand({4, 1, 1, 1});
  // 断言推荐内存格式应为 Contiguous
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
  // 调整张量尺寸和内存格式（此处存在一个注释和一个期待的测试修复）
  t.resize_({4, 1, 1, 1}, at::MemoryFormat::ChannelsLast);
  // TODO: Should be able to handle this after accumulated permutation is implemented;
  // Ambiguous case where we fallback to Contiguous;
  // This should be `EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);`
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
}

// 内存格式转置测试
TEST(MemoryFormatTest, TransposeMemoryFormat) {
  // 创建特定尺寸的随机数据张量
  Tensor t = at::rand({2, 3, 4, 5});
  // 断言推荐内存格式应为 Contiguous
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
  // 在指定维度上进行转置操作
  t.transpose_(1, 3);
  // 断言推荐内存格式不应为 ChannelsLast
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t.transpose_(2, 3);
  // 断言推荐内存格式应为 ChannelsLast
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
  
  // 更改张量尺寸并进行转置操作，依次类推进行测试
  t = at::rand({2, 3, 4, 5});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({2, 3, 4, 5});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);

  // 角点测试：
  t = at::rand({1, 4, 1, 4});
  t.transpose_(1, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(2, 3);
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);

  t = at::rand({1, 4, 4, 1});
  t.transpose_(1, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(2, 3);
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
}

// 在指定维度上进行间隔为2的切片操作
inline void sliceStepTwo(Tensor& t, int dim, at::MemoryFormat format) {
  t = t.slice(dim, 0, 3, 2);
  // 断言切片后的张量推荐内存格式是否符合期望的格式
  EXPECT_TRUE(t.suggest_memory_format() == format);
  t = t.slice(dim, 0, 3, 2);
  // 断言再次切片后的张量推荐内存格式是否符合期望的格式
  EXPECT_TRUE(t.suggest_memory_format() == format);
}
// 定义一个名为 MemoryFormatTest 的测试案例，用于测试 sliceStepTwo 函数在不同内存格式下的行为
TEST(MemoryFormatTest, SliceStepTwoMemoryFormat) {
  // 创建一个大小为 [4, 4, 4, 4] 的随机张量 t
  Tensor t = at::rand({4, 4, 4, 4});
  // 对 t 进行 slice 操作，步长为 2，内存格式为 Contiguous
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，进行类似的 slice 操作，顺序不同
  t = at::rand({4, 4, 4, 4});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式
  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式，进行类似的 slice 操作
  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);

  // 创建一个大小为 [4, 4, 1, 1] 的随机张量 t，进行 slice 操作，内存格式为 Contiguous
  t = at::rand({4, 4, 1, 1});
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式
  t = at::rand({4, 4, 1, 1});
  t.resize_({4, 4, 1, 1}, at::MemoryFormat::ChannelsLast);
  // 对 t 在第 1 维度上进行 slice 操作，从索引 0 到 3，步长为 2
  t = t.slice(1, 0, 3, 2);
  // 验证 t 的推荐内存格式是否为 ChannelsLast
  EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
  // 再次对 t 在第 1 维度上进行 slice 操作，验证其推荐内存格式是否为 Contiguous
  t = t.slice(1, 0, 3, 2);
  // TODO: 在累积置换实现后应该能处理这种情况；
  // 无法确定如何到达此处的细节
  // 在第 3 维度上对张量进行两次切片
  // 在第 1 维度上对张量进行两次切片
  // 验证 t 的推荐内存格式是否为 Contiguous
  EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);

  // 创建一个大小为 [4, 1, 4, 4] 的随机张量 t，进行 slice 操作，内存格式为 Contiguous
  t = at::rand({4, 1, 4, 4});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式
  t = at::rand({4, 1, 4, 4});
  t.resize_({4, 1, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  // 创建一个大小为 [4, 1, 1, 4] 的随机张量 t，进行 slice 操作，内存格式为 Contiguous
  t = at::rand({4, 1, 1, 4});
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式
  t = at::rand({4, 1, 1, 4});
  t.resize_({4, 1, 1, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  // 创建一个大小为 [4, 1, 4, 1] 的随机张量 t，进行 slice 操作，内存格式为 Contiguous
  t = at::rand({4, 1, 4, 1});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);

  // 重新生成随机张量 t，并将其调整为 ChannelsLast 内存格式
  t = at::rand({4, 1, 4, 1});
  t.resize_({4, 1, 4, 1}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
}
// 定义一个名为 MemoryFormatTest 的测试，测试切片操作的内存格式
TEST(MemoryFormatTest, SliceFirstMemoryFormat) {
  // 创建一个大小为 [4, 4, 4, 4] 的随机张量 t
  Tensor t = at::rand({4, 4, 4, 4});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 1, MemoryFormat::Contiguous);
  // 再次对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  // 再次对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 3, MemoryFormat::Contiguous);

  // 重新生成一个大小为 [4, 4, 4, 4] 的随机张量 t
  t = at::rand({4, 4, 4, 4});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 1, MemoryFormat::Contiguous);

  // 重新生成一个大小为 [4, 4, 4, 4] 的随机张量 t
  t = at::rand({4, 4, 4, 4});
  // 将 t 重新调整为大小 [4, 4, 4, 4]，并指定内存格式为 ChannelsLast
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  // 重新生成一个大小为 [4, 4, 4, 4] 的随机张量 t
  t = at::rand({4, 4, 4, 4});
  // 将 t 重新调整为大小 [4, 4, 4, 4]，并指定内存格式为 ChannelsLast
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);

  // 生成一个大小为 [4, 4, 1, 1] 的随机张量 t
  t = at::rand({4, 4, 1, 1});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 1, MemoryFormat::Contiguous);
  // 重新生成一个大小为 [4, 4, 1, 1] 的随机张量 t
  t = at::rand({4, 4, 1, 1});
  // 将 t 重新调整为大小 [4, 4, 1, 1]，并指定内存格式为 ChannelsLast
  t.resize_({4, 4, 1, 1}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);

  // 生成一个大小为 [4, 1, 4, 4] 的随机张量 t
  t = at::rand({4, 1, 4, 4});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  // 重新生成一个大小为 [4, 1, 4, 4] 的随机张量 t
  t = at::rand({4, 1, 4, 4});
  // 将 t 重新调整为大小 [4, 1, 4, 4]，并指定内存格式为 ChannelsLast
  t.resize_({4, 1, 4, 4}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  // 生成一个大小为 [4, 1, 1, 4] 的随机张量 t
  t = at::rand({4, 1, 1, 4});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  // 重新生成一个大小为 [4, 1, 1, 4] 的随机张量 t
  t = at::rand({4, 1, 1, 4});
  // 将 t 重新调整为大小 [4, 1, 1, 4]，并指定内存格式为 ChannelsLast
  t.resize_({4, 1, 1, 4}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 ChannelsLast
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  // 生成一个大小为 [4, 1, 4, 1] 的随机张量 t
  t = at::rand({4, 1, 4, 1});
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  // 重新生成一个大小为 [4, 1, 4, 1] 的随机张量 t
  t = at::rand({4, 1, 4, 1});
  // 将 t 重新调整为大小 [4, 1, 4, 1]，并指定内存格式为 ChannelsLast
  t.resize_({4, 1, 4, 1}, at::MemoryFormat::ChannelsLast);
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  // 对 t 进行第一维度的切片操作，指定内存格式为 Contiguous
  // TODO: 在实现累积置换后，应该能够处理这种情况；
  // [4, 1, 4, 1]@[4, 1, 1, 1] 在切片后变成 [4, 1, 1, 1]@[4, 1, 1, 1]
  // sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  sliceFirst(t, 2, MemoryFormat::Contiguous);
}
```