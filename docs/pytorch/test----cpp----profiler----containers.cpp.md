# `.\pytorch\test\cpp\profiler\containers.cpp`

```py
// 包含标准库头文件
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 C10 库的头文件
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/util.h>

// 定义 ProfilerTest 测试套件，测试 AppendOnlyList 类
TEST(ProfilerTest, AppendOnlyList) {
  const int n = 4096;
  // 创建 AppendOnlyList 对象，存储整数，最大容量为 1024
  torch::profiler::impl::AppendOnlyList<int, 1024> list;
  // 向列表中插入 n 个整数，每次验证插入后列表的大小是否正确
  for (const auto i : c10::irange(n)) {
    list.emplace_back(i);
    ASSERT_EQ(list.size(), i + 1);
  }

  int expected = 0;
  // 遍历列表，验证元素值是否按顺序递增
  for (const auto i : list) {
    ASSERT_EQ(i, expected++);
  }
  // 验证期望值是否等于 n，即列表中是否有 n 个元素
  ASSERT_EQ(expected, n);

  // 清空列表，验证列表大小是否为 0
  list.clear();
  ASSERT_EQ(list.size(), 0);
}

// 定义 ProfilerTest 测试套件，测试 AppendOnlyList 类（引用方式）
TEST(ProfilerTest, AppendOnlyList_ref) {
  const int n = 512;
  // 创建 AppendOnlyList 对象，存储键值对，最大容量为 64
  torch::profiler::impl::AppendOnlyList<std::pair<int, int>, 64> list;
  // 创建存储指向列表元素的指针的向量
  std::vector<std::pair<int, int>*> refs;
  // 向列表中插入 n 个键值对，并将每个元素的指针存储到 refs 中
  for (const auto _ : c10::irange(n)) {
    refs.push_back(list.emplace_back());
  }

  // 遍历列表，为每个元素赋值 {i, 0}
  for (const auto i : c10::irange(n)) {
    *refs.at(i) = {i, 0};
  }

  int expected = 0;
  // 遍历列表，验证每个元素的第一个值是否按顺序递增
  for (const auto& i : list) {
    ASSERT_EQ(i.first, expected++);
  }
}

// Test that we can convert TSC measurements back to wall clock time.
// 测试 TSC 测量结果能否转换为墙钟时间
TEST(ProfilerTest, clock_converter) {
  const int n = 10001;
  // 创建 ApproximateClockToUnixTimeConverter 对象
  c10::ApproximateClockToUnixTimeConverter converter;
  // 创建存储 TSC 测量结果和对应墙钟时间的向量
  std::vector<
      c10::ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair>
      pairs;
  // 进行 n 次 TSC 测量，并将结果存储到 pairs 中
  for (const auto i : c10::irange(n)) {
    pairs.push_back(c10::ApproximateClockToUnixTimeConverter::measurePair());
  }
  // 创建计数转换器
  auto count_to_ns = converter.makeConverter();
  // 创建存储测量结果差值的向量
  std::vector<int64_t> deltas;
  // 计算每对测量结果的差值，并将结果存储到 deltas 中
  for (const auto& i : pairs) {
    deltas.push_back(i.t_ - count_to_ns(i.approx_t_));
  }
  // 对差值向量进行排序
  std::sort(deltas.begin(), deltas.end());

  // 在单元测试中使用时钟可能导致不稳定性，以下方式来减少不稳定性：
  //   1) 验证时钟本身。虽然完成任务的时间可能有所变化，但两个测量相同时间的时钟应该更为一致。
  //   2) 只测试四分位距。在调用两个定时器之间发生的上下文切换可能导致数百纳秒的噪声，但这种切换只占几个百分点的情况。
  //   3) 我们愿意接受可能由于调用每个时钟的成本差异而产生的较大偏差。
  EXPECT_LT(std::abs(deltas[n / 2]), 200);
  EXPECT_LT(deltas[n * 3 / 4] - deltas[n / 4], 50);
}

// 测试软断言功能
TEST(ProfilerTest, soft_assert) {
  // 验证 SOFT_ASSERT(true) 是否为真
  EXPECT_TRUE(SOFT_ASSERT(true));
  // 设置软断言触发错误时抛出异常
  torch::profiler::impl::setSoftAssertRaises(true);
  // 验证 SOFT_ASSERT(false) 是否会抛出异常
  EXPECT_ANY_THROW(SOFT_ASSERT(false));
  // 取消软断言触发异常设置
  torch::profiler::impl::setSoftAssertRaises(false);
  // 验证 SOFT_ASSERT(false) 是否不会抛出异常
  EXPECT_NO_THROW(SOFT_ASSERT(false));
  // 恢复软断言默认行为
  torch::profiler::impl::setSoftAssertRaises(c10::nullopt);
  // 验证 SOFT_ASSERT(false) 是否不会抛出异常（恢复后）
  EXPECT_NO_THROW(SOFT_ASSERT(false));
}
```