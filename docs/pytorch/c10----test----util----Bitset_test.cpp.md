# `.\pytorch\c10\test\util\Bitset_test.cpp`

```py
# 包含 Google 测试框架的头文件
#include <gtest/gtest.h>

# 包含 Bitset 类的头文件
#include <c10/util/Bitset.h>
# 包含 irange 工具函数的头文件
#include <c10/util/irange.h>

# 使用 c10 命名空间中的 Bitset 类
using c10::utils::bitset;

# 测试用例：当空 Bitset 时，检查获取位是否为零
TEST(BitsetTest, givenEmptyBitset_whenGettingBit_thenIsZero) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 遍历每一个位，验证都为假
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 测试用例：当空 Bitset 时，检查取消设置位后是否为零
TEST(BitsetTest, givenEmptyBitset_whenUnsettingBit_thenIsZero) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 取消设置第 4 位
  b.unset(4);
  # 遍历每一个位，验证都为假
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 测试用例：当空 Bitset 时，检查设置和取消设置位后是否为零
TEST(BitsetTest, givenEmptyBitset_whenSettingAndUnsettingBit_thenIsZero) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 4 位
  b.set(4);
  # 取消设置第 4 位
  b.unset(4);
  # 遍历每一个位，验证都为假
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 测试用例：当空 Bitset 时，检查设置某位后是否为真
TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenIsSet) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 位
  b.set(6);
  # 验证第 6 位是否为真
  EXPECT_TRUE(b.get(6));
}

# 测试用例：当空 Bitset 时，检查设置某位后其他位保持未设置
TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenOthersStayUnset) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 位
  b.set(6);
  # 遍历比 6 小的每一个位，验证都为假
  for (const auto i : c10::irange(6)) {
    EXPECT_FALSE(b.get(i));
  }
  # 遍历大于 6 的每一个位，验证都为假
  for (size_t i = 7; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 测试用例：当非空 Bitset 时，检查设置某位后是否为真
TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenIsSet) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 和第 30 位
  b.set(6);
  b.set(30);
  # 验证第 30 位是否为真
  EXPECT_TRUE(b.get(30));
}

# 测试用例：当非空 Bitset 时，检查设置某位后其他位保持旧值未改变
TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenOthersStayAtOldValue) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 和第 30 位
  b.set(6);
  b.set(30);
  # 遍历比 6 小的每一个位，验证都为假
  for (const auto i : c10::irange(6)) {
    EXPECT_FALSE(b.get(i));
  }
  # 遍历比 30 小且大于等于 7 的每一个位，验证都为假
  for (const auto i : c10::irange(7, 30)) {
    EXPECT_FALSE(b.get(i));
  }
  # 遍历大于 30 的每一个位，验证都为假
  for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 测试用例：当非空 Bitset 时，检查取消设置某位后是否为假
TEST(BitsetTest, givenNonemptyBitset_whenUnsettingBit_thenIsUnset) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 和第 30 位
  b.set(6);
  b.set(30);
  # 取消设置第 6 位
  b.unset(6);
  # 验证第 6 位是否为假
  EXPECT_FALSE(b.get(6));
}

# 测试用例：当非空 Bitset 时，检查取消设置某位后其他位保持旧值未改变
TEST(
    BitsetTest,
    givenNonemptyBitset_whenUnsettingBit_thenOthersStayAtOldValue) {
  # 创建一个空的 Bitset 对象
  bitset b;
  # 设置第 6 和第 30 位
  b.set(6);
  b.set(30);
  # 取消设置第 6 位
  b.unset(6);
  # 遍历比 30 小的每一个位，验证都为假
  for (const auto i : c10::irange(30)) {
    EXPECT_FALSE(b.get(i));
  }
  # 验证第 30 位是否为真
  EXPECT_TRUE(b.get(30));
  # 遍历大于 30 的每一个位，验证都为假
  for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

# 定义一个 Mock 类，用于记录回调函数被调用的索引
struct IndexCallbackMock final {
  std::vector<size_t> called_for_indices;

  # 回调函数运算符，记录调用的索引
  void operator()(size_t index) {
    called_for_indices.push_back(index);
  }

  # 期望回调函数被调用的索引与预期索引相符
  void expect_was_called_for_indices(std::vector<size_t> expected_indices) {
    EXPECT_EQ(expected_indices.size(), called_for_indices.size());
    for (const auto i : c10::irange(expected_indices.size())) {
      EXPECT_EQ(expected_indices[i], called_for_indices[i]);
    }
  }
};

# 测试用例：当空 Bitset 时，调用 for_each_set_bit 函数不应调用回调函数
TEST(BitsetTest, givenEmptyBitset_whenCallingForEachBit_thenDoesntCall) {
  # 创建一个空的 IndexCallbackMock 对象作为回调
  IndexCallbackMock callback;
  # 创建一个空的 Bitset 对象
  bitset b;
  # 对空 Bitset 调用 for_each_set_bit 函数
  b.for_each_set_bit(callback);
  # 验证回调函数被调用的索引为空
  callback.expect_was_called_for_indices({});
}

# 测试用例：当只有一个位被设置时，调用 for_each_set_bit 函数应调用回调函数
TEST(
    BitsetTest,
    givenBitsetWithOneBitSet_whenCallingForEachBit_thenCallsForEachBit) {
  # 创建一个空的 IndexCallbackMock 对象作为回调
  IndexCallbackMock callback;
  # 创建一个空的 Bitset 对象，并设置第 5 位
  bitset b;
  b.set(5);
  # 对设置了第 5 位的 Bitset 调用 for_each_set_bit 函数
  b.for_each_set_bit(callback);
  # 验证回调函数被调用的索引为 {5}
  callback.expect_was_called_for_indices({5});
}

# 测试用例：继续定义下一个测试用例（未完成）
TEST(
    BitsetTest,
    givenBitsetWithMultipleBitsSet_whenCallingForEachBit_thenCallsForEachBit) {
  // 创建 IndexCallbackMock 实例用于模拟回调函数
  IndexCallbackMock callback;
  // 创建位集合对象 bitset
  bitset b;
  // 设置位集合中的特定位
  b.set(5);
  b.set(2);
  b.set(25);
  b.set(32);
  b.set(50);
  b.set(0);
  // 清除位集合中的特定位
  b.unset(25);
  b.set(10);
  // 对位集合中设置的每个位调用回调函数
  b.for_each_set_bit(callback);
  // 验证回调函数是否按预期调用，期望传递的参数为索引集合 {0, 2, 5, 10, 32, 50}
  callback.expect_was_called_for_indices({0, 2, 5, 10, 32, 50});
}


注释：


# 这行代码表示一个代码块的结束，通常与一个以关键字（如if、for、while等）开始的代码块配对使用。
```