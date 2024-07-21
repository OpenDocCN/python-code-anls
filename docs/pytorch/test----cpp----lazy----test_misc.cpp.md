# `.\pytorch\test\cpp\lazy\test_misc.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <string>  // 引入 string 头文件

#include <c10/util/int128.h>  // 引入 c10 库的 int128 实用工具头文件
#include <torch/csrc/lazy/core/hash.h>  // 引入 Torch 的哈希核心头文件

namespace torch {
namespace lazy {

template <typename T>
void test_hash_repeatable_sensitive(const T& example_a, const T& example_b) {
  // 测试哈希函数的可重复性
  EXPECT_EQ(Hash(example_a), Hash(example_a));
  EXPECT_EQ(MHash(example_a), MHash(example_a));
  EXPECT_EQ(MHash(example_a, example_a), MHash(example_a, example_a));

  // 测试哈希函数的敏感性
  EXPECT_NE(Hash(example_a), Hash(example_b));
  EXPECT_NE(MHash(example_a), MHash(example_b));
  EXPECT_NE(MHash(example_a, example_a), MHash(example_a, example_b));
}

TEST(HashTest, Scalar) {
  GTEST_SKIP()  // 跳过此单元测试，因为测试有问题，详见 GitHub Issue
      << "Broken test. See https://github.com/pytorch/pytorch/issues/99883";
  c10::Scalar a(0);  // 创建 c10 标量对象 a，初始值为 0
  c10::Scalar b(0);  // 创建 c10 标量对象 b，初始值为 0

  // 模拟 c10::Scalar 中未使用位中的垃圾数据
  *((uint8_t*)&b) = 1;
  // c10::Scalar 中实际的 'value' 应当是一个 64 位整数，对它的 toLong() 结果不应改变
  EXPECT_EQ(a.toLong(), b.toLong());
  // 哈希函数应忽略这些垃圾数据
  EXPECT_EQ(Hash(a), Hash(b));
  EXPECT_EQ(MHash(a), MHash(b));
  EXPECT_EQ(MHash(a, a), MHash(a, b));
}
TEST(HashTest, Sanity) {
  // String类型的测试，比较两个相似但不同的字符串的哈希是否可重复且敏感
  test_hash_repeatable_sensitive(
      std::string(
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut at suscipit purus."),
      std::string(
          "Lorem Jpsum dolor sit amet, consectetur adipiscing elit. Ut at suscipit purus."));

  // 不同数值类型的测试
  test_hash_repeatable_sensitive(true, false);
  test_hash_repeatable_sensitive((int8_t)0xfa, (int8_t)0xfb);
  test_hash_repeatable_sensitive((int16_t)0xface, (int16_t)0xfade);
  test_hash_repeatable_sensitive((int32_t)0xfaceb000, (int32_t)0xfadeb000);
  test_hash_repeatable_sensitive((int64_t)0x1faceb000, (int64_t)0x1fadeb000);
  test_hash_repeatable_sensitive((uint8_t)0xfa, (uint8_t)0xfb);
  test_hash_repeatable_sensitive((uint16_t)0xface, (uint16_t)0xfade);
  test_hash_repeatable_sensitive((uint32_t)0xfaceb000, (uint32_t)0xfadeb000);
  test_hash_repeatable_sensitive((uint64_t)0x1faceb000, (uint64_t)0x1fadeb000);

  // c10类型的测试
  test_hash_repeatable_sensitive(c10::ScalarType::Bool, c10::ScalarType::Byte);
  test_hash_repeatable_sensitive(c10::Scalar(1.334), c10::Scalar(1.335));
  test_hash_repeatable_sensitive(c10::Scalar(true), c10::Scalar(false));
  test_hash_repeatable_sensitive(c10::Scalar(12345), c10::Scalar(12354));

  // std::optional类型的测试
  test_hash_repeatable_sensitive(
      std::optional<std::string>("I have value!"),
      std::optional<std::string>(c10::nullopt));

  // 容器类型的测试
  auto a = std::vector<int32_t>({0, 1, 1, 2, 3, 5, 8});
  auto b = std::vector<int32_t>({1, 1, 2, 3, 5, 8, 12});
  test_hash_repeatable_sensitive(a, b);
  test_hash_repeatable_sensitive(
      c10::ArrayRef<int32_t>(a), c10::ArrayRef<int32_t>(b));

  // vector<bool>是一个特殊情况，因为它实现为vector<bit>
  auto bool_a = std::vector<bool>({true, false, false, true});
  auto bool_b = std::vector<bool>({true, true, false, true});
  test_hash_repeatable_sensitive(bool_a, bool_b);
}

} // namespace lazy
} // namespace torch
```