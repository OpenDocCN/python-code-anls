# `.\pytorch\test\cpp\lazy\test_permutation_util.cpp`

```py
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入 C10 库中的异常处理头文件
#include <c10/util/Exception.h>
// 引入 Torch 模块中懒加载模块的排列工具头文件
#include <torch/csrc/lazy/core/permutation_util.h>

// Torch 命名空间
namespace torch {
// Torch lazy 命名空间
namespace lazy {

// 定义单元测试 PermutationUtilTest.TestInversePermutation
TEST(PermutationUtilTest, TestInversePermutation) {
  // 检查单元素数组的逆置换是否正确
  EXPECT_EQ(InversePermutation({0}), std::vector<int64_t>({0}));
  // 检查连续数组的逆置换是否正确
  EXPECT_EQ(InversePermutation({0, 1, 2}), std::vector<int64_t>({0, 1, 2}));
  // 检查任意排列的逆置换是否正确
  EXPECT_EQ(InversePermutation({1, 3, 2, 0}), std::vector<int64_t>({3, 0, 2, 1}));
  // 检查包含非法元素的排列是否能抛出异常
  EXPECT_THROW(InversePermutation({-1}), c10::Error);
  // 检查重复元素的排列是否能抛出异常
  EXPECT_THROW(InversePermutation({1, 1}), c10::Error);
}

// 定义单元测试 PermutationUtilTest.TestIsPermutation
TEST(PermutationUtilTest, TestIsPermutation) {
  // 检查单元素数组是否是排列
  EXPECT_TRUE(IsPermutation({0}));
  // 检查连续数组是否是排列
  EXPECT_TRUE(IsPermutation({0, 1, 2, 3}));
  // 检查包含非法元素的数组是否不是排列
  EXPECT_FALSE(IsPermutation({-1}));
  // 检查不满足条件的数组是否不是排列
  EXPECT_FALSE(IsPermutation({5, 3}));
  // 检查非连续数组是否不是排列
  EXPECT_FALSE(IsPermutation({1, 2, 3}));
}

// 定义单元测试 PermutationUtilTest.TestPermute
TEST(PermutationUtilTest, TestPermute) {
  // 检查对单元素数组的排列是否正确
  EXPECT_EQ(
      PermuteDimensions({0}, std::vector<int64_t>({224})),
      std::vector<int64_t>({224}));
  // 检查对三维数组的排列是否正确
  EXPECT_EQ(
      PermuteDimensions({1, 2, 0}, std::vector<int64_t>({3, 224, 224})),
      std::vector<int64_t>({224, 224, 3}));
  // 检查包含非法元素的排列是否能抛出异常
  EXPECT_THROW(
      PermuteDimensions({-1}, std::vector<int64_t>({244})), c10::Error);
  // 检查超出范围的排列是否能抛出异常
  EXPECT_THROW(
      PermuteDimensions({3, 2}, std::vector<int64_t>({244})), c10::Error);
  // 检查排列与待排列向量尺寸不匹配是否能抛出异常
  EXPECT_THROW(
      PermuteDimensions({0, 1}, std::vector<int64_t>({244})), c10::Error);
  // 检查排列与待排列向量维度不匹配是否能抛出异常
  EXPECT_THROW(
      PermuteDimensions({0}, std::vector<int64_t>({3, 244, 244})), c10::Error);
}

} // namespace lazy
} // namespace torch


这段代码是用于测试排列工具类函数的单元测试。每个测试函数对应不同的功能测试，并使用 Google Test 框架提供的宏来验证函数的预期行为，包括正确的排列、非法排列的异常处理等。
```