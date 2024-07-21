# `.\pytorch\test\cpp\api\expanding-array.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <c10/util/irange.h> // 包含用于生成整数范围的头文件
#include <torch/torch.h> // 包含 PyTorch 核心头文件

#include <test/cpp/api/support.h> // 包含测试支持函数的头文件

#include <cstddef> // 包含标准库的头文件，定义了 nullptr_t 等
#include <initializer_list> // 包含定义了 std::initializer_list 类型的头文件
#include <vector> // 包含定义了 std::vector 类型的头文件

// 定义测试结构体，继承自 torch::test::SeedingFixture，用于测试 ExpandingArray 类
struct ExpandingArrayTest : torch::test::SeedingFixture {};

// 测试用例，验证从初始化列表构造 ExpandingArray 的功能
TEST_F(ExpandingArrayTest, CanConstructFromInitializerList) {
  torch::ExpandingArray<5> e({1, 2, 3, 4, 5}); // 创建 ExpandingArray 对象 e，使用初始化列表 {1, 2, 3, 4, 5} 初始化
  ASSERT_EQ(e.size(), 5); // 断言 e 的大小为 5
  // 遍历 e 的每个元素，验证其值是否符合预期
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

// 测试用例，验证从 std::vector 构造 ExpandingArray 的功能
TEST_F(ExpandingArrayTest, CanConstructFromVector) {
  torch::ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5}); // 创建 ExpandingArray 对象 e，使用 std::vector 初始化
  ASSERT_EQ(e.size(), 5); // 断言 e 的大小为 5
  // 遍历 e 的每个元素，验证其值是否符合预期
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

// 测试用例，验证从 std::array 构造 ExpandingArray 的功能
TEST_F(ExpandingArrayTest, CanConstructFromArray) {
  torch::ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5})); // 创建 ExpandingArray 对象 e，使用 std::array 初始化
  ASSERT_EQ(e.size(), 5); // 断言 e 的大小为 5
  // 遍历 e 的每个元素，验证其值是否符合预期
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

// 测试用例，验证从单个值构造 ExpandingArray 的功能
TEST_F(ExpandingArrayTest, CanConstructFromSingleValue) {
  torch::ExpandingArray<5> e(5); // 创建 ExpandingArray 对象 e，使用单个值 5 初始化
  ASSERT_EQ(e.size(), 5); // 断言 e 的大小为 5
  // 遍历 e 的每个元素，验证其值是否为初始化的单个值 5
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], 5);
  }
}

// 测试用例，验证在初始化列表中提供的参数数量不正确时是否抛出异常
TEST_F(
    ExpandingArrayTest,
    ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInInitializerList) {
  // 使用初始化列表 {1, 2, 3, 4, 5, 6, 7} 构造 ExpandingArray，断言是否抛出预期的异常信息
  ASSERT_THROWS_WITH(
      torch::ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
      "Expected 5 values, but instead got 7");
}

// 测试用例，验证在 std::vector 中提供的参数数量不正确时是否抛出异常
TEST_F(
    ExpandingArrayTest,
    ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInVector) {
  // 使用 std::vector 初始化 {1, 2, 3, 4, 5, 6, 7} 构造 ExpandingArray，断言是否抛出预期的异常信息
  ASSERT_THROWS_WITH(
      torch::ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
      "Expected 5 values, but instead got 7");
}
```