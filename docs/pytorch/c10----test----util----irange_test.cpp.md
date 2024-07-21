# `.\pytorch\c10\test\util\irange_test.cpp`

```py
// 包含头文件 <c10/util/irange.h>，这个头文件提供了用于生成整数序列的函数
#include <c10/util/irange.h>

// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 使用 testing 命名空间
using namespace ::testing;

// 定义 irangeTest 测试用例，测试 c10::irange 函数的功能
TEST(irangeTest, range_test) {
  // 创建一个空的整数向量 test_vec
  std::vector<int> test_vec;
  // 使用 c10::irange 生成整数序列 [4, 5, 6, 7, 8, 9, 10]，并将其添加到 test_vec 中
  for (const auto i : c10::irange(4, 11)) {
    test_vec.push_back(i);
  }
  // 定义预期的正确结果向量
  const std::vector<int> correct = {{4, 5, 6, 7, 8, 9, 10}};
  // 断言 test_vec 是否与 correct 相等
  ASSERT_EQ(test_vec, correct);
}

// 定义 irangeTest 测试用例，测试 c10::irange 函数的功能
TEST(irangeTest, end_test) {
  // 创建一个空的整数向量 test_vec
  std::vector<int> test_vec;
  // 使用 c10::irange 生成整数序列 [0, 1, 2, 3, 4]，并将其添加到 test_vec 中
  for (const auto i : c10::irange(5)) {
    test_vec.push_back(i);
  }
  // 定义预期的正确结果向量
  const std::vector<int> correct = {{0, 1, 2, 3, 4}};
  // 断言 test_vec 是否与 correct 相等
  ASSERT_EQ(test_vec, correct);
}

// 定义 irangeTest 测试用例，测试 c10::irange 函数的功能
TEST(irangeTest, neg_range_test) {
  // 创建一个空的整数向量 test_vec
  std::vector<int> test_vec;
  // 使用 c10::irange 生成整数序列 [-2, -1, 0, 1, 2]，并将其添加到 test_vec 中
  for (const auto i : c10::irange(-2, 3)) {
    test_vec.push_back(i);
  }
  // 定义预期的正确结果向量
  const std::vector<int> correct = {{-2, -1, 0, 1, 2}};
  // 断言 test_vec 是否与 correct 相等
  ASSERT_EQ(test_vec, correct);
}

// 定义 irange 测试用例，测试 c10::irange 函数的功能，生成空的整数序列
TEST(irange, empty_reverse_range_two_inputs) {
  // 创建一个空的整数向量 test_vec
  std::vector<int> test_vec;
  // 使用 c10::irange 生成空的整数序列，并将其添加到 test_vec 中
  for (const auto i : c10::irange(3, -3)) {
    test_vec.push_back(i);
    // 如果 i 大于 20，跳出循环以限制添加元素的数量
    if (i > 20) { // Cap the number of elements we add if something goes wrong
      break;
    }
  }
  // 定义预期的正确结果向量（空向量）
  const std::vector<int> correct = {};
  // 断言 test_vec 是否与 correct 相等
  ASSERT_EQ(test_vec, correct);
}

// 定义 irange 测试用例，测试 c10::irange 函数的功能，生成空的整数序列
TEST(irange, empty_reverse_range_one_input) {
  // 创建一个空的整数向量 test_vec
  std::vector<int> test_vec;
  // 使用 c10::irange 生成空的整数序列，并将其添加到 test_vec 中
  for (const auto i : c10::irange(-3)) {
    test_vec.push_back(i);
    // 如果 i 大于 20，跳出循环以限制添加元素的数量
    if (i > 20) { // Cap the number of elements we add if something goes wrong
      break;
    }
  }
  // 定义预期的正确结果向量（空向量）
  const std::vector<int> correct = {};
  // 断言 test_vec 是否与 correct 相等
  ASSERT_EQ(test_vec, correct);
}
```