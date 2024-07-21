# `.\pytorch\c10\test\util\accumulate_test.cpp`

```py
// 包含预处理命令，引入头文件 c10/util/accumulate.h，用于函数声明和定义
#include <c10/util/accumulate.h>

// 引入 Google Test 框架的头文件，用于测试
#include <gtest/gtest.h>

// 引入 list 和 vector 标准库
#include <list>
#include <vector>

// 使用 testing 命名空间，使测试代码更加简洁
using namespace ::testing;

// 定义测试用例 accumulateTest，针对 vector 的测试
TEST(accumulateTest, vector_test) {
  // 初始化一个包含整数的 vector
  std::vector<int> ints = {1, 2, 3, 4, 5};

  // 测试 c10::sum_integers 函数，预期结果为整数之和
  EXPECT_EQ(c10::sum_integers(ints), 1 + 2 + 3 + 4 + 5);
  // 测试 c10::multiply_integers 函数，预期结果为整数之积
  EXPECT_EQ(c10::multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

  // 测试 c10::sum_integers 函数，传入迭代器范围，预期结果为整数之和
  EXPECT_EQ(c10::sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
  // 测试 c10::multiply_integers 函数，传入迭代器范围，预期结果为整数之积
  EXPECT_EQ(
      c10::multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

  // 测试 c10::sum_integers 函数，传入部分迭代器范围，预期结果为部分整数之和
  EXPECT_EQ(c10::sum_integers(ints.begin() + 1, ints.end() - 1), 2 + 3 + 4);
  // 测试 c10::multiply_integers 函数，传入部分迭代器范围，预期结果为部分整数之积
  EXPECT_EQ(
      c10::multiply_integers(ints.begin() + 1, ints.end() - 1), 2 * 3 * 4);

  // 测试 c10::numelements_from_dim 函数，预期结果为指定维度范围内元素个数的乘积
  EXPECT_EQ(c10::numelements_from_dim(2, ints), 3 * 4 * 5);
  // 测试 c10::numelements_to_dim 函数，预期结果为从开头到指定维度范围内元素个数的乘积
  EXPECT_EQ(c10::numelements_to_dim(3, ints), 1 * 2 * 3);
  // 测试 c10::numelements_between_dim 函数，预期结果为两个指定维度之间元素个数的乘积
  EXPECT_EQ(c10::numelements_between_dim(2, 4, ints), 3 * 4);
  // 测试 c10::numelements_between_dim 函数，预期结果为两个指定维度之间元素个数的乘积
  EXPECT_EQ(c10::numelements_between_dim(4, 2, ints), 3 * 4);
}

// 定义测试用例 accumulateTest，针对 list 的测试
TEST(accumulateTest, list_test) {
  // 初始化一个包含整数的 list
  std::list<int> ints = {1, 2, 3, 4, 5};

  // 同上述 vector_test 的注释，适用于 list 的测试部分
  EXPECT_EQ(c10::sum_integers(ints), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(c10::multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(
      c10::multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::numelements_from_dim(2, ints), 3 * 4 * 5);
  EXPECT_EQ(c10::numelements_to_dim(3, ints), 1 * 2 * 3);
  EXPECT_EQ(c10::numelements_between_dim(2, 4, ints), 3 * 4);
  EXPECT_EQ(c10::numelements_between_dim(4, 2, ints), 3 * 4);
}

// 定义测试用例 accumulateTest，针对 vector 的空情况的测试
TEST(accumulateTest, base_cases) {
  // 初始化一个空的 vector
  std::vector<int> ints = {};

  // 测试 c10::sum_integers 函数，预期结果为 0
  EXPECT_EQ(c10::sum_integers(ints), 0);
  // 测试 c10::multiply_integers 函数，预期结果为 1（空集合的乘积定义为 1）
  EXPECT_EQ(c10::multiply_integers(ints), 1);
}

// 定义测试用例 accumulateTest，针对异常情况的测试
TEST(accumulateTest, errors) {
  // 初始化一个包含整数的 vector
  std::vector<int> ints = {1, 2, 3, 4, 5};

#ifndef NDEBUG
  // 在调试模式下，测试 c10::numelements_from_dim 函数抛出异常的情况
  EXPECT_THROW(c10::numelements_from_dim(-1, ints), c10::Error);
#endif

  // NOLINTNEXTLINE 注释，忽略某些代码检查，避免不必要的警告
  EXPECT_THROW(c10::numelements_to_dim(-1, ints), c10::Error);
  EXPECT_THROW(c10::numelements_between_dim(-1, 10, ints), c10::Error);
  EXPECT_THROW(c10::numelements_between_dim(10, -1, ints), c10::Error);

  // 测试 c10::numelements_from_dim 函数，当维度值大于 vector 大小时，预期结果为 1
  EXPECT_EQ(c10::numelements_from_dim(10, ints), 1);
  EXPECT_THROW(c10::numelements_to_dim(10, ints), c10::Error);
  EXPECT_THROW(c10::numelements_between_dim(10, 4, ints), c10::Error);
  EXPECT_THROW(c10::numelements_between_dim(4, 10, ints), c10::Error);
}
```