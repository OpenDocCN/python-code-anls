# `.\pytorch\test\cpp\api\memory.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 C10 库中的 Optional 实现头文件
#include <c10/util/Optional.h>

// 定义一个结构体 TestValue，包含两个 std::optional<int> 类型的成员变量
struct TestValue {
  // 使用左值引用构造 lvalue_
  explicit TestValue(const int& x) : lvalue_(x) {}
  // 使用右值引用构造 rvalue_
  explicit TestValue(int&& x) : rvalue_(x) {}

  // 左值的 optional<int> 成员
  std::optional<int> lvalue_;
  // 右值的 optional<int> 成员
  std::optional<int> rvalue_;
};

// 测试用例 MakeUniqueTest，验证 std::make_unique 对右值传递的正确性
TEST(MakeUniqueTest, ForwardRvaluesCorrectly) {
  // 使用 std::make_unique 创建 TestValue 对象的智能指针，传递右值参数 123
  auto ptr = std::make_unique<TestValue>(123);
  // 断言 lvalue_ 未包含值
  ASSERT_FALSE(ptr->lvalue_.has_value());
  // 断言 rvalue_ 包含值，并检查其值为 123
  ASSERT_TRUE(ptr->rvalue_.has_value());
  ASSERT_EQ(*ptr->rvalue_, 123);
}

// 测试用例 MakeUniqueTest，验证 std::make_unique 对左值传递的正确性
TEST(MakeUniqueTest, ForwardLvaluesCorrectly) {
  // 定义整型变量 x，并初始化为 5
  int x = 5;
  // 使用 std::make_unique 创建 TestValue 对象的智能指针，传递左值参数 x
  auto ptr = std::make_unique<TestValue>(x);
  // 断言 lvalue_ 包含值，并检查其值为 5
  ASSERT_TRUE(ptr->lvalue_.has_value());
  ASSERT_EQ(*ptr->lvalue_, 5);
  // 断言 rvalue_ 未包含值
  ASSERT_FALSE(ptr->rvalue_.has_value());
}

// 测试用例 MakeUniqueTest，验证 std::make_unique 可以构造数组的唯一指针
TEST(MakeUniqueTest, CanConstructUniquePtrOfArray) {
  // 使用 std::make_unique 创建一个包含 3 个元素的整型数组的唯一指针
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  auto ptr = std::make_unique<int[]>(3);
  // 根据标准要求，需要对数组元素进行值初始化，此处断言三个元素的值为 0
  ASSERT_EQ(ptr[0], 0);
  ASSERT_EQ(ptr[1], 0);
  ASSERT_EQ(ptr[2], 0);
}
```