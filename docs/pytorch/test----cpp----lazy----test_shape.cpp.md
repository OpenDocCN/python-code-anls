# `.\pytorch\test\cpp\lazy\test_shape.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <sstream>  // 包含使用 stringstream 所需的头文件

#include <torch/csrc/lazy/core/shape.h>  // 包含定义在 torch 的 lazy 模块中的 shape 类的头文件

namespace torch {
namespace lazy {

TEST(ShapeTest, Basic1) {  // 定义 Shape 类的单元测试 Basic1
  auto shape = Shape();  // 创建一个空的 Shape 对象

  // 检查初始状态下的 Shape 对象的输出字符串是否为 "UNKNOWN_SCALAR[]"
  EXPECT_STREQ(shape.to_string().c_str(), "UNKNOWN_SCALAR[]");

  // 检查初始状态下的 Shape 对象的标量类型是否为 Undefined
  EXPECT_EQ(shape.scalar_type(), c10::ScalarType::Undefined);

  // 检查初始状态下的 Shape 对象的维度是否为 0
  EXPECT_EQ(shape.dim(), 0);

  // 检查初始状态下的 Shape 对象的尺寸列表是否为空
  EXPECT_TRUE(shape.sizes().empty());

  // 检查尝试在初始状态下获取 Shape 对象的尺寸是否会抛出 std::out_of_range 异常
  EXPECT_THROW(shape.size(0), std::out_of_range);
}

TEST(ShapeTest, Basic2) {  // 定义 Shape 类的单元测试 Basic2
  auto shape = Shape(c10::ScalarType::Float, {1, 2, 3});  // 创建一个具有指定类型和尺寸的 Shape 对象

  // 检查 Shape 对象的元素总数是否为 6
  EXPECT_EQ(shape.numel(), 6);

  // 检查 Shape 对象的输出字符串是否为 "Float[1,2,3]"
  EXPECT_STREQ(shape.to_string().c_str(), "Float[1,2,3]");

  // 检查 Shape 对象的标量类型是否为 Float
  EXPECT_EQ(shape.scalar_type(), c10::ScalarType::Float);

  // 检查 Shape 对象的维度是否为 3
  EXPECT_EQ(shape.dim(), 3);

  // 检查 Shape 对象的尺寸列表的大小是否为 3
  EXPECT_EQ(shape.sizes().size(), 3);

  // 遍历 Shape 对象的维度，检查每个维度的尺寸是否符合预期
  for (int64_t i = 0; i < shape.dim(); i++) {
    EXPECT_EQ(shape.sizes()[i], i + 1);  // 检查尺寸列表中第 i 维的尺寸是否为 i + 1
    EXPECT_EQ(shape.size(i), i + 1);     // 检查获取第 i 维尺寸的方法是否正确
  }
}

TEST(ShapeTest, Basic3) {  // 定义 Shape 类的单元测试 Basic3
  auto shape = Shape(c10::ScalarType::Float, {});  // 创建一个空维度的 Shape 对象

  // 检查 Shape 对象的输出字符串是否为 "Float[]"
  EXPECT_STREQ(shape.to_string().c_str(), "Float[]");

  // 检查 Shape 对象的标量类型是否为 Float
  EXPECT_EQ(shape.scalar_type(), c10::ScalarType::Float);

  // 检查 Shape 对象的维度是否为 0
  EXPECT_EQ(shape.dim(), 0);

  // 检查 Shape 对象的元素总数是否为 1，即空维度的张量的元素总数为 1
  EXPECT_EQ(shape.numel(), 1);

  // 检查 Shape 对象的尺寸列表是否为空
  EXPECT_TRUE(shape.sizes().empty());

  // 检查尝试在空维度的 Shape 对象中获取尺寸是否会抛出 std::out_of_range 异常
  EXPECT_THROW(shape.size(0), std::out_of_range);
}

TEST(ShapeTest, SetScalarType) {  // 定义 Shape 类的单元测试 SetScalarType
  auto shape = Shape();  // 创建一个空的 Shape 对象

  shape.set_scalar_type(c10::ScalarType::Long);  // 设置 Shape 对象的标量类型为 Long
  EXPECT_EQ(shape.scalar_type(), c10::ScalarType::Long);  // 检查设置后的标量类型是否正确
}

TEST(ShapeTest, SetSize) {  // 定义 Shape 类的单元测试 SetSize
  auto shape1 = Shape();  // 创建一个空的 Shape 对象
  EXPECT_THROW(shape1.set_size(0, 0), std::out_of_range);  // 检查在空 Shape 对象中设置尺寸是否会抛出异常

  auto shape2 = Shape(c10::ScalarType::Float, {1, 2, 3});  // 创建一个具有指定尺寸的 Shape 对象
  shape2.set_size(0, 3);  // 设置 Shape 对象中第 0 维的尺寸为 3
  EXPECT_EQ(shape2.sizes()[0], 3);  // 检查设置后的第 0 维尺寸是否正确
  EXPECT_EQ(shape2.size(0), 3);     // 检查获取第 0 维尺寸的方法是否正确
}

TEST(ShapeTest, Equal) {  // 定义 Shape 类的单元测试 Equal
  auto shape1 = Shape(c10::ScalarType::Float, {});      // 创建不同的 Shape 对象
  auto shape2 = Shape(c10::ScalarType::Float, {1, 2, 3});
  auto shape3 = Shape(c10::ScalarType::Long, {1, 2, 3});
  auto shape4 = Shape(c10::ScalarType::Float, {1, 2, 3});

  // 检查不同的 Shape 对象之间的相等性判断是否正确
  EXPECT_FALSE(shape1 == shape2);
  EXPECT_FALSE(shape2 == shape3);
  EXPECT_FALSE(shape1 == shape3);
  EXPECT_TRUE(shape2 == shape4);  // shape2 和 shape4 应当相等
}

TEST(ShapeTest, Ostream) {  // 定义 Shape 类的单元测试 Ostream
  auto shape = Shape();  // 创建一个空的 Shape 对象
  std::stringstream ss;
  ss << shape;  // 将 Shape 对象输出到 stringstream 中

  // 检查输出到 stringstream 的内容是否与 Shape 对象的输出字符串相匹配
  EXPECT_EQ(shape.to_string(), ss.str());
}

} // namespace lazy
} // namespace torch
```