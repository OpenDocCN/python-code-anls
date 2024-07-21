# `.\pytorch\c10\test\core\Scalar_test.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 C10 库的标量相关头文件
#include <c10/core/Scalar.h>

// 使用 c10 命名空间
using namespace c10;

// 测试 UnsignedConstructor 的单元测试
TEST(ScalarTest, UnsignedConstructor) {
  // 定义并初始化不同长度的无符号整数
  uint16_t x = 0xFFFF;
  uint32_t y = 0xFFFFFFFF;
  uint64_t z0 = 0;
  uint64_t z1 = 0x7FFFFFFFFFFFFFFF;
  uint64_t z2 = 0xFFFFFFFFFFFFFFFF;
  
  // 创建对应的标量对象
  auto sx = Scalar(x);
  auto sy = Scalar(y);
  auto sz0 = Scalar(z0);
  auto sz1 = Scalar(z1);
  auto sz2 = Scalar(z2);
  
  // 断言各标量对象均为整数类型
  ASSERT_TRUE(sx.isIntegral(false));
  ASSERT_TRUE(sy.isIntegral(false));
  ASSERT_TRUE(sz0.isIntegral(false));
  ASSERT_TRUE(sz1.isIntegral(false));
  ASSERT_TRUE(sz2.isIntegral(false));
  
  // 断言各标量对象的类型
  ASSERT_EQ(sx.type(), ScalarType::Long);
  ASSERT_EQ(sy.type(), ScalarType::Long);
  ASSERT_EQ(sz0.type(), ScalarType::Long);
  ASSERT_EQ(sz1.type(), ScalarType::UInt64);
  ASSERT_EQ(sz2.type(), ScalarType::UInt64);
  
  // 断言标量对象能正确转换为相应的整数类型
  ASSERT_EQ(sx.toUInt16(), x);
  ASSERT_EQ(sx.toInt(), x);
  ASSERT_EQ(sy.toUInt32(), y);
  
  // 预期抛出异常，因为超出了可表示的范围
  EXPECT_THROW(sy.toInt(), std::runtime_error); // overflows
  ASSERT_EQ(sy.toLong(), y);
  ASSERT_EQ(sz0.toUInt64(), z0);
  ASSERT_EQ(sz0.toInt(), z0);
  ASSERT_EQ(sz1.toUInt64(), z1);
  
  // 预期抛出异常，因为超出了可表示的范围
  EXPECT_THROW(sz1.toInt(), std::runtime_error); // overflows
  ASSERT_EQ(sz1.toLong(), z1);
  ASSERT_EQ(sz2.toUInt64(), z2);
  
  // 预期抛出异常，因为超出了可表示的范围
  EXPECT_THROW(sz2.toInt(), std::runtime_error); // overflows
  EXPECT_THROW(sz2.toLong(), std::runtime_error); // overflows
}

// 测试 Equality 的单元测试
TEST(ScalarTest, Equality) {
  // 断言标量对象与无符号 64 位整数值相等
  ASSERT_TRUE(Scalar(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF))
                  .equal(0xFFFFFFFFFFFFFFFF));
  
  // 断言标量对象与 0 不相等
  ASSERT_FALSE(Scalar(0).equal(0xFFFFFFFFFFFFFFFF));
  
  // 确保我们不会错误地将位表示转换为相等判断
  ASSERT_FALSE(Scalar(-1).equal(0xFFFFFFFFFFFFFFFF));
}

// 测试 LongsAndLongLongs 的单元测试
TEST(ScalarTest, LongsAndLongLongs) {
  // 创建长整型和长长整型的标量对象
  Scalar longOne = 1L;
  Scalar longlongOne = 1LL;
  
  // 断言它们转换为 int 值相等
  ASSERT_EQ(longOne.toInt(), longlongOne.toInt());
}
```