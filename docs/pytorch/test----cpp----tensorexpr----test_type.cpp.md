# `.\pytorch\test\cpp\tensorexpr\test_type.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include "torch/csrc/jit/tensorexpr/eval.h"  // 引入张量表达式求值相关的头文件
#include "torch/csrc/jit/tensorexpr/ir.h"    // 引入张量表达式中间表示相关的头文件
#include "torch/csrc/jit/tensorexpr/tensor.h" // 引入张量表达式张量相关的头文件

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;  // 引入张量表达式命名空间

TEST(Type, Test01) {  // 定义名为 Type 的测试套件 Test01
  {
    Dtype dt1 = kInt;  // 创建一个类型为 kInt 的 Dtype 变量 dt1
    ASSERT_EQ(dt1, kInt);  // 断言 dt1 的值等于 kInt
  }
  {
    Dtype dt2_a(kInt, 8);  // 创建一个包含 8 个 kInt 类型的 Dtype 变量 dt2_a
    Dtype dt2_b(kInt, 4);  // 创建一个包含 4 个 kInt 类型的 Dtype 变量 dt2_b
    Dtype dt2_c(ScalarType::Int, 8);  // 创建一个包含 8 个 ScalarType::Int 类型的 Dtype 变量 dt2_c
    ASSERT_EQ(dt2_a, dt2_c);  // 断言 dt2_a 等于 dt2_c
    ASSERT_NE(dt2_a, dt2_b);  // 断言 dt2_a 不等于 dt2_b
  }
  {
    ASSERT_EQ(kInt, ToDtype<int>());  // 断言 kInt 等于 ToDtype<int>() 的结果
    ASSERT_EQ(kFloat, ToDtype<float>());  // 断言 kFloat 等于 ToDtype<float>() 的结果
    ASSERT_EQ(kByte, ToDtype<uint8_t>());  // 断言 kByte 等于 ToDtype<uint8_t>() 的结果
    ASSERT_EQ(kChar, ToDtype<int8_t>());  // 断言 kChar 等于 ToDtype<int8_t>() 的结果
    ASSERT_EQ(kShort, ToDtype<int16_t>());  // 断言 kShort 等于 ToDtype<int16_t>() 的结果
    ASSERT_EQ(kLong, ToDtype<int64_t>());  // 断言 kLong 等于 ToDtype<int64_t>() 的结果
    ASSERT_EQ(kHalf, ToDtype<at::Half>());  // 断言 kHalf 等于 ToDtype<at::Half>() 的结果
    ASSERT_EQ(kDouble, ToDtype<double>());  // 断言 kDouble 等于 ToDtype<double>() 的结果
    ASSERT_EQ(kBool, ToDtype<bool>());  // 断言 kBool 等于 ToDtype<bool>() 的结果
  }
  {
    Dtype int32x8(kInt, 8);  // 创建一个包含 8 个 kInt 类型的 Dtype 变量 int32x8
    Dtype float32x8(kFloat, 8);  // 创建一个包含 8 个 kFloat 类型的 Dtype 变量 float32x8
    ASSERT_NE(int32x8, float32x8);  // 断言 int32x8 不等于 float32x8
    ASSERT_EQ(float32x8, BinaryOpDtype(int32x8, float32x8));  // 断言 float32x8 等于 BinaryOpDtype(int32x8, float32x8) 的结果
    ASSERT_EQ(float32x8, BinaryOpDtype(float32x8, int32x8));  // 断言 float32x8 等于 BinaryOpDtype(float32x8, int32x8) 的结果
    ASSERT_EQ(int32x8, BinaryOpDtype(int32x8, int32x8));  // 断言 int32x8 等于 BinaryOpDtype(int32x8, int32x8) 的结果
    ASSERT_EQ(float32x8, BinaryOpDtype(float32x8, float32x8));  // 断言 float32x8 等于 BinaryOpDtype(float32x8, float32x8) 的结果
  }
}

TEST(Type, BitCasting) {  // 定义名为 Type 的测试套件 BitCasting
  {
    VarHandle x("x", kFloat);  // 创建一个名为 x，类型为 kFloat 的变量
    ExprHandle y = bitcast<int32_t>(x);  // 对 x 进行 int32_t 类型的位转换，结果赋给 y
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    ASSERT_EQ(y.dtype(), kInt);  // 断言 y 的数据类型为 kInt
  }
  {
    VarHandle x("x", kInt);  // 创建一个名为 x，类型为 kInt 的变量
    ExprHandle y = bitcast<float>(x);  // 对 x 进行 float 类型的位转换，结果赋给 y
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    ASSERT_EQ(y.dtype(), kFloat);  // 断言 y 的数据类型为 kFloat
  }
  {
    VarHandle x("x", kShort);  // 创建一个名为 x，类型为 kShort 的变量
    ExprHandle y = bitcast<at::Half>(x);  // 对 x 进行 at::Half 类型的位转换，结果赋给 y
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    ASSERT_EQ(y.dtype(), kHalf);  // 断言 y 的数据类型为 kHalf
  }
  {
    VarHandle x("x", kHalf);  // 创建一个名为 x，类型为 kHalf 的变量
    ExprHandle y = bitcast<int16_t>(x);  // 对 x 进行 int16_t 类型的位转换，结果赋给 y
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    ASSERT_EQ(y.dtype(), kShort);  // 断言 y 的数据类型为 kShort
  }

  constexpr int32_t ref32 = 1337;  // 创建一个 constexpr 类型为 int32_t 的变量 ref32，赋值为 1337
  constexpr int64_t ref64 = 1337;  // 创建一个 constexpr 类型为 int64_t 的变量 ref64，赋值为 1337
  constexpr float reff32 = 1337.0f;  // 创建一个 constexpr 类型为 float 的变量 reff32，赋值为 1337.0f
  constexpr double reff64 = 1337.0f;  // 创建一个 constexpr 类型为 double 的变量 reff64，赋值为 1337.0f
  using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;  // 使用 SimpleIREvaluator 创建 SimpleIRExprEval 类型

  /*{
    constexpr int16_t ref16 = 1337;
    at::Half k_;
    at::Half* k = &k_;
    *reinterpret_cast<int16_t*>(k) = ref16;
    auto a = HalfImm::make(*k);
    auto b = BitCast::make(kShort, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<int16_t>(), ref16);
  }*/

  {
    float k = raw_bitcast<float>(ref32);  // 使用 raw_bitcast 将 ref32 转换为 float 类型，结果赋给 k
    auto a = FloatImm::make(k);  // 创建一个浮点数常量表达式 a，赋值为 k
    auto b = BitCast::make(kInt, a);  // 创建一个位转换表达式 b，将 a 转换为 kInt 类型
    SimpleIRExprEval cg(b);  // 使用 SimpleIRExprEval 求解表达式 b
    ASSERT_EQ(cg.value<int32_t>(), ref32);  // 断言求解结果等于 ref32
  }

  {
    double k = raw_bitcast<double>(ref64);  // 使用 raw_bitcast 将 ref64 转换为 double 类型，结果赋给 k
    auto a = DoubleImm::make(k);  // 创建一个双精度浮点数常量表达式 a，赋值为 k
    auto b = BitCast::make(kLong, a);  // 创建一个位转换表达式 b，将 a 转换为 kLong 类型
    SimpleIRExprEval cg(b);  // 使用 SimpleIRExprEval 求解表达式 b
    ASSERT_EQ(cg.value<int64_t>(), ref64
    {
        // 将 reff32 转换为 int32_t 类型的整数 k
        int32_t k = raw_bitcast<int32_t>(reff32);
        // 将整数 k 转换为 IntImm 表达式 a
        auto a = IntImm::make(k);
        // 使用类型 kFloat 将 a 转换为 BitCast 表达式 b
        auto b = BitCast::make(kFloat, a);
        // 使用 SimpleIRExprEval 计算表达式 b
        SimpleIRExprEval cg(b);
        // 断言计算得到的值等于 reff32
        ASSERT_EQ(cg.value<float>(), reff32);
      }
    
      // This segfaults :(
      /*{
        // 声明一个名为 x 的变量句柄，类型为 kDouble
        VarHandle x("x", kDouble);
        // 断言尝试将 x 强制转换为 int32_t 类型时会抛出异常
        ASSERT_ANY_THROW(ExprHandle y = bitcast<int32_t>(x));
      }
      {
        // 声明一个名为 x 的变量句柄，类型为 kFloat
        VarHandle x("x", kFloat);
        // 断言尝试将 x 强制转换为 int64_t 类型时会抛出异常
        ASSERT_ANY_THROW(ExprHandle y = bitcast<int64_t>(x));
      }
      {
        // 声明一个名为 x 的变量句柄，类型为 kLong
        VarHandle x("x", kLong);
        // 断言尝试将 x 强制转换为 float 类型时会抛出异常
        ASSERT_ANY_THROW(ExprHandle y = bitcast<float>(x));
      }
      {
        // 声明一个名为 x 的变量句柄，类型为 kShort
        VarHandle x("x", kShort);
        // 断言尝试将 x 强制转换为 float 类型时会抛出异常
        ASSERT_ANY_THROW(ExprHandle y = bitcast<float>(x));
      }
      {
        // 声明一个名为 x 的变量句柄，类型为 kInt
        VarHandle x("x", kInt);
        // 断言尝试将 x 强制转换为 at::Half 类型时会抛出异常
        ASSERT_ANY_THROW(ExprHandle y = bitcast<at::Half>(x));
      }*/
TEST(Type, Propagation) {
  // 定义一个测试用例 Type Propagation

  // Same types:
  {
    // 创建两个变量 x 和 y，类型为 kFloat
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    // 构建表达式体，计算结果为 2.0 + (x * 3.0 + 4.0 * y)
    ExprHandle body = FloatImm::make(2.f) +
        (x * FloatImm::make(3.f) + FloatImm::make(4.f) * y);
    // 断言表达式体的数据类型为 kFloat
    ASSERT_EQ(body.dtype(), kFloat);
  }

  // Int to bigger int:
  {
    // 创建两个变量 x 和 y，分别类型为 kShort 和 kLong
    VarHandle x("x", kShort);
    VarHandle y("y", kLong);
    // 构建表达式体，计算结果为 2 + (x * 3 + 4 * y)
    ExprHandle body =
        ShortImm::make(2.f) + (x * ShortImm::make(3) + ShortImm::make(4) * y);
    // 断言表达式体的数据类型为 kLong
    ASSERT_EQ(body.dtype(), kLong);
  }

  // Float to bigger float:
  {
    // 创建两个变量 x 和 y，分别类型为 kHalf 和 kDouble
    VarHandle x("x", kHalf);
    VarHandle y("y", kDouble);
    // 构建表达式体，计算结果为 2.0 + (x * 3.0 + 4.0 * y)
    ExprHandle body =
        HalfImm::make(2.f) + (x * HalfImm::make(3) + HalfImm::make(4) * y);
    // 断言表达式体的数据类型为 kDouble
    ASSERT_EQ(body.dtype(), kDouble);
  }

  // Int to Float:
  {
    // 创建两个变量 x 和 y，分别类型为 kFloat 和 kInt
    VarHandle x("x", kFloat);
    VarHandle y("y", kInt);
    // 构建表达式体，计算结果为 2 + (x * 3 + 4 * y)
    ExprHandle body =
        IntImm::make(2) + (x * IntImm::make(3) + IntImm::make(4) * y);
    // 断言表达式体的数据类型为 kFloat
    ASSERT_EQ(body.dtype(), kFloat);
  }

  // Smaller float, bigger Int:
  {
    // 创建两个变量 x 和 y，分别类型为 kHalf 和 kLong
    VarHandle x("x", kHalf);
    VarHandle y("y", kLong);
    // 构建表达式体，计算结果为 2 + (x * 3 + 4 * y)
    ExprHandle body =
        HalfImm::make(2) + (x * HalfImm::make(3) + HalfImm::make(4) * y);
    // 断言表达式体的数据类型为 kHalf
    ASSERT_EQ(body.dtype(), kHalf);
  }

  // Bigger float, smaller Int:
  {
    // 创建两个变量 x 和 y，分别类型为 kChar 和 kDouble
    VarHandle x("x", kChar);
    VarHandle y("y", kDouble);
    // 构建表达式体，计算结果为 2 + (x * 3 + 4 * y)
    ExprHandle body =
        CharImm::make(2) + (x * CharImm::make(3) + CharImm::make(4) * y);
    // 断言表达式体的数据类型为 kDouble
    ASSERT_EQ(body.dtype(), kDouble);
  }

  // Sign change char/byte upgrades to short:
  {
    // 创建两个变量 x 和 y，分别类型为 kChar 和 kByte
    VarHandle x("x", kChar);
    VarHandle y("y", kByte);
    // 构建表达式体，计算结果为 2 + (x * 3 + 4 * y)
    ExprHandle body =
        CharImm::make(2) + (x * CharImm::make(3) + CharImm::make(4) * y);
    // 断言表达式体的数据类型为 kShort
    ASSERT_EQ(body.dtype(), kShort);
  }
}
```