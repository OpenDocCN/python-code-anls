# `.\pytorch\test\cpp\tensorexpr\test_llvm.cpp`

```
#ifdef TORCH_ENABLE_LLVM
// 如果 TORCH_ENABLE_LLVM 定义了，编译以下代码

#include <gtest/gtest.h>
// 包含 Google Test 的头文件

#include <test/cpp/tensorexpr/test_base.h>
#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>
// 包含各种测试和表达式计算相关的头文件

#include <cmath>
#include <numeric>
// 包含数学函数和数值计算所需的头文件

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
// 命名空间声明，使用 Torch 的 JIT 和 TensorExpr 功能

using LLVMExprEval = ExprEval<LLVMCodeGen>;
// 定义 LLVMExprEval 类型为 ExprEval<LLVMCodeGen>

// Typed tests, can't use gtest params here due to the way we instantiate tests.
#define TEST_LLVM_SCALAR_TYPES(_) \
  _(uint8_t, Byte, 24)            \
  _(int8_t, Char, -20)            \
  _(int16_t, Short, 3332)         \
  _(int, Int, 123456)             \
  _(int64_t, Long, 2631563121321) \
  _(float, Float, 0.122)          \
  _(double, Double, 0.21312)      \
  _(at::Half, Half, 0.128f)
// 定义一系列具体类型的测试宏，用于测试 LLVM 表达式计算功能

#define IMM_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##ImmTest) {                      \
    auto a = Name##Imm::make(Val);                 \
    LLVMExprEval cg(a);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val, 0.1);     \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val);            \
    }                                              \
  }
// 定义一个宏，用于测试立即数表达式的计算

TEST_LLVM_SCALAR_TYPES(IMM_TEST)
#undef IMM_TEST
// 展开测试所有定义的立即数类型

#define ADD_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##AddTest) {                      \
    auto a = Name##Imm::make(Val);                 \
    auto b = Name##Imm::make(Val * 2);             \
    auto c = Add::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val * 3, 0.1); \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val * 3);        \
    }                                              \
  }
// 定义一个宏，用于测试加法表达式的计算

TEST_LLVM_SCALAR_TYPES(ADD_TEST)
#undef ADD_TEST
// 展开测试所有定义的加法类型

#define SUB_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##SubTest) {                      \
    auto a = Name##Imm::make(Val * 2);             \
    auto b = Name##Imm::make(Val);                 \
    auto c = Sub::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val, 0.1);     \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val);            \
    }                                              \
  }
// 定义一个宏，用于测试减法表达式的计算

TEST_LLVM_SCALAR_TYPES(SUB_TEST)
#undef SUB_TEST
// 展开测试所有定义的减法类型

#endif // TORCH_ENABLE_LLVM
// 结束 ifdef TORCH_ENABLE_LLVM 的条件编译块
#define MUL_TEST(Type, Name, Val)                  \
  // 定义测试宏，用于测试乘法操作                        \
  TEST(LLVM, Name##MulTest) {                      \
    // 创建常量表达式 a，值为 Val                     \
    auto a = Name##Imm::make(Val);                 \
    // 创建常量表达式 b，值为 4                      \
    auto b = Name##Imm::make((Type)4);             \
    // 创建乘法表达式 c = a * b                      \
    auto c = Mul::make(a, b);                      \
    // 创建表达式求值对象 cg                         \
    LLVMExprEval cg(c);                            \
    // 根据类型是否为浮点数进行断言，浮点数使用近似比较  \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val * 4, 0.1); \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val * 4);        \
    }                                              \
  }
// 对所有 LLVM 标量类型进行乘法测试
TEST_LLVM_SCALAR_TYPES(MUL_TEST)
// 取消乘法测试宏定义
#undef MUL_TEST

#define DIV_TEST(Type, Name, Val)                  \
  // 定义测试宏，用于测试除法操作                        \
  TEST(LLVM, Name##DivTest) {                      \
    // 创建常量表达式 a，值为 6                      \
    auto a = Name##Imm::make((Type)6);             \
    // 创建常量表达式 b，值为 3                      \
    auto b = Name##Imm::make((Type)3);             \
    // 创建除法表达式 c = a / b                      \
    auto c = Div::make(a, b);                      \
    // 创建表达式求值对象 cg                         \
    LLVMExprEval cg(c);                            \
    // 根据类型是否为浮点数进行断言，浮点数使用近似比较  \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), 2, 0.1);       \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), 2);              \
    }                                              \
  }
// 对所有 LLVM 标量类型进行除法测试
TEST_LLVM_SCALAR_TYPES(DIV_TEST)
// 取消除法测试宏定义
#undef DIV_TEST

// 整型到浮点型转换测试
TEST(LLVM, IntToFloatCastTest) {
  // 创建整型常量表达式 a，值为 2
  auto a = IntImm::make(2);
  // 创建类型转换表达式 b，将 a 转换为浮点型
  auto b = Cast::make(kFloat, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b, {});
  // 断言转换后的值为浮点数 2.0
  ASSERT_EQ(cg.value<float>(), 2.0);
}

// 浮点型到整型转换测试
TEST(LLVM, FloatToIntCastTest) {
  // 创建浮点数常量表达式 a，值为 2.0
  auto a = FloatImm::make(2.0);
  // 创建类型转换表达式 b，将 a 转换为整型
  auto b = Cast::make(kInt, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为整数 2
  ASSERT_EQ(cg.value<int>(), 2);
}

// 整型到长整型转换测试
TEST(LLVM, IntToLongCastTest) {
  // 创建整型常量表达式 a，值为 12345
  auto a = IntImm::make(12345);
  // 创建类型转换表达式 b，将 a 转换为长整型
  auto b = Cast::make(kLong, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为长整数 12345
  ASSERT_EQ(cg.value<int64_t>(), 12345);
}

// 字节到字符型转换测试
TEST(LLVM, ByteToCharCastTest) {
  // 创建字节常量表达式 a，值为 250
  auto a = ByteImm::make(250);
  // 创建类型转换表达式 b，将 a 转换为字符型
  auto b = Cast::make(kChar, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为字符型数值 250
  ASSERT_EQ(cg.value<int8_t>(), (int8_t)250);
}

// 半精度浮点型到长整型转换测试
TEST(LLVM, HalfToLongCastTest) {
  // 创建半精度浮点数常量表达式 a，值为 2.0
  auto a = HalfImm::make(2.0);
  // 创建类型转换表达式 b，将 a 转换为长整型
  auto b = Cast::make(kLong, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为长整数 2
  ASSERT_EQ(cg.value<int64_t>(), 2);
}

// 字节到双精度浮点型转换测试
TEST(LLVM, ByteToDoubleCastTest) {
  // 创建字节常量表达式 a，值为 2
  auto a = ByteImm::make(2);
  // 创建类型转换表达式 b，将 a 转换为双精度浮点型
  auto b = Cast::make(kDouble, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为双精度浮点数 2.0
  ASSERT_EQ(cg.value<double>(), 2);
}

// 浮点型到字节型转换测试
TEST(LLVM, FloatToByteCastTest) {
  // 创建浮点数常量表达式 a，值为 254.0
  auto a = FloatImm::make(254.0);
  // 创建类型转换表达式 b，将 a 转换为字节型
  auto b = Cast::make(kByte, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为无符号字节型数值 254
  ASSERT_EQ(cg.value<uint8_t>(), 254);
}

// 浮点型到字符型转换测试
TEST(LLVM, FloatToCharCastTest) {
  // 创建浮点数常量表达式 a，值为 -2.0
  auto a = FloatImm::make(-2.0);
  // 创建类型转换表达式 b，将 a 转换为字符型
  auto b = Cast::make(kChar, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为字符型数值 -2
  ASSERT_EQ(cg.value<int8_t>(), -2);
}

// 字节型到浮点型转换测试
TEST(LLVM, ByteToFloatCastTest) {
  // 创建字节常量表达式 a，值为 254
  auto a = ByteImm::make(254);
  // 创建类型转换表达式 b，将 a 转换为浮点型
  auto b = Cast::make(kFloat, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为浮点数 254.0
  ASSERT_EQ(cg.value<float>(), 254.0);
}

// 字符型到浮点型转换测试
TEST(LLVM, CharToFloatCastTest) {
  // 创建字符常量表达式 a，值为 -2
  auto a = CharImm::make(-2);
  // 创建类型转换表达式 b，将 a 转换为浮点型
  auto b = Cast::make(kFloat, a);
  // 创建表达式求值对象 cg
  LLVMExprEval cg(b);
  // 断言转换后的值为浮点数 -2.0
  ASSERT_EQ(cg.value<float>(), -2.0);
}
TEST(LLVM, BitCast) {
  /* constexpr int16_t ref16 = 1337; */
  // 定义一个 constexpr 的 32 位整数 ref32
  constexpr int32_t ref32 = 1337;
  // 定义一个 constexpr 的 64 位整数 ref64
  constexpr int64_t ref64 = 1337;
  // 定义一个 constexpr 的 32 位浮点数 reff32
  constexpr float reff32 = 1337.0f;
  // 定义一个 constexpr 的 64 位浮点数 reff64
  constexpr double reff64 = 1337.0f;

  // this is broken
  /*{
    at::Half k_;
    at::Half* k = &k_;
    *reinterpret_cast<int16_t*>(k) = ref16;
    auto a = HalfImm::make(k);
    auto b = BitCast::make(kShort, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int16_t>(), ref16);
  }*/

  {
    // 使用 raw_bitcast 将 ref32 转换为 float 类型 k
    float k = raw_bitcast<float>(ref32);
    // 创建 FloatImm 表达式 a
    auto a = FloatImm::make(k);
    // 使用 BitCast::make 将 a 转换为 kInt 类型表达式 b
    auto b = BitCast::make(kInt, a);
    // 使用 LLVMExprEval 计算表达式 b
    LLVMExprEval cg(b);
    // 断言 cg 的值与 ref32 相等
    ASSERT_EQ(cg.value<int32_t>(), ref32);
  }

  {
    // 使用 raw_bitcast 将 ref64 转换为 double 类型 k
    double k = raw_bitcast<double>(ref64);
    // 创建 DoubleImm 表达式 a
    auto a = DoubleImm::make(k);
    // 使用 BitCast::make 将 a 转换为 kLong 类型表达式 b
    auto b = BitCast::make(kLong, a);
    // 使用 LLVMExprEval 计算表达式 b
    LLVMExprEval cg(b);
    // 断言 cg 的值与 ref64 相等
    ASSERT_EQ(cg.value<int64_t>(), ref64);
  }

  {
    // 使用 raw_bitcast 将 reff64 转换为 int64_t 类型 k
    int64_t k = raw_bitcast<int64_t>(reff64);
    // 创建 LongImm 表达式 a
    auto a = LongImm::make(k);
    // 使用 BitCast::make 将 a 转换为 kDouble 类型表达式 b
    auto b = BitCast::make(kDouble, a);
    // 使用 LLVMExprEval 计算表达式 b
    LLVMExprEval cg(b);
    // 断言 cg 的值与 reff64 相等
    ASSERT_EQ(cg.value<double>(), reff64);
  }

  {
    // 使用 raw_bitcast 将 reff32 转换为 int32_t 类型 k
    int32_t k = raw_bitcast<int32_t>(reff32);
    // 创建 IntImm 表达式 a
    auto a = IntImm::make(k);
    // 使用 BitCast::make 将 a 转换为 kFloat 类型表达式 b
    auto b = BitCast::make(kFloat, a);
    // 使用 LLVMExprEval 计算表达式 b
    LLVMExprEval cg(b);
    // 断言 cg 的值与 reff32 相等
    ASSERT_EQ(cg.value<float>(), reff32);
  }
}

TEST(LLVM, fastLogFloat) {
  const int kTotalSize = 128 * 128;
  // 创建名为 A 的 BufHandle 对象
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 B 的 BufHandle 对象
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建名为 index 的 VarHandle 对象
  VarHandle index = VarHandle("index", kInt);
  // 加载 a_buf 的第 index 个元素到 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建用于存储 fast_log(load_a) 的 StmtPtr 对象 store_b
  StmtPtr store_b = b_buf.store({index}, fast_log(load_a));
  // 创建 For 循环的 StmtPtr 对象 stmt
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建 kTotalSize 大小的 PaddedBuffer<float> 对象 a_v 和 b_v
  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  // 使用随机数填充 a_v
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  // 创建 LLVMCodeGen 对象 ir_eval，用于评估 stmt
  LLVMCodeGen ir_eval(stmt, {a_buf, b_buf});
  // 调用 ir_eval 对象，传入 a_v 和 b_v
  ir_eval.call({a_v, b_v});

  // 检查 b_v 中的结果与标准对数函数 std::log(a_v(i)) 的值是否一致
  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    auto ref = std::log(a_v(i));
    if (std::isnan(ref)) {
      // 断言 b_v(i) 是否为 NaN
      ASSERT_EQ(std::isnan(test), true);
    } else {
      // 断言 b_v(i) 与 ref 的浮点值是否相等
      ASSERT_FLOAT_EQ(test, ref);
    }
  }
}

TEST(LLVM, LetTest01) {
  // 创建名为 A 的 BufHandle 对象，大小为 {1}，类型为 kFloat
  BufHandle a("A", {1}, kFloat);
  // 创建 float 类型的向量 v
  std::vector<float> v = {1, 0};
  // 创建 void* 类型的参数向量 args，包含 v 的数据指针
  std::vector<void*> args({v.data()});
  // 创建名为 x 的 VarHandle 对象，类型为 kFloat
  VarHandle x("x", kFloat);
  // 创建 Block 对象 block，包含 Let::make 和 a.store 的表达式
  auto block = Block::make({
      Let::make(x, 3.f),
      a.store({0}, ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f))),
  });

  // 创建 LLVMCodeGen 对象 cg，用于评估 block，参数为 a
  LLVMCodeGen cg(block, {a});
  // 断言 cg 的返回值为 0
  ASSERT_EQ(cg.value<int>(args), 0);
  // 断言 v[0] 的值为 2.f + 3.f * 3.f + 4.f
  ASSERT_EQ(v[0], 2.f + 3.f * 3.f + 4.f);
}
TEST(LLVM, CondNoFalseBlockTest) {
  // 创建名为 "X" 的缓冲区，数据类型为整型
  BufHandle x("X", {1}, kInt);
  // 创建比较操作，判断 x.load(0) 是否小于 10
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  // 根据比较结果创建条件块，如果比较成立则执行 x.store({0}, x.load(0) + 1)，否则为 nullptr
  auto cond = Cond::make(cmp, x.store({0}, x.load(0) + 1), nullptr);

  // 针对 x_value 在 {0, 10, 20} 中的每个值进行测试
  for (int32_t x_value : {0, 10, 20}) {
    // 创建整型向量 x_buffer，初始化为 x_value
    std::vector<int32_t> x_buffer = {x_value};
    // 创建参数向量 args，包含 x_buffer 的数据指针
    std::vector<void*> args({x_buffer.data()});
    // 使用 LLVMCodeGen 对象 cg，处理条件块 cond，传入缓冲区 x
    # 使用断言检查调用 cg 对象的 value 方法返回的整数是否等于 0
    ASSERT_EQ(cg.value<int>(args), 0);
    # 如果 x_value 小于 10，则执行以下代码块
    if (x_value < 10) {
      # 使用断言检查 x_buffer[0] 的值是否等于 x_value + 1
      ASSERT_EQ(x_buffer[0], x_value + 1);
    } else {
      # 如果 x_value 不小于 10，则执行以下代码块
      # 使用断言检查 x_buffer[0] 的值是否等于 x_value
      ASSERT_EQ(x_buffer[0], x_value);
    }
  }
// 定义一个名为 `TEST` 的测试用例，名称为 "LLVM"，测试条件为 `CondTest`
TEST(LLVM, CondTest) {
  // 创建名为 `x` 的缓冲区，维度为 {1}，类型为整型
  BufHandle x("X", {1}, kInt);
  // 创建比较条件，比较 `x` 的加载值是否小于 10
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  // 创建条件执行块，如果比较结果为真，则将 `x` 加载值加 1 后存储回 `x`，否则减 1 后存储
  auto cond =
      Cond::make(cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  // 创建代码块，包含上述条件执行块和将 `x` 加载值乘以 2 后存储回 `x`
  auto block = Block::make({
      cond,
      x.store({0}, x.load(0) * 2),
  });

  // 对于 x_value 取值为 {0, 10, 20} 的循环
  for (int32_t x_value : {0, 10, 20}) {
    // 创建整型数组 `x_buffer`，初始化为 x_value 的值
    std::vector<int32_t> x_buffer = {x_value};
    // 创建指针数组 `args`，包含 `x_buffer` 的数据指针
    std::vector<void*> args({x_buffer.data()});
    // 使用 `block` 和 `x` 创建 LLVM 代码生成器 `cg`
    LLVMCodeGen cg(block, {x});
    // 断言执行 `cg` 并返回整型结果为 0
    ASSERT_EQ(cg.value<int>(args), 0);
    // 根据 `x_value` 的值进行条件断言
    if (x_value < 10) {
      // 如果 `x_value` 小于 10，则断言 `x_buffer[0]` 的值为 `(x_value + 1) * 2`
      ASSERT_EQ(x_buffer[0], (x_value + 1) * 2);
    } else {
      // 如果 `x_value` 不小于 10，则断言 `x_buffer[0]` 的值为 `(x_value - 1) * 2`
      ASSERT_EQ(x_buffer[0], (x_value - 1) * 2);
    }
  }
}

// 定义一个名为 `TEST` 的测试用例，名称为 "LLVM"，测试条件为 `CondNestedTest`
TEST(LLVM, CondNestedTest) {
  // 创建名为 `x` 的缓冲区，维度为 {1}，类型为整型
  BufHandle x("X", {1}, kInt);
  // 创建 `x` 的加载值与 5 比较的条件
  auto true_cmp =
      CompareSelect::make(x.load(0), 5, CompareSelectOperation::kGT);
  // 创建条件执行块，如果比较结果为真，则将 `x` 加载值加 1 后存储回 `x`，否则减 1 后存储
  auto true_cond = Cond::make(
      true_cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  // 创建 `x` 的加载值与 15 比较的条件
  auto false_cmp =
      CompareSelect::make(x.load(0), 15, CompareSelectOperation::kLE);
  // 创建条件执行块，如果比较结果为真，则将 `x` 加载值加 2 后存储回 `x`，否则减 2 后存储
  auto false_cond = Cond::make(
      false_cmp, x.store({0}, x.load(0) + 2), x.store({0}, x.load(0) - 2));
  // 创建 `x` 的加载值与 10 比较的条件
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  // 创建条件执行块，根据 `cmp` 的结果选择执行 `true_cond` 或 `false_cond`
  auto cond = Cond::make(cmp, true_cond, false_cond);

  // 对于 x_value 取值为 {0, 8, 15, 20} 的循环
  for (int32_t x_value : {0, 8, 15, 20}) {
    // 创建整型数组 `x_buffer`，初始化为 x_value 的值
    std::vector<int32_t> x_buffer = {x_value};
    // 创建指针数组 `args`，包含 `x_buffer` 的数据指针
    std::vector<void*> args({x_buffer.data()});
    // 使用 `cond` 和 `x` 创建 LLVM 代码生成器 `cg`
    LLVMCodeGen cg(cond, {x});
    // 断言执行 `cg` 并返回整型结果为 0
    ASSERT_EQ(cg.value<int>(args), 0);
    // 根据 `x_value` 的值进行嵌套条件断言
    if (x_value < 10) {
      if (x_value > 5) {
        // 如果 `x_value` 大于 5，则断言 `x_buffer[0]` 的值为 `x_value + 1`
        ASSERT_EQ(x_buffer[0], x_value + 1);
      } else {
        // 如果 `x_value` 不大于 5，则断言 `x_buffer[0]` 的值为 `x_value - 1`
        ASSERT_EQ(x_buffer[0], x_value - 1);
      }
    } else {
      if (x_value <= 15) {
        // 如果 `x_value` 小于等于 15，则断言 `x_buffer[0]` 的值为 `x_value + 2`
        ASSERT_EQ(x_buffer[0], x_value + 2);
      } else {
        // 如果 `x_value` 大于 15，则断言 `x_buffer[0]` 的值为 `x_value - 2`
        ASSERT_EQ(x_buffer[0], x_value - 2);
      }
    }
  }
}

// 定义一个名为 `TEST` 的测试用例，名称为 "LLVM"，测试直接向量化
TEST(LLVM, DirectVectorization) {
  // 定义常量 M 和 N 的值分别为 3 和 64
  constexpr int M = 3;
  constexpr int N = 64;
  // 创建名为 `a`、`b`、`c` 的缓冲区，维度分别为 {M, N}，类型为浮点型
  BufHandle a("a", {M, N}, kFloat);
  BufHandle b("b", {M, N}, kFloat);
  BufHandle c("c", {M, N}, kFloat);
  // 创建整型变量 `m` 和 `n`
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  // 创建循环语句 `s`，循环范围为 [0, M)，执行向量化计算并存储到 `c` 中
  StmtPtr s = For::make(
      m,
      0,
      M,
      Store::make(
          c,
          {Ramp::make(m * 64, 1, 64)},
          Load::make({kFloat, 64}, a, {Ramp::make(m * 64, 1, 64)}) *
              Load::make({kFloat, 64}, b, {Ramp::make(m * 64, 1, 64)})));
  // 使用 `s`、`a`、`b`、`c` 创建 LLVM 代码生成器 `cg`
  LLVMCodeGen cg(s, {a, b, c});
}
TEST(LLVM, VecLoadStoreTest) {
  // 创建缓冲区对象 a 和 b，分别表示名为 "A" 和 "B" 的整数类型缓冲区
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  // 初始化整数类型的向量 a_buffer 和 b_buffer，每个元素值为 1 和 2
  std::vector<int32_t> a_buffer = {1, 1, 1, 1};
  std::vector<int32_t> b_buffer = {2, 2, 2, 2};

  // 构造一个存储操作，将缓冲区 a 中的数据按照步进为 1 的方式加载，并存储到缓冲区 b 中
  auto store = b.store({Ramp::make(0, 1, 4)}, a.load({Ramp::make(0, 1, 4)}));
  // 使用 LLVMCodeGen 对象 cg 进行代码生成，处理存储操作 store，操作涉及缓冲区 a 和 b
  LLVMCodeGen cg(store, {a, b});
  // 准备参数列表 args，包含 a_buffer 和 b_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  // 断言生成代码 cg 返回的整数值等于 0
  ASSERT_EQ(cg.value<int>(args), 0);
  // 检查 a_buffer 中的数据是否为预期的值 1
  ASSERT_EQ(a_buffer[0], 1);
  ASSERT_EQ(a_buffer[1], 1);
  ASSERT_EQ(a_buffer[2], 1);
  ASSERT_EQ(a_buffer[3], 1);
  // 检查 b_buffer 中的数据是否为预期的值 1
  ASSERT_EQ(b_buffer[0], 1);
  ASSERT_EQ(b_buffer[1], 1);
  ASSERT_EQ(b_buffer[2], 1);
  ASSERT_EQ(b_buffer[3], 1);
}

#define FLOAT_INTRINSICS_TEST(Name, Lanes)                                   \
  TEST(LLVM, VecFloat_##Name##Lane##Lanes##Test) {                           \
    // 创建缓冲区对象 a 和 b，分别表示名为 "A" 和 "B" 的浮点类型缓冲区
    BufHandle a("A", {1}, kFloat);                                          
    BufHandle b("B", {1}, kFloat);                                          
    float val = 0.5f;                                                       
    // 初始化长度为 Lanes 的浮点类型向量 a_buffer 和 b_buffer，每个元素值为 val
    std::vector<float> a_buffer(Lanes, val);                                 
    std::vector<float> b_buffer(Lanes, val);                                 
    // 构造一个存储操作，将缓冲区 a 中的数据按照步进为 1 的方式加载，并存储到缓冲区 b 中
    auto store = b.store(                                                   
        {Ramp::make(0, 1, Lanes)}, Name(a.load({Ramp::make(0, 1, Lanes)}))); 
    // 使用 LLVMCodeGen 对象 cg 进行代码生成，处理存储操作 store，操作涉及缓冲区 a 和 b
    LLVMCodeGen cg(store, {a, b});                                           
    // 准备参数列表 args，包含 a_buffer 和 b_buffer 的数据指针
    std::vector<void*> args({a_buffer.data(), b_buffer.data()});             
    // 断言生成代码 cg 返回的整数值等于 0
    ASSERT_EQ(cg.value<int>(args), 0);                                       
    // 遍历 a_buffer 中的每个元素，检查是否与预期的值 val 相等
    for (const auto i : c10::irange(Lanes)) {                                
      ASSERT_FLOAT_EQ(a_buffer[i], val);                                     
    }                                                                        
  } // namespace jit

// 下面是一系列的浮点数内置函数测试，每个函数以不同的名称和向量长度调用 FLOAT_INTRINSICS_TEST 宏
FLOAT_INTRINSICS_TEST(erf, 4)
FLOAT_INTRINSICS_TEST(erfc, 4)
FLOAT_INTRINSICS_TEST(acos, 4)
FLOAT_INTRINSICS_TEST(asin, 4)
FLOAT_INTRINSICS_TEST(atan, 4)
FLOAT_INTRINSICS_TEST(cosh, 4)
FLOAT_INTRINSICS_TEST(sinh, 4)
FLOAT_INTRINSICS_TEST(tanh, 4)
FLOAT_INTRINSICS_TEST(expm1, 4)
FLOAT_INTRINSICS_TEST(lgamma, 4)
FLOAT_INTRINSICS_TEST(erf, 8)
FLOAT_INTRINSICS_TEST(erfc, 8)
FLOAT_INTRINSICS_TEST(acos, 8)
FLOAT_INTRINSICS_TEST(asin, 8)
FLOAT_INTRINSICS_TEST(atan, 8)
FLOAT_INTRINSICS_TEST(cosh, 8)
FLOAT_INTRINSICS_TEST(sinh, 8)
FLOAT_INTRINSICS_TEST(tanh, 8)
FLOAT_INTRINSICS_TEST(expm1, 8)
FLOAT_INTRINSICS_TEST(lgamma, 8)
#undef FLOAT_INTRINSICS_TEST

#define DOUBLE_INTRINSICS_TEST(Name, Lanes)                                  
  TEST(LLVM, VecDouble_##Name##Lane##Lanes##Test) {                          
    // 创建缓冲区对象 a 和 b，分别表示名为 "A" 和 "B" 的双精度浮点类型缓冲区
    BufHandle a("A", {1}, kDouble);                                          
    BufHandle b("B", {1}, kDouble);                                          
    float val = 0.5f;                                                        
    // 初始化长度为 Lanes 的双精度浮点类型向量 a_buffer，每个元素值为 val
    std::vector<double> a_buffer(Lanes, val);                                
    // 创建一个大小为 Lanes 的 double 类型向量 b_buffer，并用 val 初始化
    std::vector<double> b_buffer(Lanes, val);
    // 调用 b 的 store 方法，将结果存储到 store 中，
    // store 的值是一个 LLVM IR 表示的存储操作
    auto store = b.store(
        // 使用 Ramp::make 创建一个从 0 开始，步长为 1，长度为 Lanes 的向量
        {Ramp::make(0, 1, Lanes)},
        // 使用 a 的 load 方法加载数据，形成一个 LLVM IR 表示的加载操作，并命名为 Name
        Name(a.load({Ramp::make(0, 1, Lanes)})));
    // 使用 LLVMCodeGen 类构造一个 LLVM IR 代码生成器，传入 store 和操作数 a, b
    LLVMCodeGen cg(store, {a, b});
    // 创建一个 void* 类型的指针向量 args，存储 a_buffer 和 b_buffer 的数据地址
    std::vector<void*> args({a_buffer.data(), b_buffer.data()});
    // 断言调用 cg 的 value 方法，并期望返回值为 0
    ASSERT_EQ(cg.value<int>(args), 0);
    // 遍历 c10 命名空间中长度为 Lanes 的范围
    for (const auto i : c10::irange(Lanes)) {
      // 断言 a_buffer[i] 的值等于 val
      ASSERT_FLOAT_EQ(a_buffer[i], val);
    }
  } // namespace jit
// 宏展开，调用 DOUBLE_INTRINSICS_TEST 宏来测试各种双精度数学函数
DOUBLE_INTRINSICS_TEST(erf, 2)
DOUBLE_INTRINSICS_TEST(erfc, 2)
DOUBLE_INTRINSICS_TEST(acos, 2)
DOUBLE_INTRINSICS_TEST(asin, 2)
DOUBLE_INTRINSICS_TEST(atan, 2)
DOUBLE_INTRINSICS_TEST(cosh, 2)
DOUBLE_INTRINSICS_TEST(sinh, 2)
DOUBLE_INTRINSICS_TEST(tanh, 2)
DOUBLE_INTRINSICS_TEST(expm1, 2)
DOUBLE_INTRINSICS_TEST(lgamma, 2)
DOUBLE_INTRINSICS_TEST(erf, 4)
DOUBLE_INTRINSICS_TEST(erfc, 4)
DOUBLE_INTRINSICS_TEST(acos, 4)
DOUBLE_INTRINSICS_TEST(asin, 4)
DOUBLE_INTRINSICS_TEST(atan, 4)
DOUBLE_INTRINSICS_TEST(cosh, 4)
DOUBLE_INTRINSICS_TEST(sinh, 4)
DOUBLE_INTRINSICS_TEST(tanh, 4)
DOUBLE_INTRINSICS_TEST(expm1, 4)
DOUBLE_INTRINSICS_TEST(lgamma, 4)
#undef DOUBLE_INTRINSICS_TEST

// 定义一个测试用例 LLVM.VectorizerLoadStoreTest
TEST(LLVM, VectorizerLoadStoreTest) {
  // 创建一个名为 "A" 的缓冲区，包含一个整数
  BufHandle a("A", {1}, kInt);

  // 定义张量 c，计算张量 c 的值，依赖于变量 i，从缓冲区 a 中加载数据
  Tensor c = Compute("c", {4}, [&](const VarHandle& i) { return a.load(i); });

  // 获取张量 c 的缓冲区
  BufHandle c_buf(c.buf());

  // 创建一个循环嵌套对象 l，包含张量 c
  LoopNest l({c});

  // 获取循环嵌套对象的根语句
  StmtPtr s = l.root_stmt();

  // 断言循环向量化以优化根语句的第一个循环
  ASSERT_TRUE(LoopNest::vectorize(to<For>(to<Block>(s)->front())));

  // 断言根语句的第一个块不包含 for 循环（已被优化）
  ASSERT_TRUE(to<For>(to<Block>(s)->front()) == nullptr);

  // 使用根语句 s 和缓冲区 a、c_buf 创建 LLVM 代码生成对象 cg
  LLVMCodeGen cg(s, {a, c_buf});

  // 初始化向量 a_vec，包含 4 个整数 21
  std::vector<int> a_vec(4, 21);

  // 初始化向量 c_vec，包含 4 个整数 0
  std::vector<int> c_vec(4, 0);

  // 初始化参数列表 args，包含指向 a_vec 和 c_vec 的指针
  std::vector<void*> args({a_vec.data(), c_vec.data()});

  // 断言调用 cg 的值方法返回 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量 c_vec 中所有元素均等于 21
  assertAllEqual(c_vec, 21);
}

// 定义一个测试用例 LLVM.VectorizeBitCast
TEST(LLVM, VectorizeBitCast) {
  // 创建一个名为 "A" 的缓冲区，包含 128 个整数
  BufHandle a("A", {128}, kInt);

  // 定义张量 c，计算张量 c 的值，依赖于变量 i，将缓冲区 a 中的整数位级转换为 float 类型
  Tensor c = Compute("c", {128}, [&](const VarHandle& i) {
    return bitcast<float>(a.load(i));
  });

  // 获取张量 c 的缓冲区
  BufHandle c_buf(c.buf());

  // 创建一个循环嵌套对象 l，包含张量 c
  LoopNest l({c});

  // 获取循环嵌套对象的根语句
  StmtPtr s = l.root_stmt();

  // 断言循环向量化以优化根语句的第一个循环
  ASSERT_TRUE(LoopNest::vectorize(to<For>(to<Block>(s)->front())));

  // 断言根语句的第一个块不包含 for 循环（已被优化）
  ASSERT_TRUE(to<For>(to<Block>(s)->front()) == nullptr);

  // 使用根语句 s 和缓冲区 a、c_buf 创建 LLVM 代码生成对象 cg
  LLVMCodeGen cg(s, {a, c_buf});

  // 初始化向量 a_vec，包含 128 个整数，每个整数将 1337.0 转换为整数
  std::vector<int> a_vec(128);
  for (const auto i : c10::irange(128)) {
    a_vec[i] = raw_bitcast<int>(1337.f);
  }

  // 初始化向量 c_vec，包含 128 个浮点数，初始值为 0
  std::vector<float> c_vec(128);

  // 初始化参数列表 args，包含指向 a_vec 和 c_vec 的指针
  std::vector<void*> args({a_vec.data(), c_vec.data()});

  // 断言调用 cg 的值方法返回 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量 c_vec 中所有元素均等于 1337.0
  assertAllEqual(c_vec, 1337.f);
}

// 定义一个测试用例 LLVM.MemcpyTest
TEST(LLVM, MemcpyTest) {
  // 定义常量 N 为 32
  constexpr int N = 32;

  // 创建一个名为 "A" 的缓冲区，包含 32 个整数
  BufHandle a("A", {N}, kInt);

  // 创建一个名为 "B" 的缓冲区，包含 32 个整数
  BufHandle b("B", {N}, kInt);

  // 初始化向量 a_buffer，包含 32 个整数，每个整数为 42
  std::vector<int32_t> a_buffer(N, 42);

  // 初始化向量 b_buffer，包含 32 个整数，每个整数为 0
  std::vector<int32_t> b_buffer(N, 0);

  // 定义变量 i，类型为整数
  VarHandle i("i", kInt);

  // 创建循环表达式 expr，遍历 i 从 0 到 N-1，将缓冲区 a 中的元素存储到缓冲区 b
  auto expr = For::make(i, 0, N, b.store({i}, a.load(i)));

  // 使用表达式 expr 和缓冲区 a、b 创建 LLVM 代码生成对象 cg
  LLVMCodeGen cg(expr, {a, b});

  // 初始化参数列表 args，包含指向 a_buffer 和 b_buffer 的指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});

  // 断言调用 cg 的值方法返回 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量 a_buffer 和 b_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);

  // 断言向量 a_buffer 中所有元素均等于 42
  assertAllEqual(a_buffer, 42);

  // 断言向量 b_buffer 中所有元素均等于 42，由于进行了存储操作，因此 b_buffer 也应全为 42
  assertAllEqual(b_buffer, 42);
}

// 定义一个测试用例 LLVM.BzeroTest
TEST(LLVM, BzeroTest) {
  // 定义常量 N 为 32
  constexpr int N = 32;

  // 创建一个名为 "B" 的缓冲区，包含 32 个整数
  BufHandle b("B", {N}, kInt);

  // 初始化向量 b_buffer，包含 32 个整数，每个整数为 11
  std::vector<int32_t> b_buffer(N, 11);

  // 定义变量 i，类型为整数
  VarHandle i("i", kInt);

  // 创建循环表达式 expr，遍历 i 从 0 到 N-1，将缓冲区 b 中的每个元素设为 0
  auto expr = For::make(i, 0, N, b.store({i},
// 定义名为 TEST 的测试函数，测试 LLVM 后端的 ElemwiseAdd 函数
TEST(LLVM, ElemwiseAdd) {
  // 定义常量 N，表示向量大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区对象，存储整数类型数据
  BufHandle a("A", {N}, kInt);
  // 创建名为 b 的缓冲区对象，存储整数类型数据
  BufHandle b("B", {N}, kInt);
  // 创建名为 c 的缓冲区对象，存储整数类型数据
  BufHandle c("C", {N}, kInt);
  // 初始化整数向量 a_buffer，每个元素为 41
  std::vector<int32_t> a_buffer(N, 41);
  // 初始化整数向量 b_buffer，每个元素为 1
  std::vector<int32_t> b_buffer(N, 1);
  // 初始化整数向量 c_buffer，每个元素为 1
  std::vector<int32_t> c_buffer(N, 1);

  // 定义名为 i 的循环变量，类型为整数
  VarHandle i("i", kInt);
  // 创建表达式 expr，用于计算向量 c 的每个元素
  auto expr = For::make(i, 0, N, c.store({i}, Add::make(a.load(i), b.load(i))));

  // 创建 LLVM 代码生成对象 cg，用于生成 LLVM 代码并编译
  LLVMCodeGen cg(expr, {a, b, c});

  // 准备参数列表 args，包含 a_buffer、b_buffer 和 c_buffer 的指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 LLVM 生成的代码并返回结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量大小为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言所有 a_buffer 元素值均为 41
  assertAllEqual(a_buffer, 41);
  // 断言所有 b_buffer 元素值均为 1
  assertAllEqual(b_buffer, 1);
  // 断言所有 c_buffer 元素值均为 42
  assertAllEqual(c_buffer, 42);
}

// 定义名为 TEST 的测试函数，测试 LLVM 后端的 ElemwiseAddFloat 函数
TEST(LLVM, ElemwiseAddFloat) {
  // 定义常量 N，表示向量大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区对象，存储浮点数类型数据
  BufHandle a("A", {N}, kFloat);
  // 创建名为 b 的缓冲区对象，存储浮点数类型数据
  BufHandle b("B", {N}, kFloat);
  // 创建名为 c 的缓冲区对象，存储浮点数类型数据
  BufHandle c("C", {N}, kFloat);
  // 初始化浮点数向量 a_buffer，每个元素为 41.0f
  std::vector<float> a_buffer(N, 41);
  // 初始化浮点数向量 b_buffer，每个元素为 1.0f
  std::vector<float> b_buffer(N, 1);
  // 初始化浮点数向量 c_buffer，每个元素为 1.0f
  std::vector<float> c_buffer(N, 1);

  // 定义名为 i 的循环变量，类型为整数
  VarHandle i("i", kInt);
  // 创建表达式 expr，用于计算向量 c 的每个元素
  auto expr = For::make(i, 0, N, c.store({i}, a.load(i) + b.load(i)));

  // 创建 LLVM 代码生成对象 cg，用于生成 LLVM 代码并编译
  LLVMCodeGen cg(expr, {a, b, c});

  // 准备参数列表 args，包含 a_buffer、b_buffer 和 c_buffer 的指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 LLVM 生成的代码并返回结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量大小为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言所有 a_buffer 元素值均为 41.0f
  assertAllEqual(a_buffer, 41.0f);
  // 断言所有 b_buffer 元素值均为 1.0f
  assertAllEqual(b_buffer, 1.0f);
  // 断言所有 c_buffer 元素值均为 42.0f
  assertAllEqual(c_buffer, 42.0f);
}

// 定义名为 TEST 的测试函数，测试 LLVM 后端的 ElemwiseLog10Float 函数
TEST(LLVM, ElemwiseLog10Float) {
  // 定义常量 N，表示向量大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区对象，存储浮点数类型数据
  BufHandle a("A", {N}, kFloat);
  // 创建名为 b 的缓冲区对象，存储浮点数类型数据
  BufHandle b("B", {N}, kFloat);
  // 初始化浮点数向量 a_buffer，每个元素为 10.0f
  std::vector<float> a_buffer(N, 10.0f);
  // 初始化浮点数向量 b_buffer，每个元素为 2.0f
  std::vector<float> b_buffer(N, 2.0f);

  // 定义名为 i 的循环变量，类型为整数
  VarHandle i("i", kInt);
  // 创建表达式 expr，用于计算向量 b 的每个元素
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.store(
          {Ramp::make(i * 4, 1, 4)}, log10(a.load({Ramp::make(i * 4, 1, 4)}))));

  // 创建 LLVM 代码生成对象 cg，用于生成 LLVM 代码并编译
  LLVMCodeGen cg(expr, {a, b});

  // 准备参数列表 args，包含 a_buffer 和 b_buffer 的指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  // 断言调用 LLVM 生成的代码并返回结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言向量大小为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  // 断言所有 a_buffer 元素值均为 10.0f
  assertAllEqual(a_buffer, 10.0f);
  // 断言所有 b_buffer 元素值近似为 1.0f，精度为 1e-5f
  ExpectAllNear(b_buffer, 1.0f, 1e-5f);
}

// 定义名为 TEST 的测试函数，测试 LLVM 后端的 ElemwiseLog1pFloat 函数
TEST(LLVM, ElemwiseLog1pFloat) {
  // 定义常量 N，表示向量大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区对象，存储浮点数类型数据
  BufHandle a("A", {N}, kFloat);
  // 创建名为 b 的缓冲区对象，存储浮点数类型数据
  BufHandle b("B", {N}, kFloat);
  // 初始化浮点数向量 a_buffer，每个元素为 expf(3.0f) - 1
  std::vector<float> a_buffer(N, expf(3.0f) - 1);
  // 初始化浮点数向量 b_buffer，每个元素为 42.0f
  std::vector<float> b_buffer(N, 42.0f);

  // 定义名为 i 的循环变量，类型为整数
  VarHandle i("i", kInt);
  // 创建表达式 expr，用于计算向量 b 的每个元素
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.store(
          {R
// 定义名为 TEST 的测试用例，测试整数类型的元素级最大值操作
TEST(LLVM, ElemwiseMaxInt) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a、b、c 的 BufHandle 对象，分别表示大小为 N 的整数数组 A、B、C
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  // 初始化大小为 N 的整数数组 a_buffer、b_buffer、c_buffer，分别填充为 41、1、1
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  // 定义名为 i 的整数变量
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，遍历 i 从 0 到 N-1，对 C 中第 i 个元素赋值为 A[i] 和 B[i] 的最大值
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  // 创建 LLVMCodeGen 对象 cg，编译 expr，关联的缓冲区为 a、b、c
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建指针数组 args，存储 a_buffer、b_buffer、c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言编译生成的代码执行结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer、c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 a_buffer 中所有元素均为 41
  assertAllEqual(a_buffer, 41);
  // 断言 b_buffer 中所有元素均为 1
  assertAllEqual(b_buffer, 1);
  // 断言 c_buffer 中所有元素均为 41
  assertAllEqual(c_buffer, 41);
}

// 定义名为 TEST 的测试用例，测试整数类型的元素级最小值操作
TEST(LLVM, ElemwiseMinInt) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a、b、c 的 BufHandle 对象，分别表示大小为 N 的整数数组 A、B、C
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  // 初始化大小为 N 的整数数组 a_buffer、b_buffer、c_buffer，分别填充为 41、1、1
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  // 定义名为 i 的整数变量
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，遍历 i 从 0 到 N-1，对 C 中第 i 个元素赋值为 A[i] 和 B[i] 的最小值
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  // 创建 LLVMCodeGen 对象 cg，编译 expr，关联的缓冲区为 a、b、c
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建指针数组 args，存储 a_buffer、b_buffer、c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言编译生成的代码执行结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer、c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 a_buffer 中所有元素均为 41
  assertAllEqual(a_buffer, 41);
  // 断言 b_buffer 中所有元素均为 1
  assertAllEqual(b_buffer, 1);
  // 断言 c_buffer 中所有元素均为 1
  assertAllEqual(c_buffer, 1);
}

// 定义名为 TEST 的测试用例，测试浮点数类型的元素级最大值操作
TEST(LLVM, ElemwiseMaxFloat) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a、b、c 的 BufHandle 对象，分别表示大小为 N 的浮点数数组 A、B、C
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
  // 初始化大小为 N 的浮点数数组 a_buffer、b_buffer、c_buffer，分别填充为 41.0、1.0、1.0
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  // 定义名为 i 的整数变量
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，遍历 i 从 0 到 N-1，对 C 中第 i 个元素赋值为 A[i] 和 B[i] 的最大值
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  // 创建 LLVMCodeGen 对象 cg，编译 expr，关联的缓冲区为 a、b、c
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建指针数组 args，存储 a_buffer、b_buffer、c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言编译生成的代码执行结果为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer、c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 a_buffer 中所有元素均为 41.0
  assertAllEqual(a_buffer, 41.0f);
  // 断言 b_buffer 中所有元素均为 1.0
  assertAllEqual(b_buffer, 1.0f);
  // 断言 c_buffer 中所有元素均为 41.0
  assertAllEqual(c_buffer, 41.0f);
}

// 定义名为 TEST 的测试用例，测试浮点数类型的元素级最大值操作，其中包含 NaN 值
TEST(LLVM, ElemwiseMaxNaNFloat) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a、b、c 的 BufHandle 对象，分别表示大小为 N 的浮点数数组 A、B、C
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
  // 初始化大小为 N 的浮点数数组 a_buffer、b_buffer、c_buffer，a_buffer 填充为 NaN，b_buffer 和 c_buffer 均填充为 1.0
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  // 定义名为 i 的整数变量
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，遍历 i 从 0 到 N-1，对 C 中第 i 个元素赋值为 A[i] 和 B[i] 的最大值
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  // 创建 LLVMCodeGen 对象 cg，编译 expr，关联的缓
TEST(LLVM, ElemwiseMinFloat) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建三个缓冲区对象，每个对象包含 N 个浮点数
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
  // 初始化 a_buffer、b_buffer 和 c_buffer 分别为 N 个浮点数，初始值分别为 41.0、1.0 和 1.0
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  // 定义整型变量 i
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，对变量 i 从 0 到 N-1 进行循环，执行 c.store({i}, Min::make(a.load(i), b.load(i), false))
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  // 使用 LLVMCodeGen 对象 cg，传入表达式和缓冲区对象数组 {a, b, c} 进行初始化
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，包含 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg.value<int>(args) 返回值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 a_buffer 中所有元素均为 41.0
  assertAllEqual(a_buffer, 41.0f);
  // 断言 b_buffer 中所有元素均为 1.0
  assertAllEqual(b_buffer, 1.0f);
  // 断言 c_buffer 中所有元素均为 1.0
  assertAllEqual(c_buffer, 1.0f);
}

TEST(LLVM, ElemwiseMinNaNFloat) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建三个缓冲区对象，每个对象包含 N 个浮点数
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
  // 初始化 a_buffer 为 N 个 NaN（非数字）、b_buffer 为 N 个 1.0、c_buffer 为 N 个 1.0
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  // 定义整型变量 i
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，对变量 i 从 0 到 N-1 进行循环，执行 c.store({i}, Min::make(a.load(i), b.load(i), false))
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  // 使用 LLVMCodeGen 对象 cg，传入表达式和缓冲区对象数组 {a, b, c} 进行初始化
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，包含 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg.value<int>(args) 返回值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 b_buffer 中所有元素均为 1.0
  assertAllEqual(b_buffer, 1.0f);
  // 遍历 c_buffer 中的每个元素，断言其为 NaN
  for (auto const& elt : c_buffer) {
    ASSERT_TRUE(std::isnan(elt));
  }
}

TEST(LLVM, ElemwiseMod) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建三个缓冲区对象，每个对象包含 N 个整数
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  // 初始化 a_buffer 为 N 个整数 41、b_buffer 为 N 个整数 23、c_buffer 为 N 个整数 18
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 23);
  std::vector<int32_t> c_buffer(N, 18);

  // 定义整型变量 i
  VarHandle i("i", kInt);
  // 创建 For 循环表达式，对变量 i 从 0 到 N-1 进行循环，执行 c.store({i}, Mod::make(a.load(i), b.load(i)))
  auto expr = For::make(i, 0, N, c.store({i}, Mod::make(a.load(i), b.load(i))));

  // 使用 LLVMCodeGen 对象 cg，传入表达式和缓冲区对象数组 {a, b, c} 进行初始化
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，包含 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg.value<int>(args) 返回值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  // 断言 a_buffer 中所有元素均为 41
  assertAllEqual(a_buffer, 41);
  // 断言 b_buffer 中所有元素均为 23
  assertAllEqual(b_buffer, 23);
  // 断言 c_buffer 中所有元素均为 18
  assertAllEqual(c_buffer, 18);
}

TEST(LLVM, CompareSelectIntEQ) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建三个缓冲区对象，每个对象包含 N 个整数
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  // 初始化 a_buffer 和 b_buffer 为 N 个整数 1，c_buffer 和 c_ref 分别为 N 个整数 0 和 1
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  // 对前 N/2 个元素，将 b_buffer 中的元素设为 0
  for (int i = 0; i < N / 2; i++) {
    b_buffer[i] = 0;
  }


以上是对给定代码块进行的详细注释，按照要求包括每行代码的解释和作用。
    c_ref[i] = 0;


// 将 c_ref 数组的第 i 个元素设置为 0
c_ref[i] = 0;



  }

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));


// 定义一个名为 i 的整型变量句柄，用于循环迭代
VarHandle i("i", kInt);
// 创建一个 For 循环表达式，从 i = 0 开始，循环到 i = N-1
// 在每次迭代中，使用 CompareSelect::make() 比较 a 和 b 的值，并根据比较结果存储到 c 中
auto expr = For::make(
    i,
    0,
    N,
    c.store(
        {i},
        CompareSelect::make(
            a.load(i), b.load(i), CompareSelectOperation::kEQ)));



  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);


// 使用表达式 expr 和变量集合 {a, b, c} 创建 LLVMCodeGen 对象 cg
LLVMCodeGen cg(expr, {a, b, c});

// 准备参数向量，包括 a_buffer, b_buffer, c_buffer 的数据指针
std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
// 断言调用 cg 的 value 函数返回值为 0
ASSERT_EQ(cg.value<int>(args), 0);



  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);


// 断言 a_buffer, b_buffer, c_buffer 的大小均为 N
ASSERT_EQ(a_buffer.size(), N);
ASSERT_EQ(b_buffer.size(), N);
ASSERT_EQ(c_buffer.size(), N);



  assertAllEqual(a_buffer, 1);


// 断言 a_buffer 中所有元素都等于 1
assertAllEqual(a_buffer, 1);



  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }


// 遍历 c_ref 和 c_buffer 数组，断言它们对应位置的元素相等
for (const auto i : c10::irange(N)) {
  ASSERT_EQ(c_ref[i], c_buffer[i]);
}
}

// 定义一个名为 TEST 的测试用例，测试 LLVM 中的 CompareSelectFloatEQ 函数
TEST(LLVM, CompareSelectFloatEQ) {
  // 定义常量 N，表示数组大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，存储 float 类型数据
  BufHandle a("A", {N}, kFloat);
  // 创建名为 b 的缓冲区，存储 float 类型数据
  BufHandle b("B", {N}, kFloat);
  // 创建名为 c 的缓冲区，存储 int 类型数据
  BufHandle c("C", {N}, kInt);
  // 初始化包含 float 值 1.0 的数组 a_buffer 和 b_buffer
  std::vector<float> a_buffer(N, 1.0f);
  std::vector<float> b_buffer(N, 1.0f);
  // 初始化包含 int 值 0 的数组 c_buffer
  std::vector<int> c_buffer(N, 0);

  // 创建名为 i 的变量，表示循环索引
  VarHandle i("i", kInt);
  // 创建一个 For 循环表达式，遍历 i 从 0 到 N-1，将 CompareSelect::make 的结果存入 c
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  // 使用 expr 和缓冲区 a、b、c 创建 LLVMCodeGen 对象 cg
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，存储 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg 的 value 方法返回的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  // 断言 a_buffer 中所有元素均为 1.0f
  assertAllEqual(a_buffer, 1.0f);
  // 断言 b_buffer 中所有元素均为 1.0f
  assertAllEqual(b_buffer, 1.0f);
  // 断言 c_buffer 中所有元素均为 0
  assertAllEqual(c_buffer, 0);
}

// 定义一个名为 TEST 的测试用例，测试 LLVM 中的 CompareSelectByteGT 函数
TEST(LLVM, CompareSelectByteGT) {
  // 定义常量 N，表示数组大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，存储 uint8_t 类型数据
  BufHandle a("A", {N}, kByte);
  // 创建名为 b 的缓冲区，存储 uint8_t 类型数据
  BufHandle b("B", {N}, kByte);
  // 创建名为 c 的缓冲区，存储 int 类型数据
  BufHandle c("C", {N}, kInt);
  // 初始化包含 uint8_t 值 0 的数组 a_buffer 和 b_buffer
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 0);
  // 初始化包含 int 值 0 的数组 c_buffer 和 c_ref
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  // 将前 N/2 个元素设置为 128，并将对应的 c_ref 元素设置为 1
  for (int i = 0; i < N / 2; i++) {
    a_buffer[i] = 128;
    c_ref[i] = 1;
  }

  // 创建名为 i 的变量，表示循环索引
  VarHandle i("i", kInt);
  // 创建一个 For 循环表达式，遍历 i 从 0 到 N-1，将 CompareSelect::make 的结果存入 c
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGT)));

  // 使用 expr 和缓冲区 a、b、c 创建 LLVMCodeGen 对象 cg
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，存储 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg 的 value 方法返回的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  // 断言 b_buffer 中所有元素均为 0
  assertAllEqual(b_buffer, uint8_t(0));
  // 遍历 c_ref 数组，逐个断言 c_ref[i] 与 c_buffer[i] 相等
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

// 定义一个名为 TEST 的测试用例，测试 LLVM 中的 CompareSelectByteGE 函数
TEST(LLVM, CompareSelectByteGE) {
  // 定义常量 N，表示数组大小为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，存储 uint8_t 类型数据
  BufHandle a("A", {N}, kByte);
  // 创建名为 b 的缓冲区，存储 uint8_t 类型数据
  BufHandle b("B", {N}, kByte);
  // 创建名为 c 的缓冲区，存储 int 类型数据
  BufHandle c("C", {N}, kInt);
  // 初始化包含 uint8_t 值 0 的数组 a_buffer 和 b_buffer
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 0);
  // 初始化包含 int 值 0 的数组 c_buffer 和 c_ref
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  // 创建名为 i 的变量，表示循环索引
  VarHandle i("i", kInt);
  // 创建一个 For 循环表达式，遍历 i 从 0 到 N-1，将 CompareSelect::make 的结果存入 c
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGE)));

  // 使用 expr 和缓冲区 a、b、c 创建 LLVMCodeGen 对象 cg
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数数组 args，存储 a_buffer、b_buffer 和 c_buffer 的数据指针
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言调用 cg 的 value 方法返回的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer、b_buffer 和 c_buffer 的大小均为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  // 断言 b_buffer 中所有元素均为 0
  assertAllEqual(b_buffer, uint8_t(0));
  // 遍历 c_ref 数组，逐个断言 c_ref[i] 与 c_buffer[i] 相等
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}
TEST(LLVM, CompareSelectByteLT) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建大小为 N 的 BufHandle 对象 a, b, c，类型分别为 kByte 和 kInt
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
  // 初始化长度为 N 的 uint8_t 类型的向量 a_buffer 和 b_buffer，以及长度为 N 的 int 类型的向量 c_buffer 和 c_ref
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 128);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  // 设置前 N/2 个元素为 128，并将 c_ref 对应位置的值设为 0
  for (int i = 0; i < N / 2; i++) {
    a_buffer[i] = 128;
    c_ref[i] = 0;
  }

  // 定义 VarHandle 对象 i，类型为 kInt
  VarHandle i("i", kInt);
  // 创建 For 循环表达式 expr，遍历 i 从 0 到 N-1，将 CompareSelect 操作的结果存储到 c 中
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLT)));

  // 用 LLVMCodeGen 对象 cg 编译 expr，传入 BufHandle 对象 a, b, c
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数向量 args，分别指向 a_buffer, b_buffer, c_buffer 的数据
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言 cg 的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer, b_buffer, c_buffer 的大小为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  // 断言 b_buffer 中所有元素都等于 128
  assertAllEqual(b_buffer, uint8_t(128));
  // 遍历 c10 命名空间中 0 到 N-1 的所有元素，断言 c_ref 和 c_buffer 对应位置的值相等
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteLE) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建大小为 N 的 BufHandle 对象 a, b, c，类型分别为 kByte 和 kInt
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
  // 初始化长度为 N 的 uint8_t 类型的向量 a_buffer 和 b_buffer，以及长度为 N 的 int 类型的向量 c_buffer 和 c_ref
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 128);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  // 定义 VarHandle 对象 i，类型为 kInt
  VarHandle i("i", kInt);
  // 创建 For 循环表达式 expr，遍历 i 从 0 到 N-1，将 CompareSelect 操作的结果存储到 c 中
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLE)));

  // 用 LLVMCodeGen 对象 cg 编译 expr，传入 BufHandle 对象 a, b, c
  LLVMCodeGen cg(expr, {a, b, c});

  // 创建 void* 类型的参数向量 args，指向 a_buffer, b_buffer, c_buffer 的数据
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  // 断言 cg 的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 断言 a_buffer, b_buffer, c_buffer 的大小为 N
  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  // 断言 b_buffer 中所有元素都等于 128
  assertAllEqual(b_buffer, uint8_t(128));
  // 遍历 c10 命名空间中 0 到 N-1 的所有元素，断言 c_ref 和 c_buffer 对应位置的值相等
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, StoreFloat) {
  // 创建大小为 1 的 BufHandle 对象 result，类型为 kFloat
  BufHandle result("result", {1}, kFloat);
  // 初始化长度为 1 的 float 类型的向量 result_buffer，值为 0.0f
  std::vector<float> result_buffer = {0.0f};
  // 创建将 FloatImm::make(3.14f) 存储到 result 的表达式 expr
  auto expr = result.store({0}, FloatImm::make(3.14f));
  // 用 LLVMCodeGen 对象 cg 编译 expr，传入 BufHandle 对象 result
  LLVMCodeGen cg(expr, {result});
  // 创建 void* 类型的参数向量 args，指向 result_buffer 的数据
  std::vector<void*> args({result_buffer.data()});
  // 断言 cg 的值为 0
  ASSERT_EQ(cg.value<int>(args), 0);
  // 断言 result_buffer 的第一个元素值为 3.14f
  ASSERT_EQ(result_buffer[0], 3.14f);
}

TEST(LLVM, SimpleMath01) {
  // 定义常量 N 为 1024
  const int N = 1024;
  // 创建 Tensor 对象 tensor，大小为 N，计算表达式为 cast<float>(i * i + 1)
  Tensor tensor = Compute(
      "f", {N}, [](const VarHandle& i) { return cast<float>(i * i + 1); });
  // 创建 LoopNest 对象 l，包含 tensor
  LoopNest l({tensor});
  // 获取 l 的根语句 stmt
  StmtPtr stmt = l.root_stmt();
  // 创建大小为 N 的 BufHandle 对象 f_buf，使用 tensor 的缓冲区
  BufHandle f_buf(tensor.buf());
  // 用 LLVMCodeGen 对象 cg 编译 stmt，传入 BufHandle 对象 f_buf
  LLVMCodeGen cg(stmt, {f_buf});

  // 创建 PaddedBuffer<float> 类型的对象 f_v 和 f_ref，大小都为 N
  PaddedBuffer<float> f_v(N, "f_v");
  // 创建 void* 类型的参数向量 args，指向 f_v 的数据
  std::vector<void*> args({f_v.data()});
  // 获取 cg 的值，断言其为 0
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);

  // 创建 PaddedBuffer<float> 类型的对象 f_ref，计算 f_ref(i) = i * i + 1
  PaddedBuffer<float> f_ref(N, "f_ref");
  for (const auto i : c10::irange(N)) {
    f_ref(i) = i * i + 1;
  }
  // 断言 f_v 和 f_ref 的所有元素在 1e-5 范围内相等
  ExpectAllNear(f_v, f_ref, 1e-5);
}
// 测试用例：ComputeMul
TEST(LLVM, ComputeMul) {
  const int N = 1024;
  // 创建缓冲区对象a，表示大小为N的浮点数数组
  BufHandle a("a", {N}, kFloat);
  // 创建缓冲区对象b，表示大小为N的浮点数数组
  BufHandle b("b", {N}, kFloat);
  // 计算张量c，大小为N，使用lambda函数对a和b的元素进行乘法操作
  Tensor c = Compute(
      "c", {N}, [&](const VarHandle& i) { return a.load(i) * b.load(i); });

  // 获取张量c对应的缓冲区对象
  BufHandle c_buf(c.buf());
  // 创建循环嵌套对象，包含张量c
  LoopNest l({c});
  // 准备循环嵌套对象进行代码生成
  l.prepareForCodegen();
  // 获取根语句的指针
  StmtPtr s = l.root_stmt();

  // 使用LLVMCodeGen类对s，a，b，c_buf进行LLVM代码生成
  LLVMCodeGen cg(s, {a, b, c_buf});

  // 创建大小为N的浮点数向量a_vec，每个元素初始化为21.0
  std::vector<float> a_vec(N, 21.0f);
  // 创建大小为N的浮点数向量b_vec，每个元素初始化为2.0
  std::vector<float> b_vec(N, 2.0f);
  // 创建大小为N的浮点数向量c_vec，每个元素初始化为0.0
  std::vector<float> c_vec(N, 0.0f);
  // 创建指针向量args，包含a_vec、b_vec、c_vec的数据指针
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  // 断言cg对象对args进行求值结果为0
  ASSERT_EQ(cg.value<int>(args), 0);
  // 断言c_vec中所有元素与42.0f相等
  assertAllEqual(c_vec, 42.0f);
}

// 测试用例：BroadcastAdd
TEST(LLVM, BroadcastAdd) {
  const int M = 32;
  const int N = 1024;
  // 创建二维缓冲区对象a，大小为MxN的浮点数数组
  BufHandle a("a", {M, N}, kFloat);
  // 创建缓冲区对象b，大小为N的浮点数数组
  BufHandle b("b", {N}, kFloat);
  // 计算张量c，大小为MxN，使用lambda函数对a和b的元素进行加法操作
  Tensor c = Compute("c", {M, N}, [&](const VarHandle& i, const VarHandle& j) {
    return a.load(i, j) + b.load(j);
  });

  // 获取张量c对应的缓冲区对象
  BufHandle c_buf(c.buf());
  // 创建循环嵌套对象，包含张量c
  LoopNest l({c});
  // 准备循环嵌套对象进行代码生成
  l.prepareForCodegen();
  // 获取根语句的指针
  StmtPtr s = l.root_stmt();

  // 使用LLVMCodeGen类对s，a，b，c_buf进行LLVM代码生成
  LLVMCodeGen cg(s, {a, b, c_buf});

  // 创建大小为M*N的浮点数向量av，元素值依次递增
  std::vector<float> av(M * N);
  std::iota(av.begin(), av.end(), 0);
  // 创建大小为N的浮点数向量bv，元素值依次递增
  std::vector<float> bv(N);
  std::iota(bv.begin(), bv.end(), 0);
  // 创建大小为M*N的浮点数向量cv，所有元素初始化为0
  std::vector<float> cv(M * N, 0);
  // 创建指针向量args，包含av、bv、cv的数据指针
  std::vector<void*> args({av.data(), bv.data(), cv.data()});
  // 断言cg对象对args进行求值结果为0
  ASSERT_EQ(cg.value<int>(args), 0);

  // 对c_vec中每个元素进行遍历检查，确保其值为av和bv对应位置元素的和
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      ASSERT_EQ(cv[i * N + j], av[i * N + j] + bv[j]);
    }
  }
}

// 测试用例：BitwiseOps
TEST(LLVM, BitwiseOps) {
  // 创建IntImm类型的常量表达式a、b、c、d
  auto a = IntImm::make(59);
  auto b = IntImm::make(11);
  auto c = IntImm::make(101);
  auto d = IntImm::make(2);

  // 创建复杂的位运算表达式f
  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;
  // 使用LLVMExprEval类对f进行LLVM表达式求值
  LLVMExprEval cg(f);

  // 断言cg对象的值等于11
  ASSERT_EQ(cg.value<int>(), 11);
}

// 测试用例：ArithmeticRightShift
TEST(LLVM, ArithmeticRightShift) {
  // 创建CharImm类型的常量表达式a、b
  auto a = CharImm::make(-4);
  auto b = CharImm::make(1);
  // 创建算术右移表达式f
  ExprHandle f = a >> b;
  // 使用LLVMExprEval类对f进行LLVM表达式求值
  LLVMExprEval cg(f);
  // 断言cg对象的值等于-2
  ASSERT_EQ(cg.value<int8_t>(), -2);
}

// 测试用例：LogicalRightShift
TEST(LLVM, LogicalRightShift) {
  // 创建ByteImm类型的常量表达式a、b
  auto a = ByteImm::make(0xfc);
  auto b = ByteImm::make(1);
  // 创建逻辑右移表达式f
  ExprHandle f = a >> b;
  // 使用LLVMExprEval类对f进行LLVM表达式求值
  LLVMExprEval cg(f);
  // 断言cg对象的值等于0x7e
  ASSERT_EQ(cg.value<uint8_t>(), 0x7e);
}

// 测试用例：DynamicShapeAdd
TEST(LLVM, DynamicShapeAdd) {
  // 定义测试函数testWithSize，接受一个整数参数size
  auto testWithSize = [](int32_t size) {
    // 创建整型变量n
    VarHandle n("n", kInt);
    // 创建大小为n的浮点数数组缓冲区对象a、b、c
    BufHandle a("a", {n}, kFloat);
    BufHandle b("b", {n}, kFloat);
    BufHandle c("c", {n}, kFloat);
    // 创建整型变量i
    VarHandle i("i", kInt);
    // 创建for循环语句s，用于将a和b的元素相加并存储到c中
    StmtPtr s = For::make(i, 0, n, c.store({i}, a.load(i)));
    // 创建大小为size的浮点数向量aData，所有元素初始化为1.0
    std::vector<float> aData(size, 1.0f);
    // 创建大小为size的浮点数向量bData，所有元素初始化为2.0
    std::vector<float> bData(size, 2.0f);
    // 创建大小为size的浮点数向量cData，所有元素初始化为0.0
    std::vector<float> cData(size, 0.0f);
    // 使用LLVMCodeGen类对s、a、b、c、n进行LLVM代码生成
    LLVMCodeGen cg(s, {a, b, c, n});
    // 创建指针向量args，包含aData、bData、cData和size的地址
    std::vector<void*> args({aData.data(), bData.data(), cData.data(), &size});
    // 使用cg对象对args进行求值，期望结果与期望值（3.0f）接近
    cg.value<float>(args);
    ExpectAllNear(cData, std::vector<float>(size
    // 创建一个指向 For 循环语句的指针，用于计算数组 c 的元素
    StmtPtr s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
    // 初始化数组 aData，所有元素设为 1.0
    std::vector<float> aData(size, 1.0f);
    // 初始化数组 bData，所有元素设为 2.0
    std::vector<float> bData(size, 2.0f);
    // 初始化数组 cData，所有元素设为 0.0
    std::vector<float> cData(size, 0.0f);
    // 创建 LLVMCodeGen 对象 cg，用于生成 LLVM 代码
    LLVMCodeGen cg(s, {a, b, c, n});
    // 调用 LLVMCodeGen 对象的 call 方法执行生成的 LLVM 代码
    cg.call({aData, bData, cData, size});
    // 检验数组 cData 的每个元素是否接近于 3.0，精度为 1e-7
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  // 分别用不同的数组大小调用测试函数 testWithSize
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
TEST(LLVM, TensorDynamicShapeAdd) {
  // 定义一个 lambda 函数 testWithSize，用于测试指定大小的计算
  auto testWithSize = [](int32_t size) {
    // 创建一个名为 n 的整数变量
    VarHandle n("n", kInt);
    // 创建一个名为 a 的缓冲区，其大小为 {n}，数据类型为 kFloat
    BufHandle a("a", {n}, kFloat);
    // 创建一个名为 b 的缓冲区，其大小也为 {n}，数据类型为 kFloat
    BufHandle b("b", {n}, kFloat);
    // 定义张量 c，通过 Compute 函数创建，计算结果为 a[i] + b[i]
    Tensor c = Compute(
        "c", {n}, [&](const VarHandle& i) { return a.load(i) + b.load(i); });
    // 创建一个循环嵌套对象 l，用于处理张量 c
    LoopNest l({c});
    // 获取生成的语句树的根节点
    StmtPtr s = l.root_stmt();
    // 创建 LLVMCodeGen 对象 cg，用于将语句树编译为 LLVM IR
    LLVMCodeGen cg(s, {a, b, c, n});
    // 创建大小为 size 的浮点数向量 aData，所有元素初始化为 1.0f
    std::vector<float> aData(size, 1.0f);
    // 创建大小为 size 的浮点数向量 bData，所有元素初始化为 2.0f
    std::vector<float> bData(size, 2.0f);
    // 创建大小为 size 的浮点数向量 cData，所有元素初始化为 0.0f
    std::vector<float> cData(size, 0.0f);
    // 调用 LLVMCodeGen 对象的 call 方法，执行 LLVM IR 代码生成
    cg.call({aData, bData, cData, size});
    // 断言 cData 中的每个元素都接近于大小为 size 的浮点数向量，元素值均为 3.0f
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  // 分别使用大小为 1、16 和 37 调用 testWithSize 函数进行测试
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(LLVM, DynamicShape2D) {
  // 定义一个 lambda 函数 testWithSize，用于测试指定大小的二维计算
  auto testWithSize = [](int32_t M, int32_t N) {
    // 创建名为 m 和 n 的整数变量
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    // 创建一个名为 a 的二维缓冲区，大小为 {m, n}，数据类型为 kFloat
    BufHandle a("a", {m, n}, kFloat);
    // 创建一个名为 b 的二维缓冲区，大小也为 {m, n}，数据类型为 kFloat
    BufHandle b("b", {m, n}, kFloat);
    // 定义张量 c，通过 Compute 函数创建，计算结果为 a[i, j] + b[i, j]
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    // 创建一个循环嵌套对象 l，用于处理张量 c
    LoopNest l({c});
    // 准备循环嵌套对象 l 进行代码生成
    l.prepareForCodegen();
    // 获取生成的语句树的根节点
    StmtPtr s = l.root_stmt();
    // 创建 LLVMCodeGen 对象 cg，用于将语句树编译为 LLVM IR
    LLVMCodeGen cg(s, {a, b, c, m, n});
    // 创建大小为 M * N 的浮点数向量 aData，所有元素初始化为 1.0f
    std::vector<float> aData(M * N, 1.0f);
    // 创建大小为 M * N 的浮点数向量 bData，所有元素初始化为 2.0f
    std::vector<float> bData(M * N, 2.0f);
    // 创建大小为 M * N 的浮点数向量 cData，所有元素初始化为 0.0f
    std::vector<float> cData(M * N, 0.0f);
    // 调用 LLVMCodeGen 对象的 call 方法，执行 LLVM IR 代码生成
    cg.call({aData, bData, cData, M, N});
    // 断言 cData 中的每个元素都接近于大小为 M * N 的浮点数向量，元素值均为 3.0f
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  // 分别使用大小为 (1, 8)、(16, 32) 和 (37, 11) 调用 testWithSize 函数进行测试
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

TEST(LLVM, EmptyStmt) {
  // 创建一个空的块语句，并分配给 StmtPtr 对象 s
  StmtPtr s = alloc<Block>(std::vector<StmtPtr>({}));

  // 创建 LLVMCodeGen 对象 cg，用于将空的块语句编译为 LLVM IR
  LLVMCodeGen cg(s, {});
  // 调用 LLVMCodeGen 对象的 call 方法，执行 LLVM IR 代码生成
  cg.call({});
  // 简单地断言不会崩溃
}

TEST(LLVM, EliminatedStmt) {
  // 创建一个名为 a 的一维缓冲区，大小为 {1}，数据类型为 kFloat
  BufHandle a("a", {1}, kFloat);

  // 定义张量 c，通过 Compute 函数创建，计算结果为 m
  Tensor c = Compute("c", {0}, [&](const VarHandle& m) { return m; });

  // 创建一个循环嵌套对象 l，用于处理张量 c
  LoopNest l({c});
  // 准备循环嵌套对象 l 进行代码生成
  l.prepareForCodegen();
  // 获取生成的语句树的根节点，并对其进行简化
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建 LLVMCodeGen 对象 cg，用于将简化后的语句树编译为 LLVM IR
  LLVMCodeGen cg(s, {a, c});
  // 创建大小为 1 的浮点数向量 aData，所有元素初始化为 1.0f
  std::vector<float> aData(1, 1.0f);
  // 创建大小为 0 的浮点数向量 cData
  std::vector<float> cData(0, 0.0f);
  // 调用 LLVMCodeGen 对象的 call 方法，执行 LLVM IR 代码生成
  cg.call({aData, cData});
}

TEST(LLVM, SimpleReduction) {
  // 定义常量 M 和 N
  int M = 128;
  int N = 64;

  // 创建一个名为 a 的三维缓冲区，大小为 {1, M, N}，数据类型为 kFloat
  BufHandle a("a", {1, M, N}, kFloat);

  // 定义张量 b，通过 Reduce 函数创建，计算结果为 a 中所有元素的和
  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});
  // 创建循环嵌套对象 loop，用于处理张量 b
  LoopNest loop({b});

  // 准备循环嵌套对象 loop 进行代码生成
  loop.prepareForCodegen();
  // 获取生成的语句树的根节点，并对其进行简化
  StmtPtr s = IRSimplifier::simplify(loop.root_stmt());

  // 创建 LLVMCodeGen 对象 cg，用于将简化后的语句树编译为 LLVM IR
  LLVMCodeGen cg(s, {a, b});

  // 创建三个填充缓冲区对象，用于存储测试数据和参考结果
  PaddedBuffer<float> a_v(1, M, N
TEST(LLVM, RFactorReduction) {
  // 设置矩阵维度
  int M = 128;
  int N = 64;

  // 创建名为 "a" 的缓冲区对象，尺寸为 {1, M, N}，元素类型为 kFloat
  BufHandle a("a", {1, M, N}, kFloat);

  // 对缓冲区 a 进行求和约简操作，结果存入张量 b
  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});

  // 创建循环嵌套对象，并将张量 b 加入其中
  LoopNest loop({b});

  // 获取张量 b 相关的循环语句
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(b);

  // 获取循环嵌套中的第二个和第三个循环语句
  ForPtr loop_m = loops.at(1);
  ForPtr loop_n = loops.at(2);

  // 重新排序循环轴，将 m 和 n 循环交换顺序
  loop.reorderAxis(loop_m, loop_n);

  // 重新获取循环语句
  loops = loop.getLoopStmtsFor(b);
  loop_m = loops.at(2);
  loop_n = loops.at(1);

  // 获取张量 b 的写入操作
  auto b_body = loop.getAllWritesToBuf(b.buf())[1];

  // 将循环 n 进行因子重构，并断言操作成功
  ASSERT_TRUE(loop.rfactor(b_body, loop_n));

  // 为代码生成做准备工作
  loop.prepareForCodegen();

  // 获取最终的语句表示
  StmtPtr s = loop.root_stmt();

  // 对生成的 IR 进行简化
  s = IRSimplifier::simplify(s);

  // 使用 LLVMCodeGen 进行代码生成，包括缓冲区 a 和 b
  LLVMCodeGen cg(s, {a, b});

  // 创建用于存储数据的填充缓冲区对象
  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  // 初始化参考结果缓冲区 b_ref
  b_ref(0) = 0;

  // 执行计算，填充 a_v 缓冲区，并同时计算参考结果 b_ref
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  // 调用 LLVMCodeGen 对象的调用方法，计算结果存入 b_v
  cg.call({a_v, b_v});

  // 断言 b_v 与 b_ref 在给定精度下相近
  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, RFactorVectorizedReduction) {
  // 设置矩阵维度
  int M = 128;
  int N = 64;

  // 创建名为 "a" 的缓冲区对象，尺寸为 {1, M, N}，元素类型为 kFloat
  BufHandle a("a", {1, M, N}, kFloat);

  // 对缓冲区 a 进行求和约简操作，结果存入张量 b
  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});

  // 创建循环嵌套对象，并将张量 b 加入其中
  LoopNest loopnest({b});

  // 获取张量 b 相关的循环语句
  std::vector<ForPtr> loops = loopnest.getLoopStmtsFor(b);

  // 重新排序循环轴，将第二个和第三个循环交换顺序
  loopnest.reorderAxis(loops.at(1), loops.at(2));

  // 获取张量 b 的写入操作
  auto b_body = loopnest.getAllWritesToBuf(b.buf()).at(1);

  // 获取所有写入张量 b 的循环嵌套
  auto all_loops = loopnest.getAllLoopNestsWritingToBuf(b.buf());

  // 断言所有循环嵌套的数量和第二个循环嵌套的长度符合预期
  ASSERT_TRUE(all_loops.size() == 2 && all_loops[1].size() == 3);

  // 将张量 b 的写入操作进行因子重构，并断言操作成功
  ASSERT_TRUE(loopnest.rfactor(b_body, all_loops[1][1]));

  // 分布式循环
  auto distributed_loops = loopnest.distributeLoop(all_loops[1][1]);

  // 向量化因子重构缓冲区的初始化器
  ASSERT_TRUE(LoopNest::vectorize(distributed_loops[0]));

  // 向量化因子重构缓冲区的生产者
  ASSERT_TRUE(LoopNest::vectorize(distributed_loops[1]));

  // 简化循环嵌套
  loopnest.simplify();

  // 为代码生成做准备工作
  loopnest.prepareForCodegen();

  // 获取简化后的语句表示
  StmtPtr s = IRSimplifier::simplify(loopnest.root_stmt());

  // 使用 LLVMCodeGen 进行代码生成，包括缓冲区 a 和 b
  LLVMCodeGen cg(s, {a, b});

  // 创建用于存储数据的填充缓冲区对象
  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  // 初始化参考结果缓冲区 b_ref
  b_ref(0) = 0;

  // 执行计算，填充 a_v 缓冲区，并同时计算参考结果 b_ref
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  // 调用 LLVMCodeGen 对象的调用方法，计算结果存入 b_v
  cg.call({a_v, b_v});

  // 断言 b_v 与 b_ref 在给定精度下相近
  ExpectAllNear(b_v, b_ref, 1e-5);
}

template <bool outer, bool inner>
static void testSimpleParallel() {
  // 执行一个简单的操作，并尝试所有循环轴的并行或顺序组合
  const int M = 4;
  const int N = 6;

  // 创建张量 f，大小为 {M, N}，使用 lambda 表达式计算元素值
  Tensor f = Compute("f", {M, N}, [](const VarHandle& m, const VarHandle& n) {
    return cast<float>(m + n);
  });

  // 创建循环嵌套对象，并将张量 f 加入其中
  LoopNest loop_nest({f});

  // 获取张量 f 相关的循环语句
  auto const& loops = loop_nest.getLoopStmtsFor(f);

  // 获取循环 m 和 n 的指针
  ForPtr m = loops[0];
  ForPtr n = loops[1];

  // 如果 outer 为 true，则将 m 循环设置为并行
  if (outer) {
    m->set_parallel();
  }

  // 如果 inner 为 true，则将 n 循环设置为并行
  if (inner) {
    n->set_parallel();
  }
    n->set_parallel();

# 调用对象指针 n 的 set_parallel() 方法，设置其为并行执行。

  }

# 结束了一个代码块，可能是之前的循环或条件语句的结束。

  loop_nest.prepareForCodegen();

# 调用 loop_nest 对象的 prepareForCodegen() 方法，为代码生成做准备。

  StmtPtr stmt = loop_nest.root_stmt();

# 从 loop_nest 对象中获取根语句，并将其赋给变量 stmt。

  LLVMCodeGen cg(stmt, {f});

# 使用根语句 stmt 和包含变量 f 的向量，创建 LLVMCodeGen 对象 cg。

  PaddedBuffer<float> f_v(M, N, "f_v");

# 创建一个 PaddedBuffer 对象 f_v，存储类型为 float，尺寸为 MxN，命名为 "f_v"。

  std::vector<void*> args({f_v.data()});

# 创建一个 void* 类型的向量 args，其中包含 f_v 的数据指针。

  int value = cg.value<int>(args);

# 调用 cg 对象的 value 方法，使用参数 args，返回一个 int 类型的值，并将其赋给 value。

  ASSERT_EQ(value, 0);

# 使用断言验证 value 的值是否等于 0，如果不等则会抛出异常。

  PaddedBuffer<float> f_ref(M, N, "f_ref");

# 创建另一个 PaddedBuffer 对象 f_ref，存储类型为 float，尺寸为 MxN，命名为 "f_ref"。

  for (const auto m : c10::irange(M)) {

# 对于 m 在 c10 命名空间中 irange(M) 的范围内循环，其中 M 是预定义的数量。

    for (const auto n : c10::irange(N)) {

# 对于 n 在 c10 命名空间中 irange(N) 的范围内循环，其中 N 是预定义的数量。

      f_ref(m, n) = m + n;

# 将 f_ref 的第 m 行、第 n 列位置赋值为 m 加 n 的结果。

    }
  }

# 结束内层循环。

  ExpectAllNear(f_v, f_ref, 1e-5);

# 使用 ExpectAllNear 函数验证 f_v 和 f_ref 的每个元素是否在 1e-5 的误差范围内接近。
}

TEST(LLVM, SimpleParallelSS) {
  // 调用模板函数 testSimpleParallel，参数为 false, false，即全部顺序执行
  testSimpleParallel<false, false>();
}

TEST(LLVM, SimpleParallelSP) {
  // 调用模板函数 testSimpleParallel，参数为 false, true，即第二个参数并行，第一个参数顺序执行
  testSimpleParallel<false, true>();
}

TEST(LLVM, SimpleParallelPS) {
  // 调用模板函数 testSimpleParallel，参数为 true, false，即第一个参数并行，第二个参数顺序执行
  testSimpleParallel<true, false>();
}

TEST(LLVM, SimpleParallelPP) {
  // 调用模板函数 testSimpleParallel，参数为 true, true，即全部并行执行
  testSimpleParallel<true, true>();
}

TEST(LLVM, CompositeParallel) {
  int loop_count = 6;
  int test_count = 1 << loop_count;
  // 计算一个复合操作，并尝试所有循环轴组合为并行或顺序执行。
  for (const auto test_cfg : c10::irange(test_count)) {
    int M = 5;
    int N = 7;
    // 定义张量 t1，用于存储 Compute 操作结果
    Tensor t1 = Compute("t1", {M}, [](const VarHandle& m) { return m + 1.f; });
    // 定义张量 t2，用于存储 Compute 操作结果
    Tensor t2 = Compute("t2", {N}, [](const VarHandle& n) { return n + 2.f; });
    // 定义张量 t3，用于存储 Compute 操作结果，依赖 t1 和 t2
    Tensor t3 =
        Compute("t3", {M, N}, [=](const VarHandle& m, const VarHandle& n) {
          return t1.load(m) * t2.load(n);
        });
    // 定义张量 t4，用于存储 Compute 操作结果，依赖 t3、m 和 n
    Tensor t4 =
        Compute("t4", {M, N}, [=](const VarHandle& m, const VarHandle& n) {
          return t3.load(m, n) + m + n;
        });
    // 创建循环嵌套对象 loop_nest，传入 t4、t1、t2、t3 张量
    LoopNest loop_nest({t4}, {t1, t2, t3, t4});
    std::vector<ForPtr> loop_list;
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t1);
      loop_list.push_back(loops[0]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t2);
      loop_list.push_back(loops[0]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t3);
      loop_list.push_back(loops[0]);
      loop_list.push_back(loops[1]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t4);
      loop_list.push_back(loops[0]);
      loop_list.push_back(loops[1]);
    }
    // 断言循环嵌套的循环数等于 loop_count
    ASSERT_EQ(loop_list.size(), loop_count);
    // 根据 test_cfg 的位掩码设置循环列表中对应位置的循环为并行
    for (const auto i : c10::irange(loop_count)) {
      if (test_cfg & (1 << i)) {
        loop_list[i]->set_parallel();
      }
    }
    // 准备循环嵌套对象以便于代码生成
    loop_nest.prepareForCodegen();
    // 获取根语句的指针
    StmtPtr stmt = loop_nest.root_stmt();
    // 创建 LLVMCodeGen 对象 cg，传入根语句和 t4 张量
    LLVMCodeGen cg(stmt, {t4});

    // 创建 PaddedBuffer 对象 t4_v，用于存储计算结果
    PaddedBuffer<float> t4_v(M, N, "t4_v");
    std::vector<void*> args({t4_v.data()});
    // 使用 cg 计算结果值，传入 t4_v 的数据指针
    int value = cg.value<int>(args);
    // 断言计算结果为 0
    ASSERT_EQ(value, 0);
    // 创建参考结果矩阵 t4_ref，用于与计算结果比较
    PaddedBuffer<float> t4_ref(M, N, "t4_ref");
    // 计算 t4_ref 的每个元素值
    for (const auto m : c10::irange(M)) {
      for (const auto n : c10::irange(N)) {
        t4_ref(m, n) = (m + 1) * (n + 2) + m + n;
      }
    }
    // 断言 t4_v 和 t4_ref 在给定精度下相等
    ExpectAllNear(t4_v, t4_ref, 1e-5);
  }
}

TEST(LLVM, VectorizedGEMM) {
  int M = 32;
  int N = 32;
  int K = 48;

  // 定义缓冲区 A 和 B，分别为 MxK 和 KxN 的浮点数缓冲区
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);
  // 定义张量 CT，表示矩阵乘法的累加和，维度为 MxN
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});
  // 创建 LoopNest 对象 loop，传入 CT 张量
  LoopNest loop({CT});

  // 在 CT 张量的第一个循环上应用大小为 16 的分裂
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr m = loops[0];
    loop.splitWithMask(m, 16);
  }
  // 在 CT 张量的第三个循环上应用大小为 16 的分裂
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
  // 获取循环嵌套中第二个循环的指针
  ForPtr mi = loops[1];
  // 获取循环嵌套中第三个循环的指针
  ForPtr no = loops[2];
  // 重新排序循环嵌套中的 mi 和 no 两个循环
  loop.reorderAxis(mi, no);
}

// mo, no, mi, ni, k ->
// mo, no, mi, k, ni
{
  // 获取循环嵌套中的循环语句列表
  auto const& loops = loop.getLoopStmtsFor(CT);
  // 获取循环嵌套中第四个循环的指针
  ForPtr ni = loops[3];
  // 获取循环嵌套中第五个循环的指针
  ForPtr k = loops[4];
  // 重新排序循环嵌套中的 ni 和 k 两个循环
  loop.reorderAxis(ni, k);
}

// mo, no, mi, k, ni ->
// mo, no, k, mi, ni
{
  // 获取循环嵌套中的循环语句列表
  auto const& loops = loop.getLoopStmtsFor(CT);
  // 获取循环嵌套中第三个循环的指针
  ForPtr mi = loops[2];
  // 获取循环嵌套中第四个循环的指针
  ForPtr k = loops[3];
  // 重新排序循环嵌套中的 mi 和 k 两个循环
  loop.reorderAxis(mi, k);
}

{
  // 查找循环嵌套中所有的循环语句
  auto loops = NodeFinder<For>::find(loop.root_stmt());
  // 断言第四个循环可以向量化
  ASSERT_TRUE(LoopNest::vectorize(loops[3]));
  // 断言最后一个循环可以向量化
  ASSERT_TRUE(LoopNest::vectorize(loops.back()));
}

// 准备循环嵌套以进行代码生成
loop.prepareForCodegen();

// 获取循环嵌套的根语句
StmtPtr s = loop.root_stmt();
// 简化根语句并更新 s
s = IRSimplifier::simplify(s);

// 使用 LLVMCodeGen 对象生成 LLVM 代码，传入参数为 AP, BP, CT
LLVMCodeGen cg(s, {AP, BP, CT});

// 创建缓冲区 a_v, b_v, c_v, c_ref，分别表示矩阵 a, b, c 的填充缓冲区及参考结果缓冲区
PaddedBuffer<float> a_v(M, K, "a_v");
PaddedBuffer<float> b_v(K, N, "b_v");
PaddedBuffer<float> c_v(M, N, "c_v");
PaddedBuffer<float> c_ref(M, N, "c_ref");

// 计算参考结果矩阵 c_ref
for (const auto m : c10::irange(M)) {
  for (const auto n : c10::irange(N)) {
    // 初始化 c_ref(m, n) 为 0
    c_ref(m, n) = 0.f;
    for (const auto k : c10::irange(K)) {
      // 计算矩阵乘法 c_ref(m, n) += a_v(m, k) * b_v(k, n)
      c_ref(m, n) += a_v(m, k) * b_v(k, n);
    }
  }
}

// 调用 LLVMCodeGen 对象生成的 LLVM 代码，计算 c_v 矩阵
cg.call({a_v, b_v, c_v});

// 断言 c_v 与参考结果 c_ref 的所有元素在给定精度下近似相等
ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LLVM, CallRaw) {
  const int M = 32; // 定义常量 M，表示张量维度
  VarHandle N("N", kInt); // 定义整数变量 N，并指定其类型为 kInt
  BufHandle a("a", {M, N}, kFloat); // 定义二维缓冲区变量 a，形状为 {M, N}，数据类型为 kFloat
  BufHandle b("b", {N}, kFloat); // 定义一维缓冲区变量 b，形状为 {N}，数据类型为 kFloat
  Tensor c = Compute("c", {M, N}, [&](const VarHandle& i, const VarHandle& j) { // 定义张量 c，形状为 {M, N}，通过 lambda 函数计算每个元素的值
    return a.load(i, j) + b.load(j); // 计算张量 c 的每个元素值为 a(i, j) + b(j)
  });

  LoopNest l({c}); // 创建循环嵌套对象 l，包含张量 c
  l.prepareForCodegen(); // 准备进行代码生成
  StmtPtr s = l.root_stmt(); // 获取循环嵌套的根语句

  int32_t N_value = 1024; // 初始化整数变量 N_value，值为 1024
  std::vector<float> av(M * N_value); // 创建长度为 M*N_value 的浮点数向量 av，并初始化为递增值
  std::iota(av.begin(), av.end(), 0); // 使用 iota 函数填充向量 av
  std::vector<float> bv(N_value); // 创建长度为 N_value 的浮点数向量 bv，并初始化为递增值
  std::iota(bv.begin(), bv.end(), 0); // 使用 iota 函数填充向量 bv
  std::vector<float> cv(M * N_value, 0); // 创建长度为 M*N_value 的浮点数向量 cv，并初始化为 0

  // 准备参数列表，包括 av、bv、cv 的数据指针和 N_value 的地址
  std::vector<void*> args({av.data(), bv.data(), cv.data(), &N_value});

  LLVMCodeGen cg(s, {a, b, BufHandle(c.buf()), N}); // 使用循环嵌套的根语句 s 和缓冲区变量构建 LLVM 代码生成器 cg
  cg.call_raw(args); // 调用 LLVM 代码生成器的原始调用方法，传入参数列表 args

  // 遍历张量 c 的所有元素，进行断言检查
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N_value)) {
      ASSERT_EQ(cv[i * N_value + j], av[i * N_value + j] + bv[j]); // 断言 cv[i*N_value + j] 的值等于 av[i*N_value + j] + bv[j]
    }
  }

  SimpleIREvaluator eval(s, {a, b, BufHandle(c.buf()), N}); // 创建简单的 IR 评估器 eval，用于执行计算
  eval.call_raw(args); // 使用简单的 IR 评估器执行原始调用，传入参数列表 args

  // 再次遍历张量 c 的所有元素，进行断言检查
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N_value)) {
      ASSERT_EQ(cv[i * N_value + j], av[i * N_value + j] + bv[j]); // 断言 cv[i*N_value + j] 的值等于 av[i*N_value + j] + bv[j]
    }
  }
}

TEST(LLVM, CustomTarget) {
  constexpr int M = 16; // 定义常量 M，表示张量维度
  BufHandle a("a", {M}, kFloat); // 定义一维缓冲区变量 a，形状为 {M}，数据类型为 kFloat
  BufHandle b("b", {M}, kFloat); // 定义一维缓冲区变量 b，形状为 {M}，数据类型为 kFloat
  BufHandle c("c", {M}, kFloat); // 定义一维缓冲区变量 c，形状为 {M}，数据类型为 kFloat
  Tensor d = Compute("d", {M}, [&](const VarHandle& m) { // 定义张量 d，形状为 {M}，通过 lambda 函数计算每个元素的值
    return a.load(m) * b.load(m) + c.load(m); // 计算张量 d 的每个元素值为 a(m) * b(m) + c(m)
  });
  LoopNest nest({d}); // 创建循环嵌套对象 nest，包含张量 d
  nest.prepareForCodegen(); // 准备进行代码生成

  // 使用 LLVMCodeGenBuilder 构建 LLVM 代码生成器 cg，并设置目标三元组和 CPU 类型
  auto cg = LLVMCodeGenBuilder(nest.root_stmt(), {a, b, c, d})
                .triple("i686-elf")
                .cpu("i386")
                .build();

  std::ostringstream ss; // 创建字符串流对象 ss
  ss << cg->getCodeText("asm"); // 将 LLVM 代码生成器生成的汇编代码文本输出到字符串流 ss

  // 使用 FileCheck 工具检查汇编代码文本 ss，确保包含特定的指令，但不包含其他指令
  torch::jit::testing::FileCheck()
      .check("fadds")
      ->check("fmuls")
      ->check_not("vfmadd")
      ->run(ss.str());
}

TEST(LLVM, CodeGenKernelFuncName) {
  BufHandle a("A", {1}, kInt); // 定义一维缓冲区变量 a，形状为 {1}，数据类型为 kInt
  BufHandle b("B", {1}, kInt); // 定义一维缓冲区变量 b，形状为 {1}，数据类型为 kInt
  std::vector<int32_t> a_buffer = {42}; // 创建包含一个整数 42 的整数向量 a_buffer
  std::vector<int32_t> b_buffer = {-11}; // 创建包含一个整数 -11 的整数向量 b_buffer
  auto store = b.store({0}, a.load(0)); // 定义存储操作 store，将 a 的第一个元素存储到 b 的第一个位置

  {
    LLVMCodeGen cg(store, {a, b}); // 创建 LLVM 代码生成器 cg，用于执行存储操作 store，并传入变量 a 和 b
    // 检查 LLVMCodeGen 使用的内核函数名不为空
    ASSERT_NE(cg.kernel_func_name(), "");
  }

  {
    LLVMCodeGen cg(store, {a, b}, at::kCPU, "new_func"); // 创建 LLVM 代码生成器 cg，指定 CPU 类型和自定义的内核函数名
    // 检查 LLVMCodeGen 使用的内核函数名与上面指定的一致
    ASSERT_EQ(cg.kernel_func_name(), "new_func");
  }
}

} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM
```