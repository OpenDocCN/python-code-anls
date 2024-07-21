# `.\pytorch\test\cpp\tensorexpr\test_simplify.cpp`

```py
// 包含 Google Test 框架的头文件，用于单元测试
#include <gtest/gtest.h>
// 包含 TensorExpr 测试的基础头文件
#include <test/cpp/tensorexpr/test_base.h>

// 包含 C10 库的 irange.h，提供迭代器范围的工具
#include <c10/util/irange.h>
// 包含 TensorExpr 测试工具的头文件
#include <test/cpp/tensorexpr/test_utils.h>
// 包含 TensorExpr 中的哈希提供者头文件
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
// 包含 TensorExpr 中的表达式简化器头文件
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
// 包含 TensorExpr 中的循环嵌套头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h>

// 包含数学函数库
#include <cmath>

// 定义命名空间 torch::jit
namespace torch {
namespace jit {

// 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;
// 简化 ExprEval 类型的别名，用于简单表达式求值
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

// 定义单元测试 Simplify.ConstantFoldSimple
TEST(Simplify, ConstantFoldSimple) {
  // 创建浮点数表达式 a 和 b
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  // 计算表达式 f = a + b
  ExprHandle f = (a + b);

  // 使用 IRSimplifier 简化表达式 f
  ExprHandle newF = IRSimplifier::simplify(f);
  // 断言简化后的表达式 newF 是一个 FloatImm 节点
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  // 断言 FloatImm 的值为 5
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), 5);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言评估得到的值为 5.0f
  ASSERT_EQ(eval.value<float>(), 5.f);
}

// 定义单元测试 Simplify.ConstantFoldTwoLayer
TEST(Simplify, ConstantFoldTwoLayer) {
  // 创建浮点数表达式 a, b, c, d
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  // 计算表达式 f = (a + b) - (c + d)
  ExprHandle f = (a + b) - (c + d);

  // 使用 IRSimplifier 简化表达式 f
  ExprHandle newF = IRSimplifier::simplify(f);
  // 断言简化后的表达式 newF 是一个 FloatImm 节点
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  // 断言 FloatImm 的值为 -4
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), -4);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言评估得到的值为 -4.0f
  ASSERT_EQ(eval.value<float>(), -4.f);
}

// 定义单元测试 Simplify.ConstantFoldShifts
TEST(Simplify, ConstantFoldShifts) {
  // 创建整数表达式 a, b, c
  ExprHandle a(7);
  ExprHandle b(2);
  ExprHandle c(3);
  // 计算表达式 f = ((a << b) << b) >> c
  ExprHandle f = ((a << b) << b) >> c;

  // 使用 IRSimplifier 简化表达式 f
  ExprHandle newF = IRSimplifier::simplify(f);
  // 断言简化后的表达式 newF 是一个 IntImm 节点
  ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
  // 断言 IntImm 的值为 14
  ASSERT_EQ(newF.AsNode<IntImm>()->value(), 14);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言评估得到的值为 7 << (4 - 3) = 14
  ASSERT_EQ(eval.value<int>(), 7 << (4 - 3));
}

// 定义单元测试 Simplify.ConstantFoldBitwise
TEST(Simplify, ConstantFoldBitwise) {
  // 创建整数表达式 a, b, c
  ExprHandle a(59);
  ExprHandle b(22);
  ExprHandle c(101);
  // 计算表达式 f = (a ^ b) & c
  ExprHandle f = (a ^ b) & c;

  // 使用 IRSimplifier 简化表达式 f
  ExprHandle newF = IRSimplifier::simplify(f);
  // 断言简化后的表达式 newF 是一个 IntImm 节点
  ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
  // 断言 IntImm 的值为 37
  ASSERT_EQ(newF.AsNode<IntImm>()->value(), 37);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言评估得到的值为 (59 ^ 22) & 101 = 37
  ASSERT_EQ(eval.value<int>(), (59 ^ 22) & 101);
}

// 定义单元测试 Simplify.ConstantFoldMultiOp
TEST(Simplify, ConstantFoldMultiOp) {
  // 创建浮点数表达式 a, b, c, d, e, f
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle e(6.0f);
  ExprHandle f(7.0f);
  // 计算表达式 fn = ((a / e) - (c + d)) * (f / b)
  ExprHandle fn = ((a / e) - (c + d)) * (f / b);

  // 使用 IRSimplifier 简化表达式 fn
  ExprHandle newF = IRSimplifier::simplify(fn);
  // 断言简化后的表达式 newF 是一个 FloatImm 节点
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 创建 SimpleIRExprEval 对象，用于评估原始表达式 fn
  SimpleIRExprEval ref(fn);

  // 断言评估得到的值与原始表达式的值相等
  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

// 定义单元测试 Simplify.ConstantFoldMinMax
TEST(Simplify, ConstantFoldMinMax) {
  // 创建浮点数表达式 a, b, c
  ExprHandle a(12.0f);
  ExprHandle b(15.0f);
  ExprHandle c(17.0f);

  // 创建 Min 和 Max 表达式
  ExprHandle minHandle = Min::make(b, c, true);
  ExprHandle fn = Max::make(a, minHandle, false);

  // 断言 fn 的数据类型为 Float
  ASSERT_EQ(fn.dtype().scalar_type(), ScalarType::Float);

  // 使用 IRSimplifier 简化表达式 fn
  ExprHandle newF = IRSimplifier::simplify(fn);
  // 断言简化后的表达式 newF 是一个 FloatImm 节点
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);

  // 创建 SimpleIRExprEval 对象，用于评估表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言评估得到的值为 15.0f
  ASSERT_EQ(eval.value<float>(), 15.f);
}

// 结束命名空间 torch::jit
} // namespace jit
} // namespace torch
TEST(Simplify, ConstantFoldIntrinsics) {
  // 创建常数表达式节点 a = 2.0f
  ExprHandle a(2.0f);
  // 创建常数表达式节点 b = 3.0f
  ExprHandle b(3.0f);
  // 创建常数表达式节点 c = 4.0f
  ExprHandle c(4.0f);
  // 创建幂运算的表达式节点 powHandle = pow(a, b)
  ExprHandle powHandle = Intrinsics::make(kPow, a, b);
  // 创建正弦函数的表达式节点 sinHandle = sin(powHandle)
  ExprHandle sinHandle = Intrinsics::make(kSin, powHandle);
  // 创建浮点取模函数的表达式节点 modHandle = fmod(c, sinHandle)
  ExprHandle modHandle = Intrinsics::make(kFmod, c, sinHandle);
  // 创建对数函数的表达式节点 logHandle = log10(modHandle)
  ExprHandle logHandle = Intrinsics::make(kLog10, modHandle);
  // 创建四舍五入函数的表达式节点 rndHandle = round(logHandle)
  ExprHandle rndHandle = Intrinsics::make(kRound, logHandle);
  // 创建绝对值函数的表达式节点 fn = abs(rndHandle)
  ExprHandle fn = Intrinsics::make(kAbs, rndHandle);

  // 对表达式 fn 进行简化
  ExprHandle newF = IRSimplifier::simplify(fn);
  // 断言简化后的结果是一个 FloatImm 类型的节点
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  // 断言简化后的 FloatImm 节点的值为 1
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), 1);

  // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
  SimpleIRExprEval eval(newF);
  // 创建 SimpleIRExprEval 对象用于评估原始表达式 fn
  SimpleIRExprEval ref(fn);

  // 断言简化后的表达式和原始表达式的值相等
  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

TEST(Simplify, ConstantFoldCastToBool) {
  // 创建将整数常量 0 强制转换为布尔类型的表达式节点 f
  ExprHandle f = Cast::make(kBool, IntImm::make(0));
  // 对表达式 f 进行简化
  ExprHandle newF = IRSimplifier::simplify(f);
  // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
  SimpleIRExprEval eval(newF);
  // 断言简化后的表达式的布尔值为 false
  ASSERT_EQ(eval.value<bool>(), false);
}

TEST(Simplify, ConstantFoldWithVar) {
  {
    // 创建整型变量 x
    VarHandle x("x", kInt);
    // 创建带有变量 x 的表达式 body = x * (2 + 4)
    ExprHandle body = x * (ExprHandle(2) + ExprHandle(4));

    // 对表达式 body 进行简化
    ExprHandle newF = IRSimplifier::simplify(body);
    // 断言简化后的根节点是一个乘法节点 Mul
    MulPtr root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    // 断言乘法节点的左子节点是一个整数常量节点 IntImm
    ASSERT_NE(to<IntImm>(root->lhs()), nullptr);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 绑定变量 x 的值为 3，断言评估结果等于 3 * (2 + 4)
    eval.bindVar(x, ExprHandle(3));
    ASSERT_EQ(eval.value<int>(), 3 * (2 + 4));
  }

  {
    // 创建浮点型变量 x
    VarHandle x("x", kFloat);
    // 创建带有变量 x 的表达式 body = x * (2.f + 4.f)
    ExprHandle body = x * (ExprHandle(2.f) + ExprHandle(4.f));

    // 对表达式 body 进行简化
    ExprHandle newF = IRSimplifier::simplify(body);
    // 断言简化后的根节点是一个乘法节点 Mul
    MulPtr root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    // 断言乘法节点的右子节点是一个浮点常量节点 FloatImm
    ASSERT_NE(to<FloatImm>(root->rhs()), nullptr);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 绑定变量 x 的值为 3.f，断言评估结果等于 3 * (2 + 4)
    eval.bindVar(x, ExprHandle(3.f));
    ASSERT_EQ(eval.value<float>(), 3 * (2 + 4));
  }
}

TEST(Simplify, ConditionalSelectFoldSimple) {
  // 创建常数表达式节点 a = 3.0f
  ExprHandle a(3.0f);
  // 创建常数表达式节点 b = 4.0f
  ExprHandle b(4.0f);
  // 创建常数表达式节点 c = 3.0f
  ExprHandle c(3.0f);
  {
    // 创建比较表达式节点 f = (a > b)
    ExprHandle f = (a > b);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的结果是一个整数常量节点 IntImm
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    // 断言简化后的 IntImm 节点的值为 0
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 断言评估结果等于 0
    ASSERT_EQ(eval.value<int>(), 0);
  }
  {
    // 创建比较表达式节点 f = (a < b)
    ExprHandle f = (a < b);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的结果是一个整数常量节点 IntImm
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    // 断言简化后的 IntImm 节点的值为 1
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 断言评估结果等于 1
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    // 创建比较表达式节点 f = (a == c)
    ExprHandle f = (a == c);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的结果是一个整数常量节点 IntImm
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    // 断言简化后的 IntImm 节点的值为 1
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 断言评估结果等于 1
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    // 创建比较表达式节点 f = (a != c)
    ExprHandle f = (a != c);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的结果是一个整数常量节点 IntImm
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    // 断言简化后的 IntImm 节点的值为 0
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    // 创建 SimpleIRExprEval 对象用于评估简化后的表达式 newF
    SimpleIRExprEval eval(newF);
    // 断言评估结果等于 0
    ASSERT_EQ(eval.value<int>(), 0);
  }
}
TEST(Simplify, ConditionalSelectFoldTwoLayer) {
  // 创建四个表达式句柄，分别表示常量表达式3.0，2.0，2.0和1.0
  ExprHandle a(3.0f);
  ExprHandle b(2.0f);
  ExprHandle c(2.0f);
  ExprHandle d(1.0f);
  
  {
    // 构造条件表达式 f，判断 (a + b) 是否小于 (c + d)
    ExprHandle f = (a + b < c + d);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的表达式是 IntImm 类型且其值为 0
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 断言求值结果为 0
    ASSERT_EQ(eval.value<int>(), 0);
  }
  
  {
    // 构造条件表达式 f，判断 (a + b) 是否大于 (c + d)
    ExprHandle f = (a + b > c + d);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的表达式是 IntImm 类型且其值为 1
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 断言求值结果为 1
    ASSERT_EQ(eval.value<int>(), 1);
  }
  
  {
    // 构造条件表达式 f，判断 (a + d) 是否等于 (b + c)
    ExprHandle f = (a + d == b + c);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的表达式是 IntImm 类型且其值为 1
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 断言求值结果为 1
    ASSERT_EQ(eval.value<int>(), 1);
  }
  
  {
    // 构造条件表达式 f，判断 (a + d) 是否不等于 (b + c)
    ExprHandle f = (a + d != b + c);

    // 对表达式 f 进行简化
    ExprHandle newF = IRSimplifier::simplify(f);
    // 断言简化后的表达式是 IntImm 类型且其值为 0
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 断言求值结果为 0
    ASSERT_EQ(eval.value<int>(), 0);
  }
}

TEST(Simplify, ConditionalSelectFoldWithVar) {
  // 创建一个变量表达式句柄 x，类型为 kFloat
  VarHandle x("x", kFloat);
  // 构造一个条件表达式 f，判断 x 是否小于 4.0
  ExprHandle f = x < 4.f;

  // 对表达式 f 进行简化
  ExprHandle newF = IRSimplifier::simplify(f);
  // 断言简化后的表达式不是 IntImm 类型（即未折叠为常量）
  IntImmPtr folded = newF.AsNode<IntImm>();
  ASSERT_EQ(folded, nullptr);

  {
    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 绑定变量 x 的值为 3.0，断言求值结果为 1
    eval.bindVar(x, ExprHandle(3.f));
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    // 使用简化后的表达式进行表达式求值
    SimpleIRExprEval eval(newF);
    // 绑定变量 x 的值为 5.0，断言求值结果为 0
    eval.bindVar(x, ExprHandle(5.f));
    ASSERT_EQ(eval.value<int>(), 0);
  }
}

TEST(Simplify, UnFoldableExpr) {
  // 创建两个变量表达式句柄 x 和 y，类型均为 kFloat
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  // 构造一个复杂的表达式 body，包含乘法和加法
  ExprHandle body = (ExprHandle(3) * x) + (ExprHandle(5) * y);

  // 对表达式 body 进行简化
  ExprHandle newF = IRSimplifier::simplify(body);
  // 断言简化后的根节点是 Add 类型
  AddPtr root = newF.AsNode<Add>();
  ASSERT_NE(root, nullptr);
  // 断言根节点的左右子节点不是 FloatImm 类型（即未完全折叠为常量）
  ASSERT_EQ(to<FloatImm>(root->lhs()), nullptr);
  ASSERT_EQ(to<FloatImm>(root->rhs()), nullptr);

  // 使用简化后的表达式进行表达式求值
  SimpleIRExprEval eval(newF);
  // 绑定变量 x 和 y 的值，断言求值结果为 9 + 5 = 14
  eval.bindVar(x, ExprHandle(3.f));
  eval.bindVar(y, ExprHandle(2.f));
  ASSERT_EQ(eval.value<float>(), 9 + 10);
}

TEST(Simplify, HashSimple) {
  // 创建一个变量表达式句柄 x，类型为 kFloat
  VarHandle x("x", kFloat);
  // 创建两个常量表达式句柄 a 和 b，分别为 2.0 和 3.0
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  // 构造一个复合表达式 f，包含加法和乘法
  ExprHandle f = a + b * x;

  // 创建哈希提供器对象
  HashProvider hasher;

  // 计算变量 x、常量 a 和表达式 f 的哈希值
  auto hash_x = hasher.hash(x.node());
  auto hash_a = hasher.hash(a.node());
  auto hash_f = hasher.hash(f.node());

  // 断言哈希值不为 0，表明哈希计算成功
  ASSERT_NE(hash_x, (size_t)0);
  ASSERT_NE(hash_a, (size_t)0);
  ASSERT_NE(hash_f, (size_t)0);
  // 断言变量 x、常量 a 和表达式 f 的哈希值均不相等
  ASSERT_NE(hash_x, hash_a);
  ASSERT_NE(hash_x, hash_f);
  ASSERT_NE(hash_a, hash_f);
}
TEST(Simplify, HashEquivalence) {
  // 创建名为 x 和 y 的浮点类型变量
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  // 构建表达式 f = x * y + x * y
  ExprHandle f = (x * y) + (x * y);

  // 将表达式 f 转换为 Add 类型的指针 root
  AddPtr root = f.AsNode<Add>();
  // 断言 root 不为空
  ASSERT_NE(root, nullptr);

  // 创建一个 HashProvider 对象 hasher
  HashProvider hasher;
  // 计算表达式 f 和其根节点的哈希值
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // 断言根节点的哈希值不等于其左右子节点的哈希值
  ASSERT_NE(hash_f, hash_l);
  ASSERT_NE(hash_f, hash_r);
  // 但是左右子节点的哈希值相等
  ASSERT_EQ(hash_l, hash_r);

  // 再次验证，即使是分离的表达式也应该相等
  ExprHandle a(2);
  ExprHandle f2 = x + a / y;
  ExprHandle b(2);
  ExprHandle f3 = x + b / y;
  ASSERT_EQ(hasher.hash(f2.node()), hasher.hash(f3.node()));

  // 如果变量不同（即使名称相同），则表达式不等
  VarHandle z("x", kFloat);
  ExprHandle f4 = z + b / y;
  ASSERT_NE(hasher.hash(f2.node()), hasher.hash(f4.node()));

  // 内置函数的合理性检查
  ExprHandle f5 = Intrinsics::make(kSin, x) * Intrinsics::make(kCos, x);
  ASSERT_NE(hasher.hash(f5.node()), (size_t)0);
}

TEST(Simplify, HashEquivalenceRand) {
  // 创建随机数函数的表达式
  ExprHandle f =
      Intrinsics::make(kRand, kFloat) + Intrinsics::make(kRand, kInt);

  // 将表达式 f 转换为 Add 类型的指针 root
  AddPtr root = f.AsNode<Add>();
  // 断言 root 不为空
  ASSERT_NE(root, nullptr);

  // 创建一个 HashProvider 对象 hasher
  HashProvider hasher;
  // 计算表达式 f 和其根节点的哈希值
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // 断言根节点的哈希值不等于其左右子节点的哈希值
  ASSERT_NE(hash_f, hash_l);
  ASSERT_NE(hash_f, hash_r);
  // 并且左右子节点的哈希值也不相等
  ASSERT_NE(hash_l, hash_r);
}

TEST(Simplify, HashEquivalenceAfterFolding) {
  // 创建变量和常量
  VarHandle x("x", kFloat);
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(5.0f);

  // 创建两个表达式 f1 和 f2
  ExprHandle f1 = ((a + b) * x);
  ExprHandle f2 = (c * x);

  // 创建一个 HashProvider 对象 hasher
  HashProvider hasher;
  // 计算表达式 f1 和 f2 的哈希值
  auto hash_l = hasher.hash(f1.node());
  auto hash_r = hasher.hash(f2.node());

  // 断言根节点的哈希值不等于其左右子节点的哈希值
  ASSERT_NE(hash_l, hash_r);

  // 简化表达式 f1 和 f2
  ExprHandle ff1 = IRSimplifier::simplify(f1);
  ExprHandle ff2 = IRSimplifier::simplify(f2);

  // 再次计算简化后的表达式的哈希值
  auto hash_l_n = hasher.hash(ff1.node());
  auto hash_r_n = hasher.hash(ff2.node());
  // 现在左右子节点的哈希值应该相等
  ASSERT_EQ(hash_l_n, hash_r_n);
}

TEST(Simplify, HashDifferenceTypes) {
  // 创建一个 HashProvider 对象 hasher
  HashProvider hasher;
  std::vector<ExprPtr> immediates;

  // 添加不同类型的立即数到 immediates 中
  immediates.push_back(alloc<DoubleImm>(1));
  immediates.push_back(alloc<FloatImm>(1));
  immediates.push_back(alloc<HalfImm>(1));
  // NOLINTNEXTLINE(modernize-use-bool-literals)
  immediates.push_back(alloc<BoolImm>(1));
  immediates.push_back(alloc<CharImm>(1));
  immediates.push_back(alloc<ByteImm>(1));
  immediates.push_back(alloc<ShortImm>(1));
  immediates.push_back(alloc<IntImm>(1));
  immediates.push_back(alloc<LongImm>(1));

  // 不同类型的立即数不相等
  for (unsigned int i = 0; i < immediates.size(); ++i) {
    for (unsigned int j = i + 1; j < immediates.size(); ++j) {
      ASSERT_NE(hasher.hash(immediates[i]), hasher.hash(immediates[j]));
  }
}

// 但是如果强制转换的立即数是相同类型的话：
// 创建一个表达式处理器，表示浮点数 2.0 加上一个字符立即数 1
ExprHandle f1 = ExprHandle(2.f) + CharImm::make(1);
// 创建一个表达式处理器，表示将整数立即数 3 强制转换为浮点数
ExprHandle f2 = Cast::make(kFloat, IntImm::make(3));

// 对表达式进行简化，返回简化后的表达式处理器
ExprHandle ff1 = IRSimplifier::simplify(f1);
ExprHandle ff2 = IRSimplifier::simplify(f2);

// 使用哈希函数对简化后的表达式节点进行哈希，并断言它们的哈希值相等
ASSERT_EQ(hasher.hash(ff1.node()), hasher.hash(ff2.node()));
TEST(Simplify, HashLargeExpression) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle a("A", {N}, kInt);
  // 创建名为 b 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle b("B", {N}, kInt);
  // 创建名为 c 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle c("C", {N}, kInt);
  // 创建名为 i 的变量，数据类型为 kInt
  VarHandle i("i", kInt);
  // 创建 memcpy_stmt 语句，用于比较加载 a 和 b 的值，并存储结果到 c 中
  auto memcpy_stmt = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          {i},
          CompareSelect::make(
              Load::make(a, {i}),
              Load::make(b, {i}),
              CompareSelectOperation::kEQ)));

  // 创建名为 d 的缓冲区，大小为 1，数据类型为 kInt
  BufHandle d("D", {1}, kInt);
  // 创建名为 e 的缓冲区，大小为 1，数据类型为 kInt
  BufHandle e("E", {1}, kInt);
  // 创建 store_ramp_stmt 语句，将 d 中 Ramp(0, 1, 4) 的加载结果存储到 e 中
  auto store_ramp_stmt = Store::make(
      e, {Ramp::make(0, 1, 4)}, Load::make(d, {Ramp::make(0, 1, 4)}));

  // 创建 if_stmt 语句，根据比较 a 和 b 的值选择执行 memcpy_stmt 或 store_ramp_stmt
  auto if_stmt = Cond::make(
      CompareSelect::make(
          Load::make(a, {i}), Load::make(b, {i}), CompareSelectOperation::kGE),
      memcpy_stmt,
      store_ramp_stmt);

  // 创建 HashProvider 对象 hasher
  HashProvider hasher;
  // 计算 if_stmt 的哈希值，并存储为 hash_r
  auto hash_r = hasher.hash(if_stmt);
  // 断言 memcpy_stmt 的哈希值已被缓存
  ASSERT_TRUE(hasher.cachedHash(memcpy_stmt));
  // 计算 memcpy_stmt 的哈希值，并存储为 hash_t
  auto hash_t = hasher.hash(memcpy_stmt);
  // 断言 store_ramp_stmt 的哈希值已被缓存
  ASSERT_TRUE(hasher.cachedHash(store_ramp_stmt));
  // 计算 store_ramp_stmt 的哈希值，并存储为 hash_f
  auto hash_f = hasher.hash(store_ramp_stmt);

  // 断言 hash_r 与 hash_t 不相等
  ASSERT_NE(hash_r, hash_t);
  // 断言 hash_r 与 hash_f 不相等
  ASSERT_NE(hash_r, hash_f);
  // 断言 hash_t 与 hash_f 不相等
  ASSERT_NE(hash_t, hash_f);
}

TEST(Simplify, HashForLoopOptions) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle a("A", {N}, kInt);
  // 创建名为 b 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle b("B", {N}, kInt);
  // 创建名为 c 的缓冲区，大小为 N，数据类型为 kInt
  BufHandle c("C", {N}, kInt);
  // 创建名为 i 的变量，数据类型为 kInt
  VarHandle i("i", kInt);
  // 创建 for_stmt 循环语句，用于比较加载 a 和 b 的值，并存储结果到 c 中
  auto for_stmt = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          {i},
          CompareSelect::make(
              Load::make(a, {i}),
              Load::make(b, {i}),
              CompareSelectOperation::kEQ)));

  // 创建 HashProvider 对象 hasher
  HashProvider hasher;
  // 计算 for_stmt 的哈希值，并存储为 hash_before
  auto hash_before = hasher.hash(for_stmt);
  // 清空哈希缓存
  hasher.clearCache();

  // 设置 for_stmt 的 GPU 块索引为 LoopOptions::IDX_X
  for_stmt->set_gpu_block_index(LoopOptions::IDX_X);
  // 计算修改后的 for_stmt 的哈希值，并存储为 hash_block_idx
  auto hash_block_idx = hasher.hash(for_stmt);
  // 清空哈希缓存
  hasher.clearCache();

  // 断言 hash_before 与 hash_block_idx 不相等
  ASSERT_NE(hash_before, hash_block_idx);

  // 重置 for_stmt 的 GPU 块索引为 LoopOptions::IDX_UNSET
  for_stmt->set_gpu_block_index(LoopOptions::IDX_UNSET);
  // 计算修改后的 for_stmt 的哈希值，并存储为 hash_reset
  auto hash_reset = hasher.hash(for_stmt);
  // 清空哈希缓存
  hasher.clearCache();

  // 断言 hash_before 与 hash_reset 相等
  ASSERT_EQ(hash_before, hash_reset);

  // 设置 for_stmt 的 GPU 线程索引为 LoopOptions::IDX_X
  for_stmt->set_gpu_thread_index(LoopOptions::IDX_X);
  // 计算修改后的 for_stmt 的哈希值，并存储为 hash_thread_idx
  auto hash_thread_idx = hasher.hash(for_stmt);

  // 断言 hash_before 与 hash_thread_idx 不相等
  ASSERT_NE(hash_before, hash_thread_idx);
  // 断言 hash_block_idx 与 hash_thread_idx 不相等
  ASSERT_NE(hash_block_idx, hash_thread_idx);
}

/// (2 + x) + 4 => x + 6
TEST(Simplify, SimplifyAdd) {
  // 创建整型变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  // 创建整型变量 m
  VarHandle m("m", kInt);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  // 创建整型变量 n
  VarHandle n("n", kInt);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  // 创建整型变量 n_1
  VarHandle n_1("n_1", kInt);
  
  // 创建表达式体：(2 + x) + 4
  ExprHandle body = (ExprHandle(2) + x) + ExprHandle(4);

  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式视为 Add 类型节点
  AddPtr root = simplified.AsNode<Add>();
  ASSERT_NE(root, nullptr);

  // 获取左操作数，并检查其为变量类型，且名称为 "x"
  VarPtr lhs = to<Var>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->name_hint(), "x");

  // 获取右操作数，并检查其为整数常量类型，其值为 6
  IntImmPtr rhs = to<IntImm>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->value(), 6.f);
}

/// (2 - x) - 4 => -2 - x
TEST(Simplify, SimplifySub) {
  // 创建整型变量 x
  VarHandle x("x", kInt);

  // 创建表达式体：(2 - x) - 4
  ExprHandle body = (ExprHandle(2) - x) - ExprHandle(4);

  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式视为 Sub 类型节点
  SubPtr root = simplified.AsNode<Sub>();
  ASSERT_NE(root, nullptr);

  // 获取左操作数，并检查其为整数常量类型，其值为 -2
  IntImmPtr lhs = to<IntImm>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), -2.f);

  // 获取右操作数，并检查其为变量类型，且名称为 "x"
  VarPtr rhs = to<Var>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// 2 * (1 - x) - 4 => 2 * (-3 - x)
TEST(Simplify, SimplifyMultiLayer) {
  // 创建整型变量 x
  VarHandle x("x", kInt);

  // 创建表达式体：2 * ((1 - x) - 4)
  ExprHandle body = ExprHandle(2) * ((ExprHandle(1) - x) - ExprHandle(4));

  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 检查简化后的表达式结构
  IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
  IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
  IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
  IS_IMM_WITH_VAL(Int, sub->lhs(), -3);
  IS_VAR_WITH_NAME(sub->rhs(), "x");
}

/// 2 * (3 * x) - (x * 4) => 2 * x
TEST(Simplify, SimplifyMultiTerm) {
  // 创建整型变量 x
  VarHandle x("x", kInt);

  // 创建表达式体：2 * (3 * x) - (x * 4)
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式视为 Mul 类型节点
  MulPtr root = simplified.AsNode<Mul>();
  ASSERT_NE(root, nullptr);

  // 获取左操作数，并检查其为整数常量类型，其值为 2
  IntImmPtr lhs = to<IntImm>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), 2);

  // 获取右操作数，并检查其为变量类型，且名称为 "x"
  VarPtr rhs = to<Var>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// 2 * (3 * (long)x) - (x * 4) => 2 * x
TEST(Simplify, SimplifyCasts) {
  // 创建长整型变量 x
  VarHandle x("x", kLong);

  // 创建表达式体：2 * (3 * (long)x) - (x * 4)
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式视为 Mul 类型节点
  MulPtr root = simplified.AsNode<Mul>();
  ASSERT_NE(root, nullptr);

  // 获取左操作数，并检查其为长整数常量类型，其值为 2
  LongImmPtr lhs = to<LongImm>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), 2);

  // 获取右操作数，并检查其为变量类型，且名称为 "x"
  VarPtr rhs = to<Var>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// (x + 0) * 1 => x
// 测试用例：SimplifyEliminatesNoOps
TEST(Simplify, SimplifyEliminatesNoOps) {
  // 创建一个名为"x"的整型变量
  VarHandle x("x", kInt);
  // 构造一个表达式，其结果仍为"x"
  ExprHandle body = (x + ExprHandle(0)) * 1;

  // 调用 IRSimplifier 的 simplify 函数简化表达式
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 将简化后的表达式转换为 Var 类型指针
  VarPtr root = simplified.AsNode<Var>();
  // 断言根节点不为空
  ASSERT_NE(root, nullptr);
  // 断言根节点的名称提示为 "x"
  ASSERT_EQ(root->name_hint(), "x");
}

// 测试用例：SimplifyMultiVar
TEST(Simplify, SimplifyMultiVar) {
  // 创建名为"x"和"y"的整型变量
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 构造一个包含多个变量的乘法加法表达式
  ExprHandle body = x * 24 + y * 34;

  // 调用 IRSimplifier 的 simplify 函数简化表达式
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式转换为 Add 类型指针
  AddPtr root = simplified.AsNode<Add>();
  // 断言根节点不为空
  ASSERT_NE(root, nullptr);

  // 将根节点的左子节点转换为 Mul 类型指针
  MulPtr lhs = to<Mul>(root->lhs());
  // 断言左子节点不为空
  ASSERT_NE(lhs, nullptr);
  // 将左子节点的右操作数转换为 Var 类型指针
  VarPtr varX = to<Var>(lhs->rhs());
  // 断言 varX 不为空，并且名称提示为 "x"
  ASSERT_NE(varX, nullptr);
  ASSERT_EQ(varX->name_hint(), "x");

  // 将根节点的右子节点转换为 Mul 类型指针
  MulPtr rhs = to<Mul>(root->rhs());
  // 断言右子节点不为空
  ASSERT_NE(rhs, nullptr);
  // 将右子节点的右操作数转换为 Var 类型指针
  VarPtr varY = to<Var>(rhs->rhs());
  // 断言 varY 不为空，并且名称提示为 "y"
  ASSERT_NE(varY, nullptr);
  ASSERT_EQ(varY->name_hint(), "y");
}

// 测试用例：SimplifyReorderings（已禁用）
TEST(Simplify, DISABLED_SimplifyReorderings) {
  // 创建名为"x"和"y"的整型变量
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 构造一个加法表达式 x + 2 + y
  ExprHandle body = x + 2 + y;
  // 调用 IRSimplifier 的 simplify 函数简化表达式
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 将简化后的表达式转换为 Add 类型指针
  AddPtr root = simplified.AsNode<Add>();
  // 断言根节点不为空
  ASSERT_NE(root, nullptr);

  // 将 root 的左子节点作为 Add 类型的节点，命名为 rhs
  IS_NODE_WITH_NAME(Add, root->lhs(), rhs);
  // 断言 rhs 的左子节点为 Var 类型，名称提示为 "x"
  IS_VAR_WITH_NAME(rhs->lhs(), "x");
  // 断言 rhs 的右子节点为 Var 类型，名称提示为 "y"
  IS_VAR_WITH_NAME(rhs->rhs(), "y");
  // 断言 root 的右子节点为 Int 类型，数值为 2
  IS_IMM_WITH_VAL(Int, root->rhs(), 2);
}

// 测试用例：SimplifyEliminatesVar
TEST(Simplify, SimplifyEliminatesVar) {
  // 创建名为"x"和"y"的整型变量
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 构造一个包含变量乘法的加法表达式
  ExprHandle body = y + x * ExprHandle(0);

  // 调用 IRSimplifier 的 simplify 函数简化表达式
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 断言简化后的表达式的根节点为 Var 类型，名称提示为 "y"
  IS_VAR_WITH_NAME(simplified.node(), "y");
}

// 测试用例：SimplifyAdds
TEST(Simplify, SimplifyAdds) {
  // 创建名为"x"和"y"的整型变量
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // 构造一个加法表达式 (x + y) + (x + y)
    ExprHandle body = (x + y) + (x + y);
    // 调用 IRSimplifier 的 simplify 函数简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式的根节点为 Mul 类型
    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    // 断言根节点的左操作数为 Int 类型，数值为 2
    IS_IMM_WITH_VAL(Int, root->lhs(), 2);
    // 将根节点的右操作数作为 Add 类型的节点，命名为 add
    IS_NODE_WITH_NAME(Add, root->rhs(), add);
    // 断言 add 的左操作数为 Var 类型，名称提示为 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");
    // 断言 add 的右操作数为 Var 类型，名称提示为 "y"
    IS_VAR_WITH_NAME(add->rhs(), "y");
  }

  {
    // 构造一个乘法加法表达式 (x * y) + (x * y)
    ExprHandle body = (x * y) + (x * y);
    // 调用 IRSimplifier 的 simplify 函数简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式的根节点为 Mul 类型
    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    // 断言根节点的左操作数为 Int 类型，数值为 2
    IS_IMM_WITH_VAL(Int, root->lhs(), 2);
    // 将根节点的右操作数作为 Mul 类型的节点，命名为 mul
    IS_NODE_WITH_NAME(Mul, root->rhs(), mul);
    // 断言 mul 的左操作数为 Var 类型，名称提示为 "x"
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    // 断言 mul 的右操作数为 Var 类型，名称提示为 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 构造一个减法加法表达式 (x - y) + (x - y)
    ExprHandle body = (x - y) + (x - y);
    // 调用 IRSimplifier 的 simplify 函数简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 将简化后的表达式的根节点作为 Mul 类型的节点，命名为 mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言 mul 的左操作数为 Int 类型，数值为 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    // 将 mul 的右操作数作为 Sub 类型的节点，命名为 rhs
    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);
    // 断言 rhs 的左操作数为 Var 类型，名称提示为 "x"
    IS_VAR_WITH_NAME(rhs->lhs(), "x");
    // 断言 rhs 的右操作数为 Var 类型，名称提示为 "y"
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // 构造一个连续加法表达式 (x + x + x + x)
    ExprHandle body = (x + x + x + x);
    // 调用 IRSimplifier 的 simplify 函数简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 将简化后的表达式的根节点作为 Mul 类型的节点，命名为 root
    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    // 断言 root 的左操作数为 Int
    // 检查根节点的左子树是否是整数类型，并且其值为4
    IS_IMM_WITH_VAL(Int, root->lhs(), 4);
    // 检查根节点的右子树是否是变量，并且变量名为"x"
    IS_VAR_WITH_NAME(root->rhs(), "x");
  }

  {
    // 构建表达式 (x + 0)，即对变量 x 加 0 的操作
    ExprHandle body = x + 0;
    // 简化表达式，去除冗余操作
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 检查简化后的节点是否是变量，并且变量名为"x"
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // 构建表达式 (x + 0.f)，即对变量 x 加 0.0 的操作
    ExprHandle body = x + 0.f;
    // 简化表达式，去除冗余操作
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 检查简化后的节点是否是 Cast 类型，并且目标类型是 float
    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    // 断言目标 Cast 类型的标量类型是 Float
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    // 检查 Cast 操作的源值是否是变量"x"
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }
}

TEST(Simplify, SimplifyMuls) {
  VarHandle x("x", kInt);  // 创建一个名为"x"的整数类型变量句柄
  VarHandle y("y", kInt);  // 创建一个名为"y"的整数类型变量句柄

  {
    // (x + y) * (x + y) => (x + y) * (x + y)
    // 我们不尝试简化多项式的乘法，因为结果很少更有效。
    ExprHandle body = (x + y) * (x + y);  // 创建一个表达式，计算 (x + y) * (x + y)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 确保简化后的表达式是一个乘法节点
    IS_NODE_WITH_NAME(Add, mul->lhs(), lhs);  // 确保乘法的左子节点是一个加法节点
    IS_VAR_WITH_NAME(lhs->lhs(), "x");  // 确保加法的左操作数是变量"x"
    IS_VAR_WITH_NAME(lhs->rhs(), "y");  // 确保加法的右操作数是变量"y"
    IS_NODE_WITH_NAME(Add, mul->rhs(), rhs);  // 确保乘法的右子节点是一个加法节点
    IS_VAR_WITH_NAME(rhs->lhs(), "x");  // 确保加法的左操作数是变量"x"
    IS_VAR_WITH_NAME(rhs->rhs(), "y");  // 确保加法的右操作数是变量"y"
  }

  {
    // x * y * x * y => x * x * y * y
    // 这些仅重新排序。
    ExprHandle body = x * y * x * y;  // 创建一个表达式，计算 x * y * x * y
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul1);  // 确保简化后的表达式是一个乘法节点
    IS_NODE_WITH_NAME(Mul, mul1->lhs(), mul2);  // 确保乘法的左操作数也是一个乘法节点
    IS_NODE_WITH_NAME(Mul, mul2->lhs(), mul3);  // 确保乘法的左操作数也是一个乘法节点
    IS_VAR_WITH_NAME(mul1->rhs(), "y");  // 确保乘法的右操作数是变量"y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");  // 确保乘法的右操作数是变量"y"
    IS_VAR_WITH_NAME(mul3->lhs(), "x");  // 确保乘法的左操作数是变量"x"
    IS_VAR_WITH_NAME(mul3->rhs(), "x");  // 确保乘法的右操作数是变量"x"
  }

  {
    // 1 * (x * 1) => x
    // 1的乘法可以直接取消。
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(1));  // 创建一个表达式，计算 1 * (x * 1)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_VAR_WITH_NAME(simplified.node(), "x");  // 确保简化后的表达式是变量"x"
  }

  {
    // 1.f * (x * 1.f) => x
    // 即使是浮点数，乘法1也可以直接取消，但会保留类型信息。
    ExprHandle body = ExprHandle(1.f) * (x * ExprHandle(1.f));  // 创建一个表达式，计算 1.f * (x * 1.f)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);  // 确保简化后的表达式是一个类型转换节点
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);  // 确保转换成浮点数类型
    IS_VAR_WITH_NAME(cast->src_value(), "x");  // 确保转换的源值是变量"x"
  }

  {
    // 1 * (x * 1.f) => x
    // 单个浮点数足以转换表达式的类型。
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(1.f));  // 创建一个表达式，计算 1 * (x * 1.f)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);  // 确保简化后的表达式是一个类型转换节点
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);  // 确保转换成浮点数类型
    IS_VAR_WITH_NAME(cast->src_value(), "x");  // 确保转换的源值是变量"x"
  }

  {
    // 1 * (x * 0) => 0
    // 零可以直接消除。
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(0));  // 创建一个表达式，计算 1 * (x * 0)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_IMM_WITH_VAL(Int, simplified.node(), 0);  // 确保简化后的表达式是一个整数常量节点，值为0
  }

  {
    // 1 * (x * 0) => 0
    // 但对于浮点数来说不行，因为nan * 0 = nan。
    ExprHandle body = ExprHandle(1.f) * (x * ExprHandle(0.f));  // 创建一个表达式，计算 1.f * (x * 0.f)
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 确保简化后的表达式是一个乘法节点
    IS_NODE_WITH_NAME(Cast, mul->lhs(), cast);  // 确保乘法的左操作数是一个类型转换节点
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);  // 确保转换成浮点数类型
    IS_VAR_WITH_NAME(cast->src_value(), "x");  // 确保转换的源值是变量"x"
    IS_IMM_WITH_VAL(Float, mul->rhs(), 0.0);  // 确保乘法的右操作数是浮点数常量节点，值为0.0
  }

  {
    // (x - y) * (x - y) => (x - y) * (x - y)
    // 和加法一样，我们不尝试简化这个。
    ExprHandle body = (x - y) * (x - y);  // 创建一个表达式，计算 (x - y) * (x - y)
    // 对表达式进行简化，返回简化后的表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Mul 类型，并获取其左侧子节点作为 mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    // 确保 mul 的左侧子节点为 Sub 类型，并获取其作为 lhs
    IS_NODE_WITH_NAME(Sub, mul->lhs(), lhs);

    // 确保 lhs 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(lhs->lhs(), "x");

    // 确保 lhs 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(lhs->rhs(), "y");

    // 确保 mul 的右侧子节点为 Sub 类型，并获取其作为 rhs
    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);

    // 确保 rhs 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(rhs->lhs(), "x");

    // 确保 rhs 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // 对表达式进行简化，返回简化后的表达式
    ExprHandle body = (x + y) * (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Mul 类型，并获取其左侧子节点作为 mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    // 确保 mul 的左侧子节点为 Add 类型，并获取其作为 lhs
    IS_NODE_WITH_NAME(Add, mul->lhs(), lhs);

    // 确保 lhs 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(lhs->lhs(), "x");

    // 确保 lhs 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(lhs->rhs(), "y");

    // 确保 mul 的右侧子节点为 Sub 类型，并获取其作为 rhs
    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);

    // 确保 rhs 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(rhs->lhs(), "x");

    // 确保 rhs 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // 将多项式乘以一个项
    // x * (y + 1) => x + x * y
    ExprHandle body = x * (y + ExprHandle(1));
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Add 类型，并获取其左侧子节点作为 add
    IS_NODE_WITH_NAME(Add, simplified.node(), add);

    // 确保 add 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");

    // 确保 add 的右侧子节点为 Mul 类型，并获取其作为 mul
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul);

    // 确保 mul 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(mul->lhs(), "x");

    // 确保 mul 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 将多项式乘以一个项
    // (x * 1) * (y + 1) => x + x * y
    ExprHandle body = (x * ExprHandle(1)) * (y + ExprHandle(1));
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Add 类型，并获取其左侧子节点作为 add
    IS_NODE_WITH_NAME(Add, simplified.node(), add);

    // 确保 add 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");

    // 确保 add 的右侧子节点为 Mul 类型，并获取其作为 mul
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul);

    // 确保 mul 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(mul->lhs(), "x");

    // 确保 mul 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 将多项式乘以一个项
    // (x * 2) * (y + 1) => 2 * (x + x * y)
    ExprHandle body = (x * ExprHandle(2)) * (y + ExprHandle(1));
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Mul 类型，并获取其作为 mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    // 确保 mul 的左侧子节点为整数 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 确保 mul 的右侧子节点为 Add 类型，并获取其作为 add
    IS_NODE_WITH_NAME(Add, mul->rhs(), add);

    // 确保 add 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");

    // 确保 add 的右侧子节点为 Mul 类型，并获取其作为 mul2
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul2);

    // 确保 mul2 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(mul2->lhs(), "x");

    // 确保 mul2 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // 将多项式乘以一个项
    // (x * 2) * (y + 0) => 2 * (x * y)
    ExprHandle body = (x * ExprHandle(2)) * (y + ExprHandle(0));
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式节点为 Mul 类型，并获取其作为 mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    // 确保 mul 的左侧子节点为整数 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 确保 mul 的右侧子节点为 Mul 类型，并获取其作为 mul2
    IS_NODE_WITH_NAME(Mul, mul->rhs(), mul2);

    // 确保 mul2 的左侧子节点为变量 "x"
    IS_VAR_WITH_NAME(mul2->lhs(), "x");

    // 确保 mul2 的右侧子节点为变量 "y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // 将多项式乘以一个项
    {
        // Multiply a term by a polynomial.
        //   - term with identity scalar, poly with identity scalar.
        // (x * 1) * (y + 0) => x * y
        ExprHandle body = (x * ExprHandle(1)) * (y + ExprHandle(0));
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 确保简化后的表达式是乘法节点，且左右操作数分别是 "x" 和 "y"
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        IS_VAR_WITH_NAME(mul->lhs(), "x");
        IS_VAR_WITH_NAME(mul->rhs(), "y");
      }
    
      {
        // Multiply a polynomial by a term.
        //   - term with no scalar, poly with identity scalar.
        // x * (y + 0) => x * y
        ExprHandle body = x * (y + ExprHandle(0));
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 确保简化后的表达式是乘法节点，且左右操作数分别是 "x" 和 "y"
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        IS_VAR_WITH_NAME(mul->lhs(), "x");
        IS_VAR_WITH_NAME(mul->rhs(), "y");
      }
// 定义测试函数 SimplifySubs，用于测试表达式简化功能
TEST(Simplify, SimplifySubs) {
  // 定义整型变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (x + y) - (x + y) => 0
    // 构造表达式 (x + y) - (x + y)
    ExprHandle body = (x + y) - (x + y);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式为整数常量 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x * y) - (x * y) => 0
    // 构造表达式 (x * y) - (x * y)
    ExprHandle body = (x * y) - (x * y);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式为整数常量 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x - y) - (x - y) => 0
    // 构造表达式 (x - y) - (x - y)
    ExprHandle body = (x - y) - (x - y);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式为整数常量 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x + y) - 2 * (x + y) => -1 * x - y
    // 构造表达式 (x + y) - 2 * (x + y)
    ExprHandle body = (x + y) - ExprHandle(2) * (x + y);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为 Sub 节点
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    // 断言 Sub 节点左侧是 Mul 节点
    IS_NODE_WITH_NAME(Mul, sub->lhs(), mul);
    // 断言 Mul 节点左侧是整数常量 -1
    IS_IMM_WITH_VAL(Int, mul->lhs(), -1);
    // 断言 Mul 节点右侧是变量 "x"
    IS_VAR_WITH_NAME(mul->rhs(), "x");
    // 断言 Sub 节点右侧是变量 "y"
    IS_VAR_WITH_NAME(sub->rhs(), "y");
  }

  {
    // (x + y) - y => x
    // 构造表达式 (x + y) - y
    ExprHandle body = (x + y) - y;
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是变量 "x"
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x - 0) => x.
    // 构造表达式 x - 0
    ExprHandle body = x - 0;
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是变量 "x"
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x - 0.f) => x.
    // 浮点数中直接可以简化
    // 构造表达式 x - 0.f
    ExprHandle body = x - ExprHandle(0.f);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是 Cast 节点，目标类型为 Float
    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    // 断言 Cast 节点的源值是变量 "x"
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // (x - (float)(y - y)) => x.
    // 构造表达式 x - Cast::make(kFloat, y - y)
    ExprHandle body = x - Cast::make(kFloat, y - y);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是 Cast 节点，目标类型为 Float
    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    // 断言 Cast 节点的源值是变量 "x"
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // (x - y) - y => x - 2 * y
    // 构造表达式 (x - y) - y
    ExprHandle body = (x - y) - y;
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式是 Sub 节点
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    // 断言 Sub 节点左侧是变量 "x"
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    // 断言 Sub 节点右侧是 Mul 节点
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul);
    // 断言 Mul 节点左侧是整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    // 断言 Mul 节点右侧是变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 2 * x - x => x
    // 构造表达式 2 * x - x
    ExprHandle body = (ExprHandle(2) * x) - x;
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是变量 "x"
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // x - 2 * x = -1 * x
    // 我们没有一元负数，但可以理解为 0 - x
    // 构造表达式 x - (ExprHandle(2) * x)
    ExprHandle body = x - (ExprHandle(2) * x);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是 Mul 节点
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言 Mul 节点的左侧是整数常量 -1
    IS_IMM_WITH_VAL(Int, mul->lhs(), -1);
    // 断言 Mul 节点的右侧是变量 "x"
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // (x + y + 5) * (x - x) => 0
    // 构造表达式 (x + y + 5) * (x - x)
    ExprHandle body = (x + y + 5) * (x - x);
    // 调用 IRSimplifier 的 simplify 方法简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式是整数常量 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }
}
    // 在表达式中消除了一个乘法的一侧，导致整个表达式结果为0。
    ExprHandle body = (x + y + 5) * (x - x);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // 消除了一个不透明的取模操作
    ExprHandle body = (x % y + 2) - (x % y);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量2
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // 消除了一个复杂的不透明取模操作
    ExprHandle body = (x % y + (x * 2 - x - y * 0) - x + 2) - (x % y);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量2
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // 结果为负数的减法操作
    ExprHandle body = x - (x + 1);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量-1
    IS_IMM_WITH_VAL(Int, simplified.node(), -1);
  }

  {
    // 右侧的负数标量导致结果为正数的减法操作
    ExprHandle body = x - (x - 1);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量1
    IS_IMM_WITH_VAL(Int, simplified.node(), 1);
  }

  {
    // 多项式减法操作，需要对右侧的多项式取反
    ExprHandle body = (x * 2) - (x * 2 + 1);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为整数常量-1
    IS_IMM_WITH_VAL(Int, simplified.node(), -1);
  }

  {
    // 结果为项的多项式减法操作
    ExprHandle body = (y * x * 2) - (x * y);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为乘法节点，其中左侧为变量"x"，右侧为变量"y"
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 结果为多项式的多项式减法操作
    ExprHandle body = (x * 2) - (x + 1);
    // 对表达式进行简化处理
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式节点为减法节点，其中左侧为变量"x"乘以整数2，右侧为整数常量1
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_IMM_WITH_VAL(Int, sub->rhs(), 1);
  }
}

TEST(Simplify, SimplifyDiv) {
  // 创建一个名为 x 的整数类型变量
  VarHandle x("x", kInt);

  {
    // 创建一个除法表达式 0 / x
    ExprHandle body = ExprHandle(0) / x;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 验证简化后的表达式是否为整数常量 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // 创建一个除法表达式 x / 1
    ExprHandle body = x / 1;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 验证简化后的表达式是否为变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }
}

TEST(Simplify, SimplifyDivWithLoopContext0) {
  // 待简化的语句:
  // for (int i = 0; i < 100; i++) {
  //  A[i] = i / 100;
  //}
  // 创建整数类型变量 i
  VarHandle i("i", kInt);
  // 创建整数类型缓冲区 A，大小为 100
  BufHandle a_buf("A", {100}, kInt);
  // 创建循环语句，对数组 A 中的元素进行赋值，值为 i / 100
  auto for_stmt = For::make(i, 0, 100, Store::make(a_buf, {i}, (i / 100)));

  // 对循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

  // 创建一个字符串流用于捕获简化后的语句内容
  std::ostringstream oss;
  oss << *(simplified);
  // 验证捕获的字符串是否符合特定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = 0;
      )IR";
  // 使用FileCheck工具验证捕获的字符串是否与预期模式匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext1) {
  // 待简化的语句:
  // for (const auto i : c10::irange(6)) {
  //  A[i] = (i + 24) / 6;
  //}
  // 创建整数类型变量 i
  VarHandle i("i", kInt);
  // 创建整数类型缓冲区 A，大小为 6
  BufHandle a_buf("A", {6}, kInt);
  // 创建循环语句，对数组 A 中的元素进行赋值，值为 (i + 24) / 6
  auto for_stmt = For::make(i, 0, 6, Store::make(a_buf, {i}, (i + 24) / 6));

  // 对循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

  // 创建一个字符串流用于捕获简化后的语句内容
  std::ostringstream oss;
  oss << *(simplified);
  // 验证捕获的字符串是否符合特定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = 4;
      )IR";
  // 使用FileCheck工具验证捕获的字符串是否与预期模式匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext2) {
  // 待简化的语句:
  // for (const auto i : c10::irange(5)) {
  //  A[i] = (i + 25) / 6;
  //}
  // 创建整数类型变量 i
  VarHandle i("i", kInt);
  // 创建整数类型缓冲区 A，大小为 5
  BufHandle a_buf("A", {5}, kInt);
  // 创建循环语句，对数组 A 中的元素进行赋值，值为 (i + 25) / 6
  auto for_stmt = For::make(i, 0, 5, Store::make(a_buf, {i}, (i + 25) / 6));

  // 对循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

  // 创建一个字符串流用于捕获简化后的语句内容
  std::ostringstream oss;
  oss << *(simplified);
  // 验证捕获的字符串是否符合特定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = 4;
      )IR";
  // 使用FileCheck工具验证捕获的字符串是否与预期模式匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext3) {
  // 待简化的语句:
  // for (const auto i : c10::irange(6)) {
  //  A[i] = (i + 24) / (-6);
  //}
  // 创建整数类型变量 i
  VarHandle i("i", kInt);
  // 创建整数类型缓冲区 A，大小为 6
  BufHandle a_buf("A", {6}, kInt);
  // 创建循环语句，对数组 A 中的元素进行赋值，值为 (i + 24) / (-6)
  auto for_stmt = For::make(i, 0, 6, Store::make(a_buf, {i}, (i + 24) / (-6)));

  // 对循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

  // 创建一个字符串流用于捕获简化后的语句内容
  std::ostringstream oss;
  oss << *(simplified);
  // 验证捕获的字符串是否符合特定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NOT:   A[i] = -4;
      )IR";
  // 使用FileCheck工具验证捕获的字符串是否与预期模式匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Simplify, SimplifyDivWithLoopContext4) {
  // 测试简化：
  // for (const auto i : c10::irange(5)) {
  //  A[i] = (i - 5) / 6;
  //}
  
  // 定义变量 i，表示循环索引
  VarHandle i("i", kInt);
  // 定义缓冲区 A，形状为 {5}，数据类型为整型
  BufHandle a_buf("A", {5}, kInt);
  // 创建 for 循环语句，初始化 A[i] = (i + (-5)) / 6 的存储操作
  auto for_stmt = For::make(i, 0, 5, Store::make(a_buf, {i}, (i + (-5)) / 6));

  // 对 for 循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

  // 创建字符串流 oss，用于捕获简化后语句的输出
  std::ostringstream oss;
  oss << *(simplified);
  // 验证模式字符串，用于检查输出结果是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NOT:   A[i] = 0;
      )IR";
  // 运行文件检查工具，验证输出结果是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext5) {
  // 测试简化：
  // for (const auto i : c10::irange(6)) {
  //  for (const auto j : c10::irange(10)) {
  //    A[i, j] = (i + 6*j) / 6;
  //  }
  //}
  
  // 定义变量 i 和 j，表示双重循环的索引
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 定义缓冲区 A，形状为 {6, 10}，数据类型为整型
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环语句，初始化 A[i, j] = (i + j * 6) / 6 的存储操作
  auto for_j = For::make(j, 0, 10, Store::make(a_buf, {i, j}, (i + j * 6) / 6));
  // 创建外部循环语句，嵌套内部循环
  auto for_i = For::make(i, 0, 6, for_j);

  // 对外部循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建字符串流 oss，用于捕获简化后语句的输出
  std::ostringstream oss;
  oss << *(simplified);
  // 验证模式字符串，用于检查输出结果是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NEXT:   A[i, j] = j;
      )IR";
  // 运行文件检查工具，验证输出结果是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext6) {
  // 测试简化：
  // for (const auto i : c10::irange(6)) {
  //  for (int j = -1; j < 9; j++) {
  //    A[i, j+1] = (i + 6*j) / 6;
  //  }
  //}
  
  // 定义变量 i 和 j，表示双重循环的索引
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 定义缓冲区 A，形状为 {6, 10}，数据类型为整型
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环语句，初始化 A[i, j+1] = (i + j * 6) / 6 的存储操作
  auto for_j =
      For::make(j, -1, 9, Store::make(a_buf, {i, j + 1}, (i + j * 6) / 6));
  // 创建外部循环语句，嵌套内部循环
  auto for_i = For::make(i, 0, 6, for_j);

  // 对外部循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建字符串流 oss，用于捕获简化后语句的输出
  std::ostringstream oss;
  oss << *(simplified);
  // 验证模式字符串，用于检查输出结果是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NOT:   A[i, j] = j;
      )IR";
  // 运行文件检查工具，验证输出结果是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyDivWithLoopContext7) {
  // 测试简化：
  // for (const auto i : c10::irange(6)) {
  //  for (const auto j : c10::irange(10)) {
  //    A[i, j] = (i + 6*j) / (-6);
  //  }
  //}
  
  // 定义变量 i 和 j，表示双重循环的索引
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 定义缓冲区 A，形状为 {6, 10}，数据类型为整型
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环语句，初始化 A[i, j] = (i + j * 6) / (-6) 的存储操作
  auto for_j =
      For::make(j, 0, 10, Store::make(a_buf, {i, j}, (i + j * 6) / (-6)));
  // 创建外部循环语句，嵌套内部循环
  auto for_i = For::make(i, 0, 6, for_j);

  // 对外部循环语句进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建字符串流 oss，用于捕获简化后语句的输出
  std::ostringstream oss;
  oss << *(simplified);
  // 验证模式字符串，用于检查输出结果是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NOT:   A[i, j] = -j;
      )IR";
  // 运行文件检查工具，验证输出结果是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Simplify, SimplifyModWithLoopContext0) {
    // Stmt to simplify:
    // for (const auto i : c10::irange(100)) {
    //  A[i] = i % 100;
    //}

    // 声明变量 i，表示循环索引
    VarHandle i("i", kInt);
    // 声明缓冲区 A，大小为 100，元素类型为整数
    BufHandle a_buf("A", {100}, kInt);
    // 构建 for 循环语句，遍历 i 从 0 到 99，将 i 对 100 取模后存储到 A[i] 中
    auto for_stmt = For::make(i, 0, 100, Store::make(a_buf, {i}, (i % 100)));

    // 对 for 循环语句进行简化
    const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

    // 创建输出流对象，用于将简化后的语句输出为字符串
    std::ostringstream oss;
    oss << *(simplified);

    // 验证输出的字符串是否匹配预期的模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = i;
        )IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext1) {
    // Stmt to simplify:
    // for (const auto i : c10::irange(6)) {
    //  A[i] = (i + 24) % 6;
    //}

    // 声明变量 i，表示循环索引
    VarHandle i("i", kInt);
    // 声明缓冲区 A，大小为 6，元素类型为整数
    BufHandle a_buf("A", {6}, kInt);
    // 构建 for 循环语句，遍历 i 从 0 到 5，将 (i + 24) 对 6 取模后存储到 A[i] 中
    auto for_stmt = For::make(i, 0, 6, Store::make(a_buf, {i}, (i + 24) % 6));

    // 对 for 循环语句进行简化
    const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

    // 创建输出流对象，用于将简化后的语句输出为字符串
    std::ostringstream oss;
    oss << *(simplified);

    // 验证输出的字符串是否匹配预期的模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = i;
        )IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext2) {
    // Stmt to simplify:
    // for (const auto i : c10::irange(5)) {
    //  A[i] = (i + 25) % 6;
    //}

    // 声明变量 i，表示循环索引
    VarHandle i("i", kInt);
    // 声明缓冲区 A，大小为 5，元素类型为整数
    BufHandle a_buf("A", {5}, kInt);
    // 构建 for 循环语句，遍历 i 从 0 到 4，将 (i + 25) 对 6 取模后存储到 A[i] 中
    auto for_stmt = For::make(i, 0, 5, Store::make(a_buf, {i}, (i + 25) % 6));

    // 对 for 循环语句进行简化
    const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

    // 创建输出流对象，用于将简化后的语句输出为字符串
    std::ostringstream oss;
    oss << *(simplified);

    // 验证输出的字符串是否匹配预期的模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int i
# CHECK-NEXT:   A[i] = i + 1;
        )IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext3) {
    // Stmt to simplify:
    // for (const auto i : c10::irange(6)) {
    //  A[i] = (i + 24) % (-6);
    //}

    // 声明变量 i，表示循环索引
    VarHandle i("i", kInt);
    // 声明缓冲区 A，大小为 6，元素类型为整数
    BufHandle a_buf("A", {6}, kInt);
    // 构建 for 循环语句，遍历 i 从 0 到 5，将 (i + 24) 对 -6 取模后存储到 A[i] 中
    auto for_stmt = For::make(i, 0, 6, Store::make(a_buf, {i}, (i + 24) % (-6)));

    // 对 for 循环语句进行简化
    const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

    // 创建输出流对象，用于将简化后的语句输出为字符串
    std::ostringstream oss;
    oss << *(simplified);

    // 验证输出的字符串是否匹配预期的模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int i
# CHECK-NOT:   A[i] = i;
        )IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext4) {
    // Stmt to simplify:
    // for (const auto i : c10::irange(5)) {
    //  A[i] = (i - 5) % 6;
    //}

    // 声明变量 i，表示循环索引
    VarHandle i("i", kInt);
    // 声明缓冲区 A，大小为 5，元素类型为整数
    BufHandle a_buf("A", {5}, kInt);
    // 构建 for 循环语句，遍历 i 从 0 到 4，将 (i - 5) 对 6 取模后存储到 A[i] 中
    auto for_stmt = For::make(i, 0, 5, Store::make(a_buf, {i}, (i + (-5)) % 6));

    // 对 for 循环语句进行简化
    const StmtPtr simplified = IRSimplifier::simplify(for_stmt);

    // 创建输出流对象，用于将简化后的语句输出为字符串
    std::ostringstream oss;
    oss << *(simplified);

    // 验证输出的字符串是否匹配预期的模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int i
# CHECK-NOT:   A[i] = i - 5;
        )IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Simplify, SimplifyModWithLoopContext5) {
  // Stmt to simplify:
  // for (const auto i : c10::irange(6)) {
  //  for (const auto j : c10::irange(10)) {
  //    A[i, j] = (i + 6*j) % 6;
  //  }
  //}
  
  // 声明变量 i 和 j 作为循环变量
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建一个表示数组 A 的缓冲区，并指定其形状为 {6, 10}，元素类型为整数
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环体，对数组 A 的元素进行赋值操作，计算 (i + 6*j) % 6 的结果并存储
  auto for_j = For::make(j, 0, 10, Store::make(a_buf, {i, j}, (i + j * 6) % 6));
  // 创建外部循环体，对变量 i 进行循环，内嵌上述的内部循环体
  auto for_i = For::make(i, 0, 6, for_j);

  // 使用 IRSimplifier 对整体循环结构进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建一个输出流对象，用于将简化后的循环结构转换成字符串
  std::ostringstream oss;
  oss << *(simplified);
  // 验证输出的字符串是否符合预期的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NEXT:   A[i, j] = i;
      )IR";
  // 使用 FileCheck 工具进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext6) {
  // Stmt to simplify:
  // for (const auto i : c10::irange(6)) {
  //  for (int j = -1; j < 9; j++) {
  //    A[i, j+1] = (i + 6*j) % 6;
  //  }
  //}
  
  // 声明变量 i 和 j 作为循环变量
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建一个表示数组 A 的缓冲区，并指定其形状为 {6, 10}，元素类型为整数
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环体，对数组 A 的元素进行赋值操作，计算 (i + 6*j) % 6 的结果并存储
  auto for_j =
      For::make(j, -1, 9, Store::make(a_buf, {i, j + 1}, (i + j * 6) % 6));
  // 创建外部循环体，对变量 i 进行循环，内嵌上述的内部循环体
  auto for_i = For::make(i, 0, 6, for_j);

  // 使用 IRSimplifier 对整体循环结构进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建一个输出流对象，用于将简化后的循环结构转换成字符串
  std::ostringstream oss;
  oss << *(simplified);
  // 验证输出的字符串是否符合预期的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NOT:   A[i, j] = i;
      )IR";
  // 使用 FileCheck 工具进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyModWithLoopContext7) {
  // Stmt to simplify:
  // for (const auto i : c10::irange(6)) {
  //  for (const auto j : c10::irange(10)) {
  //    A[i, j] = (i + 6*j) % (-6);
  //  }
  //}
  
  // 声明变量 i 和 j 作为循环变量
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建一个表示数组 A 的缓冲区，并指定其形状为 {6, 10}，元素类型为整数
  BufHandle a_buf("A", {6, 10}, kInt);
  // 创建内部循环体，对数组 A 的元素进行赋值操作，计算 (i + 6*j) % (-6) 的结果并存储
  auto for_j =
      For::make(j, 0, 10, Store::make(a_buf, {i, j}, (i + j * 6) % (-6)));
  // 创建外部循环体，对变量 i 进行循环，内嵌上述的内部循环体
  auto for_i = For::make(i, 0, 6, for_j);

  // 使用 IRSimplifier 对整体循环结构进行简化
  const StmtPtr simplified = IRSimplifier::simplify(for_i);

  // 创建一个输出流对象，用于将简化后的循环结构转换成字符串
  std::ostringstream oss;
  oss << *(simplified);
  // 验证输出的字符串是否符合预期的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK:   for (int j
# CHECK-NOT:   A[i, j] = i;
      )IR";
  // 使用 FileCheck 工具进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Simplify, SimplifyMod) {
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  {
    // Constant folding works.
    // 常数折叠，计算 10 % 8 的结果
    ExprHandle body = ExprHandle(10) % 8;
    // 使用 IRSimplifier 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 验证简化后的结果是否为预期的常数值 2
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // x % x => 0
    // 对 x % x 的表达式进行简化
    ExprHandle body = x % x;
    // 使用 IRSimplifier 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 验证简化后的结果是否为预期的常数值 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // 0 % x => 0
    // 对 0 % x 的表达式进行简化
    ExprHandle body = ExprHandle(0) % x;
    // 使用 IRSimplifier 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 验证简化后的结果是否为预期的常数值 0
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // x % 1 => 0
    // 对 x % 1 的表达式进行简化
    ExprHandle body = x % 1;
    {
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 不对未知模数进行修改
      // x % y => x % y
      ExprHandle body = x % y;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Mod 类型，并且其名称为 mod
      IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
      // 确保 mod 的左操作数是名为 "x" 的变量
      IS_VAR_WITH_NAME(mod->lhs(), "x");
      // 确保 mod 的右操作数是名为 "y" 的变量
      IS_VAR_WITH_NAME(mod->rhs(), "y");
    }
    
    {
      // 如果右操作数未知，则不进行修改
      // 4 % x => 4 % x
      ExprHandle body = ExprHandle(4) % x;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Mod 类型，并且其名称为 mod
      IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
      // 确保 mod 的左操作数是整数常量 4
      IS_IMM_WITH_VAL(Int, mod->lhs(), 4);
      // 确保 mod 的右操作数是名为 "x" 的变量
      IS_VAR_WITH_NAME(mod->rhs(), "x");
    }
    
    {
      // 如果左操作数未知，则不进行修改
      // x % 4 => x % 4
      ExprHandle body = x % 4;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Mod 类型，并且其名称为 mod
      IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
      // 确保 mod 的左操作数是名为 "x" 的变量
      IS_VAR_WITH_NAME(mod->lhs(), "x");
      // 确保 mod 的右操作数是整数常量 4
      IS_IMM_WITH_VAL(Int, mod->rhs(), 4);
    }
    
    {
      // 如果左操作数是右操作数的倍数，则结果为 0
      // 2 * x % x => 0
      ExprHandle body = (x * 2) % x;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 即使倍数不是常数也成立
      // x * y % x => 0
      ExprHandle body = (x * y) % x;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 左操作数有多个未知值时也成立
      // x * y * z % x => 0
      ExprHandle body = (x * y * z) % x;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 如果分母是复合的，则成立
      // x * y * z % y * z => 0
      ExprHandle body = (x * y * z) % (y * z);
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 对于倍数为标量的基本检查
      // 12 * x % 4 => 0
      ExprHandle body = (x * 12) % 4;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
    
    {
      // 如果较小的标量在左操作数，则不成立
      // 4 * x % 12 => 4 * x % 12
      ExprHandle body = (x * 4) % 12;
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Mod 类型，并且其名称为 mod
      IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
      // 确保 mod 的左操作数是乘法表达式，其左操作数为整数常量 4
      IS_NODE_WITH_NAME(Mul, mod->lhs(), mul);
      // 确保乘法表达式的左操作数是整数常量 4
      IS_IMM_WITH_VAL(Int, mul->lhs(), 4);
      // 确保乘法表达式的右操作数是名为 "x" 的变量
      IS_VAR_WITH_NAME(mul->rhs(), "x");
      // 确保 mod 的右操作数是整数常量 12
      IS_IMM_WITH_VAL(Int, mod->rhs(), 12);
    }
    
    {
      // 同时包含标量和符号乘法表达式的情况
      // (6 * x * y) % (3 * x * y) => 0
      ExprHandle body = (ExprHandle(6) * x * y) % (x * y * 3);
      // 对表达式进行简化处理，得到简化后的表达式
      ExprHandle simplified = IRSimplifier::simplify(body);
      // 确保简化后的节点类型是 Int 类型的常量，并且其值为 0
      IS_IMM_WITH_VAL(Int, simplified.node(), 0);
    }
// Test that mixing ops together simplifies as expected.
TEST(Simplify, SimplifyMultiOp) {
  VarHandle x("x", kInt); // 创建名为"x"的整型变量
  VarHandle y("y", kInt); // 创建名为"y"的整型变量

  {
    // (x * y) + (x - y) => (x + x * y) - y
    ExprHandle body = (x * y) + (x - y); // 定义表达式 body = (x * y) + (x - y)
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub); // 确保简化后的表达式为 Sub 节点
    IS_NODE_WITH_NAME(Add, sub->lhs(), add); // 确保 sub 节点的左侧是 Add 节点
    IS_VAR_WITH_NAME(add->lhs(), "x"); // 确保 Add 节点的左操作数是变量"x"
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul); // 确保 Add 节点的右操作数是 Mul 节点
    IS_VAR_WITH_NAME(mul->lhs(), "x"); // 确保 Mul 节点的左操作数是变量"x"
    IS_VAR_WITH_NAME(mul->rhs(), "y"); // 确保 Mul 节点的右操作数是变量"y"
    IS_VAR_WITH_NAME(sub->rhs(), "y"); // 确保 sub 节点的右操作数是变量"y"
  }

  {
    // (x + y) - x * y => (x + y) - x * y
    ExprHandle body = (x + y) - x * y; // 定义表达式 body = (x + y) - x * y
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub); // 确保简化后的表达式为 Sub 节点
    IS_NODE_WITH_NAME(Add, sub->lhs(), add); // 确保 sub 节点的左侧是 Add 节点
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul); // 确保 sub 节点的右侧是 Mul 节点
    IS_VAR_WITH_NAME(add->lhs(), "x"); // 确保 Add 节点的左操作数是变量"x"
    IS_VAR_WITH_NAME(add->rhs(), "y"); // 确保 Add 节点的右操作数是变量"y"
    IS_VAR_WITH_NAME(mul->lhs(), "x"); // 确保 Mul 节点的左操作数是变量"x"
    IS_VAR_WITH_NAME(mul->rhs(), "y"); // 确保 Mul 节点的右操作数是变量"y"
  }

  {
    // (x - y) - (x + y) => -2 * y
    ExprHandle body = (x - y) - (x + y); // 定义表达式 body = (x - y) - (x + y)
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul); // 确保简化后的表达式为 Mul 节点
    IS_IMM_WITH_VAL(Int, mul->lhs(), -2); // 确保 Mul 节点的左操作数是整数常量 -2
    IS_VAR_WITH_NAME(mul->rhs(), "y"); // 确保 Mul 节点的右操作数是变量"y"
  }

  {
    // (x - 0) + (x * 1) - (x + 0) => x
    ExprHandle body = (x - 0) + (x * 1) - (x + 0); // 定义表达式 body = (x - 0) + (x * 1) - (x + 0)
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化

    IS_VAR_WITH_NAME(simplified.node(), "x"); // 确保简化后的表达式是变量"x"
  }

  {
    // (x - 0.f) + (x * 1.f) - (x + 0.f) => float(x) + float(x) - float(x)
    // Even in Float simple terms cancel out, but the variable ones cannot.
    ExprHandle body =
        (x - ExprHandle(0.f)) + (x * ExprHandle(1.f)) - (x + ExprHandle(0.f)); // 定义表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub); // 确保简化后的表达式为 Sub 节点
    IS_NODE_WITH_NAME(Add, sub->lhs(), add); // 确保 sub 节点的左侧是 Add 节点
    IS_NODE_WITH_NAME(Cast, add->lhs(), cast1); // 确保 Add 节点的左操作数是 Cast 节点
    IS_VAR_WITH_NAME(cast1->src_value(), "x"); // 确保 Cast 节点的源值是变量"x"
    IS_NODE_WITH_NAME(Cast, add->rhs(), cast2); // 确保 Add 节点的右操作数是 Cast 节点
    IS_VAR_WITH_NAME(cast2->src_value(), "x"); // 确保 Cast 节点的源值是变量"x"
    IS_NODE_WITH_NAME(Cast, sub->rhs(), cast3); // 确保 sub 节点的右操作数是 Cast 节点
    IS_VAR_WITH_NAME(cast3->src_value(), "x"); // 确保 Cast 节点的源值是变量"x"
  }
}

// Test that chaining many ops together works as expected.
TEST(Simplify, SimplifyManyOps) {
  VarHandle x("x", kInt); // 创建名为"x"的整型变量
  VarHandle y("y", kInt); // 创建名为"y"的整型变量

  {
    // x + y + x + x + y + y + x + y + x = 4 * y + 5 * x
    ExprHandle body = x + y + x + x + y + y + x + y + x; // 定义表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化

    IS_NODE_WITH_NAME(Add, simplified.node(), add); // 确保简化后的表达式为 Add 节点

    IS_NODE_WITH_NAME(Mul, add->lhs(), lhs); // 确保 Add 节点的左操作数是 Mul 节点
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 4); // 确保 Mul 节点的左操作数是整数常量 4
    IS_VAR_WITH_NAME(lhs->rhs(), "y"); // 确保 Mul 节点的右操作数是变量"y"

    IS_NODE_WITH_NAME(Mul, add->rhs(), rhs); // 确保 Add 节点的右操作数是 Mul 节点
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 5); // 确保 Mul 节点的左操作数是整数常量 5
    IS_VAR_WITH_NAME(rhs->rhs(), "x"); // 确保 Mul 节点的右操作数是变量"x"
  }

  {
    // x - y + x + x - y - y + x - y + x = 5 * x - 4 * y
    ExprHandle body = x - y + x + x - y - y + x - y + x; // 定义表达式 body
    // 这行代码以下的部分缺失，无法进行注释
    {
        // 对表达式进行简化，得到简化后的表达式对象
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 检查 simplified 中是否有名为 "add" 的 Sub 节点
        IS_NODE_WITH_NAME(Sub, simplified.node(), add);
    
        // 检查 add 节点的左子节点是否为名为 "lhs" 的 Mul 节点
        IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
        // 检查 lhs 的左子节点是否为整数常量 5
        IS_IMM_WITH_VAL(Int, lhs->lhs(), 5);
        // 检查 lhs 的右子节点是否为名为 "x" 的变量
        IS_VAR_WITH_NAME(lhs->rhs(), "x");
    
        // 检查 add 节点的右子节点是否为名为 "rhs" 的 Mul 节点
        IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
        // 检查 rhs 的左子节点是否为整数常量 4
        IS_IMM_WITH_VAL(Int, rhs->lhs(), 4);
        // 检查 rhs 的右子节点是否为名为 "y" 的变量
        IS_VAR_WITH_NAME(rhs->rhs(), "y");
      }
    
      {
        // 定义表达式 body: x + y + x - x - y - y + x + y + x
        ExprHandle body = x + y + x - x - y - y + x + y + x;
        // 对表达式 body 进行简化，得到简化后的表达式对象
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 检查 simplified 中是否有名为 "mul" 的 Mul 节点
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        // 检查 mul 的左子节点是否为整数常量 3
        IS_IMM_WITH_VAL(Int, mul->lhs(), 3);
        // 检查 mul 的右子节点是否为名为 "x" 的变量
        IS_VAR_WITH_NAME(mul->rhs(), "x");
      }
}

// 定义测试用例 SimplifyFactorization 的测试函数
TEST(Simplify, SimplifyFactorization) {
  // 创建整数变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // 表达式 (2 * x) + (2 * y) => 2 * (x + y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(2) * y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 断言乘法节点右侧为加法节点 Add
    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    // 断言加法节点左侧为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");
    // 断言加法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(add->rhs(), "y");
  }

  {
    // 当标量有公共因子时的因式分解
    // 表达式 (2 * x) + (4 * y) => 2 * (2 * y + x)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(4) * y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 断言乘法节点右侧为加法节点 Add
    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    // 断言加法节点左侧为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");

    // 断言加法节点右侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul2);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    // 断言乘法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // 未有公共因子尝试因式分解
    // 表达式 (2 * x) + (5 * y) =>  (5 * y) + (2 * x)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(5) * y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为加法节点 Add
    IS_NODE_WITH_NAME(Add, simplified.node(), add);

    // 断言加法节点左侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
    // 断言乘法节点右侧为变量 "x"
    IS_VAR_WITH_NAME(lhs->rhs(), "x");

    // 断言加法节点右侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
    // 断言乘法节点左侧为整数常量 5
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 5);
    // 断言乘法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // 合并后的因式分解
    // 表达式 (2 * x) + (4 * y) + (8 * x + 6 * y) => 10 * (x + y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(4) * y) +
        (ExprHandle(8) * x + ExprHandle(6) * y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言乘法节点左侧为整数常量 10
    IS_IMM_WITH_VAL(Int, mul->lhs(), 10);

    // 断言乘法节点右侧为加法节点 Add
    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    // 断言加法节点左侧为变量 "x"
    IS_VAR_WITH_NAME(add->lhs(), "x");
    // 断言加法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(add->rhs(), "y");
  }

  {
    // 具有公共因子但符号不同的因式分解
    // 表达式 (2 * x) + (-4 * y) => 2 * (x - 2 * y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(-4) * y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 断言乘法节点右侧为减法节点 Sub
    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    // 断言减法节点左侧为变量 "x"
    IS_VAR_WITH_NAME(sub->lhs(), "x");

    // 断言减法节点右侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul2);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    // 断言乘法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // 所有数值为负数的因式分解
    // 表达式 (-2 * x) + (-4 * y) => 2 * (-1 * x - 2 * y)
    ExprHandle body = ExprHandle(-2) * x + ExprHandle(-4) * y;
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 断言简化后的表达式为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    // 断言乘法节点右侧为减法节点 Sub
    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    // 断言减法节点左侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, sub->lhs(), mul_neg);
    // 断言乘法节点左侧为整数常量 -1
    IS_IMM_WITH_VAL(Int, mul_neg->lhs(), -1);
    // 断言乘法节点右侧为变量 "x"
    IS_VAR_WITH_NAME(mul_neg->rhs(), "x");

    // 断言减法节点右侧为乘法节点 Mul
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul2);
    // 断言乘法节点左侧为整数常量 2
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    // 断言乘法节点右侧为变量 "y"
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }
}
    // 检查 sub 的左子节点是否是 Mul 类型，并将其赋值给 mul2
    IS_NODE_WITH_NAME(Mul, sub->lhs(), mul2);
    // 检查 mul2 的左子节点是否是 Int 类型，并且其值为 -1
    IS_IMM_WITH_VAL(Int, mul2->lhs(), -1);
    // 检查 mul2 的右子节点是否是变量 "x"
    IS_VAR_WITH_NAME(mul2->rhs(), "x");
    // 检查 sub 的右子节点是否是 Mul 类型，并将其赋值给 mul3
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul3);
    // 检查 mul3 的左子节点是否是 Int 类型，并且其值为 2
    IS_IMM_WITH_VAL(Int, mul3->lhs(), 2);
    // 检查 mul3 的右子节点是否是变量 "y"
    IS_VAR_WITH_NAME(mul3->rhs(), "y");
  }

  {
    // 下面的测试确保在涉及负数时，在因式分解过程中没有无限递归。
    // 声明一系列整数类型变量
    VarHandle a("a", kInt);
    VarHandle b("b", kInt);
    VarHandle c("c", kInt);
    VarHandle d("d", kInt);
    VarHandle e("e", kInt);
    VarHandle f("f", kInt);
    VarHandle g("g", kInt);
    VarHandle h("h", kInt);

    // 定义一个复杂的表达式 body，包含多个变量的乘法和加法运算
    ExprHandle body = a * 1024 + 0 + b * (-1) + c * (-1) + d * 1 + e * 1 +
        f * 32 + g * (-1024) + h * (-32);
    // 对表达式 body 进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式 simplified 是否等于指定的字符串
    checkExprIR(
        simplified,
        "((((((d + e) + 1024 * a) + 32 * f) - b) - c) - 1024 * g) - 32 * h");
  }
// 定义一个测试用例 SimplifyFactorizeUneven，测试表达式简化和因式分解
TEST(Simplify, SimplifyFactorizeUneven) {
  // 定义整数类型的变量 x, y, z
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  // 构造表达式体 body：(4 * x + y + z * 2) + (4 * x + y + z * 4)
  ExprHandle body =
      (ExprHandle(4) * x + y + z * 2) + (ExprHandle(4) * x + y + z * 4);
  // 使用 IRSimplifier 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 确保 simplified 是一个乘法节点 Mul，并且其左侧是整数常量 2
  IS_NODE_WITH_NAME(Mul, simplified.node(), root);
  IS_IMM_WITH_VAL(Int, root->lhs(), 2);
  // 确保 root 的右侧是一个加法节点 Add
  IS_NODE_WITH_NAME(Add, root->rhs(), add1);
  // 确保 add1 的左侧是一个加法节点 Add
  IS_NODE_WITH_NAME(Add, add1->lhs(), add2);

  // 确保 add2 的左侧是变量 y
  IS_VAR_WITH_NAME(add2->lhs(), "y");
  // 确保 add2 的右侧是一个乘法节点 Mul
  IS_NODE_WITH_NAME(Mul, add2->rhs(), zmul);
  // 确保 add1 的右侧是一个乘法节点 Mul
  IS_NODE_WITH_NAME(Mul, add1->rhs(), xmul);

  // 确保 xmul 的左侧是整数常量 4，右侧是变量 x
  IS_IMM_WITH_VAL(Int, xmul->lhs(), 4);
  IS_VAR_WITH_NAME(xmul->rhs(), "x");

  // 确保 zmul 的左侧是整数常量 3，右侧是变量 z
  IS_IMM_WITH_VAL(Int, zmul->lhs(), 3);
  IS_VAR_WITH_NAME(zmul->rhs(), "z");
}

// 定义一个测试用例 SimplifyDeeperTerms，测试更深层次的表达式简化
TEST(Simplify, SimplifyDeeperTerms) {
  // 定义整数类型的变量 x, y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 构造表达式体 body：(x * y) + (2 * x) * (x + y)
  ExprHandle body = (x * y) + (ExprHandle(2) * x) * (x + y);
  // 使用 IRSimplifier 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 确保 simplified 是一个加法节点 Add
  IS_NODE_WITH_NAME(Add, simplified.node(), add);

  // 确保 add 的左侧是一个乘法节点 Mul，其左侧是整数常量 2，右侧是 x * x
  IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
  IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
  IS_NODE_WITH_NAME(Mul, lhs->rhs(), xxTerm);
  IS_VAR_WITH_NAME(xxTerm->lhs(), "x");
  IS_VAR_WITH_NAME(xxTerm->rhs(), "x");

  // 确保 add 的右侧是一个乘法节点 Mul，其左侧是整数常量 3，右侧是 x * y
  IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
  IS_IMM_WITH_VAL(Int, rhs->lhs(), 3);
  IS_NODE_WITH_NAME(Mul, rhs->rhs(), xyTerm);
  IS_VAR_WITH_NAME(xyTerm->lhs(), "x");
  IS_VAR_WITH_NAME(xyTerm->rhs(), "y");
}

// 定义一个测试用例 SimplifyDeeperDifference，测试更深层次的表达式差异
TEST(Simplify, SimplifyDeeperDifference) {
  // 定义整数类型的变量 n, n_1, m
  VarHandle n("n", kInt);
  VarHandle n_1("n_1", kInt);
  VarHandle m("m", kInt);
  // 构造表达式体 body：(m * (1 * n_1) + (n + 1)) - (m * (1 * n_1) + n)
  ExprHandle body =
      (m * (ExprHandle(1) * n_1) + (n + 1)) - (m * (ExprHandle(1) * n_1) + n);
  // 使用 IRSimplifier 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 确保 simplified 是一个整数常量节点，其值为 1
  IS_IMM_WITH_VAL(Int, simplified.node(), 1);
}

// 定义一个测试用例 SimplifyFoldComplexDifference，测试复杂表达式的常量折叠
TEST(Simplify, SimplifyFoldComplexDifference) {
  // 定义整数类型的变量 n, n_1, m
  VarHandle n("n", kInt);
  VarHandle n_1("n_1", kInt);
  VarHandle m("m", kInt);
  // 构造表达式体 body：2 + char((m * (1 * n_1) + (n + 1)) - (m * (1 * n_1) + n))
  ExprHandle body =
      (IntImm::make(2) +
       (Cast::make(
           kChar,
           (m * (ExprHandle(1) * n_1) + (n + 1)) -
               (m * (ExprHandle(1) * n_1) + n))));
  // 使用 IRSimplifier 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 确保 simplified 是一个整数常量节点，其值为 3
  IS_IMM_WITH_VAL(Int, simplified.node(), 3);
}
TEST(Simplify, SimplifyIfComponents) {
  VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
  VarHandle y("y", kInt);  // 创建一个名为"y"的整数变量
  ExprHandle body = IfThenElse::make(
      ((ExprHandle(5) - ExprHandle(4)) * x) > y,  // 如果表达式 (5 - 4) * x > y 成立
      ExprHandle(2) * x - x,  // 则返回表达式 2 * x - x
      ExprHandle(2) * y - y);  // 否则返回表达式 2 * y - y

  ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

  IS_NODE_WITH_NAME(IfThenElse, simplified.node(), ifexpr);  // 确保简化后的表达式是 IfThenElse 类型，并命名为 ifexpr

  IS_NODE_WITH_NAME(CompareSelect, ifexpr->condition(), cmp);  // 确保条件表达式是 CompareSelect 类型，并命名为 cmp
  ASSERT_EQ(cmp->compare_select_op(), kGT);  // 断言条件表达式的比较操作是大于号
  IS_VAR_WITH_NAME(cmp->lhs(), "x");  // 确保比较的左操作数是变量"x"
  IS_VAR_WITH_NAME(cmp->rhs(), "y");  // 确保比较的右操作数是变量"y"

  IS_VAR_WITH_NAME(ifexpr->true_value(), "x");  // 确保条件成立时返回值为变量"x"
  IS_VAR_WITH_NAME(ifexpr->false_value(), "y");  // 确保条件不成立时返回值为变量"y"
}

TEST(Simplify, SimplifyOpaqueTerms) {
  VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
  VarHandle y("y", kInt);  // 创建一个名为"y"的整数变量

  {
    // 2 * x/y * y - x/y * y => x/y * y
    ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);  // 定义一个包含数学运算的表达式
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 确保简化后的表达式是乘法节点，并命名为 mul
    IS_NODE_WITH_NAME(Div, mul->lhs(), div);  // 确保乘法的左操作数是除法节点，并命名为 div
    IS_VAR_WITH_NAME(div->lhs(), "x");  // 确保除法的被除数是变量"x"
    IS_VAR_WITH_NAME(div->rhs(), "y");  // 确保除法的除数是变量"y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");  // 确保乘法的右操作数是变量"y"
  }

  {
    // x%y - (x%y - 1) => 1
    ExprHandle body = (x % y) - ((x % y) - 1);  // 定义一个包含求余运算的表达式
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_IMM_WITH_VAL(Int, simplified.node(), 1);  // 确保简化后的表达式是整数常量 1
  }
}

TEST(Simplify, SimplifySymbolicMinMax) {
  {
    // Minimum with constant difference between terms.
    VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
    ExprHandle body = Min::make(x + 3, x + 7, true);  // 定义一个最小化表达式，选择常量差异项
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Add, simplified.node(), add);  // 确保简化后的表达式是加法节点，并命名为 add
    IS_VAR_WITH_NAME(add->lhs(), "x");  // 确保加法的左操作数是变量"x"
    IS_IMM_WITH_VAL(Int, add->rhs(), 3);  // 确保加法的右操作数是整数常量 3
  }

  {
    // Maximum with constant difference between terms.
    VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
    ExprHandle body = Max::make(x + 3, x + 7, true);  // 定义一个最大化表达式，选择常量差异项
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Add, simplified.node(), add);  // 确保简化后的表达式是加法节点，并命名为 add
    IS_VAR_WITH_NAME(add->lhs(), "x");  // 确保加法的左操作数是变量"x"
    IS_IMM_WITH_VAL(Int, add->rhs(), 7);  // 确保加法的右操作数是整数常量 7
  }

  {
    // Can't simplify multiples because of signedness of variable component.
    // TODO: maybe we could for unsigned types?
    VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
    ExprHandle body = Max::make(x * 3, x * 7, true);  // 定义一个最大化表达式，选择乘法项
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE(Max, simplified.node());  // 确保简化后的表达式是最大化节点
  }
}

TEST(Simplify, SimplifyNestedMax) {
  VarHandle x("x", kInt);  // 创建一个名为"x"的整数变量
  VarHandle y("y", kInt);  // 创建一个名为"y"的整数变量
  VarHandle z("z", kInt);  // 创建一个名为"z"的整数变量

  {
    // Max(x + y, x + y) => x + y
    ExprHandle body = Max::make(x + y, x + y, true);  // 定义一个包含嵌套最大化的表达式
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    IS_BINOP_W_VARS(Add, simplified.node(), add, "x", "y");  // 确保简化后的表达式是加法，并且操作数为变量"x"和"y"
  }

  {
    // Max(x + y, Max(x + y, z)) => Max(x + y, z)
    ExprHandle body = Max::make(x + y, Max::make(x + y, z, true), true);  // 定义一个包含嵌套最大化的表达式
    ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化

    IS_NODE_WITH_NAME(Max, simplified.node(), max);  // 确保简化后的表达式是最大化节点，并命名为 max
    // 调用宏IS_BINOP_W_VARS，检查表达式是否是二元操作符Add，且其左操作数是max->lhs()，右操作数是"x"和"y"
    IS_BINOP_W_VARS(Add, max->lhs(), add, "x", "y");
    // 调用宏IS_VAR_WITH_NAME，检查max->rhs()是否是变量"z"
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // 对表达式 Max(x + y, Max(z, x + y)) 进行简化，得到简化后的表达式body
    ExprHandle body = Max::make(x + y, Max::make(z, x + y, true), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否是Max节点，将结果保存在max中
    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    // 使用宏IS_BINOP_W_VARS检查max->lhs()是否是二元操作符Add，左操作数是"x"和"y"
    IS_BINOP_W_VARS(Add, max->lhs(), add, "x", "y");
    // 使用宏IS_VAR_WITH_NAME检查max->rhs()是否是变量"z"
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // 对表达式 Max(Max(x + y, z), x + y) 进行简化，得到简化后的表达式body
    ExprHandle body = Max::make(Max::make(x + y, z, true), x + y, true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否是Max节点，将结果保存在max中
    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    // 使用宏IS_BINOP_W_VARS检查max->lhs()是否是二元操作符Add，左操作数是"x"和"y"
    IS_BINOP_W_VARS(Add, max->lhs(), add, "x", "y");
    // 使用宏IS_VAR_WITH_NAME检查max->rhs()是否是变量"z"
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // 对表达式 Max(Max(z, x + y), x + y) 进行简化，得到简化后的表达式body
    ExprHandle body = Max::make(Max::make(z, x + y, true), x + y, true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否是Max节点，将结果保存在max中
    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    // 使用宏IS_BINOP_W_VARS检查max->lhs()是否是二元操作符Add，左操作数是"x"和"y"
    IS_BINOP_W_VARS(Add, max->lhs(), add, "x", "y");
    // 使用宏IS_VAR_WITH_NAME检查max->rhs()是否是变量"z"
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // 对表达式 Max(Max(x, y), x) 进行简化，得到简化后的表达式body
    // 这里由于嵌套的Max操作具有不同的propagate_nans属性，不应进行简化
    ExprHandle body = Max::make(Max::make(x, y, true), x, false);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否是Max节点，将结果保存在max中
    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    // 使用宏IS_BINOP_W_VARS检查max->lhs()是否是Max操作符，左操作数是"x"和"y"
    IS_BINOP_W_VARS(Max, max->lhs(), max1, "x", "y");
    // 使用ASSERT_TRUE检查max1->propagate_nans()的值为true
    ASSERT_TRUE(max1->propagate_nans());
    // 使用宏IS_VAR_WITH_NAME检查max->rhs()是否是变量"x"
    IS_VAR_WITH_NAME(max->rhs(), "x");
    // 使用ASSERT_FALSE检查max->propagate_nans()的值为false
    ASSERT_FALSE(max->propagate_nans());
  }

  {
    // 对表达式 Max(Min(x, y), Min(x, z)) 进行简化，得到简化后的表达式body
    ExprHandle body =
        Max::make(Min::make(x, y, true), Min::make(x, z, true), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式simplified是否等于指定的IR字符串
    checkExprIR(simplified, "Min(Max(y, z, 1), x, 1)");
  }

  {
    // 对表达式 Max(Min(x, y), Min(z, x)) 进行简化，得到简化后的表达式body
    ExprHandle body =
        Max::make(Min::make(x, y, true), Min::make(z, x, true), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式simplified是否等于指定的IR字符串
    checkExprIR(simplified, "Min(Max(y, z, 1), x, 1)");
  }

  {
    // 对表达式 Max(Min(y, x), Min(x, z)) 进行简化，得到简化后的表达式body
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(x, z, true), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式simplified是否等于指定的IR字符串
    checkExprIR(simplified, "Min(Max(y, z, 1), x, 1)");
  }

  {
    // 对表达式 Max(Min(y, x), Min(z, x)) 进行简化，得到简化后的表达式body
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(z, x, true), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式simplified是否等于指定的IR字符串
    checkExprIR(simplified, "Min(Max(y, z, 1), x, 1)");
  }

  {
    // 对表达式 Max(Min(y, x), Min(z, x)) 进行简化，得到简化后的表达式body
    // 当模式中的所有操作符propagate_nans属性不同时，不应进行简化
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(z, x, false), true);
    // 使用IRSimplifier对body进行进一步简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否为Max节点，并将结果存储在max中
    IS_NODE_WITH_NAME(Max, simplified.node(), max);

    // 使用宏IS_BINOP_W_VARS检查max->lhs()是否为Min节点，并将结果存储在min1中，变量名分别为"x"和"y"
    IS_BINOP_W_VARS(Min, max->lhs(), min1, "x", "y");

    // 断言min1->propagate_nans()返回true
    ASSERT_TRUE(min1->propagate_nans());

    // 使用宏IS_BINOP_W_VARS检查max->rhs()是否为Min节点，并将结果存储在min2中，变量名分别为"x"和"z"
    IS_BINOP_W_VARS(Min, max->rhs(), min2, "x", "z");

    // 断言min2->propagate_nans()返回false
    ASSERT_FALSE(min2->propagate_nans());

    // 断言max->propagate_nans()返回true
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // 创建表达式body: Max(5, Max(x, 8))，并对其进行简化得到simplified
    ExprHandle body = Max::make(5, Max::make(x, 8, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_BINOP_W_CONST检查simplified.node()是否为Max节点，并将结果存储在max中，常量为8
    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);

    // 断言max->propagate_nans()返回true
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // 创建表达式body: Max(8, Max(x, 5))，并对其进行简化得到simplified
    ExprHandle body = Max::make(8, Max::make(x, 5, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_BINOP_W_CONST检查simplified.node()是否为Max节点，并将结果存储在max中，常量为8
    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);

    // 断言max->propagate_nans()返回true
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // 创建表达式body: Max(Max(x, 8), 5)，并对其进行简化得到simplified
    ExprHandle body = Max::make(Max::make(x, 8, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_BINOP_W_CONST检查simplified.node()是否为Max节点，并将结果存储在max中，常量为8
    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);

    // 断言max->propagate_nans()返回true
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // 创建表达式body: Max(Max(x, 5), 8)，并对其进行简化得到simplified
    ExprHandle body = Max::make(Max::make(x, 5, true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_BINOP_W_CONST检查simplified.node()是否为Max节点，并将结果存储在max中，常量为8
    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);

    // 断言max->propagate_nans()返回true
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // 创建表达式body: Max(5, Max(x, Max(y, Max(z, 8))))，并对其进行简化得到simplified
    ExprHandle body = Max::make(
        5, Max::make(x, Max::make(y, Max::make(z, 8, true), true), true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否为Max节点，并将结果存储在max1中
    IS_NODE_WITH_NAME(Max, simplified.node(), max1);

    // 使用宏IS_NODE_WITH_NAME检查max1->lhs()是否为Max节点，并将结果存储在max2中
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);

    // 使用宏IS_BINOP_W_CONST检查max2->lhs()是否为Max节点，并将结果存储在max3中，常量为8
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);

    // 断言max3->propagate_nans()返回true
    ASSERT_TRUE(max3->propagate_nans());

    // 使用宏IS_VAR_WITH_NAME检查max2->rhs()的变量名是否为"y"
    IS_VAR_WITH_NAME(max2->rhs(), "y");

    // 使用宏IS_VAR_WITH_NAME检查max1->rhs()的变量名是否为"z"
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // 创建表达式body: Max(8, Max(Max(y, Max(z, 5)), x))，并对其进行简化得到simplified
    ExprHandle body = Max::make(
        8, Max::make(Max::make(y, Max::make(z, 5, true), true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否为Max节点，并将结果存储在max1中
    IS_NODE_WITH_NAME(Max, simplified.node(), max1);

    // 使用宏IS_NODE_WITH_NAME检查max1->lhs()是否为Max节点，并将结果存储在max2中
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);

    // 使用宏IS_BINOP_W_CONST检查max2->lhs()是否为Max节点，并将结果存储在max3中，常量为8
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);

    // 断言max3->propagate_nans()返回true
    ASSERT_TRUE(max3->propagate_nans());

    // 使用宏IS_VAR_WITH_NAME检查max2->rhs()的变量名是否为"y"
    IS_VAR_WITH_NAME(max2->rhs(), "y");

    // 使用宏IS_VAR_WITH_NAME检查max1->rhs()的变量名是否为"z"
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // 创建表达式body: Max(5, Max(Max(Max(z, 8), y), x))，并对其进行简化得到simplified
    ExprHandle body = Max::make(
        5, Max::make(Max::make(Max::make(z, 8, true), y, true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 使用宏IS_NODE_WITH_NAME检查simplified.node()是否为Max节点，并将结果存储在max1中
    IS_NODE_WITH_NAME(Max, simplified.node(), max1);

    // 使用宏IS_NODE_WITH_NAME检查max1->lhs()是否为Max节点，并将结果存储在max2中
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);

    // 使用宏IS_BINOP_W_CONST检查max2->lhs()是否为Max节点，并将结果存储在max3中，常量为8
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);

    // 断言max3->propagate_nans()返回true
    ASSERT_TRUE(max3->propagate_nans());

    // 使用宏IS_VAR_WITH_NAME检查max2->rhs()的变量名是否为"y"
    IS_VAR_WITH_NAME(max2->rhs(), "y");

    // 使用宏IS_VAR_WITH_NAME检查max1->rhs()的变量名是否为"z"
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }
    {
        // 构造表达式：Max(Max(x, Max(y, Max(5, z))), 8)
        ExprHandle body = Max::make(
            Max::make(x, Max::make(y, Max::make(5, z, true), true), true), 8, true);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 验证简化后的表达式结构
        IS_NODE_WITH_NAME(Max, simplified.node(), max1); // 确认节点为 Max，并将其命名为 max1
        IS_NODE_WITH_NAME(Max, max1->lhs(), max2);      // 确认 max1 的左子节点为 Max，并将其命名为 max2
        IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8); // 确认 max2 的左子节点为一个带有常量的二元操作符 Max，常量为 8
        ASSERT_TRUE(max3->propagate_nans()); // 断言 max3 支持 NaN 传播
        IS_VAR_WITH_NAME(max2->rhs(), "y"); // 确认 max2 的右子节点为变量 "y"
        IS_VAR_WITH_NAME(max1->rhs(), "z"); // 确认 max1 的右子节点为变量 "z"
    }
    
    {
        // 构造表达式：Max(Max(Max(y, Max(8, z)), x), 5)
        ExprHandle body = Max::make(
            Max::make(Max::make(y, Max::make(z, 8, true), true), x, true), 5, true);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 验证简化后的表达式结构
        IS_NODE_WITH_NAME(Max, simplified.node(), max1); // 确认节点为 Max，并将其命名为 max1
        IS_NODE_WITH_NAME(Max, max1->lhs(), max2);      // 确认 max1 的左子节点为 Max，并将其命名为 max2
        IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8); // 确认 max2 的左子节点为一个带有常量的二元操作符 Max，常量为 8
        ASSERT_TRUE(max3->propagate_nans()); // 断言 max3 支持 NaN 传播
        IS_VAR_WITH_NAME(max2->rhs(), "y"); // 确认 max2 的右子节点为变量 "y"
        IS_VAR_WITH_NAME(max1->rhs(), "z"); // 确认 max1 的右子节点为变量 "z"
    }
    
    {
        // 构造表达式：Max(Max(Max(Max(5, z), y), x), 8)
        ExprHandle body = Max::make(
            Max::make(Max::make(Max::make(z, 5, true), y, true), x, true), 8, true);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 验证简化后的表达式结构
        IS_NODE_WITH_NAME(Max, simplified.node(), max1); // 确认节点为 Max，并将其命名为 max1
        IS_NODE_WITH_NAME(Max, max1->lhs(), max2);      // 确认 max1 的左子节点为 Max，并将其命名为 max2
        IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8); // 确认 max2 的左子节点为一个带有常量的二元操作符 Max，常量为 8
        ASSERT_TRUE(max3->propagate_nans()); // 断言 max3 支持 NaN 传播
        IS_VAR_WITH_NAME(max2->rhs(), "y"); // 确认 max2 的右子节点为变量 "y"
        IS_VAR_WITH_NAME(max1->rhs(), "z"); // 确认 max1 的右子节点为变量 "z"
    }
    
    {
        // 构造表达式：Max(Max(Max(Max(z, 5), y), x), 8)
        // 当所有 Max 操作符的 propagate_nans 不同时，不进行简化
        ExprHandle body = Max::make(
            Max::make(Max::make(Max::make(z, 5, true), y, false), x, true),
            8,
            false);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 检查简化后的表达式是否符合预期
        checkExprIR(simplified, "Max(Max(Max(Max(z, 5, 1), y, 0), x, 1), 8, 0)");
    }
    
    {
        // 构造表达式：Max(8, Max(Max(x, 5), Max(y, z)))
        ExprHandle body = Max::make(
            8, Max::make(Max::make(x, 5, true), Max::make(y, z, true), true), true);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 验证简化后的表达式结构
        IS_NODE_WITH_NAME(Max, simplified.node(), max1); // 确认节点为 Max，并将其命名为 max1
        IS_NODE_WITH_NAME(Max, max1->lhs(), max2);      // 确认 max1 的左子节点为 Max，并将其命名为 max2
        IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8); // 确认 max2 的左子节点为一个带有常量的二元操作符 Max，常量为 8
        ASSERT_TRUE(max3->propagate_nans()); // 断言 max3 支持 NaN 传播
        IS_VAR_WITH_NAME(max2->rhs(), "y"); // 确认 max2 的右子节点为变量 "y"
        IS_VAR_WITH_NAME(max1->rhs(), "z"); // 确认 max1 的右子节点为变量 "z"
    }
    
    {
        // 构造表达式：Max(Max(Max(x, 5), Max(y, z)), 8)
        ExprHandle body = Max::make(
            Max::make(Max::make(x, 5, true), Max::make(y, z, true), true), 8, true);
        // 简化表达式
        ExprHandle simplified = IRSimplifier::simplify(body);
    
        // 验证简化后的表达式结构
        IS_NODE_WITH_NAME(Max, simplified.node(), max1); // 确认节点为 Max，并将其命名为 max1
        IS_NODE_WITH_NAME(Max, max1->lhs(), max2);      // 确认 max1 的左子节点为 Max，并将其命名为 max2
        IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8); // 确认 max2 的左子节点为一个带有常量的二元操作符 Max，常量为 8
        ASSERT_TRUE(max3->propagate_nans()); // 断言 max3 支持 NaN 传播
    }
    # 检查 max2 对象的右手边(rhs)是否为变量，并且变量名应为 "y"
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    # 检查 max1 对象的右手边(rhs)是否为变量，并且变量名应为 "z"
    IS_VAR_WITH_NAME(max1->rhs(), "z");
}

TEST(Simplify, SimplifyNestedMin) {
  VarHandle x("x", kInt);  // 创建一个整数变量 x
  VarHandle y("y", kInt);  // 创建一个整数变量 y
  VarHandle z("z", kInt);  // 创建一个整数变量 z

  {
    // Min(x + y, x + y) => x + y
    // 创建一个表达式，计算 Min(x + y, x + y)，并进行简化
    ExprHandle body = Min::make(x + y, x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    // 确保 simplified 是一个加法操作，其操作数为 x 和 y
    IS_BINOP_W_VARS(Add, simplified.node(), add, "x", "y");
  }

  {
    // Min(x + y, Min(x + y, z)) => Min(x + y, z)
    // 创建一个表达式，计算 Min(x + y, Min(x + y, z))，并进行简化
    ExprHandle body = Min::make(x + y, Min::make(x + y, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保 simplified 是一个 Min 操作，左操作数是 x + y，右操作数是 z
    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "x", "y");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(x + y, Min(z, x + y)) => Min(x + y, z)
    // 创建一个表达式，计算 Min(x + y, Min(z, x + y))，并进行简化
    ExprHandle body = Min::make(x + y, Min::make(z, x + y, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保 simplified 是一个 Min 操作，左操作数是 x + y，右操作数是 z
    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "x", "y");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(x + y, z), x + y) => Min(x + y, z)
    // 创建一个表达式，计算 Min(Min(x + y, z), x + y)，并进行简化
    ExprHandle body = Min::make(Min::make(x + y, z, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保 simplified 是一个 Min 操作，左操作数是 x + y，右操作数是 z
    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "x", "y");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(z, x + y), x + y) => Min(x + y, z)
    // 创建一个表达式，计算 Min(Min(z, x + y), x + y)，并进行简化
    ExprHandle body = Min::make(Min::make(z, x + y, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保 simplified 是一个 Min 操作，左操作数是 x + y，右操作数是 z
    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "x", "y");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(x, y), x) => Min(Min(x, y), x)
    // 嵌套的 Min 操作，propagate_nans 设置为 false，不应该进行简化
    ExprHandle body = Min::make(Min::make(x, y, true), x, false);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保 simplified 是一个 Min 操作，左操作数是 Min(x, y)，右操作数是 x
    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_BINOP_W_VARS(Min, min1->lhs(), min2, "x", "y");
    ASSERT_TRUE(min2->propagate_nans());
    IS_VAR_WITH_NAME(min1->rhs(), "x");
    ASSERT_FALSE(min1->propagate_nans());
  }

  {
    // Min(Max(x, y), Max(x, z)) => Max(Min(y, z), x)
    // 创建一个表达式，计算 Min(Max(x, y), Max(x, z))，并进行简化
    ExprHandle body =
        Min::make(Max::make(x, y, true), Max::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);
    checkExprIR(simplified, "Max(Min(y, z, 1), x, 1)");
  }

  {
    // Min(Max(x, y), Max(z, x)) => Max(Min(y, z), x)
    // 创建一个表达式，计算 Min(Max(x, y), Max(z, x))，并进行简化
    ExprHandle body =
        Min::make(Max::make(x, y, true), Max::make(z, x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);
    checkExprIR(simplified, "Max(Min(y, z, 1), x, 1)");
  }

  {
    // Min(Max(y, x), Max(x, z)) => Max(Min(y, z), x)
    // 创建一个表达式，计算 Min(Max(y, x), Max(x, z))，并进行简化
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);
  {
    // Min(Max(y, x), Max(z, x)) => Max(Min(y, z), x)
    // 创建表达式，使用最大值运算符和最小值运算符重组表达式
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(z, x, true), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否与预期的字符串表示相匹配
    checkExprIR(simplified, "Max(Min(y, z, 1), x, 1)");
  }

  {
    // Min(Max(y, x), Max(z, x)) => Min(Max(x, y), Max(x, z))
    // 当模式中的所有操作符的传播 NaNs 属性不相同时，不应进行简化
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(z, x, false), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并分别检查其左右子节点
    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Max, min->lhs(), max1, "x", "y");
    ASSERT_TRUE(max1->propagate_nans());
    IS_BINOP_W_VARS(Max, min->rhs(), max2, "x", "z");
    ASSERT_FALSE(max2->propagate_nans());
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(5, Min(x, 8)) => Min(x, 8)
    // 创建包含常量和最小值运算符的表达式
    ExprHandle body = Min::make(5, Min::make(x, 8, true), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并检查其包含的常量和变量
    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(8, Min(x, 5)) => Min(x, 8)
    // 创建包含常量和最小值运算符的表达式
    ExprHandle body = Min::make(8, Min::make(x, 5, true), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并检查其包含的常量和变量
    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Min(x, 8), 5) => Min(x, 8)
    // 创建包含常量和最小值运算符的表达式
    ExprHandle body = Min::make(Min::make(x, 8, true), 5, true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并检查其包含的常量和变量
    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Min(x, 5), 8) => Min(x, 8)
    // 创建包含常量和最小值运算符的表达式
    ExprHandle body = Min::make(Min::make(x, 5, true), 8, true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并检查其包含的常量和变量
    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(5, Min(Min(y, Min(z, 8)), x)) => Min(Min(Min(x, 5), y), z)
    // 创建嵌套最小值运算符的表达式
    ExprHandle body = Min::make(
        5, Min::make(x, Min::make(y, Min::make(z, 8, true), true), true), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并逐层检查其嵌套结构和包含的常量和变量
    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(5, Min(Min(y, Min(z, 8)), x)) => Min(Min(Min(x, 5), y), z)
    // 创建嵌套最小值运算符的表达式
    ExprHandle body = Min::make(
        5, Min::make(Min::make(y, Min::make(z, 8, true), true), x, true), true);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认简化后的表达式为 Min 类型，并逐层检查其嵌套结构和包含的常量和变量
    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2->rhs() 是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1->rhs() 是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(5, Min(Min(Min(z, 8), y), x)) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      5, Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(x, Min(y, Min(8, z))), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(x, Min::make(y, Min::make(8, z, true), true), true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(y, Min(8, z)), x), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(Min::make(y, Min::make(z, 8, true), true), x, true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(Min(8, z), y), x), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(Min(z, 5), y), x), 8) => Min(Min(Min(Min(z, 5), y), x), 8)
  // 当所有 Min 操作符的 propagate_nans 属性不同时，不进行简化
  ExprHandle body = Min::make(
      Min::make(Min::make(Min::make(z, 5, true), y, false), x, true),
      8,
      false);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 检查简化后的表达式是否与预期的字符串匹配
  checkExprIR(simplified, "Min(Min(Min(Min(z, 5, 1), y, 0), x, 1), 8, 0)");
}

{
  // 表达式 Min(8, Min(Min(x, 5), Min(y, z))) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      8, Min::make(Min::make(x, 5, true), Min::make(y, z, true), true), true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);



  // 检查 min2->rhs() 是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1->rhs() 是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(5, Min(Min(Min(z, 8), y), x)) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      5, Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(x, Min(y, Min(8, z))), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(x, Min::make(y, Min::make(8, z, true), true), true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(Min(z, 8), y), x), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(Min(8, z), y), x), 5) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), 5, true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
  // 验证 min2 的左子节点是否为名为 "x"，值为 5 的二元操作符
  IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
  // 验证 min3 是否支持 NaN 传播
  ASSERT_TRUE(min3->propagate_nans());
  // 检查 min2 的右子节点是否为名为 "y" 的变量
  IS_VAR_WITH_NAME(min2->rhs(), "y");
  // 检查 min1 的右子节点是否为名为 "z" 的变量
  IS_VAR_WITH_NAME(min1->rhs(), "z");
}

{
  // 表达式 Min(Min(Min(Min(z, 5), y), x), 8) => Min(Min(Min(Min(z, 5), y), x), 8)
  // 当所有 Min 操作符的 propagate_nans 属性不同时，不进行简化
  ExprHandle body = Min::make(
      Min::make(Min::make(Min::make(z, 5, true), y, false), x, true),
      8,
      false);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 检查简化后的表达式是否与预期的字符串匹配
  checkExprIR(simplified, "Min(Min(Min(Min(z, 5, 1), y, 0), x, 1), 8, 0)");
}

{
  // 表达式 Min(8, Min(Min(x, 5), Min(y, z))) => Min(Min(Min(x, 5), y), z)
  ExprHandle body = Min::make(
      8, Min::make(Min::make(x, 5, true), Min::make(y, z, true), true), true);
  // 对表达式进行简化
  ExprHandle simplified = IRSimplifier::simplify(body);

  // 验证 simplified 的根节点是否为 Min 类型
  IS_NODE_WITH_NAME(Min, simplified.node(), min1);
  // 验证 min1 的左子节点是否为 Min 类型，并将其保存为 min2
  IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    // 调用宏 IS_BINOP_W_CONST，验证表达式 Min(min2->lhs(), min3) 是否为二元操作符且左操作数为 min2->lhs()，右操作数为常数 "x" 的 5
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    // 断言 min3 是否支持 NaN 传播
    ASSERT_TRUE(min3->propagate_nans());
    // 调用宏 IS_VAR_WITH_NAME，验证 min2->rhs() 是否为变量 "y"
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    // 调用宏 IS_VAR_WITH_NAME，验证 min1->rhs() 是否为变量 "z"
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // 构造表达式 Min(Min(Min(x, 5), Min(y, z)), 8)
    ExprHandle body = Min::make(
        Min::make(Min::make(x, 5, true), Min::make(y, z, true), true), 8, true);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 调用宏 IS_NODE_WITH_NAME，验证 simplified 的节点类型为 Min，且命名为 min1
    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    // 调用宏 IS_NODE_WITH_NAME，验证 min1 的左操作数节点类型为 Min，且命名为 min2
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    // 调用宏 IS_BINOP_W_CONST，验证 min2 的左操作数是否为 x，右操作数为常数 5
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    // 断言 min3 是否支持 NaN 传播
    ASSERT_TRUE(min3->propagate_nans());
    // 调用宏 IS_VAR_WITH_NAME，验证 min2 的右操作数是否为变量 "y"
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    // 调用宏 IS_VAR_WITH_NAME，验证 min1 的右操作数是否为变量 "z"
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }
}

TEST(Simplify, SimplifyWontReorderFloat) {
  {
    // 3 * (3 * x) - 3 * (3 * y) => 9 * (x - y)
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);

    // 构建表达式：3 * (3 * x) - 3 * (3 * y)
    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结构为乘法节点，并检查左右子节点
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 9);
    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_VAR_WITH_NAME(sub->rhs(), "y");
  }

  {
    // 3 * (3 * x) - 3 * (3 * y) => 3 * (3 * x) - 3 * (3 * y).
    // 如果变量是浮点型，则操作不可交换，不能重排。
    // 定义浮点型变量 x 和 y
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);

    // 构建表达式：3 * (3 * x) - 3 * (3 * y)
    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结构为减法节点，检查左侧和右侧子节点的结构
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Float, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, lhsMul->rhs(), lhsVarMul);
    IS_IMM_WITH_VAL(Float, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, sub->rhs(), rhsMul);
    IS_IMM_WITH_VAL(Float, rhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, rhsMul->rhs(), rhsVarMul);
    IS_IMM_WITH_VAL(Float, rhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(rhsVarMul->rhs(), "y");
  }

  {
    // 3 * (3 * x) - 3 * (3 * y) => 3 * (3 * x) - (9 * y).
    // 如果不重排浮点操作数，则我们将简化子表达式。
    // 定义双精度浮点型变量 x 和整型变量 y
    VarHandle x("x", kDouble);
    VarHandle y("y", kInt);

    // 构建表达式：3 * (3 * x) - 3 * (3 * y)
    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结构为减法节点，检查左侧和右侧子节点的结构
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Double, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, lhsMul->rhs(), lhsVarMul);
    IS_IMM_WITH_VAL(Double, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME_AND_CAST(Mul, sub->rhs(), rhsMul, Double);
    IS_IMM_WITH_VAL(Int, rhsMul->lhs(), 9);
    IS_VAR_WITH_NAME(rhsMul->rhs(), "y");
  }

  {
    // 防止从数据类型传播的浮点数重排。
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);

    // 构建表达式：3.f * (3 * x) - 3 * (3.f * y)
    ExprHandle body = ExprHandle(3.f) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3.f) * y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结构为减法节点，检查左侧和右侧子节点的结构
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Float, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME_AND_CAST(Mul, lhsMul->rhs(), lhsVarMul, Float);
    IS_IMM_WITH_VAL(Int, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, sub->rhs(), rhsMul);



    IS_IMM_WITH_VAL(Float, rhsMul->lhs(), 3);
    IS_NODE_WITH_NAME_AND_CAST(Mul, rhsMul->rhs(), rhsVarMul, Float);
    IS_IMM_WITH_VAL(Int, rhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(rhsVarMul->rhs(), "y");
  }
}
    // 检查 rhsMul->lhs() 是否是 Float 类型的常量 3
    IS_IMM_WITH_VAL(Float, rhsMul->lhs(), 3);
    // 检查 rhsMul->rhs() 是否是名为 Mul 的节点，将结果保存在 rhsVarMul 中
    IS_NODE_WITH_NAME(Mul, rhsMul->rhs(), rhsVarMul);
    // 检查 rhsVarMul->lhs() 是否是 Float 类型的常量 3
    IS_IMM_WITH_VAL(Float, rhsVarMul->lhs(), 3);
    // 检查 rhsVarMul->rhs() 是否是名为 Cast 的节点，将结果保存在 yCast 中
    IS_NODE_WITH_NAME(Cast, rhsVarMul->rhs(), yCast);
    // 检查 yCast->src_value() 是否是名为 "y" 的变量
    IS_VAR_WITH_NAME(yCast->src_value(), "y");

  }

  {
    // 创建名为 x 的浮点数变量句柄
    VarHandle x("x", kFloat);
    // 创建名为 y 的浮点数变量句柄
    VarHandle y("y", kFloat);
    // 构造表达式：x % y - (x % y - 1)，存储在 body 中
    // 不对 FP（浮点数）类型的不透明操作进行重新排序
    ExprHandle body = (x % y) - ((x % y) - 1);
    // 使用 IRSimplifier 对表达式进行简化，简化后的表达式存储在 simplified 中
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 检查 simplified.node() 是否是名为 Sub 的节点，将结果保存在 sub 中
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    // 检查 sub->lhs() 是否是名为 Mod 的节点，将结果保存在 lhsMod 中
    IS_NODE_WITH_NAME(Mod, sub->lhs(), lhsMod);
    // 检查 lhsMod->lhs() 是否是名为 "x" 的变量
    IS_VAR_WITH_NAME(lhsMod->lhs(), "x");
    // 检查 lhsMod->rhs() 是否是名为 "y" 的变量
    IS_VAR_WITH_NAME(lhsMod->rhs(), "y");

    // 检查 sub->rhs() 是否是名为 Sub 的节点，将结果保存在 rhsSub 中
    IS_NODE_WITH_NAME(Sub, sub->rhs(), rhsSub);
    // 检查 rhsSub->lhs() 是否是名为 Mod 的节点，将结果保存在 rhsMod 中
    IS_NODE_WITH_NAME(Mod, rhsSub->lhs(), rhsMod);
    // 检查 rhsMod->lhs() 是否是名为 "x" 的变量
    IS_VAR_WITH_NAME(rhsMod->lhs(), "x");
    // 检查 rhsMod->rhs() 是否是名为 "y" 的变量
    IS_VAR_WITH_NAME(rhsMod->rhs(), "y");
    // 检查 rhsSub->rhs() 是否是 Float 类型的常量 1
    IS_IMM_WITH_VAL(Float, rhsSub->rhs(), 1);
  }
}

TEST(Simplify, SimplifyRoundModPattern) {
  {
    // (x/y)*y + x%y => x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 (x/y)*y + x%y
    ExprHandle body = ((x / y) * y) + (x % y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Reverse order.
    // x%y + (x/y)*y => x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 x%y + (x/y)*y
    ExprHandle body = (x % y) + ((x / y) * y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Non opaque denominator.
    // (x / (4+y)) * (4+y)) + (x % (y + 4)) => x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 (x / (4+y)) * (4+y)) + (x % (y + 4))
    ExprHandle body = ((x / (ExprHandle(4) + y)) * (ExprHandle(4) + y)) +
        (x % (y + ExprHandle(4)));
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Reverse order.
    // (x % (y + 4)) + (x / (4+y)) * (4+y)) => x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 (x % (y + 4)) + (x / (4+y)) * (4+y))
    ExprHandle body = (x % (y + ExprHandle(4))) +
        ((x / (ExprHandle(4) + y)) * (ExprHandle(4) + y));
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Opaque denominator.
    // (x / (2/y)) * (2/y)) + (x % (2/y)) => x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 (x / (2/y)) * (2/y)) + (x % (2/y))
    ExprHandle body = ((x / (ExprHandle(2) / y)) * (ExprHandle(2) / y)) +
        (x % (ExprHandle(2) / y));
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是变量 x
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Non opaque numerator
    // ((2*x)/y * y) + ((2*x) % y) => 2 * x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 ((2*x)/y * y) + ((2*x) % y)
    ExprHandle body =
        (((ExprHandle(2) * x) / y) * y) + ((ExprHandle(2) * x) % y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确保简化后的表达式结果是 2 * x
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Opaque numerator.
    // ((x/2) / y * y) + (x/2 % y) => x / 2.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 ((x/2) / y * y) + (x/2 % y)
    ExprHandle body =
        (((x / ExprHandle(2)) / y) * y) + ((x / ExprHandle(2)) % y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结果是 x / 2
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_IMM_WITH_VAL(Int, div->rhs(), 2);
  }

  {
    // Numerator and denominator.
    // ((2*x)/(2*y) * (2*y)) + ((2*x) % (2*y)) => 2 * x.
    // 定义变量 x 和 y，类型为整数
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式 ((2*x)/(2*y) * (2*y)) + ((2*x) % (2*y))
    ExprHandle body =
        (((ExprHandle(2) * x) / (ExprHandle(2) * y)) * (ExprHandle(2) * y)) +
        ((ExprHandle(2) * x) % (ExprHandle(2) * y));
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确保简化后的表达式结果是 2 * x
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Reverse order.
  {
    // ((2*x) % (2*y)) + ((2*x)/(2*y) * (2*y)) => 2 * x.
    // 定义名为 x 和 y 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 定义表达式 body，代表 ((2*x) % (2*y)) + ((2*x)/(2*y) * (2*y))
    ExprHandle body = ((ExprHandle(2) * x) % (ExprHandle(2) * y)) +
        (((ExprHandle(2) * x) / (ExprHandle(2) * y)) * (ExprHandle(2) * y));
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);

    // 确认 simplified 是乘法节点，并且左操作数是整数常量 2，右操作数是变量 "x"
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Negated Subtraction of Round Mod.
    // (x/y) * y - (0 - x%y) => x.
    // 定义名为 x 和 y 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 定义表达式 body，代表 (x/y) * y - (0 - x%y)
    ExprHandle body = ((x / y) * y) - (ExprHandle(0) - (x % y));
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认 simplified 是变量 "x"
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Other terms are preserved.
    // (x/y)*y + x%y + (y * x) => x + (y * x).
    // 定义名为 x 和 y 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 定义表达式 body，代表 (x/y)*y + x%y + (y * x)
    ExprHandle body = ((x / y) * y) + (x % y) + (y * x);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认 simplified 是加法节点，并且左操作数是变量 "x"，右操作数是乘法节点（左操作数是变量 "x"，右操作数是变量 "y"）
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // Sanity checking we wont do the optimization on floats.
    // 定义名为 x 和 y 的浮点类型变量
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    // 定义表达式 body，代表 (x/y)*y + x%y
    ExprHandle body = ((x / y) * y) + (x % y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认 simplified 是加法节点，并且左操作数是乘法节点（左操作数是除法节点（左操作数是变量 "x"，右操作数是变量 "y"），右操作数是变量 "y"）
    // 右操作数是取模节点（左操作数是变量 "x"，右操作数是变量 "y"）
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), roundMul);
    IS_NODE_WITH_NAME(Div, roundMul->lhs(), roundDiv);
    IS_VAR_WITH_NAME(roundDiv->lhs(), "x");
    IS_VAR_WITH_NAME(roundDiv->rhs(), "y");
    IS_VAR_WITH_NAME(roundMul->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "y");
  }

  {
    // Sanity check we wont do it if the mod term doesn't match.
    // 定义名为 x、y 和 z 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    // 定义表达式 body，代表 (x/y)*y + x%z
    ExprHandle body = ((x / y) * y) + (x % z);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查 simplified 是否等于原始表达式 "(x / y) * y + x % z"
    checkExprIR(simplified, "(x / y) * y + x % z");
  }

  {
    // Sanity check we wont do it if the div term doesn't match.
    // 定义名为 x、y 和 z 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    // 定义表达式 body，代表 (y * (x / z)) + (x % y)
    ExprHandle body = (y * (x / z)) + (x % y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查 simplified 是否等于原始表达式 "x % y + (x / z) * y"
    checkExprIR(simplified, "x % y + (x / z) * y");
  }

  {
    // Sanity check we wont do it if the mul term doesn't match.
    // 定义名为 x、y 和 z 的整数类型变量
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    // 定义表达式 body，代表 ((x / y) * z) + (x % y)
    ExprHandle body = ((x / y) * z) + (x % y);
    // 简化表达式 body
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查 simplified 是否等于原始表达式 "x % y + (x / y) * z"
    checkExprIR(simplified, "x % y + (x / y) * z");
  }
}
  // 定义一个测试用例，名为 SimplifyRoundModPatternFactorization
  TEST(Simplify, SimplifyRoundModPatternFactorization) {
    {
      // 完全因式分解。
      // 2 * (x/y * y) + 2 * (x%y) => 2 * x.
      VarHandle x("x", kInt);  // 定义整型变量 x
      VarHandle y("y", kInt);  // 定义整型变量 y
      ExprHandle body = ExprHandle(2) * ((x / y) * y) + ExprHandle(2) * (x % y);  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 断言简化后的表达式是乘法节点
      IS_IMM_WITH_VAL(Int, mul->lhs(), 2);  // 断言乘法的左操作数是整数常量 2
      IS_VAR_WITH_NAME(mul->rhs(), "x");  // 断言乘法的右操作数是变量 x
    }

    {
      // 部分因式分解。
      // 32 * (x/8) + 4 * (x % 8) => 4 * x.
      VarHandle x("x", kInt);  // 定义整型变量 x
      VarHandle y("y", kInt);  // 定义整型变量 y
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks,cppcoreguidelines-avoid-magic-numbers)
      ExprHandle body = ExprHandle(32) * (x / 8) + ExprHandle(4) * (x % 8);  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 断言简化后的表达式是乘法节点
      IS_IMM_WITH_VAL(Int, mul->lhs(), 4);  // 断言乘法的左操作数是整数常量 4
      IS_VAR_WITH_NAME(mul->rhs(), "x");  // 断言乘法的右操作数是变量 x
    }

    {
      // 需要常量折叠的因式分解。
      // 20 * (x  / (16 / 2)) * 2 + (11 % 6) * (x % (7+1)) => 5 * x.
      VarHandle x("x", kInt);  // 定义整型变量 x
      ExprHandle body = ExprHandle(40) * (x / (ExprHandle(16) / 2)) +
          (ExprHandle(11) % 6) * (x % (ExprHandle(7) + 1));  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 断言简化后的表达式是乘法节点
      IS_IMM_WITH_VAL(Int, mul->lhs(), 5);  // 断言乘法的左操作数是整数常量 5
      IS_VAR_WITH_NAME(mul->rhs(), "x");  // 断言乘法的右操作数是变量 x
    }

    {
      VarHandle x("x", kInt);  // 定义整型变量 x
      ExprHandle body = (x / 5) * 10 + ExprHandle(2) * (x % 5);  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Mul, simplified.node(), mul);  // 断言简化后的表达式是乘法节点
      IS_IMM_WITH_VAL(Int, mul->lhs(), 2);  // 断言乘法的左操作数是整数常量 2
      IS_VAR_WITH_NAME(mul->rhs(), "x");  // 断言乘法的右操作数是变量 x
    }

    {
      VarHandle x("x", kInt);  // 定义整型变量 x
      ExprHandle body = (x / 10) * 0 + x % 5;  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Mod, simplified.node(), mod);  // 断言简化后的表达式是取模节点
      IS_VAR_WITH_NAME(mod->lhs(), "x");  // 断言取模操作的左操作数是变量 x
      IS_IMM_WITH_VAL(Int, mod->rhs(), 5);  // 断言取模操作的右操作数是整数常量 5
    }
  }

  // 定义一个测试用例，名为 SimplifyRoundModPatternMultivar
  TEST(Simplify, SimplifyRoundModPatternMultivar) {
    {
      // 多变量情况。
      // (x/8) * 8 + (y/5)*5 + x%8 + y%5 => x + y.
      VarHandle x("x", kInt);  // 定义整型变量 x
      VarHandle y("y", kInt);  // 定义整型变量 y
      ExprHandle body = (x / ExprHandle(8) * ExprHandle(8)) +
          (y / ExprHandle(5) * ExprHandle(5)) + (x % 8) + (y % 5);  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Add, simplified.node(), add);  // 断言简化后的表达式是加法节点
      IS_VAR_WITH_NAME(add->lhs(), "x");  // 断言加法操作的左操作数是变量 x
      IS_VAR_WITH_NAME(add->rhs(), "y");  // 断言加法操作的右操作数是变量 y
    }

    {
      // 找到正确的变量。
      // (y/8) * 8  x%8 + y%8 + z%8 => x%8 + y + z%8
      VarHandle x("x", kInt);  // 定义整型变量 x
      VarHandle y("y", kInt);  // 定义整型变量 y
      VarHandle z("z", kInt);  // 定义整型变量 z
      ExprHandle body =
          (y / ExprHandle(8) * ExprHandle(8)) + (x % 8) + (y % 8) + (z % 8);  // 构建表达式
      ExprHandle simplified = IRSimplifier::simplify(body);  // 对表达式进行简化
      IS_NODE_WITH_NAME(Add, simplified.node(), add);  // 断言简化后的表达式是加法节点
      IS_NODE_WITH_NAME(Add, add->lhs(), add2);  // 断言加法的左操作数是加法节点
      IS_NODE_WITH_NAME(Mod, add2->lhs(), xMod);  // 断言加法节点的左操作数是取模节点
      IS_VAR_WITH_NAME(xMod->lhs(), "x");  // 断言取模操作的左操作数是变量 x
    }
  }
}
    // 检查 Int 类型的 xMod 右手边是否是一个立即数，并且其值为 8
    IS_IMM_WITH_VAL(Int, xMod->rhs(), 8);
    // 检查 add2 的右手边是否是一个变量，并且其变量名为 "y"
    IS_VAR_WITH_NAME(add2->rhs(), "y");
    // 检查 add 的右手边是否是一个 Mod 节点，并将结果保存在 zMod 中
    IS_NODE_WITH_NAME(Mod, add->rhs(), zMod);
    // 检查 zMod 的左手边是否是一个变量，并且其变量名为 "z"
    IS_VAR_WITH_NAME(zMod->lhs(), "z");
    // 检查 zMod 的右手边是否是一个立即数，并且其值为 8
    IS_IMM_WITH_VAL(Int, zMod->rhs(), 8);
  }

  {
    // 复合表达式。
    // 计算表达式 (x + (z + 512 * y) % 16) + 16 * ((z + 512 * y) / 16)
    // => 简化为 (z + 512 * y) + x
    // 定义整型变量 x, y, z
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);

    // 构建表达式体，包括加法、取模、整数除法和乘法
    ExprHandle body = x + (z + y * 512) % 16 + ((z + y * 512) / 16 * 16);
    // 使用 IRSimplifier 简化表达式体
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否与 "x + (z + 512 * y)" 相符
    checkExprIR(simplified, "x + (z + 512 * y)");
  }
TEST(Simplify, SimplifyModRoundModPattern) {
  {
    // t/7 % 9 * 7 + t % 7 => t%63
    // 定义名为 t 的变量，类型为整数
    VarHandle t("t", kInt);
    // 构造表达式：(t / 7 % 9) * 7 + t % 7
    ExprHandle body = (t / 7 % 9) * 7 + t % 7;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式结构为 Mod 类型
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    // 确认 Mod 的左子树为变量 "t"
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    // 确认 Mod 的右子树为整数常量 63
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // 2*t/7 % 9 * 7 + 2*t % 7 => 2*t % 63
    // 定义名为 t 的变量，类型为整数
    VarHandle t("t", kInt);
    // 构造表达式：(ExprHandle(2) * t / 7 % 9) * 7 + ExprHandle(2) * t % 7
    ExprHandle body = (ExprHandle(2) * t / 7 % 9) * 7 + ExprHandle(2) * t % 7;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式结构为 Mod 类型
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    // 确认 Mod 的左子树为 Mul 类型
    IS_NODE_WITH_NAME(Mul, mod->lhs(), mul);
    // 确认 Mul 的左子树为整数常量 2
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    // 确认 Mul 的右子树为变量 "t"
    IS_VAR_WITH_NAME(mul->rhs(), "t");
    // 确认 Mod 的右子树为整数常量 63
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // t/x % y * x + t % x => t%(x*y)
    // 定义名为 t, x, y 的变量，类型为整数
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 构造表达式：(t / x % y) * x + t % x
    ExprHandle body = (t / x % y) * x + t % x;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式结构为 Mod 类型
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    // 确认 Mod 的左子树为变量 "t"
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    // 确认 Mod 的右子树为 Mul 类型
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    // 确认 Mul 的左子树为变量 "x"
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    // 确认 Mul 的右子树为变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // k*t/x % y * x + k*t % x => k*t%(x*y)
    // 定义名为 t, x, y, k 的变量，类型为整数
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    // 构造表达式：(k * t / x % y) * x + k * t % x
    ExprHandle body = (k * t / x % y) * x + k * t % x;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 调用自定义函数检查简化后的表达式结构是否符合预期
    checkExprIR(simplified, "(k * t) % (x * y)");
  }

  {
    // t/k/x % y * x + t/k % x => t/k%(x*y)
    // 定义名为 t, x, y, k 的变量，类型为整数
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    // 构造表达式：(t / k / x % y) * x + t / k % x
    ExprHandle body = (t / k / x % y) * x + t / k % x;
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式结构为 Mod 类型
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    // 确认 Mod 的左子树为 Div 类型
    IS_NODE_WITH_NAME(Div, mod->lhs(), div);
    // 确认 Div 的左子树为变量 "t"
    IS_VAR_WITH_NAME(div->lhs(), "t");
    // 确认 Div 的右子树为变量 "k"
    IS_VAR_WITH_NAME(div->rhs(), "k");
    // 确认 Mod 的右子树为 Mul 类型
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    // 确认 Mul 的左子树为变量 "x"
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    // 确认 Mul 的右子树为变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // Sanity checking we wont do the optimization on floats.
    // 定义名为 x, y, z 的变量，类型为浮点数
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    VarHandle z("z", kFloat);
    // 构造表达式：((x / y % z) * y) + (x % y)
    ExprHandle body = ((x / y % z) * y) + (x % y);
    // 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 确认简化后的表达式结构为 Add 类型
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    // 确认 Add 的左子树为 Mul 类型
    IS_NODE_WITH_NAME(Mul, add->lhs(), mul);
    // 确认 Mul 的左子树为 Mod 类型
    IS_NODE_WITH_NAME(Mod, mul->lhs(), mod);
    // 确认 Mod 的左子树为 Div 类型
    IS_NODE_WITH_NAME(Div, mod->lhs(), div);
    // 确认 Div 的左子树为变量 "x"
    IS_VAR_WITH_NAME(div->lhs(), "x");
    // 确认 Div 的右子树为变量 "y"
    IS_VAR_WITH_NAME(div->rhs(), "y");
    // 确认 Mod 的右子树为变量 "z"
    IS_VAR_WITH_NAME(mod->rhs(), "z");
    // 确认 Mul 的右子树为变量 "y"
    IS_VAR_WITH_NAME(mul->rhs(), "y");
    // 确认 Add 的右子树为 Mod 类型
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod2);
    // 确认 Mod 的左子树为变量 "x"
    IS_VAR_WITH_NAME(mod2->lhs(), "x");
    // 确认 Mod 的右子树为变量 "y"
    IS_VAR_WITH_NAME(mod2->rhs(), "y");
  }
}

TEST(Simplify, SimplifyModRoundModPatternFactorization) {
  {
    {
        // 2 * (t / 7 % 9 * 7) + 2 * (t % 7) => 2 * (t % 63)
        // 定义整型变量 t
        VarHandle t("t", kInt);
        // 构造表达式体，按照给定的算术表达式计算
        ExprHandle body =
            ExprHandle(2) * ((t / 7 % 9) * 7) + ExprHandle(2) * (t % 7);
        // 对表达式进行简化
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 确认简化后的表达式结构为乘法节点，并验证其左操作数为整数常量 2
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
        // 确认右操作数为模运算节点，并验证其左操作数为变量 t，右操作数为整数常量 63
        IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
        IS_VAR_WITH_NAME(mod->lhs(), "t");
        IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
    }
    
    {
        // t / 7 % 9 * 14 + 2 * (t % 7) => 2 * (t % 63)
        // 定义整型变量 t
        VarHandle t("t", kInt);
        // 构造表达式体，按照给定的算术表达式计算
        ExprHandle body = (t / 7 % 9) * 14 + ExprHandle(2) * (t % 7);
        // 对表达式进行简化
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 确认简化后的表达式结构为乘法节点，并验证其左操作数为整数常量 2
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
        // 确认右操作数为模运算节点，并验证其左操作数为变量 t，右操作数为整数常量 63
        IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
        IS_VAR_WITH_NAME(mod->lhs(), "t");
        IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
    }
    
    {
        // t / 14 % 9 * 7 + t / 2 % 7 => t / 2 % 63
        // 定义整型变量 t
        VarHandle t("t", kInt);
        // 构造表达式体，按照给定的算术表达式计算
        ExprHandle body = (t / 14 % 9) * 7 + t / 2 % 7;
        // 对表达式进行简化
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 确认简化后的表达式结构为模运算节点，并验证其左操作数为除法节点，左操作数为变量 t，右操作数为整数常量 2
        IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
        IS_NODE_WITH_NAME(Div, mod->lhs(), div);
        IS_VAR_WITH_NAME(div->lhs(), "t");
        IS_IMM_WITH_VAL(Int, div->rhs(), 2);
        // 验证模运算节点的右操作数为整数常量 63
        IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
    }
    
    {
        // t / (7 * 3) % 9 * 7 * 3 + t % (7 * 3) => t % 189
        // 定义整型变量 t
        VarHandle t("t", kInt);
        // 构造表达式体，按照给定的算术表达式计算
        ExprHandle body = (t / (ExprHandle(7) * ExprHandle(3)) % 9) * 7 * 3 +
            t % (ExprHandle(7) * ExprHandle(3));
        // 对表达式进行简化
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 确认简化后的表达式结构为模运算节点，并验证其左操作数为变量 t，右操作数为整数常量 189
        IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
        IS_VAR_WITH_NAME(mod->lhs(), "t");
        IS_IMM_WITH_VAL(Int, mod->rhs(), 189);
    }
    
    {
        // 2 * (t / x % y * x) + 2 * (t % x) => 2 * (t % (x * y))
        // 定义整型变量 t, x, y
        VarHandle t("t", kInt);
        VarHandle x("x", kInt);
        VarHandle y("y", kInt);
        // 构造表达式体，按照给定的算术表达式计算
        ExprHandle body =
            ExprHandle(2) * ((t / x % y) * x) + ExprHandle(2) * (t % x);
        // 对表达式进行简化
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 确认简化后的表达式结构为乘法节点，并验证其左操作数为整数常量 2
        IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
        IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
        // 确认右操作数为模运算节点，验证其左操作数为变量 t，右操作数为乘法节点，左操作数为变量 x，右操作数为变量 y
        IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
        IS_VAR_WITH_NAME(mod->lhs(), "t");
        IS_NODE_WITH_NAME(Mul, mod->rhs(), mul2);
        IS_VAR_WITH_NAME(mul2->lhs(), "x");
        IS_VAR_WITH_NAME(mul2->rhs(), "y");
    }
TEST(Simplify, SimplifyModRoundModPatternMultivar) {
  {
    // 定义整数类型变量 t
    VarHandle t("t", kInt);
    // 构建表达式：(t/7 % 9) * 7 + t % 7 + t
    ExprHandle body = (t / 7 % 9) * 7 + t % 7 + t;
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否等于 "t % 63 + t"
    checkExprIR(simplified, "t % 63 + t");
  }

  {
    // 定义整数类型变量 t
    VarHandle t("t", kInt);
    // 构建表达式：(t/7 % 9) * 7 + (t/8 % 9) * 8 + t % 7 + t % 8
    ExprHandle body = (t / 7 % 9) * 7 + (t / 8 % 9) * 8 + t % 7 + t % 8;
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式结构
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mod, add->lhs(), mod1);
    IS_VAR_WITH_NAME(mod1->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod1->rhs(), 63);
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod2);
    IS_VAR_WITH_NAME(mod2->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod2->rhs(), 72);
  }

  {
    // 定义整数类型变量 t, x, y, k
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    // 构建表达式：k + (t/x % y) * x + t % x
    ExprHandle body = k + (t / x % y) * x + t % x;
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式结构
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "k");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 定义整数类型变量 t, x, y, k
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    // 构建表达式：(t/x % y) * x + t % x + (t/k / x % y) * x + t/k % x
    ExprHandle body = (t / x % y) * x + t % x + (t / k / x % y) * x + t / k % x;
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否等于 "(t / k) % (x * y) + t % (x * y)"
    checkExprIR(simplified, "(t / k) % (x * y) + t % (x * y)");
  }

  {
    // 定义整数类型变量 io_flat
    VarHandle t("io_flat", kInt);
    // 构建表达式：7 * (t/7 % 9) + t % 7 + 63 * (t/63)
    ExprHandle body = ExprHandle(7) * (t / 7 % 9) + t % 7 + ExprHandle(63) * (t / 63);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否为变量 "io_flat"
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }

  {
    // 定义整数类型变量 io_flat
    VarHandle t("io_flat", kInt);
    // 构建复杂表达式
    ExprHandle body = (t / (ExprHandle(11) * 10 * 9 * 7)) * (7 * 9 * 10 * 11) +
        (t / (ExprHandle(10) * 9 * 7) % 11) * 7 * 9 * 10 +
        (t / (ExprHandle(9) * 7) % 10) * 7 * 9 + (t / 7 % 9) * 7 + t % 7;
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否为变量 "io_flat"
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }

  {
    // 定义整数类型变量 io_flat, m, n
    VarHandle t("io_flat", kInt);
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    // 构建表达式：m * (t/m % n) + t % m + (m * n) * (t/(m * n))
    ExprHandle body = m * (t / m % n) + t % m + (m * n) * (t / (m * n));
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 检查简化后的表达式是否为变量 "io_flat"
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }
}
  // 创建一个表达式，计算 io_flat 的值
  ExprHandle body = m * (t / m % n) + t % m + (m * n) * (t / (m * n));
  // 简化表达式 body
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 检查简化后的表达式是否包含名为 "io_flat" 的变量
  IS_VAR_WITH_NAME(simplified.node(), "io_flat");
}

{ // 5D: i0_flat / (k * l * n * m)  * (m * n * l * k) +
  // (i0_flat / (l * n * m) % k)  * m * n * l +
  // (i0_flat / (n * m) % l) * m * n +
  // (i0_flat / m % n)  * m +
  // i0_flat % m => io_flat
  // 声明变量 t, m, n, l, k，分别表示 io_flat, m, n, l, k
  VarHandle t("io_flat", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle l("l", kInt);
  VarHandle k("k", kInt);
  // 构建复杂的表达式 body，计算 io_flat 的值
  ExprHandle body = (t / (k * l * n * m)) * (m * n * l * k) +
      (t / (l * n * m) % k) * m * n * l + (t / (n * m) % l) * m * n +
      (t / m % n) * m + t % m;
  // 简化表达式 body
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 检查简化后的表达式是否包含名为 "io_flat" 的变量
  IS_VAR_WITH_NAME(simplified.node(), "io_flat");
}
}

TEST(Simplify, SimplifyDivisionScalarFactorization) {
  {
    // 简单因式分解分子和分母。
    // 8x / 4y => 2x / y.
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 创建表达式，进行数学运算
    ExprHandle body = (x * 8) / (y * 4);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }

  {
    // 如果无法进行因式分解，则不做任何更改。
    // 不能因式分解时保持不变。
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 创建表达式，进行数学运算
    ExprHandle body = (x * 7) / (y * 4);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 7);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_NODE_WITH_NAME(Mul, div->rhs(), rhs);
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 4);
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // 不要重新排序浮点数。
    // 不能重新排序浮点数。
    // 定义浮点型变量 x 和 y
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    // 创建表达式，进行数学运算
    ExprHandle body = (x * 8) / (y * 4);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_VAR_WITH_NAME(lhs->lhs(), "x");
    IS_IMM_WITH_VAL(Float, lhs->rhs(), 8.f);
    IS_NODE_WITH_NAME(Mul, div->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "y");
    IS_IMM_WITH_VAL(Float, rhs->rhs(), 4.f);
  }

  {
    // 确保如果只有标量部分，则不进行任何操作。
    // 如果只有标量部分，则保持不变。
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 创建表达式，进行数学运算
    ExprHandle body = (x * 1) / (y * 1);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }

  {
    // 可以因式分解变量的数量。
    // 可以因式分解变量的数量。
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 创建表达式，进行数学运算
    ExprHandle body = (x + x + x + x) / (y + y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }
}

TEST(Simplify, SimplifyConstantBranches) {
  {
    // 如果条件恒为真，则选择 true_value。
    // 1 ? x : y => x
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 定义常量表达式 t 为 1
    ExprHandle t(1);
    // 创建条件表达式，根据条件进行选择
    ExprHandle body = IfThenElse::make(t, x, y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
    // 断言简化后的表达式结构
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // 如果条件恒为假，则选择 false_value。
    // 0 ? x : y => y
    // 定义整型变量 x 和 y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    // 定义常量表达式 t 为 0
    ExprHandle t(0);
    // 创建条件表达式，根据条件进行选择
    ExprHandle body = IfThenElse::make(t, x, y);
    // 简化表达式
    ExprHandle simplified = IRSimplifier::simplify(body);
  {
    // 检查表达式是否简化后，条件为真（非零）
    IS_VAR_WITH_NAME(simplified.node(), "y");
  }

  {
    // 在检查条件前先简化表达式。
    // (x-x) ? x : y => y
    VarHandle x("x", kInt); // 创建名为"x"的整数类型变量
    VarHandle y("y", kInt); // 创建名为"y"的整数类型变量
    ExprHandle body = IfThenElse::make(x - x, x, y); // 构造条件表达式，如果(x-x)为真，则选择x，否则选择y
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化
    IS_VAR_WITH_NAME(simplified.node(), "y"); // 检查简化后的表达式结果是否是变量"y"
  }

  {
    // 如果两个分支表达式相同，则不进行条件判断。
    // y ? x : x => x
    VarHandle x("x", kInt); // 创建名为"x"的整数类型变量
    VarHandle y("y", kInt); // 创建名为"y"的整数类型变量
    ExprHandle body = IfThenElse::make(y, x, x); // 构造条件表达式，如果y为真，则选择第一个x，否则选择第二个x
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化
    IS_VAR_WITH_NAME(simplified.node(), "x"); // 检查简化后的表达式结果是否是变量"x"
  }

  {
    // 如果两个分支表达式简化为相同结果，也仍然有效。
    // y ? (x + x) : (2 * x) => x
    VarHandle x("x", kInt); // 创建名为"x"的整数类型变量
    VarHandle y("y", kInt); // 创建名为"y"的整数类型变量
    ExprHandle body = IfThenElse::make(y, x + x, ExprHandle(2) * x); // 构造条件表达式，如果y为真，则选择x+x，否则选择2*x
    ExprHandle simplified = IRSimplifier::simplify(body); // 对表达式进行简化
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul); // 检查简化后的表达式结果是否是乘法节点，并且存储在变量mul中
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2); // 检查乘法节点的左子节点是否是整数值2
    IS_VAR_WITH_NAME(mul->rhs(), "x"); // 检查乘法节点的右子节点是否是变量"x"
  }
}

TEST(Simplify, SimplifyConstantCond) {
  {
    // 如果条件恒为真，则取 true_value。
    // 1 ? A[0] = 1 : B[0] = 1 => A[0] = 1
    BufHandle a("A", {1}, kInt);
    BufHandle b("B", {1}, kInt);
    ExprHandle condition(1); // 定义条件为常量 1
    StmtPtr true_val = Store::make(a, {0}, 1); // 存储操作：将值 1 存入数组 A 的索引 0 处
    StmtPtr false_val = Store::make(b, {0}, 1); // 存储操作：将值 1 存入数组 B 的索引 0 处

    CondPtr body = alloc<Cond>(condition.node(), true_val, false_val); // 创建条件语句对象
    StmtPtr simplified = IRSimplifier::simplify(body); // 简化条件语句
    BlockPtr block = to<Block>(simplified); // 转换为块语句对象
    IS_NODE_WITH_NAME(Store, block->front(), store); // 确保块的第一个语句是存储操作
    IS_VAR_WITH_NAME(store->base_handle(), "A"); // 确保存储操作的目标变量是 A
  }

  {
    // 如果条件恒为假，则取 false_value。
    // 0 ? A[0] = 1 : B[0] = 1 => B[0] = 1
    BufHandle a("A", {1}, kInt);
    BufHandle b("B", {1}, kInt);
    ExprHandle condition(0); // 定义条件为常量 0
    StmtPtr true_val = Store::make(a, {0}, 1); // 存储操作：将值 1 存入数组 A 的索引 0 处
    StmtPtr false_val = Store::make(b, {0}, 1); // 存储操作：将值 1 存入数组 B 的索引 0 处

    StmtPtr body = alloc<Cond>(condition.node(), true_val, false_val); // 创建条件语句对象
    StmtPtr simplified = IRSimplifier::simplify(body); // 简化条件语句
    BlockPtr block = to<Block>(simplified); // 转换为块语句对象
    IS_NODE_WITH_NAME(Store, block->front(), store); // 确保块的第一个语句是存储操作
    IS_VAR_WITH_NAME(store->base_handle(), "B"); // 确保存储操作的目标变量是 B
  }

  {
    // 在检查之前对条件进行简化。
    // (x-x) ? A[0] = 1 : B[0] = 1 => B[0] = 1
    VarHandle x("x", kInt); // 定义整型变量 x
    BufHandle a("A", {1}, kInt); // 定义整型数组 A
    BufHandle b("B", {1}, kInt); // 定义整型数组 B
    ExprHandle condition(x - x); // 定义条件为 x - x，即常量 0
    StmtPtr true_val = Store::make(a, {0}, 1); // 存储操作：将值 1 存入数组 A 的索引 0 处
    StmtPtr false_val = Store::make(b, {0}, 1); // 存储操作：将值 1 存入数组 B 的索引 0 处

    StmtPtr body = alloc<Cond>(condition.node(), true_val, false_val); // 创建条件语句对象
    StmtPtr simplified = IRSimplifier::simplify(body); // 简化条件语句
    BlockPtr block = to<Block>(simplified); // 转换为块语句对象
    IS_NODE_WITH_NAME(Store, block->front(), store); // 确保块的第一个语句是存储操作
    IS_VAR_WITH_NAME(store->base_handle(), "B"); // 确保存储操作的目标变量是 B
  }

  {
    // 如果两个分支相同则不执行条件判断。
    // x ? A[0] = x : A[0] = x => A[0] = x
    VarHandle x("x", kInt); // 定义整型变量 x
    BufHandle a("A", {1}, kInt); // 定义整型数组 A
    ExprHandle condition(x - x); // 定义条件为 x - x，即常量 0
    StmtPtr true_val = Store::make(a, {0}, x); // 存储操作：将变量 x 的值存入数组 A 的索引 0 处
    StmtPtr false_val = Store::make(a, {0}, x); // 存储操作：将变量 x 的值存入数组 A 的索引 0 处

    StmtPtr body = alloc<Cond>(condition.node(), true_val, false_val); // 创建条件语句对象
    StmtPtr simplified = IRSimplifier::simplify(body); // 简化条件语句
    BlockPtr block = to<Block>(simplified); // 转换为块语句对象
    IS_NODE_WITH_NAME(Store, block->front(), store); // 确保块的第一个语句是存储操作
    IS_VAR_WITH_NAME(store->base_handle(), "A"); // 确保存储操作的目标变量是 A
  }

  {
    // 如果两个分支简化为相同的内容仍然有效。
    // x ? (x + x) : (2 * x) => x
    VarHandle x("x", kInt); // 定义整型变量 x
    BufHandle a("A", {1}, kInt); // 定义整型数组 A
    ExprHandle condition(x - x); // 定义条件为 x - x，即常量 0
    StmtPtr true_val = Store::make(a, {0}, ExprHandle(2) * x); // 存储操作：将 2*x 的值存入数组 A 的索引 0 处
    StmtPtr false_val = Store::make(a, {0}, x + x); // 存储操作：将 x + x 的值存入数组 A 的索引 0 处

    StmtPtr body = alloc<Cond>(condition.node(), true_val, false_val); // 创建条件语句对象
    StmtPtr simplified = IRSimplifier::simplify(body); // 简化条件语句
    BlockPtr block = to<Block>(simplified); // 转换为块语句对象
    IS_NODE_WITH_NAME(Store, block->front(), store); // 确保块的第一个语句是存储操作
    IS_VAR_WITH_NAME(store->base_handle(), "A"); // 确保存储操作的目标变量是 A
  }

  {
    // 创建一个名为 x 的变量句柄，类型为整数
    VarHandle x("x", kInt);
    // 创建一个名为 a 的缓冲区句柄，大小为 {1}，元素类型为整数
    BufHandle a("A", {1}, kInt);
    // 创建一个表达式句柄，表示条件 x
    ExprHandle condition(x);
    // 创建一个将 x 存储到缓冲区 a 的语句，位置为 {0}
    StmtPtr true_val = Store::make(a, {0}, x);
    // 创建一个将 2*x 存储到缓冲区 a 的语句，位置为 {0}
    StmtPtr false_val = Store::make(a, {0}, ExprHandle(2) * x);
    
    // 创建一个条件语句块，根据 condition 决定执行 true_val 还是 false_val
    StmtPtr body = alloc<Cond>(condition.node(), true_val, false_val);
    // 对创建的条件语句块进行简化处理
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为 Block 类型
    BlockPtr block = to<Block>(simplified);
    // 断言块是否为 nullptr
    ASSERT_EQ(block, nullptr);
    
    
    
    {
    // 创建一个条件语句块，条件为 false，主体为空块
    StmtPtr cond = alloc<Cond>(
        ExprHandle(false).node(),
        alloc<Block>(std::vector<StmtPtr>({})),
        nullptr);
    // 对创建的条件语句块进行简化处理
    StmtPtr simplified = IRSimplifier::simplify(cond);
    // 断言简化后的结果是否为 nullptr
    ASSERT_EQ(simplified, nullptr);
    }
    
    
    
    {
    // 创建一个条件语句块，条件为 true，主体为空块
    StmtPtr cond = alloc<Cond>(
        ExprHandle(true).node(),
        nullptr,
        alloc<Block>(std::vector<StmtPtr>({})));
    // 对创建的条件语句块进行简化处理
    StmtPtr simplified = IRSimplifier::simplify(cond);
    // 断言简化后的结果是否为 nullptr
    ASSERT_EQ(simplified, nullptr);
    }
    // 定义一个测试用例，名为 SimplifyEliminateEmptyCond，测试条件分支为空时的简化处理
TEST(Simplify, SimplifyEliminateEmptyCond) {
  // 如果条件分支为空，不同的方式，进行消除
  {
    // 创建一个名为 x 的整型变量句柄
    VarHandle x("x", kInt);
    // 创建一个表达式句柄，表示条件为 x
    ExprHandle condition(x);
    // 创建一个空的语句块，作为真值分支
    StmtPtr true_val = alloc<Block>(std::vector<StmtPtr>({}));

    // 创建一个条件语句，条件为 condition，真值分支为 true_val，假值分支为空
    StmtPtr body = alloc<Cond>(condition.node(), true_val, nullptr);
    // 对该条件语句进行简化处理
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的结果转换为 Block 类型
    BlockPtr block = to<Block>(simplified);
    // 断言 block 不为 nullptr
    ASSERT_NE(block, nullptr);
    // 断言 block 中的语句数为 0
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // 创建一个名为 x 的整型变量句柄
    VarHandle x("x", kInt);
    // 创建一个表达式句柄，表示条件为 x
    ExprHandle condition(x);
    // 创建一个空的语句块，作为假值分支
    StmtPtr false_val = alloc<Block>(std::vector<StmtPtr>({}));

    // 创建一个条件语句，条件为 condition，真值分支为空，假值分支为 false_val
    StmtPtr body = alloc<Cond>(condition.node(), nullptr, false_val);
    // 对该条件语句进行简化处理
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的结果转换为 Block 类型
    BlockPtr block = to<Block>(simplified);
    // 断言 block 不为 nullptr
    ASSERT_NE(block, nullptr);
    // 断言 block 中的语句数为 0
    ASSERT_EQ(block->nstmts(), 0);
  }
}

// 定义一个测试用例，名为 SimplifyConstantComparisons，测试常量比较的简化处理
TEST(Simplify, SimplifyConstantComparisons) {
  // 定义一个 lambda 表达式 ComparisonTest，用于测试具体的比较操作
  auto ComparisonTest =
      [](ExprHandle a, ExprHandle b, CompareSelectOperation op, int result) {
        // 创建一个比较操作的表达式
        ExprHandle body = CompareSelect::make(a, b, op);
        // 对该表达式进行简化处理
        ExprHandle simplified = IRSimplifier::simplify(body);
        // 断言简化后的结果是一个立即数，并且其值为 result
        IS_IMM_WITH_VAL(Int, simplified.node(), result);
      };

  // Equals.
  // 测试相等比较操作
  ComparisonTest(2, 2, kEQ, 1);
  ComparisonTest(1, 2, kEQ, 0);
  ComparisonTest(2, 1, kEQ, 0);

  // Greater than.
  // 测试大于比较操作
  ComparisonTest(2, 2, kGT, 0);
  ComparisonTest(1, 2, kGT, 0);
  ComparisonTest(2, 1, kGT, 1);

  // Greater or Equal.
  // 测试大于等于比较操作
  ComparisonTest(2, 2, kGE, 1);
  ComparisonTest(1, 2, kGE, 0);
  ComparisonTest(2, 1, kGE, 1);

  // Less Than.
  // 测试小于比较操作
  ComparisonTest(2, 2, kLT, 0);
  ComparisonTest(1, 2, kLT, 1);
  ComparisonTest(2, 1, kLT, 0);

  // Less or Equal.
  // 测试小于等于比较操作
  ComparisonTest(2, 2, kLE, 1);
  ComparisonTest(1, 2, kLE, 1);
  ComparisonTest(2, 1, kLE, 0);

  // Not equal.
  // 测试不等比较操作
  ComparisonTest(2, 2, kNE, 0);
  ComparisonTest(1, 2, kNE, 1);
  ComparisonTest(2, 1, kNE, 1);

  // With specified results:
  // 测试带有指定结果的比较操作
  ExprHandle body = CompareSelect::make(2, 2, 5, 42, kNE);
  // 对该表达式进行简化处理
  ExprHandle simplified = IRSimplifier::simplify(body);
  // 断言简化后的结果是一个立即数，并且其值为 42
  IS_IMM_WITH_VAL(Int, simplified.node(), 42);
}

// 定义一个测试用例，名为 SimplifySymbolicComparisons，测试符号比较的简化处理
TEST(Simplify, SimplifySymbolicComparisons) {
  // 创建名为 x 和 y 的整型变量句柄
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  // 定义一个 lambda 表达式 TookTrueBranch，用于断言表达式为立即数 1
  auto TookTrueBranch = [](ExprHandle a) { IS_IMM_WITH_VAL(Int, a.node(), 1); };
  // 定义一个 lambda 表达式 TookFalseBranch，用于断言表达式为立即数 0
  auto TookFalseBranch = [](ExprHandle a) {
    IS_IMM_WITH_VAL(Int, a.node(), 0);
  };

  // EQ

  // x == x => 1
  // 测试 x 等于 x 的情况
  ExprHandle body = CompareSelect::make(x, x, kEQ);
  // 对该表达式进行简化处理，并断言结果为立即数 1
  TookTrueBranch(IRSimplifier::simplify(body));

  // x == x+1 => 0
  // 测试 x 等于 x+1 的情况
  body = CompareSelect::make(x, x + 1, kEQ);
  // 对该表达式进行简化处理，并断言结果为立即数 0
  TookFalseBranch(IRSimplifier::simplify(body));

  // x == x * 2 cannot simplify since we don't know x is nonzero.
  // 测试 x 等于 x * 2 的情况，由于无法确定 x 是否为非零，因此无法进行简化处理

  body = CompareSelect::make(x, x * 2, kEQ);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  // 断言简化后的结果是一个 CompareSelect 类型的节点
  IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

  // x == x * 1 => 1
  // 测试 x 等于 x * 1 的情况
  body = CompareSelect::make(x, x * 1, kEQ);
  // 对该表达式进行简化处理，并断言结果为立即数 1
  TookTrueBranch(IRSimplifier::simplify(body));

  {
    // x == y => x == y
    // 测试 x 等于 y 的情况
    body = CompareSelect::make(x, y, kEQ);
  // 对表达式进行简化，并将结果赋给simplified
  ExprHandle simplified = IRSimplifier::simplify(body);
  
  // 确认简化后的节点是CompareSelect类型，并将其赋给cmp
  IS_NODE_WITH_NAME(CompareSelect, simplified.node(), cmp);
  
  // 断言比较操作为相等
  ASSERT_EQ(cmp->compare_select_op(), kEQ);
  
  // 确认左操作数为变量"x"
  IS_VAR_WITH_NAME(cmp->lhs(), "x");
  
  // 确认右操作数为整数常量5
  IS_IMM_WITH_VAL(Int, cmp->rhs(), 5);
}

{
  // 构造比较操作 x == 5，并将结果赋给body
  body = CompareSelect::make(x, 5, kEQ);
  
  // 对表达式进行简化，并将结果赋给simplified
  ExprHandle simplified = IRSimplifier::simplify(body);
  
  // 确认简化后的节点是CompareSelect类型，并将其赋给cmp
  IS_NODE_WITH_NAME(CompareSelect, simplified.node(), cmp);
  
  // 断言比较操作为相等
  ASSERT_EQ(cmp->compare_select_op(), kEQ);
  
  // 确认左操作数为变量"x"
  IS_VAR_WITH_NAME(cmp->lhs(), "x");
  
  // 确认右操作数为整数常量5
  IS_IMM_WITH_VAL(Int, cmp->rhs(), 5);
}

// GT

// 构造比较操作 x+1 > x，并将结果赋给body
body = CompareSelect::make(x + 1, x, kGT);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x > x+1，并将结果赋给body
body = CompareSelect::make(x, x + 1, kGT);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x > x-1，并将结果赋给body
body = CompareSelect::make(x, x - 1, kGT);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x-1 > x，并将结果赋给body
body = CompareSelect::make(x - 1, x, kGT);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x > x，并将结果赋给body
body = CompareSelect::make(x, x, kGT);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x*2 > x，并将结果赋给body
body = CompareSelect::make(x * 2, x, kGT);

// 确认简化后的节点是CompareSelect类型
IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

// GE

// 构造比较操作 x+1 >= x，并将结果赋给body
body = CompareSelect::make(x + 1, x, kGE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x >= x+1，并将结果赋给body
body = CompareSelect::make(x, x + 1, kGE);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x >= x，并将结果赋给body
body = CompareSelect::make(x, x, kGE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x*2 >= x，并将结果赋给body
body = CompareSelect::make(x * 2, x, kGE);

// 确认简化后的节点是CompareSelect类型
IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

// LT

// 构造比较操作 x+1 < x，并将结果赋给body
body = CompareSelect::make(x + 1, x, kLT);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x < x+1，并将结果赋给body
body = CompareSelect::make(x, x + 1, kLT);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x < x，并将结果赋给body
body = CompareSelect::make(x, x, kLT);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// LE

// 构造比较操作 x+1 <= x，并将结果赋给body
body = CompareSelect::make(x + 1, x, kLE);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));

// 构造比较操作 x <= x+1，并将结果赋给body
body = CompareSelect::make(x, x + 1, kLE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x <= x，并将结果赋给body
body = CompareSelect::make(x, x, kLE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// NE

// 构造比较操作 x+1 != x，并将结果赋给body
body = CompareSelect::make(x + 1, x, kNE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x != x+1，并将结果赋给body
body = CompareSelect::make(x, x + 1, kNE);

// 对表达式进行简化，并执行一个假设条件
TookTrueBranch(IRSimplifier::simplify(body));

// 构造比较操作 x != x，并将结果赋给body
body = CompareSelect::make(x, x, kNE);

// 对表达式进行简化，并执行一个假设条件
TookFalseBranch(IRSimplifier::simplify(body));
TEST(Simplify, SimplifyEliminateZeroLengthFor) {
  {
    // Will eliminate zero loop For.

    // 创建名为a的缓冲区，大小为4，类型为整数
    BufHandle a("A", {4}, kInt);
    // 创建名为c的缓冲区，大小为4，类型为整数
    BufHandle c("C", {4}, kInt);
    // 创建名为i的变量，类型为整数
    VarHandle i("i", kInt);
    // 创建一个For循环，循环范围是从0到0，循环体为将a[i]的值存入c[i]
    auto body = For::make(i, 0, 0, Store::make(c, {i}, Load::make(a, {i})));
    // 对循环进行简化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为Block类型
    BlockPtr block = to<Block>(simplified);
    // 断言Block中语句数量为0
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // still works if start is not zero.

    // 创建名为a的缓冲区，大小为4，类型为整数
    BufHandle a("A", {4}, kInt);
    // 创建名为c的缓冲区，大小为4，类型为整数
    BufHandle c("C", {4}, kInt);
    // 创建名为i的变量，类型为整数
    VarHandle i("i", kInt);
    // 创建一个For循环，循环范围是从2到2，循环体为将a[i]的值存入c[i]
    auto body = For::make(i, 2, 2, Store::make(c, {i}, Load::make(a, {i})));
    // 对循环进行简化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为Block类型
    BlockPtr block = to<Block>(simplified);
    // 断言Block中语句数量为0
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // works if both terms are variable.

    // 创建名为x的变量，类型为整数
    VarHandle x("x", kInt);
    // 创建名为a的缓冲区，大小为4，类型为整数
    BufHandle a("A", {4}, kInt);
    // 创建名为c的缓冲区，大小为4，类型为整数
    BufHandle c("C", {4}, kInt);
    // 创建名为i的变量，类型为整数
    VarHandle i("i", kInt);
    // 创建一个For循环，循环范围是从x到x，循环体为将a[i]的值存入c[i]
    auto body = For::make(i, x, x, Store::make(c, {i}, Load::make(a, {i})));
    // 对循环进行简化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为Block类型
    BlockPtr block = to<Block>(simplified);
    // 断言Block中语句数量为0
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // works if one term simplifies down.

    // 创建名为x的变量，类型为整数
    VarHandle x("x", kInt);
    // 创建名为a的缓冲区，大小为4，类型为整数
    BufHandle a("A", {4}, kInt);
    // 创建名为c的缓冲区，大小为4，类型为整数
    BufHandle c("C", {4}, kInt);
    // 创建名为i的变量，类型为整数
    VarHandle i("i", kInt);
    // 创建一个For循环，循环范围是从0到x-x，循环体为将a[i]的值存入c[i]
    auto body = For::make(i, 0, x - x, Store::make(c, {i}, Load::make(a, {i})));
    // 对循环进行简化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为Block类型
    BlockPtr block = to<Block>(simplified);
    // 断言Block中语句数量为0
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // Sanity check does nothing if the condition is not met.

    // 创建名为a的缓冲区，大小为4，类型为整数
    BufHandle a("A", {4}, kInt);
    // 创建名为c的缓冲区，大小为4，类型为整数
    BufHandle c("C", {4}, kInt);
    // 创建名为i的变量，类型为整数
    VarHandle i("i", kInt);
    // 创建一个普通的For循环，循环范围是从0到3，循环体为将a[i]的值存入c[i]
    auto body = For::make(i, 0, 3, Store::make(c, {i}, Load::make(a, {i})));
    // 对循环进行简化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句依然是For循环类型
    IS_NODE(For, simplified);
  }
}
    VarHandle i("i", kInt);
    auto body = For::make(i, x, x + 1, Store::make(c, {i}, Load::make(a, {i})));
    StmtPtr simplified = IRSimplifier::simplify(body);
    BlockPtr block = to<Block>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_VAR_WITH_NAME(store->flat_index(), "x");
    
    
    
    // 定义整型变量 i
    VarHandle i("i", kInt);
    // 创建一个 for 循环，循环体内是一个存储操作，存储 a[i] 到 c[i] 的值
    auto body = For::make(i, x, x + 1, Store::make(c, {i}, Load::make(a, {i})));
    // 简化 IR（中间表示），以便进一步分析和优化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为 BlockPtr 类型
    BlockPtr block = to<Block>(simplified);
    // 检查块的第一个语句是否是 Store 类型，并将其命名为 store
    IS_NODE_WITH_NAME(Store, block->front(), store);
    // 检查 store 操作的基本句柄是否为 "C"
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    // 检查 store 操作的平坦索引是否为 "x"
    IS_VAR_WITH_NAME(store->flat_index(), "x");
    
    
    
    // works if one term simplifies down.
    VarHandle x("x", kInt);
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body =
        For::make(i, 0, x - x + 1, Store::make(c, {i}, Load::make(a, {i})));
    StmtPtr simplified = IRSimplifier::simplify(body);
    BlockPtr block = to<Block>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
    
    
    
    // 如果一个项简化为零，则运行此段代码。
    VarHandle x("x", kInt);
    // 定义一个大小为 4 的整型数组 A
    BufHandle a("A", {4}, kInt);
    // 定义一个大小为 4 的整型数组 C
    BufHandle c("C", {4}, kInt);
    // 定义整型变量 i
    VarHandle i("i", kInt);
    // 创建一个 for 循环，循环体内是一个存储操作，存储 a[i] 到 c[i] 的值
    auto body =
        For::make(i, 0, x - x + 1, Store::make(c, {i}, Load::make(a, {i})));
    // 简化 IR（中间表示），以便进一步分析和优化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 将简化后的语句块转换为 BlockPtr 类型
    BlockPtr block = to<Block>(simplified);
    // 检查块的第一个语句是否是 Store 类型，并将其命名为 store
    IS_NODE_WITH_NAME(Store, block->front(), store);
    // 检查 store 操作的基本句柄是否为 "C"
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    // 检查 store 操作的平坦索引是否为常量 0
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
    
    
    
    // Sanity check does nothing if the condition is not met.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    // 创建一个 for 循环，循环次数为 3，循环体内是一个存储操作，存储 a[i] 到 c[i] 的值
    auto body = For::make(i, 0, 3, Store::make(c, {i}, Load::make(a, {i})));
    // 简化 IR（中间表示），以便进一步分析和优化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 检查简化后的语句是否为 For 类型
    IS_NODE(For, simplified);
    
    
    
    // 如果条件未满足，则进行健全性检查。
    // 定义一个大小为 4 的整型数组 A
    BufHandle a("A", {4}, kInt);
    // 定义一个大小为 4 的整型数组 C
    BufHandle c("C", {4}, kInt);
    // 定义整型变量 i
    VarHandle i("i", kInt);
    // 创建一个 for 循环，循环次数为 3，循环体内是一个存储操作，存储 a[i] 到 c[i] 的值
    auto body = For::make(i, 0, 3, Store::make(c, {i}, Load::make(a, {i})));
    // 简化 IR（中间表示），以便进一步分析和优化
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 检查简化后的语句是否为 For 类型
    IS_NODE(For, simplified);
}

TEST(Simplify, SimplifyForWontLoseLoopOptions) {
  {
    // 创建一个名为 a 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle a("A", {4}, kInt);
    // 创建一个名为 c 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle c("C", {4}, kInt);
    // 创建一个名为 i 的变量，数据类型为 kInt
    VarHandle i("i", kInt);
    // 创建一个循环选项对象
    LoopOptions options;
    // 设置 GPU 块索引为 LoopOptions::IDX_W
    options.set_gpu_block_index(LoopOptions::IDX_W);
    // 创建一个循环体，循环变量为 i，起始值为 0，结束值为 1，循环体内为将 a[i] 的值存储到 c[i] 中
    auto body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})), options);
    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 断言简化后的结果为 For 节点
    IS_NODE_WITH_NAME(For, simplified, for_);
    // 获取简化后的循环选项
    LoopOptions options2 = for_->loop_options();
    // 断言简化前后的 GPU 块索引相同
    ASSERT_EQ(options.gpu_block_index(), options2.gpu_block_index());
  }
}

TEST(Simplify, SimplifyMultilevelFor) {
  {
    // 多层循环将被简化
    // 创建一个名为 a 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle a("A", {4}, kInt);
    // 创建一个名为 c 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle c("C", {4}, kInt);
    // 创建一个名为 i 的变量，数据类型为 kInt
    VarHandle i("i", kInt);
    // 创建一个名为 j 的变量，数据类型为 kInt
    VarHandle j("j", kInt);
    // 创建一个循环体，循环变量为 i，起始值为 0，结束值为 1，循环体内为将 a[i] 的值存储到 c[i] 中
    auto body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})));
    // 创建一个外层循环，循环变量为 j，起始值为 0，结束值为 1，循环体为 body
    auto outer = For::make(j, 0, 1, body);
    // 简化外层循环
    StmtPtr simplified = IRSimplifier::simplify(outer);
    // 将简化后的结果转换为 Block 节点
    BlockPtr block = to<Block>(simplified);
    // 断言简化后的结果为 Store 节点
    IS_NODE_WITH_NAME(Store, block->front(), store);
    // 断言 Store 节点的 base_handle 为 "C"
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    // 断言 Store 节点的 flat_index 为 0
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // 如果内部循环被消除，将保留外部循环
    // 创建一个名为 a 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle a("A", {4}, kInt);
    // 创建一个名为 c 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle c("C", {4}, kInt);
    // 创建一个名为 i 的变量，数据类型为 kInt
    VarHandle i("i", kInt);
    // 创建一个名为 j 的变量，数据类型为 kInt
    VarHandle j("j", kInt);
    // 创建一个循环体，循环变量为 i，起始值为 0，结束值为 1，循环体内为将 a[i] 的值存储到 c[i] 中
    auto body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})));
    // 创建一个外层循环，循环变量为 j，起始值为 0，结束值为 2，循环体为 body
    auto outer = For::make(j, 0, 2, body);
    // 简化外层循环
    StmtPtr simplified = IRSimplifier::simplify(outer);
    // 将简化后的结果转换为 For 节点
    ForPtr for__ = static_to<For>(simplified);
    // 断言简化后的结果为 For 节点
    IS_NODE_WITH_NAME(For, for__, for_);
    // 断言 For 节点的循环变量为 "j"
    IS_VAR_WITH_NAME(for_->var(), "j");
    // 断言 For 节点的起始值为 0
    IS_IMM_WITH_VAL(Int, for_->start(), 0);
    // 断言 For 节点的结束值为 2
    IS_IMM_WITH_VAL(Int, for_->stop(), 2);
    // 将 For 节点的循环体转换为 Block 节点
    BlockPtr block = to<Block>(for_->body());
    // 断言 Block 节点不为空
    ASSERT_NE(block, nullptr);
    // 断言 Block 节点的第一个节点为 Store 节点
    IS_NODE_WITH_NAME(Store, block->front(), store);
    // 断言 Store 节点的 base_handle 为 "C"
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    // 断言 Store 节点的 flat_index 为 0
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // 如果外部循环被消除，将保留内部循环
    // 创建一个名为 a 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle a("A", {4}, kInt);
    // 创建一个名为 c 的缓冲区，大小为 {4}，数据类型为 kInt
    BufHandle c("C", {4}, kInt);
    // 创建一个名为 i 的变量，数据类型为 kInt
    VarHandle i("i", kInt);
    // 创建一个名为 j 的变量，数据类型为 kInt
    VarHandle j("j", kInt);
    // 创建一个循环体，循环变量为 i，起始值为 0，结束值为 2，循环体内为将 a[i] 的值存储到 c[i] 中
    auto body = For::make(i, 0, 2, Store::make(c, {i}, Load::make(a, {i})));
    // 创建一个外层循环，循环变量为 j，起始值为 0，结束值为 1，循环体为 body
    auto outer = For::make(j, 0, 1, body);
    // 简化外层循环
    StmtPtr simplified = IRSimplifier::simplify(outer);
    // 将简化后的结果转换为 Block 节点
    BlockPtr block = to<Block>(simplified);
    // 断言 Block 节点的第一个节点为 For 节点
    IS_NODE_WITH_NAME(For, block->front(), for_);
    // 断言 For 节点的循环变量为 "i"
    IS_VAR_WITH_NAME(for_->var(), "i");
    // 断言 For 节点的起始值为 0
    IS_IMM_WITH_VAL(Int, for_->start(), 0);
    // 断言 For 节点的结束值为 2
    IS_IMM_WITH_VAL(Int, for_->stop(), 2);
    // 断言 For 节点的循环体的第一个节点为 Store 节点
    IS_NODE_WITH_NAME(Store, for_->body()->front(), store);
    // 断言 Store 节点的 base_handle 为 "C"
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    // 断言 Store 节点的 flat_index 为 "i"
    IS_VAR_WITH_NAME(store->flat_index(), "i");
  }
}

TEST(Simplify, SimplifyForCleansUp) {
  {
    // 创建一个名为 a 的缓冲区，大小为 {1, 12, 1}，数据类型为 kFloat
    BufHandle a("a", {1, 12, 1}, kFloat);
    // 创建一个名为 x 的变量，数据类型为 kInt
    VarHandle x("x", kInt);
    // 调用 Compute 函数创建 Tensor 对象 b，表示一个名为 "x" 的计算，维度为 {1, 12, 1}
    Tensor b = Compute(
        "x",
        {1, 12, 1},
        [](const VarHandle& i, const VarHandle& m, const VarHandle& n) {
          // 计算并返回 i + m + n 的结果作为 Tensor b 的内容
          return i + m + n;
        });

    // 使用 Tensor b 创建 LoopNest 对象 l，用于管理循环嵌套
    LoopNest l({b});

    // 准备 LoopNest 对象 l 以进行代码生成的准备工作
    l.prepareForCodegen();

    // 对 LoopNest 的根语句进行名称清理，返回清理后的语句作为 body
    StmtPtr body = LoopNest::sanitizeNames(l.root_stmt());

    // 对 body 进行简化，返回简化后的语句作为 simplified
    StmtPtr simplified = IRSimplifier::simplify(body);

    // 将 simplified 转换为 BlockPtr 对象 block
    BlockPtr block = to<Block>(simplified);

    // 确保 block 的第一个语句是 For 对象，并将其命名为 for_
    IS_NODE_WITH_NAME(For, block->front(), for_);
    // 确保 for_ 循环是关于变量 "m" 的循环
    // 注：这里的 "m" 是指在 Compute 函数中的参数，不是普通的变量
    IS_VAR_WITH_NAME(for_->var(), "j");

    // 确保 for_ 循环的主体第一个语句是 Store 对象
    IS_NODE_WITH_NAME(Store, for_->body()->front(), store);
    // 确保 store 对象的 flat_index 是变量 "j"
    IS_VAR_WITH_NAME(store->flat_index(), "j");
    // 确保 store 对象的 value 是变量 "j"
    IS_VAR_WITH_NAME(store->value(), "j");
TEST(Simplify, SimplifyEliminateEmptyFor) {
  {
    // Flatten many layers around an empty block to an empty block.
    // 创建一个初始为空的块对象
    StmtPtr last = alloc<Block>(std::vector<StmtPtr>({}));
    // 循环11次，创建For循环，并将上一个循环的结果作为内部语句
    for (const auto i : c10::irange(11)) {
      (void)i; // 抑制未使用变量警告
      // 创建循环变量对象
      VarHandle loopVar("loopVar", kInt);
      // 更新last为新的For循环语句块
      last = For::make(loopVar, 0, 10, last);
    }

    // 对最终的语句块进行简化
    StmtPtr simplified = IRSimplifier::simplify(last);
    // 确保简化后的结果是一个Block类型的节点
    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保简化后的块中没有语句
    ASSERT_EQ(block->nstmts(), 0);
  }
}

TEST(Simplify, SimplifyFlattenBlock) {
  {
    // Flatten multiple blocks down to one.
    // 将多个嵌套块合并成一个块。
    // 创建一个数组变量a
    BufHandle a("A", {1}, kInt);
    // 创建第一个存储语句store1
    StorePtr store1 = Store::make(a, {0}, 1);
    // 创建第二个存储语句store2
    StorePtr store2 = Store::make(a, {0}, 0);

    // 创建包含store1和store2的块block1
    BlockPtr block1 = alloc<Block>(std::vector<StmtPtr>({store1, store2}));
    // 创建包含block1的块block2
    BlockPtr block2 = alloc<Block>(std::vector<StmtPtr>({block1}));

    // 创建包含block2的块enclosing
    BlockPtr enclosing = alloc<Block>(std::vector<StmtPtr>({block2}));
    // 对enclosing进行简化
    StmtPtr simplified = IRSimplifier::simplify(enclosing);

    // 确保简化后的结果是一个Block类型的节点
    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保简化后的块中包含两个语句
    ASSERT_EQ(block->nstmts(), 2);

    // 确保块中第一个语句是store1类型的节点
    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    // 确保块中最后一个语句是store2类型的节点
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    // 确保store1的值与block中的store1_节点的值相等
    ASSERT_EQ(store1->value(), store1_->value());
    // 确保store2的值与block中的store2_节点的值相等
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // Flatten multiple sub blocks containing statements.
    // 将包含语句的多个子块合并成一个块。
    // 创建一个数组变量a
    BufHandle a("A", {1}, kInt);
    // 创建第一个存储语句store1
    StorePtr store1 = Store::make(a, {0}, 1);
    // 创建第二个存储语句store2
    StorePtr store2 = Store::make(a, {0}, 0);

    // 创建只包含store1的块block1
    BlockPtr block1 = alloc<Block>(std::vector<StmtPtr>({store1}));
    // 创建只包含store2的块block2
    BlockPtr block2 = alloc<Block>(std::vector<StmtPtr>({store2}));

    // 创建包含block1和block2的块enclosing
    BlockPtr enclosing = alloc<Block>(std::vector<StmtPtr>({block1, block2}));
    // 对enclosing进行简化
    StmtPtr simplified = IRSimplifier::simplify(enclosing);

    // 确保简化后的结果是一个Block类型的节点
    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保简化后的块中包含两个语句
    ASSERT_EQ(block->nstmts(), 2);

    // 确保块中第一个语句是store1类型的节点
    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    // 确保块中最后一个语句是store2类型的节点
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    // 确保store1的值与block中的store1_节点的值相等
    ASSERT_EQ(store1->value(), store1_->value());
    // 确保store2的值与block中的store2_节点的值相等
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // Flatten sub blocks with different depths.
    // 将不同深度的子块合并成一个块。
    // 创建一个数组变量a
    BufHandle a("A", {1}, kInt);
    // 创建第一个存储语句store1
    StorePtr store1 = Store::make(a, {0}, 1);
    // 创建第二个存储语句store2
    StorePtr store2 = Store::make(a, {0}, 0);

    // 创建只包含store2的块block1
    BlockPtr block1 = alloc<Block>(std::vector<StmtPtr>({store2}));
    // 创建只包含block1的块block2
    BlockPtr block2 = alloc<Block>(std::vector<StmtPtr>({block1}));

    // 创建包含store1和block2的块enclosing
    BlockPtr enclosing = alloc<Block>(std::vector<StmtPtr>({store1, block2}));
    // 对enclosing进行简化
    StmtPtr simplified = IRSimplifier::simplify(enclosing);

    // 确保简化后的结果是一个Block类型的节点
    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保简化后的块中包含两个语句
    ASSERT_EQ(block->nstmts(), 2);

    // 确保块中第一个语句是store1类型的节点
    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    // 确保块中最后一个语句是store2类型的节点
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    // 确保store1的值与block中的store1_节点的值相等
    ASSERT_EQ(store1->value(), store1_->value());
    // 确保store2的值与block中的store2_节点的值相等
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // 创建一个指向空块的指针 `last`
    StmtPtr last = alloc<Block>(std::vector<StmtPtr>({}));

    // 循环11次，每次创建一个新的块并将上一个块作为其唯一的语句
    for (const auto i : c10::irange(11)) {
      (void)i; // 抑制未使用变量的警告

      // 用当前的 `last` 创建一个新的块，并将其作为新的 `last`
      last = alloc<Block>(std::vector<StmtPtr>({last}));
    }

    // 简化最终得到的 `last` 块
    StmtPtr simplified = IRSimplifier::simplify(last);

    // 断言：确保简化后的块确实是一个名为 `Block` 的节点，并且其中语句数为 0
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 0);
TEST(Simplify, SimplifyEliminateZeroLengthAlloc) {
  {
    // Simple positive case.
    // 创建一个名为 b 的 BufHandle 对象，长度为 0，类型为 kInt
    BufHandle b("x", {0}, kInt);

    // 创建一个 AllocatePtr 对象，用于分配 b
    AllocatePtr alloc_ = Allocate::make(b);
    // 创建一个 FreePtr 对象，用于释放 b
    FreePtr free_ = Free::make(b);

    // 创建一个 BlockPtr 对象 block1，包含 alloc_ 和 free_ 两个语句
    BlockPtr block1 = alloc<Block>(std::vector<StmtPtr>({alloc_, free_}));
    // 断言 block1 中语句数量为 2
    ASSERT_EQ(block1->nstmts(), 2);

    // 对 block1 进行简化操作
    StmtPtr simplified = IRSimplifier::simplify(block1);
    // 断言 simplified 是一个 Block，且命名为 block2
    IS_NODE_WITH_NAME(Block, simplified, block2);
    // 断言 block2 中语句数量为 0
    ASSERT_EQ(block2->nstmts(), 0);
  }

  {
    // Simple negative case.
    // 创建一个名为 b 的 BufHandle 对象，长度为 2，类型为 kInt
    BufHandle b("x", {2}, kInt);

    // 创建一个 AllocatePtr 对象，用于分配 b
    AllocatePtr alloc_ = Allocate::make(b);
    // 创建一个 FreePtr 对象，用于释放 b
    FreePtr free_ = Free::make(b);

    // 创建一个 BlockPtr 对象 block1，包含 alloc_ 和 free_ 两个语句
    BlockPtr block1 = alloc<Block>(std::vector<StmtPtr>({alloc_, free_}));
    // 断言 block1 中语句数量为 2
    ASSERT_EQ(block1->nstmts(), 2);

    // 对 block1 进行简化操作
    StmtPtr simplified = IRSimplifier::simplify(block1);
    // 断言 simplified 是一个 Block，且命名为 block2
    IS_NODE_WITH_NAME(Block, simplified, block2);
    // 断言 block2 中语句数量为 2
    ASSERT_EQ(block2->nstmts(), 2);
  }

  {
    // Finds right Alloc/Free.
    // 创建两个名为 b1 和 b2 的 BufHandle 对象，分别为 {0} 和 {2}，类型均为 kInt
    BufHandle b1("x", {0}, kInt);
    BufHandle b2("y", {2}, kInt);

    // 创建两个 AllocatePtr 对象，分别用于分配 b1 和 b2
    AllocatePtr alloc1 = Allocate::make(b1);
    AllocatePtr alloc2 = Allocate::make(b2);
    // 创建两个 FreePtr 对象，分别用于释放 b2 和 b1
    FreePtr free2_ = Free::make(b2);
    FreePtr free1_ = Free::make(b1);

    // 创建一个 BlockPtr 对象 block1，包含 alloc1, alloc2, free2_, free1_ 四个语句
    BlockPtr block1 =
        alloc<Block>(std::vector<StmtPtr>({alloc1, alloc2, free2_, free1_}));
    // 断言 block1 中语句数量为 4
    ASSERT_EQ(block1->nstmts(), 4);

    // 对 block1 进行简化操作
    StmtPtr simplified = IRSimplifier::simplify(block1);
    // 断言 simplified 是一个 Block，且命名为 block2
    IS_NODE_WITH_NAME(Block, simplified, block2);
    // 断言 block2 中语句数量为 2
    ASSERT_EQ(block2->nstmts(), 2);
    // 断言 block2 的第一个语句是一个 Allocate，命名为 simplified_alloc
    IS_NODE_WITH_NAME(Allocate, block2->stmts().front(), simplified_alloc);
    // 断言 simplified_alloc 的 buffer 变量名为 "y"
    IS_VAR_WITH_NAME(simplified_alloc->buffer_var(), "y");
    // 断言 block2 的最后一个语句是一个 Free，命名为 simplified_free
    IS_NODE_WITH_NAME(Free, block2->stmts().back(), simplified_free);
    // 断言 simplified_alloc 和 simplified_free 的 buffer 变量相同
    ASSERT_EQ(simplified_alloc->buffer_var(), simplified_free->buffer_var());
  }

  {
    // Dynamic shape.
    // 创建一个 VarHandle 对象 z，类型为 kInt
    VarHandle z("z", kInt);
    // 创建两个 BufHandle 对象 b1 和 b2，长度分别为 {0} 和 {z}，类型均为 kInt
    BufHandle b1("x", {0}, kInt);
    BufHandle b2("y", {z}, kInt);

    // 创建两个 AllocatePtr 对象，分别用于分配 b1 和 b2
    AllocatePtr alloc1 = Allocate::make(b1);
    AllocatePtr alloc2 = Allocate::make(b2);
    // 创建两个 FreePtr 对象，分别用于释放 b2 和 b1
    FreePtr free2_ = Free::make(b2);
    FreePtr free1_ = Free::make(b1);

    // 创建一个 BlockPtr 对象 block1，包含 alloc1, alloc2, free2_, free1_ 四个语句
    BlockPtr block1 =
        alloc<Block>(std::vector<StmtPtr>({alloc1, alloc2, free2_, free1_}));
    // 断言 block1 中语句数量为 4
    ASSERT_EQ(block1->nstmts(), 4);
    // 对 block1 进行简化操作
    StmtPtr simplified = IRSimplifier::simplify(block1);
    // 断言 simplified 是一个 Block，且命名为 block2
    IS_NODE_WITH_NAME(Block, simplified, block2);
    // 断言 block2 中语句数量为 2
    ASSERT_EQ(block2->nstmts(), 2);
  }
}
    # 创建一个表达式，表示两个随机整数的乘积
    ExprHandle body =
        Intrinsics::make(kRand, kInt) * Intrinsics::make(kRand, kInt);
    # 使用 IRSimplifier 对表达式进行简化
    ExprHandle simplified = IRSimplifier::simplify(body);
    # 检查简化后的表达式是否是乘法（Mul）节点，并将其存储在 mul 变量中
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    # 检查乘法节点的左子树是否为随机数节点
    IS_RAND(mul->lhs());
    # 检查乘法节点的右子树是否为随机数节点
    IS_RAND(mul->rhs());
  }
TEST(Simplify, SimplifyReorderForCond) {
  BufHandle a("A", {4}, kInt);  // 创建一个名为 "A" 的缓冲区，大小为 {4}，数据类型为 kInt
  BufHandle b("B", {1}, kInt);  // 创建一个名为 "B" 的缓冲区，大小为 {1}，数据类型为 kInt
  BufHandle c("C", {4}, kInt);  // 创建一个名为 "C" 的缓冲区，大小为 {4}，数据类型为 kInt
  VarHandle i("i", kInt);       // 创建一个名为 "i" 的变量，数据类型为 kInt
  VarHandle j("j", kInt);       // 创建一个名为 "j" 的变量，数据类型为 kInt

  {
    // for ( if ( ... ) ) => if ( for ( ... ) ).
    // 创建一个循环体，条件是 if ( j < 10 )，如果条件成立则执行将 a[i] 的值存入 c[i] 中，否则为空
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(j, 10, CompareSelectOperation::kLT),
            Store::make(c, {i}, Load::make(a, {i})),
            nullptr));

    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句是一个条件语句
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    // 确保条件语句的 true 分支是一个块
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    // 确保块的第一个语句是一个循环
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Can't reorder if condition is dependent on the loop var.
    // 创建一个循环体，条件是 if ( i == 2 )，如果条件成立则执行将 a[i] 的值存入 c[i] 中，否则为空
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(i, 2, CompareSelectOperation::kEQ),
            Store::make(c, {i}, Load::make(a, {i})),
            nullptr));

    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句是一个循环
    IS_NODE_WITH_NAME(For, simplified, loop);
    // 确保循环体的第一个语句是一个条件语句
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }

  {
    // Can't reorder if condition is dependent on a var that is modified inside
    // the loop.
    // 创建一个循环体，条件是 if ( c[0] < 10 )，如果条件成立则执行将 a[i] 的值存入 c[0] 中，否则为空
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(c, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句是一个循环
    IS_NODE_WITH_NAME(For, simplified, loop);
    // 确保循环体的第一个语句是一个条件语句
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }

  {
    // Condition based on buffer not referenced in body. Can reorder here.
    // 创建一个循环体，条件是 if ( b[0] < 10 )，如果条件成立则执行将 a[i] 的值存入 c[0] 中，否则为空
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(b, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句是一个条件语句
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    // 确保条件语句的 true 分支是一个块
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    // 确保块的第一个语句是一个循环
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Condition based on buffer read only in body. Can reorder here.
    // 创建一个循环体，条件是 if ( a[0] < 10 )，如果条件成立则执行将 a[i] 的值存入 c[0] 中，否则为空
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(a, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    // 简化循环体
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 确保简化后的语句是一个条件语句
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    // 确保条件语句的 true 分支是一个块
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    // 确保块的第一个语句是一个循环
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Condition depends on Let in the loop. Cannot reorder.
    // 创建一个循环体，条件是 if ( let in loop )，无法重新排序
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                /* Load::make( ... ) */, 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));
    {
        // 创建一个 For 循环语句，循环变量 i 从 0 到 3
        auto body = For::make(
            i,
            0,
            4,
            // 循环体是一个 Block，包含一个 Let 语句和一个 Cond 语句
            Block::make(
                {Let::make(j, 3), // 定义一个变量 j，赋值为 3
                 Cond::make( // 条件语句，根据 j 是否小于 10 进行分支
                     CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                     // 如果 j 小于 10，则执行 Store 操作
                     Store::make(c, {0}, Load::make(a, {i})),
                     // 否则执行空操作
                     nullptr)}));
    
        // 对 body 进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 确保简化后的语句是一个 For 循环
        IS_NODE_WITH_NAME(For, simplified, loop);
        // 确保 loop 的第一个子语句是一个 Let 语句
        IS_NODE_WITH_NAME(Let, loop->body()->front(), let);
        // 确保 loop 的最后一个子语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, loop->body()->back(), cond);
      }
    
      {
        // 多层级的条件语句，所有条件都是独立的。将两个 Cond 语句移出循环外部。
        auto body = For::make(
            i,
            0,
            4,
            // 循环体是一个 Cond 语句，包含两个内部的 Cond 分支
            Cond::make(
                CompareSelect::make(
                    Load::make(a, {0}), 10, CompareSelectOperation::kLT),
                // 第一个 Cond 分支
                Cond::make(
                    CompareSelect::make(j, 10, CompareSelectOperation::kEQ),
                    Store::make(c, {0}, Load::make(a, {i})),
                    nullptr),
                nullptr));
    
        // 对 body 进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 确保简化后的语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, simplified, cond);
        // 确保 cond 的 true 分支是一个 Block
        IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
        // 确保 true_block 的第一个子语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, true_block->front(), cond2);
        // 确保 cond2 的 true 分支是一个 Block
        IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_block2);
        // 确保 true_block2 的第一个子语句是一个 For 循环
        IS_NODE_WITH_NAME(For, true_block2->front(), loop);
      }
    
      {
        // 多层级的条件语句，内部条件依赖于循环变量，只重新排序第一个 Cond 语句。
        auto body = For::make(
            i,
            0,
            4,
            // 循环体是一个 Cond 语句，包含一个内部的 Cond 分支
            Cond::make(
                CompareSelect::make(
                    Load::make(a, {0}), 10, CompareSelectOperation::kLT),
                // 第一个 Cond 分支
                Cond::make(
                    CompareSelect::make(i, 3, CompareSelectOperation::kEQ),
                    Store::make(c, {0}, Load::make(a, {i})),
                    nullptr),
                nullptr));
    
        // 对 body 进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 确保简化后的语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, simplified, cond);
        // 确保 cond 的 true 分支是一个 Block
        IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
        // 确保 true_block 的第一个子语句是一个 For 循环
        IS_NODE_WITH_NAME(For, true_block->front(), loop);
        // 确保 loop 的 body 是一个 Block
        IS_NODE_WITH_NAME(Block, loop->body(), loop_body);
        // 确保 loop_body 的第一个子语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, loop_body->front(), cond2);
      }
    
      {
        // 如果 Cond 语句有一个 else 分支，则不进行重新排序。
        // 虽然可以这样做，但是否更好需要斟酌。
        auto body = For::make(
            i,
            0,
            4,
            // 循环体是一个 Cond 语句，包含一个条件分支和一个 else 分支
            Cond::make(
                CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                // 条件分支
                Store::make(c, {0}, Load::make(a, {i})),
                // else 分支
                Store::make(c, {0}, 0)));
    
        // 对 body 进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 确保简化后的语句是一个 For 循环
        IS_NODE_WITH_NAME(For, simplified, loop);
        // 确保 loop 的第一个子语句是一个 Cond 语句
        IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
      }
    
      {
        // 条件语句使用 Tensor 的不同区域。
        // 虽然可以进行更好的分析来重新排序，但此处不进行。包含以供完整性。
    
        // 创建一个 For 循环语句，循环变量 i 从 0 到 3
        auto body = For::make(
            i,
            0,
            4,
            // 循环体是一个 Cond 语句，包含一个条件分支
            Cond::make(
                CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                // 如果条件满足，执行 Store 操作
                Store::make(c, {0}, Load::make(a, {i})),
                // 否则执行 Store 操作
                Store::make(c, {0}, 0)));
    
        // 对 body 进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 确保简化后的语句是一个 For 循环
        IS_NODE_WITH_NAME(For, simplified, loop);
    // 创建一个名为 `body` 的自动变量，表示一个 `For` 循环结构
    auto body = For::make(
        // 循环变量 `i` 初始化为 0
        i,
        // 循环起始值为 0
        0,
        // 循环终止值为 4
        4,
        // 循环体条件，若 `c` 中索引为 0 的值小于 10，则执行存储操作，否则为空操作
        Cond::make(
            // 创建一个比较选择操作，比较 `c` 中索引为 0 的值是否小于 10
            CompareSelect::make(
                // 加载 `c` 中索引为 0 的值
                Load::make(c, {0}), 10, CompareSelectOperation::kLT),
            // 若条件成立，执行存储操作：将 `a` 中索引为 `i` 的值存储到 `c` 的索引为 1 的位置
            Store::make(c, {1}, Load::make(a, {i})),
            // 条件不成立时的空操作，使用 `nullptr` 表示
            nullptr));

    // 对 `body` 进行简化，得到简化后的语句 `simplified`
    StmtPtr simplified = IRSimplifier::simplify(body);

    // 判断 `simplified` 是否为 `For` 节点，并将其赋值给 `loop`
    IS_NODE_WITH_NAME(For, simplified, loop);

    // 获取 `loop` 的第一个子节点，并判断是否为 `Cond` 节点，将其赋值给 `cond`
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
}


这段代码是对一个循环结构进行操作，其中包含了条件判断和存储操作，通过注释详细解释了每一步的含义和作用。
}

TEST(Simplify, SimplifyFuseConditions) {
  BufHandle a("A", {2}, kInt);  // 创建名为"A"的缓冲区句柄a，大小为{2}，类型为kInt
  BufHandle b("B", {2}, kInt);  // 创建名为"B"的缓冲区句柄b，大小为{2}，类型为kInt
  VarHandle i("i", kInt);       // 创建名为"i"的变量句柄i，类型为kInt
  VarHandle j("j", kInt);       // 创建名为"j"的变量句柄j，类型为kInt

  {
    // 可以融合因为条件相同。
    // if (A) { X }; if (A) { Y }; => if (A) { X; Y }
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),  // 创建比较操作，如果i < 10为真
             Store::make(a, {0}, i),  // 创建将i存储到a[0]的操作
             nullptr),  // 第一个条件为nullptr，即没有假分支
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),  // 创建比较操作，如果i < 10为真
             Store::make(a, {1}, i),  // 创建将i存储到a[1]的操作
             nullptr)});  // 第二个条件为nullptr，即没有假分支

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化IR树
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    IS_NODE_WITH_NAME(Block, simplified, block);  // 断言简化后的语句块是Block类型，并且命名为block
    ASSERT_EQ(block->nstmts(), 1);  // 断言block中有一个语句
    IS_NODE_WITH_NAME(Cond, block->front(), cond);  // 断言block的第一个语句是Cond类型，并且命名为cond
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);  // 断言cond的真分支是Block类型，并且命名为true_stmt
    ASSERT_EQ(true_stmt->nstmts(), 2);  // 断言true_stmt中有两个语句
    ASSERT_EQ(cond->false_stmt(), nullptr);  // 断言cond的假分支为空
  }

  {
    // 不能融合，因为左侧条件不相同（i != j）。
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),  // 创建比较操作，如果i < 10为真
             Store::make(a, {0}, i),  // 创建将i存储到a[0]的操作
             nullptr),  // 第一个条件为nullptr，即没有假分支
         Cond::make(
             CompareSelect::make(j, 10, CompareSelectOperation::kLT),  // 创建比较操作，如果j < 10为真
             Store::make(a, {1}, i),  // 创建将i存储到a[1]的操作
             nullptr)});  // 第二个条件为nullptr，即没有假分支

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化IR树
    IS_NODE_WITH_NAME(Block, simplified, block);  // 断言简化后的语句块是Block类型，并且命名为block
    ASSERT_EQ(block->nstmts(), 2);  // 断言block中有两个语句
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);  // 断言block的第一个语句是Cond类型，并且命名为cond1
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);  // 断言block的最后一个语句是Cond类型，并且命名为cond2

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);  // 断言cond1的真分支是Block类型，并且命名为true_stmt1
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);  // 断言cond2的真分支是Block类型，并且命名为true_stmt2
    ASSERT_EQ(true_stmt1->nstmts(), 1);  // 断言true_stmt1中有一个语句
    ASSERT_EQ(true_stmt2->nstmts(), 1);  // 断言true_stmt2中有一个语句

    ASSERT_EQ(cond1->false_stmt(), nullptr);  // 断言cond1的假分支为空
    ASSERT_EQ(cond2->false_stmt(), nullptr);  // 断言cond2的假分支为空
  }
  {
    // 不能融合，因为右侧条件不相同（10 != 11）。
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),  // 创建比较操作，如果i < 10为真
             Store::make(a, {0}, i),  // 创建将i存储到a[0]的操作
             nullptr),  // 第一个条件为nullptr，即没有假分支
         Cond::make(
             CompareSelect::make(i, 11, CompareSelectOperation::kLT),  // 创建比较操作，如果i < 11为真
             Store::make(a, {1}, i),  // 创建将i存储到a[1]的操作
             nullptr)});  // 第二个条件为nullptr，即没有假分支

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化IR树
    IS_NODE_WITH_NAME(Block, simplified, block);  // 断言简化后的语句块是Block类型，并且命名为block
    ASSERT_EQ(block->nstmts(), 2);  // 断言block中有两个语句
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);  // 断言block的第一个语句是Cond类型，并且命名为cond1
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);  // 断言block的最后一个语句是Cond类型，并且命名为cond2

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);  // 断言cond1的真分支是Block类型，并且命名为true_stmt1
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);  // 断言cond2的真分支是Block类型，并且命名为true_stmt2
    ASSERT_EQ(true_stmt1->nstmts(), 1);  // 断言true_stmt1中有一个语句
    ASSERT_EQ(true_stmt2->nstmts(), 1);  // 断言true_stmt2中有一个语句

    ASSERT_EQ(cond1->false_stmt(), nullptr);  // 断言cond1的假分支为空
    ASSERT_EQ(cond2->false_stmt(), nullptr);  // 断言cond2的假分支为空
  }

  {
    // 不能融合，因为操作不同（LT vs GT）。
    {
        // 创建一个包含两个条件的代码块
        auto body = Block::make(
            {Cond::make(
                 CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                 Store::make(a, {0}, i),  // 如果 i 小于 10，则将 i 存储到数组 a 的第一个位置
                 nullptr),  // 如果条件不满足，无需执行任何操作
             Cond::make(
                 CompareSelect::make(i, 10, CompareSelectOperation::kGT),
                 Store::make(a, {1}, i),  // 如果 i 大于 10，则将 i 存储到数组 a 的第二个位置
                 nullptr)});  // 如果条件不满足，无需执行任何操作
    
        // 简化代码块中的语句
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 断言简化后的代码块确实是 Block 类型，并且赋值给 block 变量
        IS_NODE_WITH_NAME(Block, simplified, block);
        // 断言 block 中包含两个语句
        ASSERT_EQ(block->nstmts(), 2);
        // 断言 block 的第一个语句是 Cond 类型，并且赋值给 cond1 变量
        IS_NODE_WITH_NAME(Cond, block->front(), cond1);
        // 断言 block 的最后一个语句是 Cond 类型，并且赋值给 cond2 变量
        IS_NODE_WITH_NAME(Cond, block->back(), cond2);
    
        // 断言 cond1 的 true 分支是一个 Block，并且赋值给 true_stmt1 变量
        IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
        // 断言 cond2 的 true 分支是一个 Block，并且赋值给 true_stmt2 变量
        IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
        // 断言 true_stmt1 包含一条语句
        ASSERT_EQ(true_stmt1->nstmts(), 1);
        // 断言 true_stmt2 包含一条语句
        ASSERT_EQ(true_stmt2->nstmts(), 1);
    
        // 断言 cond1 的 false 分支为空
        ASSERT_EQ(cond1->false_stmt(), nullptr);
        // 断言 cond2 的 false 分支为空
        ASSERT_EQ(cond2->false_stmt(), nullptr);
    }
    
    {
        // 无法合并，因为 CompareSelect 结果不同。
        // 实际上，如果我们规范化 CompareSelect 结果，我们完全可以合并，但这是后续的 TODO。
        auto body = Block::make(
            {Cond::make(
                 CompareSelect::make(i, 10, 1, 0, CompareSelectOperation::kLT),
                 Store::make(a, {0}, i),
                 nullptr),
             Cond::make(
                 CompareSelect::make(j, 10, 2, 0, CompareSelectOperation::kLT),
                 Store::make(a, {1}, i),
                 nullptr)});
    
        // 简化代码块中的语句
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 断言简化后的代码块确实是 Block 类型，并且赋值给 block 变量
        IS_NODE_WITH_NAME(Block, simplified, block);
        // 断言 block 中包含两个语句
        ASSERT_EQ(block->nstmts(), 2);
        // 断言 block 的第一个语句是 Cond 类型，并且赋值给 cond1 变量
        IS_NODE_WITH_NAME(Cond, block->front(), cond1);
        // 断言 block 的最后一个语句是 Cond 类型，并且赋值给 cond2 变量
        IS_NODE_WITH_NAME(Cond, block->back(), cond2);
    
        // 断言 cond1 的 true 分支是一个 Block，并且赋值给 true_stmt1 变量
        IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
        // 断言 cond2 的 true 分支是一个 Block，并且赋值给 true_stmt2 变量
        IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
        // 断言 true_stmt1 包含一条语句
        ASSERT_EQ(true_stmt1->nstmts(), 1);
        // 断言 true_stmt2 包含一条语句
        ASSERT_EQ(true_stmt2->nstmts(), 1);
    
        // 断言 cond1 的 false 分支为空
        ASSERT_EQ(cond1->false_stmt(), nullptr);
        // 断言 cond2 的 false 分支为空
        ASSERT_EQ(cond2->false_stmt(), nullptr);
    }
    
    {
        // 只能与 false 分支合并。
        auto body = Block::make(
            {Cond::make(
                 CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                 nullptr,
                 Store::make(a, {0}, i)),  // 如果 i 小于 10，则将 i 存储到数组 a 的第一个位置
             Cond::make(
                 CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                 nullptr,
                 Store::make(a, {1}, i))});  // 如果 i 小于 10，则将 i 存储到数组 a 的第二个位置
    
        // 简化代码块中的语句
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 断言简化后的代码块确实是 Block 类型，并且赋值给 block 变量
        IS_NODE_WITH_NAME(Block, simplified, block);
        // 断言 block 中只包含一条语句
        ASSERT_EQ(block->nstmts(), 1);
        // 断言 block 的唯一语句是 Cond 类型，并且赋值给 cond 变量
        IS_NODE_WITH_NAME(Cond, block->front(), cond);
        // 断言 cond 的 false 分支是一个 Block，并且赋值给 false_stmt 变量
        IS_NODE_WITH_NAME(Block, cond->false_stmt(), false_stmt);
        // 断言 false_stmt 包含两条语句
        ASSERT_EQ(false_stmt->nstmts(), 2);
        // 断言 cond 的 true 分支为空
        ASSERT_EQ(cond->true_stmt(), nullptr);
    }
    // 创建一个包含多个语句的块 `body`
    auto body = Block::make(
        {
            // 创建一个条件语句 `Cond`，如果 `i < 10` 则执行 `Store::make(a, {0}, i)`，否则执行 `Store::make(b, {0}, i)`
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, i),
                Store::make(b, {0}, i)),
            // 创建另一个条件语句 `Cond`，如果 `i < 10` 则执行 `Store::make(a, {1}, i)`，否则执行 `Store::make(b, {1}, i)`
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {1}, i),
                Store::make(b, {1}, i))
        });

    // 简化 `body` 中的IR（中间表示），返回简化后的语句 `simplified`
    StmtPtr simplified = IRSimplifier::simplify(body);

    // 确保 `simplified` 是一个类型为 `Block` 的节点，并将其命名为 `block`
    IS_NODE_WITH_NAME(Block, simplified, block);

    // 断言 `block` 中语句的数量为 1
    ASSERT_EQ(block->nstmts(), 1);

    // 从 `block` 中找到第一个语句，并确保它是类型为 `Cond` 的节点，并将其命名为 `cond`
    IS_NODE_WITH_NAME(Cond, block->front(), cond);

    // 获取 `cond` 的 true 分支，确保其是一个类型为 `Block` 的节点，并将其命名为 `true_stmt`
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);

    // 断言 `true_stmt` 中语句的数量为 2
    ASSERT_EQ(true_stmt->nstmts(), 2);

    // 获取 `cond` 的 false 分支，确保其是一个类型为 `Block` 的节点，并将其命名为 `false_stmt`
    IS_NODE_WITH_NAME(Block, cond->false_stmt(), false_stmt);

    // 断言 `false_stmt` 中语句的数量为 2
    ASSERT_EQ(false_stmt->nstmts(), 2);
}

{
    // 可以与存在不匹配的 true / false 分支合并
    auto body = Block::make(
        {
            // 创建一个条件语句 `Cond`，如果 `i < 10` 则执行 `Store::make(a, {0}, i)`，否则执行 `nullptr`
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, i),
                nullptr),
            // 创建另一个条件语句 `Cond`，如果 `i < 10` 则执行 `nullptr`，否则执行 `Store::make(b, {1}, i)`
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                nullptr,
                Store::make(b, {1}, i))
        });

    // 简化 `body` 中的IR（中间表示），返回简化后的语句 `simplified`
    StmtPtr simplified = IRSimplifier::simplify(body);

    // 确保 `simplified` 是一个类型为 `Block` 的节点，并将其命名为 `block`
    IS_NODE_WITH_NAME(Block, simplified, block);

    // 断言 `block` 中语句的数量为 1
    ASSERT_EQ(block->nstmts(), 1);

    // 从 `block` 中找到第一个语句，并确保它是类型为 `Cond` 的节点，并将其命名为 `cond`
    IS_NODE_WITH_NAME(Cond, block->front(), cond);

    // 获取 `cond` 的 true 分支，确保其是一个类型为 `Block` 的节点，并将其命名为 `true_stmt`
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);

    // 断言 `true_stmt` 中语句的数量为 1
    ASSERT_EQ(true_stmt->nstmts(), 1);

    // 获取 `cond` 的 false 分支，确保其是一个类型为 `Block` 的节点，并将其命名为 `false_stmt`
    IS_NODE_WITH_NAME(Block, cond->false_stmt(), false_stmt);

    // 断言 `false_stmt` 为 `nullptr`
    ASSERT_EQ(false_stmt, nullptr);
}

{
    // 可以融合部分块内容，例如在融合之前和之后存在非融合语句的情况
    // before:
    // if (j < 10) { A[0] = j; }
    // if (i < 10) { A[0] = i; }
    // if (i < 10) { A[1] = i; }
    // if (i < 11) { A[1] = j; }
    //
    // after:
    // if (j < 10) { A[0] = j; }
    // if (i < 10) {
    //   A[0] = i;
    //   A[1] = i;
    // }
    // if (i < 11) { A[1] = j; }

    auto body = Block::make({
        // 创建一个条件语句 `Cond`，如果 `j < 10` 则执行 `Store::make(a, {0}, j)`，否则执行 `nullptr`
        Cond::make(
            CompareSelect::make(j, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, j),
            nullptr),
        // 创建一个条件语句 `Cond`，如果 `i < 10` 则执行 `Store::make(a, {0}, i)` 和 `Store::make(a, {1}, i)`，否则执行 `nullptr`
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, i),
            nullptr),
        // 创建一个条件语句 `Cond`，如果 `i < 10` 则执行 `Store::make(a, {1}, i)`，否则执行 `nullptr`
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, i),
            nullptr),
        // 创建一个条件语句 `Cond`，如果 `i < 11` 则执行 `Store::make(a, {1}, j)`，否则执行 `nullptr`
        Cond::make(
            CompareSelect::make(i, 11, CompareSelectOperation::kLT),
            Store::make(a, {1}, j),
            nullptr),
    });

    // 简化 `body` 中的IR（中间表示），返回简化后的语句 `simplified`
    StmtPtr simplified = IRSimplifier::simplify(body);

    // 确保 `simplified` 是一个类型为 `Block` 的节点，并将其命名为 `block`
    IS_NODE_WITH_NAME(Block, simplified, block);

    // 断言 `block` 中语句的数量为 3
    ASSERT_EQ(block->nstmts(), 3);

    // 获取 `block` 的迭代器，并移动到下一个位置
    auto it = block->begin();
    it++;

    // 确保迭代器指向的语句是一个类型为 `Cond` 的节点，并将其命名为 `cond`
    IS_NODE_WITH_NAME(Cond, *it, cond);

    // 获取 `cond` 的 true 分支，确保其是一个类型为 `Block` 的节点，并将其命名为 `true_stmt`
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);

    // 断言 `true_stmt` 中语句的数量为 2
    ASSERT_EQ(true_stmt->nstmts(), 2);

    // 确保 `cond` 的 false 分支为 `nullptr`
    ASSERT_EQ(cond->false_stmt(), nullptr);
}
    {
        // 可以将相同条件的较长序列融合在一起。
        auto body = Block::make({
            // 创建条件语句：如果 i 小于 10，则将 j 存储到数组 a 的位置 {0} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, j),
                nullptr),
            // 创建条件语句：如果 i 小于 10，则将 i 存储到数组 a 的位置 {0} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, i),
                nullptr),
            // 创建条件语句：如果 i 小于 10，则将 i 存储到数组 a 的位置 {1} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {1}, i),
                nullptr),
            // 创建条件语句：如果 i 小于 10，则将 j 存储到数组 a 的位置 {1} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {1}, j),
                nullptr),
        });
        // 对代码块进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 验证简化后的结果是一个名为 Block 的节点，并命名为 block
        IS_NODE_WITH_NAME(Block, simplified, block);
        // 断言 block 包含的语句数为 1
        ASSERT_EQ(block->nstmts(), 1);
        // 验证 block 的第一个语句是一个名为 Cond 的节点，并命名为 cond
        IS_NODE_WITH_NAME(Cond, block->front(), cond);
        // 验证 cond 的 true 分支是一个名为 Block 的节点，并命名为 true_stmt
        IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
        // 断言 true_stmt 包含的语句数为 4
        ASSERT_EQ(true_stmt->nstmts(), 4);
        // 验证 cond 的 false 分支为空
        ASSERT_EQ(cond->false_stmt(), nullptr);
    }
    
    {
        // 无法通过非条件语句进行融合。
        auto body = Block::make({
            // 创建条件语句：如果 i 小于 10，则将 j 存储到数组 a 的位置 {0} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, j),
                nullptr),
            // 创建条件语句：如果 i 小于 10，则将 i 存储到数组 a 的位置 {0} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {0}, i),
                nullptr),
            // 存储语句：将 i + j 存储到数组 b 的位置 {1} 中
            Store::make(b, {1}, i + j),
            // 创建条件语句：如果 i 小于 10，则将 i 存储到数组 a 的位置 {1} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {1}, i),
                nullptr),
            // 创建条件语句：如果 i 小于 10，则将 j 存储到数组 a 的位置 {1} 中，否则为空语句
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kLT),
                Store::make(a, {1}, j),
                nullptr),
        });
        // 对代码块进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
        // 验证简化后的结果是一个名为 Block 的节点，并命名为 block
        IS_NODE_WITH_NAME(Block, simplified, block);
        // 断言 block 包含的语句数为 3
        ASSERT_EQ(block->nstmts(), 3);
        // 验证 block 的第一个语句是一个名为 Cond 的节点，并命名为 cond
        IS_NODE_WITH_NAME(Cond, block->front(), cond);
        // 验证 cond 的 true 分支是一个名为 Block 的节点，并命名为 true_stmt
        IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
        // 断言 true_stmt 包含的语句数为 2
        ASSERT_EQ(true_stmt->nstmts(), 2);
        // 验证 cond 的 false 分支为空
        ASSERT_EQ(cond->false_stmt(), nullptr);
    
        // 验证 block 的最后一个语句是一个名为 Cond 的节点，并命名为 cond2
        IS_NODE_WITH_NAME(Cond, block->back(), cond2);
        // 验证 cond2 的 true 分支是一个名为 Block 的节点，并命名为 true_stmt2
        IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt2);
        // 断言 true_stmt2 包含的语句数为 2
        ASSERT_EQ(true_stmt2->nstmts(), 2);
        // 验证 cond2 的 false 分支为空
        ASSERT_EQ(cond2->false_stmt(), nullptr);
    
        // 使用迭代器检查 block 中的第二个语句，应为一个名为 Store 的节点，并命名为 middle
        auto it = block->begin();
        it++;
        IS_NODE_WITH_NAME(Store, *it, middle);
    }
    
    {
        // 如果条件简化为相同的情况，则可以融合。
        auto body = Block::make(
            // 创建包含两个条件语句的代码块
            {Cond::make(
                 // 创建一个比较选择表达式：如果 i * 2 小于 87 % 11，则将 i 存储到数组 a 的位置 {0} 中，否则为空语句
                 CompareSelect::make(
                     i * 2,
                     ExprHandle(87) % ExprHandle(11),
                     CompareSelectOperation::kLT),
                 Store::make(a, {0}, i),
                 nullptr),
             Cond::make(
                 // 创建一个比较选择表达式：如果 i * 2 小于 300 / 30，则将 i 存储到数组 a 的位置 {1} 中，否则为空语句
                 CompareSelect::make(
                     i * 2,
                     ExprHandle(300) / ExprHandle(30),
                     CompareSelectOperation::kLT),
                 Store::make(a, {1}, i),
                 nullptr)});
        // 对代码块进行简化操作
        StmtPtr simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保 `simplified` 是一个 `Block` 节点，并将其赋给 `block`
    ASSERT_EQ(block->nstmts(), 1);
    // 断言 `block` 中语句的数量为 1

    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    // 确保 `block` 的第一个语句是一个 `Cond` 节点，并将其赋给 `cond`

    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    // 确保 `cond` 的真实分支是一个 `Block` 节点，并将其赋给 `true_stmt`

    ASSERT_EQ(true_stmt->nstmts(), 2);
    // 断言 `true_stmt` 中语句的数量为 2

    ASSERT_EQ(cond->false_stmt(), nullptr);
    // 断言 `cond` 的假分支为空指针
  }

  {
    // 可以合并非比较选择节点。
    // 如果 (i) { X } 如果 (i) { Y } => 如果 (i) { X; Y }
    auto body = Block::make(
        {Cond::make(i, Store::make(a, {0}, i), nullptr),
         Cond::make(i, Store::make(a, {1}, i), nullptr)});

    StmtPtr simplified = IRSimplifier::simplify(body);
    // 简化 `body` 并将结果赋给 `simplified`

    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保 `simplified` 是一个 `Block` 节点，并将其赋给 `block`

    ASSERT_EQ(block->nstmts(), 1);
    // 断言 `block` 中语句的数量为 1

    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    // 确保 `block` 的第一个语句是一个 `Cond` 节点，并将其赋给 `cond`

    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    // 确保 `cond` 的真实分支是一个 `Block` 节点，并将其赋给 `true_stmt`

    ASSERT_EQ(true_stmt->nstmts(), 2);
    // 断言 `true_stmt` 中语句的数量为 2

    ASSERT_EQ(cond->false_stmt(), nullptr);
    // 断言 `cond` 的假分支为空指针
  }

  {
    // 确保当合并不同的非比较选择节点时仍保持合理性检查。
    auto body = Block::make(
        {Cond::make(i, Store::make(a, {0}, i), nullptr),
         Cond::make(j, Store::make(a, {1}, i), nullptr)});

    StmtPtr simplified = IRSimplifier::simplify(body);
    // 简化 `body` 并将结果赋给 `simplified`

    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保 `simplified` 是一个 `Block` 节点，并将其赋给 `block`

    ASSERT_EQ(block->nstmts(), 2);
    // 断言 `block` 中语句的数量为 2

    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    // 确保 `block` 的第一个语句是一个 `Cond` 节点，并将其赋给 `cond1`

    IS_NODE_WITH_NAME(Cond, block->back(), cond2);
    // 确保 `block` 的最后一个语句是一个 `Cond` 节点，并将其赋给 `cond2`
  }

  {
    // 确保在可能合并时仍然进行常量条件消除的合理性检查。
    auto body = Block::make(
        {Cond::make(1, Store::make(a, {0}, i), nullptr),
         Cond::make(1, Store::make(a, {1}, i), nullptr)});
    StmtPtr simplified = IRSimplifier::simplify(body);
    // 简化 `body` 并将结果赋给 `simplified`

    IS_NODE_WITH_NAME(Block, simplified, block);
    // 确保 `simplified` 是一个 `Block` 节点，并将其赋给 `block`

    ASSERT_EQ(block->nstmts(), 2);
    // 断言 `block` 中语句的数量为 2

    IS_NODE_WITH_NAME(Store, block->front(), store1);
    // 确保 `block` 的第一个语句是一个 `Store` 节点，并将其赋给 `store1`

    IS_NODE_WITH_NAME(Store, block->back(), store2);
    // 确保 `block` 的最后一个语句是一个 `Store` 节点，并将其赋给 `store2`
  }

  {
    // 确保在融合后进行 for 循环条件重新排序的合理性检查。
    auto body = For::make(
        i,
        0,
        4,
        Block::make(
            {Cond::make(
                 CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                 Store::make(a, {1}, Load::make(b, {0})),
                 nullptr),
             Cond::make(
                 CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                 Store::make(a, {2}, Load::make(b, {0})),
                 nullptr)}));

    StmtPtr simplified = IRSimplifier::simplify(body);
    // 简化 `body` 并将结果赋给 `simplified`

    IS_NODE_WITH_NAME(Cond, simplified, cond);
    // 确保 `simplified` 是一个 `Cond` 节点，并将其赋给 `cond`

    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    // 确保 `cond` 的真实分支是一个 `Block` 节点，并将其赋给 `true_block`

    IS_NODE_WITH_NAME(For, true_block->front(), loop);
    // 确保 `true_block` 的第一个语句是一个 `For` 循环节点，并将其赋给 `loop`
  }
TEST(Simplify, SimplifySyncThreads) {
  BufHandle a("A", {4}, kInt);  // 创建一个名为 "A" 的缓冲区句柄，包含4个整数，类型为kInt
  VarHandle i("i", kInt);  // 创建一个名为 "i" 的变量句柄，类型为kInt

  {
    // 合并两个内部的SyncThreads。
    auto body = Block::make(
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        {Store::make(a, {0}, 1),  // 在缓冲区a的索引0处存储值1
         alloc<SyncThreads>(),  // 分配一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         Store::make(a, {1}, 0)});  // 在缓冲区a的索引1处存储值0
    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化体(body)的IR并返回
    IS_NODE_WITH_NAME(Block, simplified, block);  // 确保简化后的结果是一个名为Block的节点类型
    ASSERT_EQ(block->nstmts(), 3);  // 断言简化后的块包含3个语句
    auto it = block->begin();  // 获取块的迭代器
    IS_NODE(Store, *it++);  // 确保第一个语句是Store类型
    IS_NODE(SyncThreads, *it++);  // 确保第二个语句是SyncThreads类型
    IS_NODE(Store, *it++);  // 确保第三个语句是Store类型
  }

  {
    // 消除外部的SyncThreads。
    auto body = Block::make(
        {alloc<SyncThreads>(),  // 分配一个SyncThreads对象
         Store::make(a, {1}, 0),  // 在缓冲区a的索引1处存储值0
         alloc<SyncThreads>()});  // 分配另一个SyncThreads对象

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化体(body)的IR并返回
    IS_NODE_WITH_NAME(Block, simplified, block);  // 确保简化后的结果是一个名为Block的节点类型
    ASSERT_EQ(block->nstmts(), 1);  // 断言简化后的块只包含1个语句
    auto it = block->begin();  // 获取块的迭代器
    IS_NODE(Store, *it);  // 确保语句是Store类型
  }

  {
    // 合并多个内部SyncThreads。
    auto body = Block::make(
        {Store::make(a, {0}, 1),  // 在缓冲区a的索引0处存储值1
         alloc<SyncThreads>(),  // 分配一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         Store::make(a, {1}, 0)});  // 在缓冲区a的索引1处存储值0

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化体(body)的IR并返回
    IS_NODE_WITH_NAME(Block, simplified, block);  // 确保简化后的结果是一个名为Block的节点类型
    ASSERT_EQ(block->nstmts(), 3);  // 断言简化后的块包含3个语句
    auto it = block->begin();  // 获取块的迭代器
    IS_NODE(Store, *it++);  // 确保第一个语句是Store类型
    IS_NODE(SyncThreads, *it++);  // 确保第二个语句是SyncThreads类型
    IS_NODE(Store, *it++);  // 确保第三个语句是Store类型
  }

  {
    // 合并多个外部SyncThreads。
    auto body = Block::make(
        {alloc<SyncThreads>(),  // 分配一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         Store::make(a, {1}, 0),  // 在缓冲区a的索引1处存储值0
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>()});  // 分配另一个SyncThreads对象

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化体(body)的IR并返回
    IS_NODE_WITH_NAME(Block, simplified, block);  // 确保简化后的结果是一个名为Block的节点类型
    ASSERT_EQ(block->nstmts(), 1);  // 断言简化后的块只包含1个语句
    auto it = block->begin();  // 获取块的迭代器
    IS_NODE(Store, *it);  // 确保语句是Store类型
  }

  {
    // 合并多个段落;
    auto body = Block::make(
        {Store::make(a, {0}, 1),  // 在缓冲区a的索引0处存储值1
         alloc<SyncThreads>(),  // 分配一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         Store::make(a, {1}, 0),  // 在缓冲区a的索引1处存储值0
         Store::make(a, {2}, 0),  // 在缓冲区a的索引2处存储值0
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         alloc<SyncThreads>(),  // 分配另一个SyncThreads对象
         Store::make(a, {3}, 0)});  // 在缓冲区a的索引3处存储值0

    StmtPtr simplified = IRSimplifier::simplify(body);  // 简化体(body)的IR并返回
    IS_NODE_WITH_NAME(Block, simplified, block);  // 确保简化后的结果是一个名为Block的节点类型
    ASSERT_EQ(block->nstmts(), 6);  // 断言简化后的块包含6个语句
    auto it = block->begin();  // 获取块的迭代器
    IS_NODE(Store, *it++);  // 确保第一个语句是Store类型
    IS_NODE(SyncThreads, *it++);  // 确保第二个语句是SyncThreads类型
    IS_NODE(Store, *it++);  // 确保第三个语句是Store类型
    IS_NODE(Store, *it++);  // 确保第四个语句是Store类型
    IS_NODE(SyncThreads, *it++);  // 确保第五个语句是SyncThreads类型
    IS_NODE(Store, *it);  // 确保第六个语句是Store类型
  }
}
TEST(Simplify, SimplifyRampSubBroadcast) {
  // 定义测试中使用的向量宽度
  int num_lanes = 4;
  // 创建一个 Ramp 表达式，起始为 0，步长为 6，向量宽度为 num_lanes
  ExprHandle ramp = Ramp::make(ExprHandle(0), ExprHandle(6), num_lanes);
  // 创建一个 Broadcast 表达式，值为 -5，向量宽度为 num_lanes
  ExprHandle broadcast = Broadcast::make(ExprHandle(-5), num_lanes);
  // 对 ramp - broadcast 进行简化
  ExprHandle simplified = IRSimplifier::simplify(ramp - broadcast);
  // 将简化后的表达式转换为 Ramp 类型
  RampPtr newRamp = simplified.AsNode<Ramp>();
  // 确认 newRamp 的 base 是 IntImm 类型，并获取其值
  IS_NODE_WITH_NAME(IntImm, newRamp->base(), base);
  // 断言 base 的值为 5
  ASSERT_EQ(base->value(), 5);
  // 确认 newRamp 的 stride 是 IntImm 类型，并获取其值
  IS_NODE_WITH_NAME(IntImm, newRamp->stride(), stride);
  // 断言 stride 的值为 6
  ASSERT_EQ(stride->value(), 6);
  // 断言 newRamp 的向量宽度与 num_lanes 相等
  ASSERT_EQ(newRamp->lanes(), num_lanes);
}

TEST(Simplify, SimplifyBroadcastTermExpander) {
  // 定义测试中使用的向量宽度
  int num_lanes = 8;
  // 创建三个 Broadcast 表达式，分别为 0, 1, 2，向量宽度为 num_lanes
  ExprHandle bc0 = Broadcast::make(ExprHandle(0), num_lanes);
  ExprHandle bc1 = Broadcast::make(ExprHandle(1), num_lanes);
  ExprHandle bc2 = Broadcast::make(ExprHandle(2), num_lanes);
  // 执行 IRSimplifier 简化操作，触发 TermExpander::mutate 的相关路径
  // bc1 + (bc0 / bc2) + bc1 不完全简化，观察其值
  ExprHandle simplified = IRSimplifier::simplify(bc1 + (bc0 / bc2) + bc1);
  // 创建一个大小为 num_lanes 的 BufHandle 对象
  BufHandle buf("buf", {num_lanes}, kInt);
  // 创建一个 Store 表达式，存储 simplified 到 buf[0:num_lanes-1] 中
  auto store = Store::make(buf, {Ramp::make(0, 1, num_lanes)}, simplified);
  // 创建 SimpleIREvaluator 对象 eval，用于评估 store 的结果
  SimpleIREvaluator eval(store, {buf});
  // 创建一个大小为 num_lanes 的整数数组 output
  std::vector<int> output(num_lanes);
  // 评估 store，将结果存储在 output 中
  eval(output);
  // 遍历 output，断言每个元素的值为 2
  for (const auto i : c10::irange(num_lanes)) {
    ASSERT_EQ(output[i], 2);
  }
}

TEST(Simplify, CompareSelectLoopBounds) {
  // 定义常量 N 为 8
  constexpr int N = 8;
  // 创建一个大小为 N 的 BufHandle 对象 b
  BufHandle b("b", {N}, kFloat);
  // 创建 VarHandle 对象 n, m, var_N, var_M，分别表示整数类型的变量
  VarHandle n("n", kInt);
  VarHandle m("m", kInt);
  VarHandle var_N("var_N", kInt);
  VarHandle var_M("var_M", kInt);

  // 定义 test_case_fn，用于生成 For 循环语句
  auto test_case_fn = [](const VarHandle& n,
                         const BufHandle& b,
                         const ExprHandle& start,
                         const ExprHandle& stop,
                         const int& cmp_val,
                         const CompareSelectOperation& cmp_op,
                         const std::string& check_string) {
    // 创建 For 循环语句，范围从 start 到 stop，对 b[n] 执行 CompareSelect 操作
    StmtPtr s = For::make(
        n,
        start,
        stop,
        b.store({n}, CompareSelect::make(n, cmp_val, 0.f, 1.0f, cmp_op)));
    // 对生成的 For 循环语句进行简化
    s = IRSimplifier::simplify(s);
    // 创建一个输出流对象 oss
    std::ostringstream oss;
    // 将简化后的语句 s 写入 oss 中
    oss << *s;
    // 创建目标字符串 target_string，以 "# CHECK: " 开头，接着是 check_string
    std::string target_string = "# CHECK: ";
    target_string += check_string;
    // 运行 FileCheck 测试，检查目标字符串在 oss 字符串中的匹配情况
    torch::jit::testing::FileCheck().run(target_string, oss.str());
  };

  // 定义嵌套循环的测试用例函数，接受多个变量和表达式参数
  auto test_case_nest_loops_fn = [](const VarHandle& n,
                                    const VarHandle& m,
                                    const BufHandle& b,
                                    const ExprHandle& n_start,
                                    const ExprHandle& n_stop,
                                    const ExprHandle& m_start,
                                    const ExprHandle& m_stop,
                                    const CompareSelectOperation& cmp_op,
                                    const std::string& check_string) {
    // 创建内部循环语句，将比较选择操作的结果存储在缓冲区 b 中
    StmtPtr s = For::make(
        m,
        m_start,
        m_stop,
        b.store({n, m}, CompareSelect::make(n, m, 0.f, 1.0f, cmp_op)));
    // 创建外部循环语句 root_s，并进行表达式简化
    StmtPtr root_s = For::make(n, n_start, n_stop, s);
    root_s = IRSimplifier::simplify(root_s);
    // 创建输出流 oss，并将简化后的循环语句 root_s 输出到 oss 中
    std::ostringstream oss;
    oss << *root_s;
    // 构建目标字符串，以便后续进行 FileCheck 检查
    std::string target_string = "# CHECK: ";
    target_string += check_string;
TEST(Simplify, CompareSelectCondAlwaysInLoopBounds) {
  // 测试函数：比较选择条件总是在循环边界内
  constexpr int N = 8;  // 定义常量 N
  BufHandle b("b", {N}, kFloat);  // 创建名为 b 的缓冲区，包含 N 个元素，每个元素为浮点数
  VarHandle n("n", kInt);  // 创建整数变量 n
  // 创建 for 循环语句，迭代变量 n 从 1 到 N，对 b[n] 进行赋值
  StmtPtr s = For::make(
      n, 1, N, b.store({n}, CompareSelect::make(n, 1, 0.f, 1.0f, kLT)));
  s = IRSimplifier::simplify(s);  // 简化 IR
  std::ostringstream oss;  // 创建字符串输出流对象 oss
  oss << *s;  // 将简化后的 IR 写入字符串流
  // 使用 FileCheck 验证 IR 输出是否符合预期
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[n] = 1.f;
)IR",
      oss.str());
}

TEST(Simplify, IfThenCondAlwaysInLoopBounds) {
  // 测试函数：条件语句总是在循环边界内
  constexpr int N = 8;  // 定义常量 N
  BufHandle b("b", {N}, kFloat);  // 创建名为 b 的缓冲区，包含 N 个元素，每个元素为浮点数
  VarHandle n("n", kInt);  // 创建整数变量 n
  // 创建 for 循环语句，迭代变量 n 从 1 到 N，对 b[n] 进行赋值
  StmtPtr s =
      For::make(n, 1, N, b.store({n}, IfThenElse::make(n < 1, 0.f, 1.0f)));
  s = IRSimplifier::simplify(s);  // 简化 IR
  std::ostringstream oss;  // 创建字符串输出流对象 oss
  oss << *s;  // 将简化后的 IR 写入字符串流
  // 使用 FileCheck 验证 IR 输出是否符合预期
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[n] = 1.f;
)IR",
      oss.str());
}

TEST(Simplify, MultiClauseCondAlwaysInLoopBounds) {
  // 测试函数：多条件子句总是在循环边界内
  // 该测试模拟了 conv2d 的未填充区域。我们希望删除整个循环范围内可证明满足（或不满足）的任何条件。
  constexpr int N = 8;  // 定义常量 N
  BufHandle b("b", {N, N}, kFloat);  // 创建名为 b 的二维缓冲区，大小为 N x N，元素类型为浮点数
  VarHandle i("i", kInt);  // 创建整数变量 i
  VarHandle j("j", kInt);  // 创建整数变量 j
  // 创建一个条件选择对象 csel，根据 i 和 j 的范围选择相应的值
  auto csel = CompareSelect::make(i, 1, kLT);
  csel = CompareSelect::make(j, 1, 1, csel, kLT);
  csel = CompareSelect::make(i, N - 1, 1, csel, kGE);
  csel = CompareSelect::make(j, N - 1, 1, csel, kGE);
  // 创建 for 循环嵌套，遍历 i 从 1 到 N-1，j 从 1 到 N-1
  StmtPtr s = b.store({i, j}, IfThenElse::make(csel, 0.f, 1.0f));
  s = For::make(j, 1, N - 1, s);
  s = For::make(i, 1, N - 1, s);
  s = IRSimplifier::simplify(s);  // 简化 IR
  std::ostringstream oss;  // 创建字符串输出流对象 oss
  oss << *s;  // 将简化后的 IR 写入字符串流
  // 使用 FileCheck 验证 IR 输出是否符合预期
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[i, j] = 1.f;
)IR",
      oss.str());
}
TEST(Simplify, DISABLED_SimplifyLoopBounds) {
    // 这个测试模拟了 conv2d 的填充区域。我们希望调整循环边界，以确保条件始终满足。
    // 注意，这可以通过拆分和应用前面测试中的基于范围的条件简化来解决。
    // Before:
    //   for (const auto i : c10::irange(3)) {
    //     for (const auto j : c10::irange(3)) {
    //       b[i, j] = (b[i, j]) + (IfThenElse(
    //         j>=7 ? 1 : (i>=7 ? 1 : (j<1 ? 1 : (i<1 ? 1 : 0))), 0.f, a[i, j]));
    // After:
    //   for (const auto i : c10::irange(1, 3)) {
    //     for (const auto j : c10::irange(1, 3)) {
    //       b[i, j] = (b[i, j]) + 1.f;
    // 定义常量 N 和 K
    constexpr int N = 8;
    constexpr int K = 3;
    // 创建缓冲区 a 和 b
    BufHandle a("a", {N, N}, kFloat);
    BufHandle b("b", {N, N}, kFloat);
    // 定义变量 i 和 j
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);
    // 创建 CompareSelect 对象 csel，用于条件选择
    auto csel = CompareSelect::make(i, 1, kLT);
    csel = CompareSelect::make(j, 1, 1, csel, kLT);
    csel = CompareSelect::make(i, N - 1, 1, csel, kGE);
    csel = CompareSelect::make(j, N - 1, 1, csel, kGE);
    // 构造语句 s，用于存储操作
    StmtPtr s = b.store(
        {i, j}, b.load({i, j}) + IfThenElse::make(csel, 0.f, a.load({i, j})));
    // 构造 j 的循环，边界为 0 到 K
    s = For::make(j, 0, K, s);
    // 构造 i 的循环，边界为 0 到 K
    s = For::make(i, 0, K, s);
    // 对生成的 IR 进行简化处理
    s = IRSimplifier::simplify(s);
    // 创建一个字符串流 oss，将简化后的 IR 输出到流中
    std::ostringstream oss;
    oss << *s;
    // 运行 FileCheck 验证生成的 IR 是否符合预期格式
    torch::jit::testing::FileCheck().run(
        R"IR(
# CHECK: for (const auto i : c10::irange(1, 3)) {
# CHECK: for (const auto j : c10::irange(1, 3)) {
# CHECK-NOT: IfThenElse
)IR",
        oss.str());
}

} // namespace jit
} // namespace torch
```