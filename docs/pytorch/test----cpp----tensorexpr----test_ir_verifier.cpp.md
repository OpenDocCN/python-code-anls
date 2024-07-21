# `.\pytorch\test\cpp\tensorexpr\test_ir_verifier.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <stdexcept>  // 包含异常处理相关的头文件
#include "test/cpp/tensorexpr/test_base.h"  // 包含测试基类的头文件

#include <torch/csrc/jit/tensorexpr/expr.h>  // 包含表达式相关的头文件
#include <torch/csrc/jit/tensorexpr/ir.h>    // 包含中间表示相关的头文件
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>  // 包含中间表示验证器的头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h>     // 包含循环嵌套相关的头文件
#include <torch/csrc/jit/tensorexpr/tensor.h>       // 包含张量相关的头文件
#include <torch/csrc/jit/testing/file_check.h>      // 包含文件检查相关的头文件

#include <sstream>  // 包含字符串流处理的头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;  // 使用 torch::jit::tensorexpr 命名空间

TEST(IRVerifier, BitwiseOps) {
  VarPtr X = alloc<Var>("x", kInt);    // 分配一个整型变量 x
  VarPtr Y = alloc<Var>("y", kFloat);  // 分配一个浮点型变量 y
  {
    auto a = alloc<And>(X, Y);  // 创建一个 And 表达式对象 a，对 X 和 Y 进行按位与操作
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
  {
    auto a = alloc<Or>(X, Y);   // 创建一个 Or 表达式对象 a，对 X 和 Y 进行按位或操作
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
  {
    auto a = alloc<Xor>(X, Y);  // 创建一个 Xor 表达式对象 a，对 X 和 Y 进行按位异或操作
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
  {
    auto a = alloc<Lshift>(X, Y);  // 创建一个 Lshift 表达式对象 a，对 X 左移 Y 位
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
  {
    auto a = alloc<Rshift>(X, Y);  // 创建一个 Rshift 表达式对象 a，对 X 右移 Y 位
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
}

TEST(IRVerifier, CompareSelect) {
  ExprPtr X = alloc<IntImm>(1);       // 创建一个整型常量表达式对象 X
  ExprPtr Y = alloc<FloatImm>(3.14f);  // 创建一个浮点型常量表达式对象 Y
  {
    auto a = alloc<CompareSelect>(X, X, X, Y, kEQ);  // 创建一个 CompareSelect 表达式对象 a
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
  {
    auto a = alloc<CompareSelect>(X, Y, X, X, kEQ);  // 创建一个 CompareSelect 表达式对象 a
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
}

TEST(IRVerifier, Ramp) {
  VarPtr I = alloc<Var>("i", kInt);    // 分配一个整型变量 i
  VarPtr J = alloc<Var>("j", kFloat);  // 分配一个浮点型变量 j
  {
    auto a = alloc<Ramp>(I, J, 4);  // 创建一个 Ramp 表达式对象 a，起始于 I，步长为 J，数量为 4
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望抛出异常
  }
}

TEST(IRVerifier, Load) {
  VarPtr I = alloc<Var>("i", kInt);    // 分配一个整型变量 i
  VarPtr J = alloc<Var>("j", kLong);   // 分配一个长整型变量 j
  VarPtr K = alloc<Var>("k", kFloat);  // 分配一个浮点型变量 k
  BufPtr B = alloc<Buf>(  // 分配一个缓冲区 B，包含两个整型常量表达式，数据类型为浮点型
      "b",
      std::vector<ExprPtr>({alloc<IntImm>(10), alloc<IntImm>(20)}),
      kFloat);
  {
    auto a = alloc<Load>(B, std::vector<ExprPtr>({I, J}));  // 创建一个 Load 表达式对象 a，从缓冲区 B 中加载数据
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));  // 验证表达式 a 是否满足验证条件，期望不抛出异常
  }
  {
    auto a = alloc<Load>(B, std::vector<ExprPtr>({K, K}));  // 创建一个 Load 表达式对象 a，从缓冲区 B 中加载数据
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    // 检查表达式 `verify(a)` 是否会抛出任何异常
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // 只有在扁平化索引中才允许使用多行索引
    // 创建一个多行索引，使用步长为1的整数常量和步长为4
    auto multilane_index = alloc<Ramp>(I, alloc<IntImm>(1), 4);
    // 创建一个加载操作，加载数组 B 中的数据，索引包括 I 和多行索引
    auto a = alloc<Load>(B, std::vector<ExprPtr>({I, multilane_index}));
    // 禁用 lint 检查，忽略与 `goto` 相关的警告，用于测试中可能需要使用 `goto`
    // 检查表达式 `verify(a)` 是否会抛出任何异常
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, IfThenElse) {
  // 创建名为I的整型变量
  VarPtr I = alloc<Var>("i", kInt);
  // 创建名为J的长整型变量
  VarPtr J = alloc<Var>("j", kLong);
  // 创建名为K的单精度浮点数变量
  VarPtr K = alloc<Var>("k", kFloat);
  {
    // 条件表达式必须是整型
    auto a = alloc<IfThenElse>(K, I, I);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // 真假分支表达式的数据类型必须匹配
    auto a = alloc<IfThenElse>(I, I, J);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // 条件表达式不能有多个通道
    auto a = alloc<IfThenElse>(alloc<Broadcast>(I, 4), I, I);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, For) {
  // 创建名为I的整型变量
  VarPtr I = alloc<Var>("i", kInt);
  // 创建名为J的整型变量
  VarPtr J = alloc<Var>("j", kInt);
  // 创建一个空语句块
  StmtPtr body = alloc<Block>(std::vector<StmtPtr>({}));
  {
    // 不能使用 nullptr 作为变量
    auto a = alloc<For>(nullptr, I, J, body);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, Block) {
  // 创建名为I的整型变量
  VarPtr I = alloc<Var>("i", kInt);
  // 创建名为B的整型缓冲区
  BufPtr B = alloc<Buf>("B", std::vector<ExprPtr>({alloc<IntImm>(10)}), kInt);
  {
    // 创建一个存储语句，将变量I存储到缓冲区B中
    StmtPtr store = alloc<Store>(B, std::vector<ExprPtr>({I}), I);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    StmtPtr block1 = alloc<Block>(std::vector<StmtPtr>({store}));
    // 创建另一个存储语句块，试图多次插入相同的语句
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    StmtPtr block2 = alloc<Block>(std::vector<StmtPtr>({store}));
    // 语句不能有多个父级，因此在多个块中插入它是非法的
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(block2));
  }
}

TEST(IRVerifier, Store) {
  // 创建名为I的整型变量
  VarPtr I = alloc<Var>("i", kInt);
  // 创建名为J的长整型变量
  VarPtr J = alloc<Var>("j", kLong);
  // 创建名为K的单精度浮点数变量
  VarPtr K = alloc<Var>("k", kFloat);
  // 创建名为B的浮点型缓冲区，包含两个整型常数表达式
  BufPtr B = alloc<Buf>(
      "b",
      std::vector<ExprPtr>({alloc<IntImm>(10), alloc<IntImm>(20)}),
      kFloat);
  {
    // 索引具有不同整型数据类型（kInt, kLong）是允许的
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I, J}), K);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));
  }
  {
    // 浮点数索引
    auto a = alloc<Store>(B, std::vector<ExprPtr>({K, K}), K);
    // 忽略以下警告，因为这里是测试代码
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // 多通道只允许在扁平化的索引中
    auto multilane_index = alloc<Ramp>(I, alloc<IntImm>(1), 4);
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I, multilane_index}), K);
  {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    // 测试：预期对 verify(a) 的调用会抛出任何异常
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Value and buf dtypes mismatch
    // 创建变量 a，分配一个 Store 类型的对象
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I}), I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    // 测试：预期对 verify(a) 的调用会抛出任何异常
    EXPECT_ANY_THROW(verify(a));
  }
} // 结束 torch 命名空间
} // 结束 jit 命名空间
} // 结束整个代码的命名空间
```