# `.\pytorch\test\cpp\tensorexpr\test_cpp_codegen.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include "test/cpp/tensorexpr/test_base.h" // 包含测试基类的头文件

#include <c10/util/irange.h> // 包含用于迭代范围的实用工具头文件
#include <torch/csrc/jit/tensorexpr/cpp_codegen.h> // 包含 C++ 代码生成器的头文件
#include <torch/csrc/jit/tensorexpr/fwd_decls.h> // 包含前向声明头文件
#include <torch/csrc/jit/tensorexpr/stmt.h> // 包含语句类的头文件
#include <torch/csrc/jit/tensorexpr/tensor.h> // 包含张量类的头文件
#include <torch/csrc/jit/testing/file_check.h> // 包含文件检查工具的头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr; // 使用 torch::jit::tensorexpr 命名空间

#define STR_CHECK(node, expected) \ // 定义宏 STR_CHECK，用于比较节点打印结果和期望值
  std::stringstream ss; \ // 创建一个字符串流对象 ss
  CppPrinter printer(&ss); \ // 创建 CppPrinter 对象 printer，用于打印节点内容到 ss
  printer.visit(node); \ // 访问节点并将结果打印到 ss
  ASSERT_EQ(ss.str(), expected) // 使用 Google Test 的断言 ASSERT_EQ 来比较 ss 的内容和期望值

#define FILE_CHECK(node, pattern) \ // 定义宏 FILE_CHECK，用于比较节点打印结果和文件检查模式
  std::stringstream ss; \ // 创建一个字符串流对象 ss
  CppPrinter printer(&ss); \ // 创建 CppPrinter 对象 printer，用于打印节点内容到 ss
  printer.visit(node); \ // 访问节点并将结果打印到 ss
  torch::jit::testing::FileCheck().run(pattern, ss.str()) // 运行文件检查工具来比较 ss 的内容和模式

TEST(CppPrinter, IntImm) { // 定义 Google Test 单元测试 IntImm
  auto i = alloc<IntImm>(10); // 分配一个整数常量节点 i，值为 10
  STR_CHECK(i, "10"); // 使用 STR_CHECK 宏来比较 i 的打印结果和 "10"
}

TEST(CppPrinter, FloatImm) { // 定义 Google Test 单元测试 FloatImm
  auto f = alloc<FloatImm>(10); // 分配一个浮点数常量节点 f，值为 10
  STR_CHECK(f, "10.f"); // 使用 STR_CHECK 宏来比较 f 的打印结果和 "10.f"
}

// 后续类似的单元测试，均使用 STR_CHECK 宏来测试不同类型的节点打印结果
TEST(CppPrinter, CompareSelect) {
  // 分配并初始化一个 CompareSelect 对象，条件为 1 <= 2，true 值为 1.f，false 值为 2.f
  auto cs = alloc<CompareSelect>(
      alloc<IntImm>(1),
      alloc<IntImm>(2),
      alloc<FloatImm>(1),
      alloc<FloatImm>(2),
      CompareSelectOperation::kLE);
  // 验证生成的字符串是否符合预期，应为 "((1 <= 2) ? 1.f : 2.f)"
  STR_CHECK(cs, "((1 <= 2) ? 1.f : 2.f)");
}

TEST(CppPrinter, IfThenElse) {
  // 创建一个加法节点作为条件：1 + 2
  auto cond = alloc<Add>(alloc<IntImm>(1), alloc<IntImm>(2));
  // 创建一个减法节点作为 true 分支的值：0 - 1
  auto true_value = alloc<Sub>(alloc<IntImm>(0), alloc<IntImm>(1));
  // 创建一个乘法节点作为 false 分支的值：2 * 3
  auto false_value = alloc<Mul>(alloc<IntImm>(2), alloc<IntImm>(3));
  // 创建一个 IfThenElse 对象，条件为 1 + 2，true 值为 0 - 1，false 值为 2 * 3
  auto v = alloc<IfThenElse>(cond, true_value, false_value);
  // 验证生成的字符串是否符合预期，应为 "((1 + 2) ? 0 - 1 : 2 * 3)"
  STR_CHECK(v, "((1 + 2) ? 0 - 1 : 2 * 3)");
}

TEST(CppPrinter, AllocateFree) {
  // 创建一个大小为 {2, 3} 的整型缓冲区 "x"
  BufHandle buf("x", {2, 3}, kInt);
  // 创建一个 Allocate 对象，用于分配缓冲区 "x"
  AllocatePtr alloc = Allocate::make(buf);
  // 创建一个 Free 对象，用于释放缓冲区 "x"
  FreePtr free = Free::make(buf);
  // 创建一个包含 Allocate 和 Free 操作的 Block 对象
  BlockPtr block = Block::make({alloc, free});

  const std::string pattern = R"(
   # CHECK: {
   # CHECK:   int* x = static_cast<int*>(malloc(24));
   # CHECK:   free(x);
   # CHECK: }
  )";
  // 验证生成的文件内容是否符合预期模式
  FILE_CHECK(block, pattern);
}

TEST(CppPrinter, LoadStore) {
  // 创建大小为 {2, 3} 的整型缓冲区 "A" 和大小为 {3, 4} 的整型缓冲区 "B"
  BufHandle a("A", {2, 3}, kInt);
  BufHandle b("B", {3, 4}, kInt);
  // 创建一个将 A[1][1] 存储到 B[2][2] 的 Store 对象
  auto store = b.store({2, 2}, a.load(1, 1));
  // 验证生成的字符串是否符合预期，应为 "B[(0 + 2 * (1 * 4)) + 2 * 1] = A[(0 + 1 * (1 * 3)) + 1 * 1];\n"
  STR_CHECK(
      store, "B[(0 + 2 * (1 * 4)) + 2 * 1] = A[(0 + 1 * (1 * 3)) + 1 * 1];\n");
}

TEST(CppPrinter, Var) {
  // 创建一个整型变量节点 "x"
  auto var = alloc<Var>("x", kInt);
  // 验证生成的字符串是否符合预期，应为 "x"
  STR_CHECK(var, "x");
}

TEST(CppPrinter, Cast) {
  // 创建一个将整型转换为浮点型的 Cast 对象，值为 1
  auto cast = alloc<Cast>(kFloat, alloc<IntImm>(1));
  // 验证生成的字符串是否符合预期，应为 "static_cast<float>(1)"
  STR_CHECK(cast, "static_cast<float>(1)");
}

TEST(CppPrinter, BitCast) {
  // 创建一个将浮点数转换为整型的 BitCast 对象，值为 20.0
  auto cast = alloc<BitCast>(kInt, alloc<FloatImm>(20));
  // 验证生成的字符串是否符合预期，应为 "std::bitcast<float, int>(20.f)"
  STR_CHECK(cast, "std::bitcast<float, int>(20.f)");
}

TEST(CppPrinter, Let) {
  // 创建一个浮点型变量节点 "x"，值为 2.0
  auto var = alloc<Var>("x", kFloat);
  auto val = alloc<FloatImm>(2);
  // 创建一个 Let 对象，定义变量 "x" 并初始化为 2.0
  auto let = alloc<Let>(var, val);
  // 验证生成的字符串是否符合预期，应为 "float x = 2.f;\n"
  STR_CHECK(let, "float x = 2.f;\n");
}

TEST(CppPrinter, For) {
  constexpr int N = 1024;
  // 创建大小为 {1024} 的整型缓冲区 "A", "B", "C" 和一个整型变量 "i"
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  VarHandle i("i", kInt);
  // 创建一个循环，遍历 i 从 0 到 1024，并在缓冲区 C 中存储 A[i] + B[i]
  auto f = For::make(i, 0, N, c.store({i}, Add::make(a.load(i), b.load(i))));
  const std::string pattern = R"(
   # CHECK: for (int i = 0; i < 1024; i++) {
   # CHECK:   C[i] = (A[i]) + (B[i]);
   # CHECK: }
  )";
  // 验证生成的文件内容是否符合预期模式
  FILE_CHECK(f, pattern);
}

TEST(CppPrinter, Cond) {
  // 创建大小为 {1} 的整型缓冲区 "X"
  BufHandle x("X", {1}, kInt);
  // 创建一个比较选择节点，比较 X[0] < 10
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  // 创建一个条件节点，根据比较结果选择执行 x[0] + 1 或 x[0] - 1
  auto cond =
      Cond::make(cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  const std::string pattern = R"(
    # CHECK: if (((X[0] < 10) ? 1 : 0)) {
    # CHECK:   X[0] = (X[0]) + 1;
    # CHECK: } else {
    # CHECK:   X[0] = (X[0]) - 1;
    # CHECK: }
  )";
  // 验证生成的文件内容是否符合预期模式
  FILE_CHECK(cond, pattern);
}

TEST(CppPrinter, Intrinsics) {
  const std::unordered_set<IntrinsicsOp, std::hash<int>> unsupported_ops{
      kRand, kSigmoid};
  // 遍历所有内部操作，跳过不支持的操作
  for (const auto i : c10::irange(static_cast<uint32_t>(kMaxIntrinsicsOp))) {
    IntrinsicsOp op = static_cast<IntrinsicsOp>(i);
    if (unsupported_ops.count(op)) {
      continue;
    }
    # 检查操作符 op 的参数数量是否为 1
    if (Intrinsics::OpArgCount(op) == 1) {
      # 如果参数数量为 1，则创建一个 Intrinsics 对象 v，
      # 使用 op 和一个浮点数常量 2.0f 进行初始化
      auto v = alloc<Intrinsics>(op, alloc<FloatImm>(2.0f));
      # 对 v 进行字符串格式检查和处理，生成形如 "std::函数名(2.f)" 的字符串
      STR_CHECK(v, "std::" + v->func_name() + "(2.f)");
    } else {
      # 如果参数数量不为 1，则创建一个 Intrinsics 对象 v，
      # 使用 op 和两个浮点数常量 1.0f 和 2.0f 进行初始化
      auto v =
          alloc<Intrinsics>(op, alloc<FloatImm>(1.0f), alloc<FloatImm>(2.0f));
      # 对 v 进行字符串格式检查和处理，生成形如 "std::函数名(1.f, 2.f)" 的字符串
      STR_CHECK(v, "std::" + v->func_name() + "(1.f, 2.f)");
    }
  }
}

// 定义一个测试用例 CppPrinter.ExternalCall
TEST(CppPrinter, ExternalCall) {
  // 创建包含两个 IntImm(2) 的 ExprPtr 向量 dims
  std::vector<ExprPtr> dims{alloc<IntImm>(2), alloc<IntImm>(2)};
  // 分别创建输出缓冲区 output，和两个缓冲区 buf_arg1, buf_arg2，每个都有两个维度，数据类型为 kFloat
  auto output = alloc<Buf>("out", dims, kFloat);
  auto buf_arg1 = alloc<Buf>("a", dims, kFloat);
  auto buf_arg2 = alloc<Buf>("b", dims, kFloat);
  // 创建一个加法表达式 Add(IntImm(1), IntImm(2))，并包装成标量参数 scalar_arg
  auto scalar_arg = alloc<Add>(alloc<IntImm>(1), alloc<IntImm>(2));
  // 创建包含 buf_arg1 和 buf_arg2 的 BufPtr 向量 buf_args，和包含 scalar_arg 的 ExprPtr 向量 scalar_args
  std::vector<BufPtr> buf_args{buf_arg1, buf_arg2};
  std::vector<ExprPtr> scalar_args{scalar_arg};
  // 创建一个外部调用 ExternalCall，输出为 output，调用名称为 "nnc_aten_matmul"，参数为 buf_args 和 scalar_args
  auto call =
      alloc<ExternalCall>(output, "nnc_aten_matmul", buf_args, scalar_args);
  // 定义字符串模式 pattern，用于检查输出
  const std::string pattern = R"(
   # CHECK: {
   # CHECK:   void* buf_ptrs[]{out, a, b};
   # CHECK:   int64_t buf_ranks[]{2, 2, 2};
   # CHECK:   int64_t buf_dims[]{2, 2, 2, 2, 2, 2};
   # CHECK:   int8_t buf_dtypes[]{6, 6, 6};
   # CHECK:   int64_t extra_args[]{1 + 2};
   # CHECK:   nnc_aten_matmul(
   # CHECK:       3,
   # CHECK:       buf_ptrs,
   # CHECK:       buf_ranks,
   # CHECK:       buf_dims,
   # CHECK:       buf_dtypes,
   # CHECK:       1,
   # CHECK:       extra_args);
   # CHECK: }
  )";
  // 调用文件检查宏 FILE_CHECK，验证 call 对象是否符合 pattern
  FILE_CHECK(call, pattern);
}

} // namespace jit
} // namespace torch
```