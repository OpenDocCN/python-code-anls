# `.\pytorch\test\cpp\tensorexpr\test_reductions.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架头文件

#include <limits> // 引入数值极限相关的头文件
#include <memory> // 引入内存管理相关的头文件
#include <sstream> // 引入字符串流处理相关的头文件
#include <stdexcept> // 引入标准异常类相关的头文件
#include <unordered_map> // 引入无序映射容器相关的头文件

#include <test/cpp/tensorexpr/test_base.h> // 引入测试基础相关的头文件

#include <c10/util/irange.h> // 引入 C10 库中的整数范围迭代器头文件
#include <test/cpp/tensorexpr/padded_buffer.h> // 引入填充缓冲区相关的头文件
#include <torch/csrc/jit/tensorexpr/analysis.h> // 引入 TensorExpr 分析功能的头文件
#include <torch/csrc/jit/tensorexpr/eval.h> // 引入 TensorExpr 评估功能的头文件
#include <torch/csrc/jit/tensorexpr/ir.h> // 引入 TensorExpr 中间表示的头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h> // 引入 TensorExpr 中间表示打印的头文件
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h> // 引入 TensorExpr 中间表示简化的头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h> // 引入 TensorExpr 循环嵌套的头文件
#include <torch/csrc/jit/tensorexpr/tensor.h> // 引入 TensorExpr 张量表示的头文件
#include <torch/csrc/jit/testing/file_check.h> // 引入文件检查功能的头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr; // 使用 TensorExpr 命名空间

TEST(Reductions, ReduceSum0D_1) {
  const int M = 10; // 定义常量 M 为 10

  BufHandle b("b", {M}, kFloat); // 创建缓冲区句柄 b，表示长度为 M 的浮点数数组
  std::vector<float> in(M); // 创建长度为 M 的浮点数向量 in
  for (const auto j : c10::irange(M)) { // 使用 C10 的整数范围迭代器遍历 M
    in[j] = j; // 将 j 赋值给 in[j]
  }

  std::vector<float> out(M, -1.f); // 创建长度为 M 的浮点数向量 out，初始值为 -1.0

  Tensor c = Reduce("sum", {M}, Sum(), b, {}); // 创建一个求和操作的 Tensor c，对应缓冲区 b
  LoopNest loop({c}); // 创建循环嵌套对象，包含 Tensor c
  loop.prepareForCodegen(); // 准备进行代码生成
  StmtPtr s = loop.root_stmt(); // 获取循环嵌套的根语句
  s = IRSimplifier::simplify(s); // 简化根语句

  SimpleIREvaluator cg(s, {b, c}); // 创建简单的中间表示求值器，传入缓冲区 b 和 Tensor c

  cg.call({in, out}); // 调用求值器，计算结果并存入 out 中
  for (const auto i : c10::irange(M)) { // 使用 C10 的整数范围迭代器遍历 M
    ASSERT_EQ(out[i], in[i]); // 断言 out[i] 等于 in[i]
  }
}

TEST(Reductions, ReduceSum0D_2) {
  BufHandle b("b", {}, kFloat); // 创建一个空的浮点数缓冲区句柄 b
  std::vector<float> in(1); // 创建长度为 1 的浮点数向量 in
  in[0] = 77.7; // 将值 77.7 存入 in[0]

  std::vector<float> out(1, -1.f); // 创建长度为 1 的浮点数向量 out，初始值为 -1.0

  Tensor c = Reduce("sum", {}, Sum(), b, {}); // 创建一个求和操作的 Tensor c，对应空缓冲区 b
  LoopNest loop({c}); // 创建循环嵌套对象，包含 Tensor c
  loop.prepareForCodegen(); // 准备进行代码生成
  StmtPtr s = loop.root_stmt(); // 获取循环嵌套的根语句
  s = IRSimplifier::simplify(s); // 简化根语句

  SimpleIREvaluator cg(s, {b, c}); // 创建简单的中间表示求值器，传入缓冲区 b 和 Tensor c

  cg.call({in, out}); // 调用求值器，计算结果并存入 out 中
  ASSERT_EQ(out[0], in[0]); // 断言 out[0] 等于 in[0]
}

// Sum an array to a single value.
TEST(Reductions, ReduceSum1D) {
  BufHandle b("b", {10}, kFloat); // 创建长度为 10 的浮点数缓冲区句柄 b
  std::vector<float> in(10); // 创建长度为 10 的浮点数向量 in
  for (const auto j : c10::irange(10)) { // 使用 C10 的整数范围迭代器遍历 10
    in[j] = j; // 将 j 赋值给 in[j]
  }

  std::vector<float> out(1, -1.f); // 创建长度为 1 的浮点数向量 out，初始值为 -1.0

  Tensor c = Reduce("sum", {}, Sum(), b, {10}); // 创建一个求和操作的 Tensor c，对应缓冲区 b
  LoopNest loop({c}); // 创建循环嵌套对象，包含 Tensor c
  loop.prepareForCodegen(); // 准备进行代码生成
  StmtPtr s = loop.root_stmt(); // 获取循环嵌套的根语句
  s = IRSimplifier::simplify(s); // 简化根语句

  SimpleIREvaluator cg(s, {b, c}); // 创建简单的中间表示求值器，传入缓冲区 b 和 Tensor c

  cg.call({in, out}); // 调用求值器，计算结果并存入 out 中
  ASSERT_EQ(out[0], 45); // 断言 out[0] 等于 45
}

// Sum a 2D tensor to a 1D tensor with dynamic shapes.
TEST(Reductions, ReduceSum2D) {
  const int M = 3; // 定义常量 M 为 3
  const int N = 7; // 定义常量 N 为 7

  VarHandle m("m", kInt); // 创建整数变量句柄 m
  VarHandle n("n", kInt); // 创建整数变量句柄 n

  BufHandle b("b", {m, n}, kFloat); // 创建形状为 {m, n} 的浮点数缓冲区句柄 b
  std::vector<float> in(M * N); // 创建长度为 M*N 的浮点数向量 in
  for (const auto i : c10::irange(M)) { // 使用 C10 的整数范围迭代器遍历 M
    for (const auto j : c10::irange(N)) { // 使用 C10 的整数范围迭代器遍历 N
      in[i * N + j] = j; // 将 j 赋值给 in[i * N + j]
    }
  }

  std::vector<float> out(M, -1.f); // 创建长度为 M 的浮点数向量 out，初始值为 -1.0

  Tensor c = Reduce("sum", {M}, Sum(), b, {N}); // 创建一个求和操作的 Tensor c，对应缓冲区 b
  LoopNest loop({c}); // 创建循环嵌套对象，包含 Tensor c
  loop.prepareForCodegen(); // 准备进行代码生成
  StmtPtr s = loop.root_stmt(); // 获取
// Sum a 3D tensor to both a 2D and 1D tensor, then reduce the 2D tensor flat to
// check our work.
TEST(Reductions, ReduceSum3D) {
  // 定义常量 M 为 10
  const int M = 10;
  // 定义整型变量 m
  VarHandle m("m", kInt);

  // 创建三维缓冲区 b，维度为 {2, 3, m}，元素类型为 float
  BufHandle b("b", {2, 3, m}, kFloat);

  // 对三维张量 b 进行求和操作，将结果保存到二维张量 c 中
  Tensor c = Reduce("sum", {2, 3}, Sum(), b, {m});
  // 创建循环嵌套对象 loop，用于代码生成前的准备工作
  LoopNest loop({c});
  loop.prepareForCodegen();
  // 获取根语句，并简化其表达式
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  // 创建简单的 IR 评估器 cg，用于求解简化后的语句 s
  SimpleIREvaluator cg(s, {b, c, m});

  // 初始化三个向量，用于测试
  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> cData(2 * 3, 6.0f);
  std::vector<float> dData(2, 1.0f);
  std::vector<float> eData(2, 1.0f);

  // 填充 bData 向量，以验证测试数据
  for (int i = 0; i < 2 * 3; ++i) {
    for (const auto j : c10::irange(M)) {
      bData[i * M + j] = j;
    }
  }

  // 调用评估器 cg 运行测试数据
  cg.call({bData, cData, M});
  float expected = 0;
  // 计算预期值
  for (const auto i : c10::irange(M)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected += i;
  }

  // 验证结果
  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(cData[i], expected);
  }

  // 对三维张量 b 进行进一步的求和操作，将结果保存到一维张量 d 中
  Tensor d = Reduce("sum2", {2}, Sum(), b, {3, m});
  LoopNest loop2({d});
  loop2.prepareForCodegen();
  // 获取简化后的根语句 s2
  StmtPtr s2 = loop2.root_stmt();
  s2 = IRSimplifier::simplify(s2);

  // 创建简单的 IR 评估器 cg2，用于求解简化后的语句 s2
  SimpleIREvaluator cg2(s2, {b, d, m});
  // 运行评估器 cg2，使用测试数据 dData 和 M
  cg2.call({bData, dData, M});

  // 更新预期值，考虑对额外维度的求和
  expected = expected * 3;

  // 验证结果
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(dData[i], expected);
  }

  // 使用自定义缓冲区 c_buf，对 c 执行降维操作，将结果保存到二维张量 e 中
  BufHandle c_buf(c.buf());
  Tensor e = Reduce("sum3", {2}, Sum(), c_buf, {3});
  LoopNest loop3({e});
  loop3.prepareForCodegen();
  // 获取简化后的根语句 s3
  StmtPtr s3 = loop3.root_stmt();
  s3 = IRSimplifier::simplify(s3);

  // 创建简单的 IR 评估器 cg3，用于求解简化后的语句 s3
  SimpleIREvaluator cg3(s3, {c, e});
  // 运行评估器 cg3，使用测试数据 cData 和 eData
  cg3.call({cData, eData});

  // 验证结果
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(eData[i], expected);
  }
}

// Sum a large (10 D) Tensor 5 dimensions in.
TEST(Reductions, ReduceSum10D) {
  // 创建十维输入缓冲区 in_，维度为 {2, 3, 2, 3, 2, 3, 2, 3, 2, 3}，元素类型为 float
  BufHandle in_("in_", {2, 3, 2, 3, 2, 3, 2, 3, 2, 3}, kFloat);
  // 定义输入大小常量 InputSize
  const int InputSize = 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3;
  // 创建输出缓冲区 out_，维度为 {2, 3, 2, 3, 2}，元素类型为 float
  BufHandle out_("out_", {2, 3, 2, 3, 2}, kFloat);
  // 定义输出大小常量 OutputSize
  const int OutputSize = 2 * 3 * 2 * 3 * 2;

  // 初始化输入向量 in，所有元素为 1.0f
  std::vector<float> in(InputSize, 1.f);
  // 初始化输出向量 out，所有元素为 -1.0f
  std::vector<float> out(OutputSize, -1.f);

  // 对十维张量 in_ 进行求和操作，将结果保存到五维张量 c 中
  Tensor c = Reduce("sum", {2, 3, 2, 3, 2}, Sum(), in_, {3, 2, 3, 2, 3});
  // 创建循环嵌套对象 loop，用于代码生成前的准备工作
  LoopNest loop({c});
  loop.prepareForCodegen();
  // 获取根语句，并简化其表达式
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  // 创建简单的 IR 评估器 cg，用于求解简化后的语句 s
  SimpleIREvaluator cg(s, {in_, c});

  // 运行评估器 cg，使用输入向量 in 和输出向量 out
  cg.call({in, out});

  // 计算期望值
  // NOLINTNEXTLINE(bugprone-integer-division)
  float expected = InputSize / OutputSize;
  // 验证结果
  for (const auto i : c10::irange(OutputSize)) {
    ASSERT_EQ(out[i], expected);
  }
}

// Reduce via Mul rather than Add using a custom Reducer.
TEST(Reductions, ReduceProduct) {
  // 定义常量 M 和 N 为 4
  const int M = 4;
  const int N = 4;

  // 创建二维缓冲区 b，维度为 {M, N}，元素类型为 float
  BufHandle b("b", {M, N}, kFloat);
  // 初始化输入向量 in，为每个元素赋值
  std::vector<float> in(M * N);
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      in[i * N + j] = 2 + j;
    }
  }
    }
  }

  std::vector<float> out(M, -1.f);  // 创建一个包含 M 个元素，每个元素初始值为 -1.0 的浮点数向量 out

  Reducer product(
      ExprHandle(1.f), [](ExprHandle a, ExprHandle b) { return a * b; });  // 创建一个以 1.0 为初始值的 Reducer 对象，用于计算乘积

  Tensor c = Reduce("product", {M}, product, b, {N});  // 使用 Reducer 对象 product 对张量 b 进行归约操作，生成一个名为 "product" 的张量 c，其形状为 {M}，并在第一维上进行归约 {N}

  LoopNest loop({c});  // 创建一个循环嵌套对象，包含张量 c

  loop.prepareForCodegen();  // 准备循环嵌套对象进行代码生成

  StmtPtr s = loop.root_stmt();  // 获取循环嵌套的根语句

  s = IRSimplifier::simplify(s);  // 对根语句进行简化处理

  SimpleIREvaluator cg(s, {b, c});  // 创建一个简单的 IR 评估器，用简化后的根语句和张量 b、c 进行初始化

  cg.call({in, out});  // 调用评估器执行计算，其中输入是 in，输出是 out

  float expected = 1;  // 初始化期望值为 1.0
  for (const auto i : c10::irange(N)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected *= 2 + i;  // 循环计算期望值，乘以 2 + i
  }

  for (const auto i : c10::irange(M)) {
    ASSERT_EQ(out[i], expected);  // 使用断言检查 out 向量的每个元素是否等于期望值
  }
// Maximum reductions.
TEST(Reductions, ReduceMax) {
  // 创建一个大小为 10 的浮点数缓冲区
  BufHandle in_("b", {10}, kFloat);

  // 初始化一个大小为 10 的浮点数向量，并设置所有元素的值
  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (const auto j : c10::irange(10)) {
    in[j] = j;
  }

  // 执行最大值归约操作，生成一个名为 dm1 的张量
  Tensor dm1 = Reduce("max", {}, Maximum(kFloat), in_, {10});

  // 创建一个循环嵌套对象并准备进行代码生成
  LoopNest loop({dm1});
  loop.prepareForCodegen();
  // 获取循环嵌套的根语句
  StmtPtr s = loop.root_stmt();
  // 对根语句进行简化
  s = IRSimplifier::simplify(s);
  // 创建一个简单的 IR 评估器，并传入相应的输入数据
  SimpleIREvaluator cg(s, {in_, dm1});

  // 调用评估器，计算结果
  cg.call({in, out});

  // 断言输出结果是否符合预期
  ASSERT_EQ(out[0], 9);

  // 创建一个大小为 [2, 5] 的浮点数缓冲区
  BufHandle in2_("b", {2, 5}, kFloat);
  // 初始化一个大小为 2 的浮点数向量，并设置所有元素的值为 -1
  std::vector<float> out2(2, -1.f);

  // 执行二维最大值归约操作，生成一个名为 m2d 的张量
  Tensor m2d = Reduce("max", {2}, Maximum(kFloat), in2_, {5});

  // 创建另一个循环嵌套对象并准备进行代码生成
  LoopNest loop2({m2d});
  loop2.prepareForCodegen();
  // 获取第二个循环嵌套的根语句
  s = loop2.root_stmt();
  // 对根语句进行简化
  s = IRSimplifier::simplify(s);

  // 创建第二个简单的 IR 评估器，并传入相应的输入数据
  SimpleIREvaluator cg2(s, {in2_, m2d});
  // 调用第二个评估器，计算结果
  cg2.call({in, out2});

  // 断言输出结果是否符合预期
  ASSERT_EQ(out2[0], 4);
  ASSERT_EQ(out2[1], 9);
}

// Minimum reduction, with custom initialization.
TEST(Reductions, ReduceMinCustomInitializer) {
  // 定义一个名为 minInit 的浮点数变量
  VarHandle minInit("minInit", kFloat);
  // 创建一个大小为 10 的浮点数缓冲区
  BufHandle in_("b", {10}, kFloat);

  // 初始化一个大小为 10 的浮点数向量，并设置所有元素的值
  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (const auto j : c10::irange(10)) {
    in[j] = 10 + j;
  }

  // 执行最小值归约操作，生成一个名为 min 的张量
  Tensor min = Reduce(
      "min",
      {},
      Minimum(ExprHandle(minInit)),
      [&](ParameterList& v) { return in_.load(v); },
      {10});

  // 创建一个循环嵌套对象并准备进行代码生成
  LoopNest loop({min});
  loop.prepareForCodegen();
  // 获取循环嵌套的根语句
  StmtPtr s = loop.root_stmt();
  // 对根语句进行简化
  s = IRSimplifier::simplify(s);

  // 创建一个简单的 IR 评估器，并传入相应的输入数据
  SimpleIREvaluator cg(s, {in_, min, minInit});

  // 使用正常的方式调用评估器，注意输出数据的初始值低于正确的最小值
  cg.call({in, out, std::numeric_limits<float>::max()});
  // 断言输出结果是否符合预期
  ASSERT_EQ(out[0], 10);

  // 使用一个低于最小值的初始化值进行调用
  cg.call({in, out, 5.f});
  // 断言输出结果是否符合预期
  ASSERT_EQ(out[0], 5);
}

// Example implementation of Any/All.
// TODO: this is very awkward without logical And/Or operators.
TEST(Reductions, ReduceAnyAll) {
  // 定义一个名为 searchValue 的整数变量
  VarHandle searchValue("searchValue", kInt);
  // 创建一个大小为 [4, 10] 的整数缓冲区
  BufHandle b("b", {4, 10}, kInt);

  // 创建一个自定义的任意值等于搜索值的归约器
  Reducer anyEqSV(ExprHandle(0), [](ExprHandle a, ExprHandle b) {
    return CompareSelect::make(a, 1, 1, b, kEQ);
  });

  // 执行任意值等于搜索值的归约操作，生成一个名为 any 的张量
  Tensor any = Reduce(
      "anyEqual",
      {4},
      anyEqSV,
      [&](const auto& i, const auto& j) {
        return CompareSelect::make(b.load(i, j), searchValue, kEQ);
      },
      {10});

  // 创建一个循环嵌套对象并准备进行代码生成
  LoopNest loop({any});
  loop.prepareForCodegen();
  // 获取循环嵌套的根语句
  StmtPtr s = loop.root_stmt();
  // 对根语句进行简化
  s = IRSimplifier::simplify(s);

  // 创建一个简单的 IR 评估器，并传入相应的输入数据
  SimpleIREvaluator cg(s, {b, any, searchValue});

  // 初始化一个大小为 40 的整数向量，设置所有元素的值
  std::vector<int> in(40, 0);
  std::vector<int> out(4, 0);

  // 输入数据中包含 0 到 39 的数，分布在 4 行中
  for (const auto i : c10::irange(40)) {
    in[i] = i;
  }
  // 使用搜索值为 1 调用评估器，计算结果
  cg.call({in, out, 1});

  // 断言输出结果是否符合预期
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 0);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  // 使用搜索值为 15 调用评估器，计算结果
  cg.call({in, out, 15});

  // 断言输出结果是否符合预期
  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  // 创建一个自定义的全部值大于搜索值的归约器
  Reducer allGTSV(ExprHandle(1), [](ExprHandle a, ExprHandle b) {
    //
    // 调用 CompareSelect 类的静态方法 make，生成一个比较选择操作，返回值为 CompareSelect::make(a, 0, 0, b, kEQ)
    return CompareSelect::make(a, 0, 0, b, kEQ);
  });

  // 创建名为 allGreaterThan 的张量对象，使用 Reduce 函数对 allGTSV 进行归约操作
  Tensor allGreaterThan = Reduce(
      "allGreaterThan",  // 归约操作的名称
      {4},  // 归约的维度列表
      allGTSV,  // 待归约的张量对象
      [&](const auto& i, const auto& j) {  // 归约操作的 lambda 函数，对每个元素执行比较操作
        // 返回 CompareSelect::make(b.load(i, j), searchValue, kGT) 的结果
        return CompareSelect::make(b.load(i, j), searchValue, kGT);
      },
      {10});  // 归约操作的配置参数

  // 创建一个 LoopNest 对象，并将 allGreaterThan 张量传递给构造函数
  LoopNest loop2({allGreaterThan});
  // 准备代码生成前的准备工作
  loop2.prepareForCodegen();
  // 获取 LoopNest 对象中的根语句
  s = loop2.root_stmt();
  // 对根语句进行简化处理
  s = IRSimplifier::simplify(s);

  // 创建 SimpleIREvaluator 对象 cg2，使用简化后的语句 s 和输入参数列表 {b, allGreaterThan, searchValue}
  SimpleIREvaluator cg2(s, {b, allGreaterThan, searchValue});

  // 使用 cg2 调用评估器的 call 方法，传递输入参数列表 {in, out, 11}
  cg2.call({in, out, 11});

  // 断言输出张量 out 的值
  // 第一个元素应为 0
  ASSERT_EQ(out[0], 0);
  // 第二个元素应为 0
  ASSERT_EQ(out[1], 0);
  // 第三个元素应为 1
  ASSERT_EQ(out[2], 1);
  // 第四个元素应为 1
  ASSERT_EQ(out[3], 1);

  // 使用 cg2 再次调用评估器的 call 方法，传递输入参数列表 {in, out, -3}
  cg2.call({in, out, -3});

  // 断言输出张量 out 的值
  // 所有元素应为正数，因此都应为 1
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 1);
  ASSERT_EQ(out[3], 1);
}

// 定义测试用例 "ReduceMatmul2D"，测试矩阵乘法的降维操作
TEST(Reductions, ReduceMatmul2D) {
  // 定义两个缓冲区 tA 和 tB，分别表示 3x2 和 2x3 的浮点数矩阵
  BufHandle tA("tA", {3, 2}, kFloat);
  BufHandle tB("tB", {2, 3}, kFloat);

  // 初始化 tA_ 和 tB_ 为长度为 6 的浮点数向量
  std::vector<float> tA_(6);
  std::vector<float> tB_(6);

  // 初始化 out 为长度为 9 的浮点数向量，每个元素值为 -1.0
  std::vector<float> out(9, -1.f);
  
  // 使用两个嵌套循环填充 tA_ 和 tB_ 的数据
  for (const auto i : c10::irange(3)) {
    for (const auto j : c10::irange(2)) {
      tA_[i * 2 + j] = i * 2 + j;
      tB_[j * 3 + i] = i * 2 + j;
    }
  }

  // 定义一个张量 mm，表示降维后的矩阵乘法结果
  Tensor mm = Reduce(
      "mm",
      {3, 3},  // 结果张量的形状为 3x3
      Sum(),   // 使用 Sum() 函数对结果进行求和
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return tA.load(m, k) * tB.load(k, n);
      },
      {2});    // 指定降维的维度为 2

  // 创建一个循环嵌套对象 loop，并准备进行代码生成
  LoopNest loop({mm});
  loop.prepareForCodegen();

  // 获取最终的语句根节点
  StmtPtr s = loop.root_stmt();

  // 对语句进行简化处理
  s = IRSimplifier::simplify(s);

  // 创建一个简单的 IR 评估器对象 cg，用于评估计算结果
  SimpleIREvaluator cg(s, {tA, tB, mm});

  // 调用评估器进行计算，传入输入数据和输出数据的引用
  cg.call({tA_, tB_, out});

  // 期望的输出结果
  std::vector<float> expected(
      {1.f, 3.f, 5.f, 3.f, 13.f, 23.f, 5.f, 23.f, 41.f});

  // 断言计算结果与期望结果一致
  for (const auto i : c10::irange(9)) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

// 定义测试用例 "ReduceRfactorLike"，测试类似 Rfactor 操作的降维
TEST(Reductions, ReduceRfactorLike) {
  // 定义一个大小为 10x10 的浮点数缓冲区 in
  BufHandle in("in", {10, 10}, kFloat);

  // 初始化 in_ 为长度为 100 的浮点数向量，填充从 0 到 99 的整数
  std::vector<float> in_(100);
  for (const auto i : c10::irange(100)) {
    in_[i] = i;
  }

  // 初始化 in_rf_ 为长度为 10 的浮点数向量，每个元素值为 -2.0
  std::vector<float> in_rf_(10, -2.f);

  // 初始化 out 为长度为 1 的浮点数向量，其值为 -1.0
  std::vector<float> out(1, -1.f);

  // 定义一个张量 l1，表示对 in 进行降维求和操作
  Tensor l1 = Reduce("l1", {10}, Sum(), in, {10});

  // 从 l1 中获取缓冲区 in_rf，并定义一个张量 l2，对 in_rf 进行降维求和操作
  BufHandle in_rf(l1.buf());
  Tensor l2 = Reduce("l2", {}, Sum(), in_rf, {10});

  // 创建一个循环嵌套对象 loop，并准备进行代码生成
  LoopNest loop({l1, l2});
  loop.prepareForCodegen();

  // 获取最终的语句根节点
  StmtPtr s = loop.root_stmt();

  // 对语句进行简化处理
  s = IRSimplifier::simplify(s);

  // 创建一个简单的 IR 评估器对象 cg，用于评估计算结果
  SimpleIREvaluator cg(s, {in, l1, l2});

  // 调用评估器进行计算，传入输入数据和输出数据的引用
  cg.call({in_, in_rf_, out});

  // 断言计算结果与期望结果一致
  ASSERT_EQ(out[0], 99 * 50);
}

// 定义测试用例 "ReduceAsProducer"，测试降维操作作为生产者的情况
TEST(Reductions, ReduceAsProducer) {
  const int M = 10;  // 定义常量 M 的值为 10
  VarHandle m("m", kInt);  // 定义整数变量 m

  // 定义两个缓冲区 a 和 b，分别表示 2x3 和 2x3xM 的浮点数张量
  BufHandle a("a", {2, 3}, kFloat);
  BufHandle b("b", {2, 3, m}, kFloat);

  // 定义一个张量 c，表示对 b 进行降维求和操作
  Tensor c = Reduce("sum", {2, 3}, Sum(), b, {m});

  // 定义一个张量 d，表示根据 c 和 a 进行计算得到的结果
  Tensor d =
      Compute("scale", {2, 3}, [&](const VarHandle& l, const VarHandle& n) {
        return c.load(l, n) * a.load(l, n);
      });

  // 创建一个循环嵌套对象 loop，并准备进行代码生成
  LoopNest loop({d}, {c, d});
  loop.prepareForCodegen();

  // 获取最终的语句根节点
  StmtPtr s = loop.root_stmt();

  // 对语句进行简化处理
  s = IRSimplifier::simplify(s);

  // 创建一个简单的 IR 评估器对象 cg，用于评估计算结果
  SimpleIREvaluator cg(s, {a, b, d, m});

  // 初始化输入数据 aData, bData, dData 和常量 M
  std::vector<float> aData(2 * 3, 0);
  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> dData(2 * 3, 6.0f);

  // 填充输入数据 aData 和 bData
  for (int i = 0; i < 2 * 3; ++i) {
    aData[i] = 6 - i;
    for (const auto j : c10::irange(M)) {
      bData[i * M + j] = j;
    }
  }

  // 调用评估器进行计算，传入输入数据和输出数据的引用
  cg.call({aData, bData, dData, M});

  // 计算期望的输出结果
  float expected = 0;
  for (const auto i : c10::irange(M)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected += i;
  }
  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(dData[i], expected * (6 - i));
  }
}
TEST(Reductions, ReorderedReductionInitializer) {
  /* From the quip:
  for k in 0..1:  // blockIdx
  */

  // 创建名为"in"的缓冲区，形状为{16, 8}，元素类型为float
  BufHandle in("in", {16, 8}, kFloat);

  // 创建包含16 * 8个元素的浮点数向量in_
  std::vector<float> in_(16 * 8);
  // 填充向量in_，使得每个元素值等于其索引i的值
  for (const auto i : c10::irange(16)) {
    for (const auto j : c10::irange(8)) {
      in_[i * 8 + j] = i;
    }
  }

  // 创建包含16个元素的浮点数向量out，每个元素初始化为-1.0
  std::vector<float> out(16, -1.f);

  // 创建一个Reduce操作，命名为"sum"，对形状为{16}的in进行求和，指定轴{8}进行操作
  Tensor tensor = Reduce("sum", {16}, Sum(), in, {8});

  // 创建循环嵌套LoopNest对象，包含tensor张量
  LoopNest l({tensor});
  // 获取tensor张量的循环语句列表
  std::vector<ForPtr> loops = l.getLoopStmtsFor(tensor);
  // 在第二个循环(loops[1])上进行尾部分割，分割大小为2
  LoopNest::splitWithTail(loops[1], 2);

  // 为代码生成做准备
  l.prepareForCodegen();

  // 获取根语句s并简化
  StmtPtr s = l.root_stmt();
  s = IRSimplifier::simplify(s);

  // 创建SimpleIREvaluator对象cg，用于评估简化后的根语句s，传入参数列表{in, tensor}
  SimpleIREvaluator cg(s, {in, tensor});

  // 调用cg对象，传入aData和bData，结果存入dData
  cg.call({in_, out});

  // 对每个元素i进行遍历，验证out[i]的值是否等于i * 8
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(out[i], i * 8);
  }
}
  // 迭代范围为0到128的循环，针对某种并行计算环境
  for (m in 0..128):
    // 迭代范围为0到64的循环，表示线程索引
    for (n in 0..64):
      // 对张量执行求和操作，更新张量c中的元素，将张量a中的部分数据加到c中
      SumOp(c(k, n), 0, a(k, m, n), {m})

  /*
    此处是注释部分，不是代码
    注意：该代码段的具体实现和环境未提供，但是它包含了对张量操作的说明和并行计算的示例。
  */

  // 创建一个名为in的缓冲区对象，形状为{1, 12, 6}，数据类型为kFloat
  BufHandle in("in", {1, 12, 6}, kFloat);
  // 创建一个包含12*6个元素，每个元素值为1.0的向量
  std::vector<float> in_(12 * 6, 1.f);

  // 创建一个对张量进行求和操作的对象tensor_
  Tensor tensor_ = Reduce("sum", {1, 12}, Sum(), in, {6});
  // 创建一个循环嵌套对象l_，包含上述张量操作
  LoopNest l_({tensor_});

  // 准备进行代码生成的准备工作
  l_.prepareForCodegen();
  // 克隆并简化根语句s_，得到简化后的语句对象s_
  StmtPtr s_ = Stmt::clone(l_.root_stmt());
  s_ = IRSimplifier::simplify(s_);

  // 创建另一个对张量进行求和操作的对象tensor
  Tensor tensor = Reduce("sum", {1, 12}, Sum(), in, {6});
  // 创建一个循环嵌套对象l，包含上述张量操作
  LoopNest l({tensor});

  // 获取张量tensor的循环语句，并设置第一个循环的GPU块索引为0
  auto loops = l.getLoopStmtsFor(tensor);
  loops[0]->set_gpu_block_index(0);
  // 设置第二个循环的GPU线程索引为0
  loops[1]->set_gpu_thread_index(0);

  // 重新排序第二个和第三个循环的轴
  LoopNest::reorderAxis(loops[1], loops[2]);

  // 获取循环嵌套对象l的根语句并赋值给s
  StmtPtr s = l.root_stmt();
  // 简化语句s
  s = IRSimplifier::simplify(s);

  // 再次准备进行代码生成的准备工作
  l.prepareForCodegen();

  // 获取循环嵌套对象l的根语句并赋值给s
  s = l.root_stmt();
  // 简化语句s
  s = IRSimplifier::simplify(s);

  // 创建一个16个元素，每个元素值为-1.0的向量out1
  std::vector<float> out1(16, -1.f);
  // 创建一个简单的IR评估器对象cg，用于评估简化后的语句s_
  SimpleIREvaluator cg(s_, {in, tensor_});
  // 调用评估器，将in_作为输入，计算结果存入out1
  cg.call({in_, out1});

  // 创建一个16个元素，每个元素值为-1.0的向量out2
  std::vector<float> out2(16, -1.f);
  // 创建另一个简单的IR评估器对象cg2，用于评估简化后的语句s
  SimpleIREvaluator cg2(s, {in, tensor});
  // 调用评估器，将in_作为输入，计算结果存入out2
  cg2.call({in_, out2});

  // 使用断言验证out1和out2中的每个元素相等
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(out1[i], out2[i]);
  }
}

TEST(Reductions, ReduceRfactor) {
  const int M = 10;  // 定义常量 M，代表数组维度
  const int N = 10;  // 定义常量 N，代表数组维度
  VarHandle m("m", kInt);  // 创建名为 m 的整型变量句柄
  VarHandle n("n", kInt);  // 创建名为 n 的整型变量句柄

  BufHandle b("b", {m, n}, kFloat);  // 创建名为 b 的浮点型缓冲区句柄，大小为 m x n
  std::vector<float> in(M * N);  // 创建包含 M*N 个元素的浮点型向量 in
  for (int j = 0; j < M * N; ++j) {  // 循环填充 in 向量
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);  // 创建一个包含单个元素 (-1.0) 的浮点型向量 out

  Tensor c = Reduce("sum", {}, Sum(), b, {m, n});  // 创建一个名为 c 的求和张量，作用在 b 上，沿着 {m, n} 的维度
  LoopNest loop({c});  // 创建一个循环嵌套对象，包含张量 c
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取张量 c 的循环语句列表
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];  // 获取写入缓冲区 c.buf() 的所有写操作的第二个（索引为1）写入语句
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0)));  // 对 c_body 在 loops 的第一个循环处进行rfactor操作，返回操作是否成功
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());  // 在根语句中查找所有 ReduceOp 节点
  ASSERT_EQ(rc.size(), 2);  // 断言找到的 ReduceOp 节点数为 2
  loop.prepareForCodegen();  // 为代码生成做准备
  StmtPtr s = loop.root_stmt();  // 获取循环嵌套的根语句
  s = IRSimplifier::simplify(s);  // 对根语句进行简化

  SimpleIREvaluator cg(s, {b, c, m, n});  // 创建一个简单的 IR 评估器对象 cg

  cg.call({in, out, M, N});  // 调用评估器 cg 进行计算
  ASSERT_EQ(out[0], 4950);  // 断言计算结果正确
}

TEST(Reductions, Reduce3DRfactorInner) {
  const int M = 10;  // 定义常量 M，代表数组维度
  const int N = 10;  // 定义常量 N，代表数组维度
  const int K = 10;  // 定义常量 K，代表数组维度
  VarHandle m("m", kInt);  // 创建名为 m 的整型变量句柄
  VarHandle n("n", kInt);  // 创建名为 n 的整型变量句柄
  VarHandle k("k", kInt);  // 创建名为 k 的整型变量句柄

  BufHandle b("b", {m, n, k}, kFloat);  // 创建名为 b 的三维浮点型缓冲区句柄，大小为 m x n x k
  std::vector<float> in(M * N * K);  // 创建包含 M*N*K 个元素的浮点型向量 in
  for (int j = 0; j < M * N * K; ++j) {  // 循环填充 in 向量
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);  // 创建一个包含单个元素 (-1.0) 的浮点型向量 out

  Tensor c = Reduce("sum", {}, Sum(), b, {m, n, k});  // 创建一个名为 c 的求和张量，作用在 b 上，沿着 {m, n, k} 的维度
  LoopNest loop({c});  // 创建一个循环嵌套对象，包含张量 c
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取张量 c 的循环语句列表
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];  // 获取写入缓冲区 c.buf() 的所有写操作的第二个（索引为1）写入语句
  ASSERT_FALSE(loop.rfactor(c_body, loops.at(2)));  // 对 c_body 在 loops 的第三个循环处尝试rfactor操作，断言操作失败
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());  // 在根语句中查找所有 ReduceOp 节点
  ASSERT_EQ(rc.size(), 1);  // 断言找到的 ReduceOp 节点数为 1
  loop.prepareForCodegen();  // 为代码生成做准备
  StmtPtr s = loop.root_stmt();  // 获取循环嵌套的根语句
  s = IRSimplifier::simplify(s);  // 对根语句进行简化

  SimpleIREvaluator cg(s, {b, c, m, n, k});  // 创建一个简单的 IR 评估器对象 cg

  cg.call({in, out, M, N, K});  // 调用评估器 cg 进行计算
  ASSERT_EQ(out[0], 499500);  // 断言计算结果正确
}

TEST(Reductions, Reduce3DRfactorOuter) {
  const int M = 10;  // 定义常量 M，代表数组维度
  const int N = 10;  // 定义常量 N，代表数组维度
  const int K = 10;  // 定义常量 K，代表数组维度
  VarHandle m("m", kInt);  // 创建名为 m 的整型变量句柄
  VarHandle n("n", kInt);  // 创建名为 n 的整型变量句柄
  VarHandle k("k", kInt);  // 创建名为 k 的整型变量句柄

  BufHandle b("b", {m, n, k}, kFloat);  // 创建名为 b 的三维浮点型缓冲区句柄，大小为 m x n x k
  std::vector<float> in(M * N * K);  // 创建包含 M*N*K 个元素的浮点型向量 in
  for (int j = 0; j < M * N * K; ++j) {  // 循环填充 in 向量
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);  // 创建一个包含单个元素 (-1.0) 的浮点型向量 out

  Tensor c = Reduce("sum", {}, Sum(), b, {m, n, k});  // 创建一个名为 c 的求和张量，作用在 b 上，沿着 {m, n, k} 的维度
  LoopNest loop({c});  // 创建一个循环嵌套对象，包含张量 c
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取张量 c 的循环语句列表
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];  // 获取写入缓冲区 c.buf() 的所有写操作的第二个（索引为1）写入语句
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0)));  // 对 c_body 在 loops 的第一个循环处进行rfactor操作，返回操作是否成功
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());  // 在根语句中查找所有 ReduceOp 节点
  ASSERT_EQ(rc.size(), 2);  // 断言找到的 ReduceOp 节点数为 2
  loop.prepareForCodegen();  // 为代码生成做准备
  StmtPtr s = loop.root_stmt();  //
    // 使用原始循环对象 orig_loop 创建 LoopNest 对象 refloop
    LoopNest refloop(orig_loop);
    // 使用原始循环对象 orig_loop 创建 LoopNest 对象 loop
    LoopNest loop(orig_loop);
    // 为 refloop 执行代码生成前的准备工作
    refloop.prepareForCodegen();
    // 使用简化后的 refloop 根语句创建 SimpleIREvaluator 对象 ref_cg，输入为 in_ 和 c
    SimpleIREvaluator ref_cg(
        IRSimplifier::simplify(refloop.root_stmt()), {in_, c});
    // 调用 ref_cg 的计算函数，参数为 in 和 ref
    ref_cg.call({in, ref});

    // 获取临时缓冲区 tmp_buf 的指针
    BufPtr tmp_buf = c.buf();

    // 对于范围在 [0, rfac_number) 的每个 idx
    for (const auto idx : c10::irange(rfac_number)) {
      // 获取 loop 对 tmp_buf 执行写操作的所有语句，并选取第二个写操作
      auto reduce = loop.getAllWritesToBuf(tmp_buf)[1];
      // 断言在 loop 中对 reduce 进行 rfactor 操作，使用 tmp_buf 的第 idx 个循环语句
      ASSERT_TRUE(loop.rfactor(
          reduce, loop.getLoopStmtsFor(tmp_buf).at(idx), &tmp_buf));
    }

    // 为 loop 执行代码生成前的准备工作
    loop.prepareForCodegen();
    // 获取简化后的 loop 根语句，并赋给变量 s
    StmtPtr s = loop.root_stmt();
    // 对 s 进行进一步简化
    s = IRSimplifier::simplify(s);

    // 使用简化后的 s 和输入 in_、c 创建 SimpleIREvaluator 对象 cg
    SimpleIREvaluator cg(s, {in_, c});
    // 调用 cg 的计算函数，参数为 in 和 out
    cg.call({in, out});

    // 断言 ref 和 out 的第一个元素相等
    ASSERT_EQ(ref[0], out[0]);
}
// 以尾部循环分割一个归约轴。
TEST(Reductions, ReduceSplitTail) {
  const int M = 10;
  const int N = 10;
  const int K = 10;

  // 创建一个名为 b 的 BufHandle，表示一个形状为 {M, N, K} 的三维浮点数数组
  BufHandle b("b", {M, N, K}, kFloat);
  // 创建一个包含 M * N * K 个元素的浮点数向量 in，并对其进行初始化
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于 c10::irange(3) 返回的每个 i 值
  for (const auto i : c10::irange(3)) {
    // 创建一个长度为 M 的浮点数向量 out，初始值为 -1
    std::vector<float> out(M, -1.f);

    // 对 b 进行 Reduce 操作，将结果存入 Tensor c 中，采用 Sum() 归约操作符，对应的轴为 {N, K}
    Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
    // 使用 Tensor c 构建循环嵌套对象 LoopNest
    LoopNest loop({c});
    // 获取与 Tensor c 相关的循环语句列表
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    // 对 loops[i] 所表示的循环进行尾部分割，分割因子为 8
    LoopNest::splitWithTail(loops[i], 8);

    // 准备循环嵌套以进行代码生成
    loop.prepareForCodegen();
    // 获取根语句并简化之
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    // 创建 SimpleIREvaluator 对象 cg，用于评估简化后的 IR（中间表示）代码
    SimpleIREvaluator cg(s, {b, c});

    // 调用评估器 cg，传入输入向量 in 和输出向量 out，执行评估
    cg.call({in, out});
    // 断言检查 out[0] 的值是否等于预期的 4950
    ASSERT_EQ(out[0], 4950);
  }
}

// 以无尾部循环分割一个归约轴。
TEST(Reductions, ReduceSplitNoTail) {
  const int M = 10;
  const int N = 10;
  const int K = 10;
  
  // 创建一个名为 b 的 BufHandle，表示一个形状为 {M, N, K} 的三维浮点数数组
  BufHandle b("b", {M, N, K}, kFloat);
  // 创建一个包含 M * N * K 个元素的浮点数向量 in，并对其进行初始化
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于 c10::irange(3) 返回的每个 i 值
  for (const auto i : c10::irange(3)) {
    // 创建一个长度为 M 的浮点数向量 out，初始值为 -1
    std::vector<float> out(M, -1.f);

    // 对 b 进行 Reduce 操作，将结果存入 Tensor c 中，采用 Sum() 归约操作符，对应的轴为 {N, K}
    Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
    // 使用 Tensor c 构建循环嵌套对象 LoopNest
    LoopNest loop({c});
    // 获取与 Tensor c 相关的循环语句列表
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    // 对 loops[i] 所表示的循环进行无尾部分割，分割因子为 5
    LoopNest::splitWithTail(loops[i], 5);

    // 准备循环嵌套以进行代码生成
    loop.prepareForCodegen();
    // 获取根语句并简化之
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    // 创建 SimpleIREvaluator 对象 cg，用于评估简化后的 IR（中间表示）代码
    SimpleIREvaluator cg(s, {b, c});

    // 调用评估器 cg，传入输入向量 in 和输出向量 out，执行评估
    cg.call({in, out});
    // 断言检查 out[0] 的值是否等于预期的 4950
    ASSERT_EQ(out[0], 4950);
  }
}

// 以只有尾部循环的方式分割一个归约轴（分割后的循环大小为 0，将被消除）。
TEST(Reductions, ReduceOverSplitTail) {
  const int M = 10;
  const int N = 10;
  const int K = 10;

  // 创建一个名为 b 的 BufHandle，表示一个形状为 {M, N, K} 的三维浮点数数组
  BufHandle b("b", {M, N, K}, kFloat);
  // 创建一个包含 M * N * K 个元素的浮点数向量 in，并对其进行初始化
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于 c10::irange(3) 返回的每个 i 值
  for (const auto i : c10::irange(3)) {
    // 创建一个长度为 M 的浮点数向量 out，初始值为 -1
    std::vector<float> out(M, -1.f);

    // 对 b 进行 Reduce 操作，将结果存入 Tensor c 中，采用 Sum() 归约操作符，对应的轴为 {N, K}
    Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
    // 使用 Tensor c 构建循环嵌套对象 LoopNest
    LoopNest loop({c});
    // 获取与 Tensor c 相关的循环语句列表
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    // 对 loops[i] 所表示的循环进行尾部分割，分割因子为 16
    LoopNest::splitWithTail(loops[i], 16);

    // 准备循环嵌套以进行代码生成
    loop.prepareForCodegen();
    // 获取根语句并简化之
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    // 创建 SimpleIREvaluator 对象 cg，用于评估简化后的 IR（中间表示）代码
    SimpleIREvaluator cg(s, {b, c});

    // 调用评估器 cg，传入输入向量 in 和输出向量 out，执行评估
    cg.call({in, out});
    // 断言检查 out[0] 的值是否等于预期的 4950
    ASSERT_EQ(out[0], 4950);
  }
}

// 以掩码分割一个归约轴。
TEST(Reductions, ReduceSplitMask) {
  const int M = 10;
  const int N = 10;
  const int K = 10;

  // 创建一个名为 b 的 BufHandle，表示一个形状为 {M, N, K} 的三维浮点数数组
  BufHandle b("b", {M, N, K}, kFloat);
  // 创建一个包含 M * N * K 个元素的浮点数向量 in，并对其进行初始化
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于 c10::irange(3) 返回的每个 i 值
  for (const auto i : c10::irange(3)) {
    // 创建一个长度为 M 的浮点数向量 out，初始值为 -1
    std::vector<float> out(M, -
// 测试在无需使用掩码的情况下分割一个归约轴。
TEST(Reductions, ReduceSplitNoMask) {
  const int M = 10;
  const int N = 10;
  const int K = 10;
  BufHandle b("b", {M, N, K}, kFloat);  // 创建一个名为 b 的缓冲区，大小为 {M, N, K}，数据类型为 kFloat
  std::vector<float> in(M * N * K);     // 创建一个大小为 M*N*K 的 float 向量 in

  // 填充向量 in，使其按顺序包含从 0 到 M*N*K-1 的数值
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于范围为 {0, 1, 2} 的每个 i
  for (const auto i : c10::irange(3)) {
    std::vector<float> out(M, -1.f);  // 创建一个大小为 M 的 float 向量 out，初始值为 -1.0

    // 执行归约操作，求和所有 b 的元素，并沿指定轴 {N, K} 归约
    Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
    LoopNest loop({c});  // 创建一个包含 c 的 LoopNest 对象
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取用于 c 的循环语句

    // 在第 i 个循环中使用 splitWithMask 方法，分割长度为 5
    LoopNest::splitWithMask(loops[i], 5);

    loop.prepareForCodegen();  // 准备进行代码生成
    StmtPtr s = loop.root_stmt();  // 获取根语句
    s = IRSimplifier::simplify(s);  // 简化中间表示的语句

    // 创建一个 SimpleIREvaluator 对象，使用简化后的语句 s 和参数 {b, c}
    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});  // 调用评估器进行计算
    ASSERT_EQ(out[0], 4950);  // 断言计算结果是否正确
  }
}

// 使用掩码完成分割一个归约轴。
TEST(Reductions, ReduceOverSplitMask) {
  const int M = 10;
  const int N = 10;
  const int K = 10;

  BufHandle b("b", {M, N, K}, kFloat);  // 创建一个名为 b 的缓冲区，大小为 {M, N, K}，数据类型为 kFloat
  std::vector<float> in(M * N * K);     // 创建一个大小为 M*N*K 的 float 向量 in

  // 填充向量 in，使其按顺序包含从 0 到 M*N*K-1 的数值
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  // 对于范围为 {0, 1, 2} 的每个 i
  for (const auto i : c10::irange(3)) {
    std::vector<float> out(M, -1.f);  // 创建一个大小为 M 的 float 向量 out，初始值为 -1.0

    // 执行归约操作，求和所有 b 的元素，并沿指定轴 {N, K} 归约
    Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
    LoopNest loop({c});  // 创建一个包含 c 的 LoopNest 对象
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取用于 c 的循环语句

    // 在第 i 个循环中使用 splitWithMask 方法，分割长度为 16
    LoopNest::splitWithMask(loops[i], 16);

    loop.prepareForCodegen();  // 准备进行代码生成
    StmtPtr s = loop.root_stmt();  // 获取根语句
    s = IRSimplifier::simplify(s);  // 简化中间表示的语句

    // 创建一个 SimpleIREvaluator 对象，使用简化后的语句 s 和参数 {b, c}
    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});  // 调用评估器进行计算
    ASSERT_EQ(out[0], 4950);  // 断言计算结果是否正确
  }
}

// 测试在存在 splitWithTail 的情况下进行 rfactor，当图中存在两个 ReduceOps 时。
TEST(Reductions, ReduceSplitRfactor) {
  const int M = 2;
  const int N = 10;
  const int K = 10;
  const int SPLIT_FACTOR = 4;

  BufHandle b("b", {M, N, K}, kFloat);  // 创建一个名为 b 的缓冲区，大小为 {M, N, K}，数据类型为 kFloat
  std::vector<float> in(M * N * K);     // 创建一个大小为 M*N*K 的 float 向量 in

  // 填充向量 in，使其包含从 0 到 M*N*K-1 的数值
  for (const auto m : c10::irange(M)) {
    for (int j = 0; j < N * K; ++j) {
      in[m * N * K + j] = j;
    }
  }

  std::vector<float> out(M, -1.f);  // 创建一个大小为 M 的 float 向量 out，初始值为 -1.0

  // 执行归约操作，求和所有 b 的元素，并沿指定轴 {N, K} 归约
  Tensor c = Reduce("sum", {M}, Sum(), b, {N, K});
  LoopNest loop({c});  // 创建一个包含 c 的 LoopNest 对象
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);  // 获取用于 c 的循环语句

  // 在第 2 个循环中使用 splitWithTail 方法，分割因子为 SPLIT_FACTOR
  LoopNest::splitWithTail(loops[2], SPLIT_FACTOR);

  auto c_body = loop.getAllWritesToBuf(c.buf())[2];  // 获取归约体 c_body
  auto all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());  // 获取所有写入缓冲区 c 的循环嵌套
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(2).size() == 3);  // 断言循环嵌套的结构
  LoopNest::reorderAxis(all_loops[2][1], all_loops[2][2]);  // 重新排序循环轴
  all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());  // 获取重新排序后的所有循环嵌套
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(2).size() == 3);  // 再次断言循环嵌套的结构
  ASSERT_TRUE(loop.rfactor(c_body, all_loops[2][1]));  // 执行 rfactor 操作
  loop.prepareForCodegen();  // 准备进行代码生成
  loop.simplify();  // 简化循环嵌套
  StmtPtr s = loop.root_stmt();  // 获取根语句

  // 创建一个 SimpleIREvaluator 对象，使用简化后的语句 s 和参数 {b, c}
  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});  // 调用评估器进行计算
  for (const auto i : c10::irange(M)) {
    (void)i;  // 抑制未使用变量警告
    ASSERT_EQ(out[0], 4950);  // 断言计算结果是否正确
  }
}
TEST(Reductions, ReduceOverSplitRfactor) {
  // 定义常量 N、K 和 SPLIT_FACTOR
  const int N = 10;
  const int K = 10;
  const int SPLIT_FACTOR = 16;

  // 创建名为 b 的缓冲区，大小为 N*K，数据类型为 float
  BufHandle b("b", {N, K}, kFloat);
  // 创建包含 N*K 个元素的 float 类型向量 in，并初始化为 0 到 N*K-1
  std::vector<float> in(N * K);
  for (int j = 0; j < N * K; ++j) {
    in[j] = j;
  }

  // 创建包含单个元素且初始化为 -1.0 的 float 类型向量 out
  std::vector<float> out(1, -1.f);

  // 使用 Reduce 类创建一个名为 c 的 Tensor 对象，计算 b 的元素之和，维度为 {N, K}
  Tensor c = Reduce("sum", {}, Sum(), b, {N, K});
  // 创建 LoopNest 对象，并将 c 添加到其中
  LoopNest loop({c});
  // 获取循环语句列表，用于处理 c
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr i, t;
  // 使用 SPLIT_FACTOR 对 loops[1] 进行分割，并返回分割后的循环变量 i 和 t
  LoopNest::splitWithTail(loops[1], SPLIT_FACTOR, &i, &t);
  // 重新排序 loops[0] 的轴，将 i 插入其中
  LoopNest::reorderAxis(loops[0], i);

  // 获取所有写入到 c 缓冲区的循环嵌套，并将其存储在 all_loops 中
  auto all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());
  // 断言检查 all_loops 的大小为 3，且第二个元素的大小为 3
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(1).size() == 3);
  // 获取所有写入到 c 缓冲区的 c_body，并存储在 c_body 中
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  // 断言检查对 c_body 进行 rfactor，以及 all_loops[1][0] 是否成功
  ASSERT_TRUE(loop.rfactor(c_body, all_loops[1][0]));
  // 重新排序 all_loops[1][0] 的轴，将 all_loops[1][2] 插入其中
  LoopNest::reorderAxis(all_loops[1][0], all_loops[1][2]);

  // 准备进行代码生成前的准备工作
  loop.prepareForCodegen();
  // 简化循环嵌套
  loop.simplify();
  // 获取根语句指针 s
  StmtPtr s = loop.root_stmt();

  // 使用 SimpleIREvaluator 对象 cg，评估 s，并传入 b 和 c 作为参数
  SimpleIREvaluator cg(s, {b, c});

  // 调用 cg 的 call 方法，传入 in 和 out 作为参数
  cg.call({in, out});
  // 断言检查 out[0] 的值是否等于 4950
  ASSERT_EQ(out[0], 4950);

  // 创建 ostringstream 对象 oss
  std::ostringstream oss;
  // 将 cg.stmt() 的内容写入 oss 中
  oss << *cg.stmt();

  // 检查 IR，验证 rfactor 后的 reduce 是否被消除
  // TODO: 由于大小为 0，此处应该消除 alloc free
  /*
  const std::string& verification_pattern =
      R"IR(
# CHECK: Allocate(tmp_buf); // dtype=float, dims=[0]
# CHECK: sum[0] = 0.f;
# CHECK: for (int n = 0; n < 10; n++) {
# CHECK:   for (int k_tail = 0; k_tail < 10; k_tail++) {
# CHECK:     sum[0] = (sum[0]) + (b[k_tail + 10 * n]);
# CHECK:   }
# CHECK: }
# CHECK: Free(tmp_buf);)IR";
  */
  // TODO: rfactor 输出尚不一致，将进行修复 (@nickg)
  // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Reductions, ReduceInlineReduction) {
  // 定义常量 M、N 和 K
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 创建名为 a_buf 和 b_buf 的缓冲区，分别大小为 {M} 和 {M, N, K}，数据类型为 float
  BufHandle a_buf("a", {M}, kFloat);
  BufHandle b_buf("b", {M, N, K}, kFloat);

  // 使用 Reduce 类创建一个名为 x 的 Tensor 对象，计算 b_buf 的元素之和，维度为 {N, K}
  Tensor x = Reduce("x", {M}, Sum(), b_buf, {N, K});
  // 使用 Compute 类创建一个名为 y 的 Tensor 对象，计算 a_buf[m] + x[m]，维度为 {M}
  Tensor y = Compute(
      "y", {M}, [&](const VarHandle& m) { return a_buf.load(m) + x.load(m); });

  // 创建 PaddedBuffer 对象 a_v 和 b_v，分别大小为 {M} 和 {M, N, K}，数据类型为 float
  PaddedBuffer<float> a_v(M);
  PaddedBuffer<float> b_v(M, N, K);

  // 初始化 a_v 中的元素为 m*m，其中 m 的范围是 0 到 M-1
  for (const auto i : c10::irange(M)) {
    a_v(i) = i * i;
  }

  // 初始化 b_v 中的元素为 j*j*k，其中 j 的范围是 0 到 N-1，k 的范围是 0 到 K-1
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      for (const auto k : c10::irange(K)) {
        b_v(i, j, k) = j * j * k;
      }
    }
  }

  // 创建 LoopNest 对象 l1，包含 y 和 x，并传入 x 和 y 作为参数
  LoopNest l1({y}, {x});
  // 无法内联一个 reduction 计算
  ASSERT_FALSE(l1.computeInline(x.buf()));
}
TEST(Reductions, ReduceInlineConsumer) {
  // 定义常量 M、N、K 分别为 4、5、6
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 创建名为 a_buf 和 b_buf 的缓冲区对象，每个对象包含形状为 {M, N, K} 的浮点数数组
  BufHandle a_buf("a", {M, N, K}, kFloat);
  BufHandle b_buf("b", {M, N, K}, kFloat);

  // 创建张量 x，形状为 {M, N, K}，通过 lambda 函数加载 a_buf 和 b_buf 的数据并求和
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n, k) + b_buf.load(m, n, k);
      });

  // 创建张量 y，形状为 {M}，对张量 x 进行 Reduce 操作，按 Sum() 函数进行求和，指定轴为 {N, K}
  Tensor y = Reduce("y", {M}, Sum(), x, {N, K});

  // 创建填充缓冲区对象 a_v 和 b_v，各自形状为 {M, N, K}，并用指定公式填充数据
  PaddedBuffer<float> a_v(M, N, K);
  PaddedBuffer<float> b_v(M, N, K);

  // 嵌套循环填充 a_v 和 b_v 的数据
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      for (const auto k : c10::irange(K)) {
        a_v(i, j, k) = i * i + k;
        b_v(i, j, k) = j * j + k;
      }
    }
  }

  // 创建 LoopNest 对象 l1 和 l2，用于管理计算和优化
  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);
  // 在 l2 上将张量 x 进行内联计算
  l2.computeInline(x.buf());

  // 对 l1 和 l2 进行代码生成前的准备工作
  l1.prepareForCodegen();
  l2.prepareForCodegen();

  // 对 l1 和 l2 的根语句进行简化
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());

  // 创建 SimpleIREvaluator 对象 eval1 和 eval2，用于评估简化后的语句
  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, y});

  // 创建填充缓冲区对象 y_1 和 y_2，形状为 {M}，用于存储评估结果
  PaddedBuffer<float> y_1(M);
  PaddedBuffer<float> y_2(M);

  // 使用 eval1 和 eval2 对填充缓冲区 a_v、b_v 和 y_1、y_2 进行评估
  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);

  // 断言 y_1 和 y_2 的结果在一定精度范围内接近
  ExpectAllNear(y_1, y_2, 1e-5);

  // 创建 ostringstream 对象 oss1 和 oss2，用于将简化后的语句转换为字符串
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;

  // 断言 oss1 的字符串长度大于 oss2 的字符串长度
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

TEST(Reductions, ReduceInlineReducerInternal) {
  // 定义常量 M、N、K 分别为 4、5、6
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 创建名为 a_buf 和 b_buf 的缓冲区对象，每个对象包含形状为 {M, N, K} 的浮点数数组
  BufHandle a_buf("a", {M, N, K}, kFloat);
  BufHandle b_buf("b", {M, N, K}, kFloat);

  // 创建张量 x，形状为 {M, N, K}，通过 lambda 函数加载 a_buf 和 b_buf 的数据并求和
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n, k) + b_buf.load(m, n, k);
      });

  // 创建 Reducer 对象 minimum，初始值为 0，用 lambda 函数定义 Reduce 操作
  Reducer minimum(ExprHandle(0.f), [&](ExprHandle a, ExprHandle b) {
    return Add::make(ExprHandle(1.f), Min::make(a, b, false));
  });

  // 创建张量 y，形状为 {M}，对张量 x 进行 Reduce 操作，按 minimum 函数进行求解，指定轴为 {N, K}
  Tensor y = Reduce("y", {M}, minimum, x, {N, K});

  // 创建填充缓冲区对象 a_v 和 b_v，各自形状为 {M, N, K}，并用指定公式填充数据
  PaddedBuffer<float> a_v(M, N, K);
  PaddedBuffer<float> b_v(M, N, K);

  // 嵌套循环填充 a_v 和 b_v 的数据
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      for (const auto k : c10::irange(K)) {
        a_v(i, j, k) = i * i + k;
        b_v(i, j, k) = j * j + k;
      }
    }
  }

  // 创建 LoopNest 对象 l1 和 l2，用于管理计算和优化
  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);
  // 在 l2 上将张量 x 进行内联计算
  l2.computeInline(x.buf());

  // 对 l1 和 l2 进行代码生成前的准备工作
  l1.prepareForCodegen();
  l2.prepareForCodegen();

  // 对 l1 和 l2 的根语句进行简化
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());

  // 创建 SimpleIREvaluator 对象 eval1 和 eval2，用于评估简化后的语句
  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, y});

  // 创建填充缓冲区对象 y_1 和 y_2，形状为 {M}，用于存储评估结果
  PaddedBuffer<float> y_1(M);
  PaddedBuffer<float> y_2(M);

  // 使用 eval1 和 eval2 对填充缓冲区 a_v、b_v 和 y_1、y_2 进行评估
  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);

  // 断言 y_1 和 y_2 的结果在一定精度范围内接近
  ExpectAllNear(y_1, y_2, 1e-5);

  // 创建 ostringstream 对象 oss1 和 oss2，用于将简化后的语句转换为字符串
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;

  // 断言 oss1 的字符串长度大于 oss2 的字符串长度
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}
TEST(Reductions, ReductionCacheAccessesOperatorAxis) {
  // 定义测试用例中的维度参数
  int L = 4;
  int N = 3;
  int M = 2;

  // 创建名为'a'和'b'的缓冲区对象，每个对象包含维度信息和数据类型
  BufHandle a("a", {L, N, M}, kFloat);
  BufHandle b("b", {L, N, M}, kFloat);

  // 定义张量'c'，根据元素级运算生成新的张量，用lambda表达式计算每个元素的值
  Tensor c = Compute(
      "scale",
      {L, N, M},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });

  // 定义张量'd'，对张量'c'进行按维度'L'的归约求和操作
  Tensor d = Reduce("sum", {L}, Sum(), c, {N, M});

  // 定义张量'e'，根据张量'd'和'b'的元素生成新的张量，用lambda表达式计算每个元素的值
  Tensor e = Compute("scale", {L}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  // 创建LoopNest对象'l'，传入需要计算的张量数组和依赖的张量数组
  LoopNest l({e}, {c, d, e});

  // 复制LoopNest对象'l'到'l_before'
  LoopNest l_before(l);

  // 为'l_before'对象准备代码生成
  l_before.prepareForCodegen();

  // 创建SimpleIREvaluator对象'cg_before'，使用'sanitizeNames'函数处理语句，传入输入数据张量数组
  SimpleIREvaluator cg_before(
      LoopNest::sanitizeNames(l_before.root_stmt()), {a, b, e});

  // 获取张量'd'对应的循环语句，并缓存访问
  StmtPtr d_loop = l.getLoopStmtsFor(d)[0];
  l.cacheAccesses(d.buf(), "d_local", d_loop);

  // 准备LoopNest对象'l'进行代码生成
  l.prepareForCodegen();

  // 简化根语句，传入的结果保存在'result'中
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));

  // 创建SimpleIREvaluator对象'cg_after'，使用简化后的语句'result'，传入输入数据张量数组
  SimpleIREvaluator cg_after(result, {a, b, e});

  // 创建ostringstream对象'oss'，将'cg_after'的语句输出到字符串流中
  std::ostringstream oss;
  oss << *cg_after.stmt();

  // 定义预期的IR字符串，使用R"IR(...)"语法，用于后续的IR检查
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(d_local); // dtype=float, dims=[4]
#CHECK: for (int i_2
#CHECK:   d_local[i_2] = 0.f
#CHECK:   for (int
#CHECK:     for (int
#CHECK:       d_local[i_2] = (d_local[i_2]) + (scale[
#CHECK:     }
#CHECK:   }
#CHECK: }
#CHECK: for (int i_3
#CHECK:   sum[i_3] = d_local[i_3]
#CHECK: Free(d_local);
#CHECK-NOT: d_local
      )IR";

  // 使用FileCheck类的run方法检查生成的IR语句是否符合预期
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  // 创建PaddedBuffer对象，用于存储测试数据
  PaddedBuffer<float> a_v(L, M, N, "a");
  PaddedBuffer<float> b_v(L, M, N, "b");
  PaddedBuffer<float> c_v(L, M, N, "c");
  PaddedBuffer<float> d_v(L, "d");
  PaddedBuffer<float> e_before(L, "e_before");
  PaddedBuffer<float> e_after(L, "e_after");

  // 使用at::randn方法生成随机数据填充a_v和b_v缓冲区
  for (const auto l : c10::irange(L)) {
    for (const auto m : c10::irange(M)) {
      for (const auto n : c10::irange(N)) {
        a_v(l, m, n) = at::randn({1}).item().to<float>();
        b_v(l, m, n) = at::randn({1}).item().to<float>();
      }
    }
  }

  // 调用cg_before对象的call方法，执行代码生成前的计算
  cg_before.call({a_v, b_v, e_before});

  // 调用cg_after对象的call方法，执行代码生成后的计算
  cg_after.call({a_v, b_v, e_after});

  // 使用ExpectAllNear方法比较e_before和e_after两个缓冲区的数值是否在指定精度内相等
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);
}
    // 返回表达式的计算结果，使用加载函数 `load` 从缓冲区 `b` 中加载数据，并乘以从缓冲区 `d` 中加载的数据
    return b.load(0, 0, l) * d.load(l);
  });

  // 创建循环嵌套对象 `l`，将变量 `e` 作为外层循环
  LoopNest l({e}, {c, d, e});
  // 复制 `l` 对象到 `l_before` 中
  LoopNest l_before(l);
  // 为 `l_before` 对象准备代码生成
  l_before.prepareForCodegen();
  // 使用 `l_before` 对象的根语句创建简单的IR求值器 `cg_before`，使用变量 `a`, `b`, `e`
  SimpleIREvaluator cg_before(l_before.root_stmt(), {a, b, e});

  // 获取循环 `d` 的循环语句，并存储在 `d_loop` 中
  StmtPtr d_loop = l.getLoopStmtsFor(d)[1];
  // 在循环嵌套 `l` 中为缓冲区 `d` 的访问缓存，命名为 "d_local"，作用域为 `d_loop`
  l.cacheAccesses(d.buf(), "d_local", d_loop);
  // 为循环嵌套 `l` 准备代码生成
  l.prepareForCodegen();

  // 对 `l` 的根语句进行简化和名字清理，得到 `result`
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  // 使用简化后的语句 `result` 创建简单的IR求值器 `cg_after`，使用变量 `a`, `b`, `e`
  SimpleIREvaluator cg_after(result, {a, b, e});

  // 创建一个字符串输出流 `oss`
  std::ostringstream oss;
  // 将 `cg_after` 的语句输出到字符串流 `oss` 中
  oss << *cg_after.stmt();
  // 定义期望的IR字符串 `expected_ir`
  const std::string& expected_ir =
      R"IR(
#`
// 分配内存给变量 d_local，数据类型为 float，维度为 [1]
#CHECK: Allocate(d_local); // dtype=float, dims=[1]
// 将 sum[i_1] 初始化为 0
#CHECK: sum[i_1] = 0
// 将 d_local[0] 设置为 sum[i_1] 的值
#CHECK: d_local[0] = sum[i_1]
// 循环遍历 j_1
#CHECK: for (int j_1
// 在 j_1 循环内部，遍历 k_1
#CHECK:   for (int k_1
// 将 d_local[0] 更新为 (d_local[0]) + (scale[ 的值
#CHECK: d_local[0] = (d_local[0]) + (scale[
// 结束 k_1 循环
#CHECK:   }
// 结束 j_1 循环
#CHECK: }
// 将 sum[i_1] 更新为 d_local[0] 的值
#CHECK: sum[i_1] = d_local[0]
// 释放变量 d_local 的内存
#CHECK: Free(d_local);
// 确保在预期的 IR 中不包含 d_local 的任何引用
#CHECK-NOT: d_local
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  PaddedBuffer<float> a_v(L, M, N, "a");
  PaddedBuffer<float> b_v(L, M, N, "b");
  PaddedBuffer<float> c_v(L, M, N, "c");
  PaddedBuffer<float> d_v(L, "d");
  PaddedBuffer<float> e_before(L, "e_before");
  PaddedBuffer<float> e_after(L, "e_after");

  // 为 a_v 和 b_v 分配随机数值
  for (const auto l : c10::irange(L)) {
    for (const auto m : c10::irange(M)) {
      for (const auto n : c10::irange(N)) {
        a_v(l, m, n) = at::randn({1}).item().to<float>();
        b_v(l, m, n) = at::randn({1}).item().to<float>();
      }
    }
  }

  // 调用 cg_before 和 cg_after，传入 a_v, b_v, e_before 和 a_v, b_v, e_after
  cg_before.call({a_v, b_v, e_before});
  cg_after.call({a_v, b_v, e_after});

  // 检查 e_before 和 e_after 之间的接近程度
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);
}
    }
  }



// 这里可能是函数或代码块的结尾，根据缩进可以推测出当前位于某个代码块的末尾



  cg_before.call({a_v, b_v, e_before});
  cg_after.call({a_v, b_v, e_after});



// 调用 cg_before 和 cg_after 函数，并传入参数 {a_v, b_v, e_before} 和 {a_v, b_v, e_after}，分别表示调用前和调用后的参数组合



  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);



// 使用 NOLINTNEXTLINE 来告知代码静态分析工具忽略此处的特定警告，具体是关于不使用魔法数字的警告。
// 调用 ExpectAllNear 函数，用来断言 e_before 和 e_after 两个值在 1e-5 的精度范围内是否接近。
TEST(Reductions, ReductionSplitCacheConsumerAccess) {
  // 创建缓冲区对象 a 和 b，各自大小为 24x32x12，数据类型为 float
  BufHandle a("a", {24, 32, 12}, kFloat);
  BufHandle b("b", {24, 32, 12}, kFloat);

  // 计算张量 c，形状为 24x32x12，元素为 b.load(l, n, m) * a.load(l, n, m)
  Tensor c = Compute(
      "scale",
      {24, 32, 12},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });

  // 对张量 c 进行减少操作，生成张量 d，形状为 24，使用 Sum() 函数进行求和，轴为 {32, 12}
  Tensor d = Reduce("sum", {24}, Sum(), c, {32, 12});

  // 计算张量 e，形状为 24，元素为 b.load(0, 0, l) * d.load(l)
  Tensor e = Compute("scale", {24}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  // 创建 LoopNest 对象 l，包含张量 {e}，依赖张量 {c, d, e}
  LoopNest l({e}, {c, d, e});

  // 在张量 e 的第一个循环语句中进行分裂，划分为 4 份
  LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4);

  // 获取张量 e 的循环语句，并为其创建缓存访问，使用缓冲区 d.buf()，缓存名称为 "sum_local"，应用于循环 e_loop
  StmtPtr e_loop = l.getLoopStmtsFor(e)[1];
  l.cacheAccesses(d.buf(), "sum_local", e_loop);

  // 准备 LoopNest 对象 l 进行代码生成
  l.prepareForCodegen();

  // 简化并清理 IR 根语句，并为其创建一个 SimpleIREvaluator 对象 cg
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {a, b, e});

  // 创建一个 ostringstream 对象 oss，将 cg.stmt() 的内容写入其中
  std::ostringstream oss;
  oss << *cg.stmt();

  // 定义预期的 IR 字符串 expected_ir
  const std::string& expected_ir =
      R"IR(
#CHECK: Alias(sum_local,scale);
#CHECK: sum[i_1] = (sum[i_1]) + (scale[
#CHECK: for (int j_2 = 0; j_2 < 4
#CHECK:   sum_local[j_2] = sum[j_2 + 4 * i_2];
#CHECK:   scale_1[j_3 + 4 * i_2] = (b[j_3 + 4 * i_2]) * (sum_local[j_3]);
      )IR";

  // 使用 FileCheck 进行预期 IR 字符串的验证，与 oss.str() 的内容比较
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}
  return b.load(0, 0, l) * d.load(l);
});

LoopNest l({e}, {c, d, e});

// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
// 声明一个内部循环指针
ForPtr inner;

// 分割外部的归约轴。
LoopNest::splitWithMask(l.getLoopStmtsFor(d)[0], 4, &inner);

// 分割归约消费者。
LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4, &inner);

// 在循环嵌套中缓存对数据对象 d 的访问，使用内部循环指针 inner
l.cacheAccesses(d.buf(), "sum_local", inner);

// 准备进行代码生成
l.prepareForCodegen();

// 对循环嵌套的根语句进行简化和名称清理
StmtPtr result = LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));

// 创建一个简单的IR求值器，用于评估 result 的计算结果，传入变量 a, b, e
SimpleIREvaluator cg(result, {a, b, e});

// reduction changes but cache does not.
// 创建一个字符串流对象 oss，并将 cg 的语句内容输出到 oss 中
std::ostringstream oss;
oss << *cg.stmt();

// 预期的中间表示字符串
const std::string& expected_ir =
    R"IR(
#CHECK: Alias(sum_local,scale);
#CHECK:         sum[j_1 + 4 * i_1] = (sum[j_1 + 4 * i_1]) + (scale[((l + 12 * k_1) + 1536 * i_1) + 384 * j_1]);
#CHECK: for (int i_2 = 0; i_2 < 6
#CHECK:   for (int j_2 = 0; j_2 < 4
#CHECK:     sum_local[j_2] = sum[j_2 + 4 * i_2];
#CHECK:   for (int j_3 = 0; j_3 < 4
#CHECK:     scale_1[j_3 + 4 * i_2] = (b[j_3 + 4 * i_2]) * (sum_local[j_3]);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionReorderCacheConsumerAccess) {
  BufHandle a("a", {24, 32, 12}, kFloat);
  BufHandle b("b", {24, 32, 12}, kFloat);

  // 定义张量 c，表示 b 和 a 的逐元素乘积
  Tensor c = Compute(
      "scale",
      {24, 32, 12},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });

  // 定义张量 d，对张量 c 沿第一个维度进行求和
  Tensor d = Reduce("sum", {24}, Sum(), c, {32, 12});

  // 定义张量 e，表示张量 b 在第一和第二维度为零时，与张量 d 的逐元素乘积
  Tensor e = Compute("scale", {24}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  // 创建循环嵌套对象
  LoopNest l({e}, {c, d, e});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;

  // 重新排序外部约简轴
  auto loops = l.getLoopStmtsFor(d);
  LoopNest::reorderAxis(loops[0], loops[1]);

  // 将约简消费者分割
  LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4, &inner);

  // 在循环嵌套中缓存访问 d 的内存块，命名为 "sum_local"，应用于内部循环
  l.cacheAccesses(d.buf(), "sum_local", inner);

  // 为代码生成做准备
  l.prepareForCodegen();

  // 简化并消毒 IR 树的变量名
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));

  // 创建简单的 IR 评估器，用于检查生成的 IR 是否符合预期
  SimpleIREvaluator cg(result, {a, b, e});

  // 检查生成的 IR 是否与期望的 IR 匹配
  std::ostringstream oss;
  oss << *cg.stmt();
  const std::string& expected_ir =
      R"IR(
#CHECK:        sum[j_1] = (sum[j_1]) + (scale[(k_1 + 12 * i_2) + 384 * j_1]);
#CHECK:  for (int i_3 = 0; i_3 < 6;
#CHECK:    for (int j_2 = 0; j_2 < 4;
#CHECK:      sum_local[j_2] = sum[j_2 + 4 * i_3];
#CHECK:    for (int j_3 = 0; j_3 < 4;
#CHECK:      scale_1[j_3 + 4 * i_3] = (b[j_3 + 4 * i_3]) * (sum_local[j_3]);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionRfactorCacheTempOuter) {
  const int M = 10;
  const int N = 10;
  const int K = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle k("k", kInt);

  BufHandle b("B", {m, n, k}, kFloat);
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;



  }



  std::vector<float> out(1, -1.f);



  Tensor c = Reduce("sum", {}, Sum(), b, {m, n, k});



  LoopNest loop({c});



  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);



  LoopNest::reorderAxis(loops.at(0), loops.at(1));



  loops = loop.getLoopStmtsFor(c);



  auto c_body = loop.getAllWritesToBuf(c.buf())[1];



  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  BufPtr rfac_buf;



  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0), &rfac_buf));



  loop.distributeLoop(loops.at(0));



  auto all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);



  ASSERT_TRUE(all_loops.size() == 2 && all_loops.at(1).size() == 3);



  LoopNest::reorderAxis(all_loops[1][0], all_loops[1][1]);



  all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);



  LoopNest::cacheAccesses(rfac_buf, "tmp", all_loops[1][1]);



  loop.simplify();



  loop.prepareForCodegen();



  StmtPtr s = LoopNest::sanitizeNames(loop.root_stmt());



  SimpleIREvaluator cg(s, {b, c, m, n, k});



  std::ostringstream oss;



  oss << *cg.stmt();



  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_rfac); // 分配存储空间给 sum_rfac 变量，数据类型为 float，维度为 [n]
#CHECK: Allocate(tmp); // 分配存储空间给 tmp 变量，数据类型为 float，维度为 [1]
#CHECK: for (int i_1 = 0; i < m
#CHECK:   for (int j = 0; j < n
#CHECK:     tmp[0] = 0
#CHECK:     for (int k
#CHECK:       tmp[0] = (tmp[0]) + (B[
#CHECK:     }
#CHECK:   sum_rfac[j] = (sum_rfac[j]) + (tmp[0]);
#CHECK:   Free(tmp);
#CHECK-NOT: tmp
    }
  }  // 结束第二个嵌套循环

  // 创建长度为8的float向量，并初始化为-1
  std::vector<float> out_before(8, -1.f);
  // 创建长度为8的float向量，并初始化为-1
  std::vector<float> out_after(8, -1.f);

  // 创建名为"in"的BufHandle对象，大小为{8, 8}，类型为kFloat
  BufHandle in("in", {8, 8}, kFloat);

  // 对输入张量进行求和约简操作，创建Tensor对象
  Tensor tensor = Reduce("sum", {8}, Sum(), in, {8});
  // 创建对Tensor对象进行代码生成的LoopNest对象l_before
  LoopNest l_before({tensor});
  // 复制l_before创建新的LoopNest对象l
  LoopNest l(l_before);
  // 准备l_before对象以便进行代码生成
  l_before.prepareForCodegen();
  // 创建SimpleIREvaluator对象cg_before，用于执行l_before中的IR代码
  SimpleIREvaluator cg_before(l_before.root_stmt(), {in, tensor});
  // 调用cg_before对象，计算输入数据in_的IR，并将结果写入out_before向量
  cg_before.call({in_, out_before});

  // 断言对tensor相关的循环进行向量化优化
  ASSERT_TRUE(LoopNest::vectorize(l.getLoopStmtsFor(tensor)[0]));

  // 获取循环嵌套结构的根语句，并进行简化和重命名
  StmtPtr s = l.root_stmt();
  s = LoopNest::sanitizeNames(IRSimplifier::simplify(s));

  // 创建ostringstream对象oss，将简化后的IR代码流输出到oss中
  std::ostringstream oss;
  oss << *s;
  // 期望的IR代码，使用R"IR(...)"语法表示
  const std::string& expected_ir =
      R"IR(
// 检查预期的IR生成代码是否包含下述语句，并将其验证
const std::string& expected_ir =
    R"IR(
#CHECK: sum = 0.f;
#CHECK: for (int i = 0; i < 8; i++) {
#CHECK:   sum_rfac[i] = 0.f;
#CHECK: }
#CHECK: for (int i_1 = 0; i_1 < 8; i_1++) {
#CHECK:   sum_rfac[Ramp(0, 1, 8)] = ReduceOp((sum_rfac[Ramp(0, 1, 8)]) + (in[Ramp(8 * i_1, 1, 8)]), reduce_args={i_1});
#CHECK: }
#CHECK: for (int i_2 = 0; i_2 < 8; i_2++) {
#CHECK:   sum = ReduceOp((sum) + (sum_rfac[i_2]), reduce_args={i_2});
#CHECK: }
    )IR";
torch::jit::testing::FileCheck().run(expected_ir, oss.str());

// 为代码生成准备，简化IR并创建简化后的评估器
l.prepareForCodegen();
s = IRSimplifier::simplify(l.root_stmt());
SimpleIREvaluator cg_after(s, {in, tensor});
// 调用评估器并比较前后的输出结果
cg_after.call({in_, out_after});

// 断言向量化后的输出与未向量化的输出相等
ASSERT_EQ(out_before[0], out_after[0]);
}
TEST(Reductions, InitFunction) {
  constexpr int M = 32;  // 定义常量 M，表示数组 A 的行数
  constexpr int N = 16;  // 定义常量 N，表示数组 B 的元素个数
  BufHandle A("A", {M, N}, kFloat);  // 创建名为 A 的缓冲区，大小为 M × N，元素类型为浮点数
  BufHandle B("B", {N}, kFloat);  // 创建名为 B 的缓冲区，大小为 N，元素类型为浮点数
  Tensor C = Reduce(
      "C",
      {N},
      Sum(),
      // 返回 B 的元素
      [&](const std::vector<VarHandle>& v) { return B.load(v[0]); },
      // 返回 A 的元素，通过参数 v[1] 表示行索引，参数 v[0] 表示列索引
      [&](const std::vector<VarHandle>& v) { return A.load(v[1], v[0]); },
      {M});  // 对于数组 A，迭代范围是 0 到 M-1
  LoopNest nest({C});  // 创建一个循环嵌套对象，包含张量 C
  nest.prepareForCodegen();  // 准备进行代码生成的循环嵌套对象
  StmtPtr s = LoopNest::sanitizeNames(IRSimplifier::simplify(nest.root_stmt()));  // 简化和重命名循环嵌套的根语句
  std::ostringstream oss;  // 创建一个字符串流对象 oss
  oss << *s << "\n";  // 将简化后的语句输出到字符串流 oss 中
  const std::string& expected_ir =
      R"IR(
#CHECK:  for (int i = 0; i < 16; i++) {
#CHECK:    C[i] = B[i];
#CHECK:    for (int j = 0; j < 32; j++) {
#CHECK:      C[i] = (C[i]) + (A[i + 16 * j]);
#CHECK:    }
#CHECK:  }
      )IR";  // 定义预期的中间表示（IR）字符串，用于检查生成的代码
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());  // 运行 IR 字符串检查，比较生成的代码和预期的 IR
}
} // namespace jit
} // namespace torch
```