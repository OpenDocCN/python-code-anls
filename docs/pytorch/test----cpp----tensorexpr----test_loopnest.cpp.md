# `.\pytorch\test\cpp\tensorexpr\test_loopnest.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入需要使用的其他自定义测试工具和类的头文件
#include <test/cpp/tensorexpr/test_base.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>

// 引入需要测试的头文件和命名空间
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

// 定义使用的命名空间
namespace torch {
namespace jit {

// 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

// 函数用于检查 StmtPtr 对象生成的 IR 是否符合给定模式
void checkIR(StmtPtr s, const std::string& pattern) {
  // 将生成的 IR 写入到字符串流中
  std::ostringstream oss;
  oss << *s;
  // 使用 FileCheck 工具运行模式匹配
  torch::jit::testing::FileCheck().run(pattern, oss.str());
}

// 函数用于检查 ExprPtr 对象生成的 IR 是否符合给定模式
void checkExprIR(ExprPtr e, const std::string& pattern) {
  // 构造带有注释的模式字符串
  std::string prefixed_pattern = "# CHECK: " + pattern + "\n";
  // 将生成的 IR 写入到字符串流中
  std::ostringstream oss;
  oss << *e << "\n";
  // 使用 FileCheck 工具运行模式匹配
  torch::jit::testing::FileCheck().run(prefixed_pattern, oss.str());
}

// 对 ExprHandle 进行重载的函数，调用 checkExprIR(ExprPtr, pattern) 进行匹配
void checkExprIR(const ExprHandle& e, const std::string& pattern) {
  checkExprIR(e.node(), pattern);
}

// 测试用例，测试简单表达式计算的情况
TEST(LoopNest, ExprSimple01) {
  // 创建一个计算张量的对象 tensor
  Tensor tensor =
      Compute("f", {16, 5}, [](const VarHandle& x, const VarHandle& y) {
        // 返回一个表达式，包含浮点数计算和类型转换
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });

  // 创建循环嵌套对象 l
  LoopNest l({tensor});

  // 获取写入到 tensor 缓冲区的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 对 loops[0] 进行循环分割操作
  LoopNest::splitWithTail(loops[0], 2);
  LoopNest::splitWithTail(loops[0], 2);
}

// 测试用例，测试简单表达式计算的情况
TEST(LoopNest, ExprLower01) {
  // 创建一个计算张量的对象 tensor
  Tensor tensor =
      Compute("f", {16, 5}, [](const VarHandle& x, const VarHandle& y) {
        // 返回一个表达式，包含浮点数计算和类型转换
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });

  // 创建循环嵌套对象 l
  LoopNest l({tensor});

  // 获取循环嵌套的根语句对象 stmt
  StmtPtr stmt = l.root_stmt();

  // 创建字符串流 oss，将 stmt 的内容写入流中
  std::ostringstream oss;
  oss << *stmt;

  // 断言生成的 IR 字符串长度大于 20 小于 200
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

// 测试用例，测试复杂表达式计算的情况
TEST(LoopNest, ExprSimple02) {
  // 创建一个 lambda 表达式 func，表示复杂的表达式计算
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };

  // 创建一个计算张量的对象 tensor
  Tensor tensor = Compute("f", {26, 5}, func);

  // 创建循环嵌套对象 l
  LoopNest l({tensor});

  // 获取写入到 tensor 缓冲区的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 对 loops[0] 进行循环分割操作
  LoopNest::splitWithTail(loops[0], 4);

  // 获取循环嵌套的根语句对象 stmt
  StmtPtr stmt = l.root_stmt();

  // 创建字符串流 oss，将 stmt 的内容写入流中
  std::ostringstream oss;
  oss << *stmt;

  // 断言生成的 IR 字符串长度大于 200 小于 600
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  // 在作用域内部进行比较以参考循环结构
  {
    // 定义外层循环变量 x_outer、内层循环变量 x_inner 和 y
    VarHandle x_outer("i_outer", kInt);
    VarHandle x_inner("i_inner", kInt);
    VarHandle y("i", kInt);
    VarHandle x_tail("i_tail", kInt);

    // 定义一个名为 f 的缓冲区，大小为 {26, 5}，元素类型为 kFloat
    BufHandle f("f", {26, 5}, kFloat);

    // 定义表达式 x_1，表示循环变量的计算方式
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(26) - 0) / 4;
    // 创建一个嵌套的 for 循环语句，对 x_outer 进行迭代
    ForPtr stmt1 = For::make(
        x_outer,
        0,
        x_outer_end,
        // 在 x_outer 的每次迭代中，再创建一个嵌套的 for 循环语句对 x_inner 进行迭代
        For::make(
            x_inner,
            0,
            4,
            // 在 x_inner 的每次迭代中，执行一个 Store 操作，将 func(x_1, y) 的结果存储到 f 中的 {x_1, y} 处
            Store::make(f, {x_1, y}, func(x_1, y))));
    
    // 计算 x_2 的值，用于后续的 for 循环
    ExprHandle x_2 = x_tail + x_outer_end * 4;
    
    // 创建第二个嵌套的 for 循环语句，对 x_tail 进行迭代
    ForPtr stmt2 = For::make(
        x_tail,
        0,
        (ExprHandle(26) - 0) % 4,
        // 在 x_tail 的每次迭代中，执行一个 Store 操作，将 func(x_2, y) 的结果存储到 f 中的 {x_2, y} 处
        Store::make(f, {x_2, y}, func(x_2, y))));
    
    // 将 stmt1 和 stmt2 封装到一个 Block 中形成复合语句
    StmtPtr stmt = Block::make({stmt1, stmt2});

    // 创建一个字符串流对象 oss_ref，并将 stmt 的内容输出到 oss_ref 中
    std::ostringstream oss_ref;
    oss_ref << *stmt;
    
    // 断言 oss 的字符串与 oss_ref 的字符串相等
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    // 创建一个大小为 (26, 5) 的 PaddedBuffer<float> 对象 f_v 和 f_ref
    PaddedBuffer<float> f_v(26, 5, "f_v");
    PaddedBuffer<float> f_ref(26, 5, "f_res");

    // 对 stmt 进行索引平铺（FlattenIndexes），以便进行后续的简单 IR 评估
    stmt = FlattenIndexes(stmt);
    SimpleIREvaluator ir_eval(stmt, {tensor});
    
    // 使用 ir_eval 对 f_v 进行评估，执行 IR 代码块中的计算
    ir_eval(f_v);

    // 使用嵌套的 for 循环初始化 f_ref，计算每个位置的值
    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    // 断言 f_v 和 f_ref 中的所有元素在一定误差范围内接近
    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

// 获取简化后的循环体
BlockPtr getSimplifiedBody(const LoopNest& l) {
  // 获取循环嵌套结构的根语句
  StmtPtr stmt = l.root_stmt();
  // 简化根语句，返回简化后的语句
  StmtPtr simplified = IRSimplifier::simplify(stmt);
  // 转换为块语句，并返回
  return to<Block>(simplified);
}

// 断言 For 循环的起始和结束范围
void assertForRange(ForPtr f, int expected_start, int expected_stop) {
  ASSERT_NE(f, nullptr);
  // 获取 For 循环的起始值，并断言其非空
  IntImmPtr start = to<IntImm>(f->start());
  ASSERT_NE(start, nullptr);
  // 断言 For 循环的起始值与期望值相等
  ASSERT_EQ(start->value(), expected_start);
  // 获取 For 循环的结束值，并断言其非空
  IntImmPtr stop = to<IntImm>(f->stop());
  ASSERT_NE(stop, nullptr);
  // 断言 For 循环的结束值与期望值相等
  ASSERT_EQ(stop->value(), expected_stop);
}

// 断言块语句中所有 For 循环的起始和结束范围
void assertForRanges(
    BlockPtr body,
    const std::vector<std::pair<int, int>>& start_stops) {
  // 断言块语句中 For 循环的数量与期望的数量相等
  ASSERT_EQ(body->nstmts(), start_stops.size());

  // 迭代器遍历块语句中的每个语句
  auto it = body->begin();
  for (size_t i = 0; i < start_stops.size(); i++, it++) {
    // 将当前语句转换为 For 循环，并断言其非空
    ForPtr loop = to<For>(*it);
    assertForRange(loop, start_stops[i].first, start_stops[i].second);
  }
}

// 测试案例：ExprSliceHeadWithLoopOptions
TEST(LoopNest, ExprSliceHeadWithLoopOptions) {
  // 定义计算函数
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建张量对象
  Tensor tensor = Compute("f", {10}, func);
  // 创建循环嵌套结构
  LoopNest l({tensor});

  // 获取写入 tensor 缓冲区的所有循环嵌套结构
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 设置第一个循环的 GPU 块索引
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  // 对第一个循环进行头部切片操作
  LoopNest::sliceHead(loops[0], 2, &head, &tail);

  // 获取简化后的循环体
  BlockPtr body = getSimplifiedBody(l);
  // 断言块语句中所有 For 循环的起始和结束范围
  assertForRanges(body, {{0, 2}, {0, 8}});

  // 断言尾部循环的 GPU 块索引设置情况
  ASSERT_TRUE(tail->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // 断言头部循环的默认循环选项
  ASSERT_TRUE(head->loop_options().isDefault());
}

// 测试案例：ExprSliceTailWithLoopOptions
TEST(LoopNest, ExprSliceTailWithLoopOptions) {
  // 定义计算函数
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建张量对象
  Tensor tensor = Compute("f", {10}, func);
  // 创建循环嵌套结构
  LoopNest l({tensor});

  // 获取写入 tensor 缓冲区的所有循环嵌套结构
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对第一个循环进行尾部切片操作
  LoopNest::sliceTail(loops[0], 4, &head, &tail);

  // 进行尾部循环的进一步切片操作
  tail->set_gpu_block_index(LoopOptions::IDX_Y);
  LoopNest::sliceTail(tail, 2, &tail_head, &tail_tail);

  // 获取简化后的循环体
  BlockPtr body = getSimplifiedBody(l);
  // 断言块语句中所有 For 循环的起始和结束范围
  assertForRanges(body, {{0, 6}, {0, 2}, {8, 10}});

  // 断言尾部头部循环的 GPU 块索引设置情况
  ASSERT_TRUE(tail_head->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail_head->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // 断言头部和尾部尾部循环的默认循环选项
  ASSERT_TRUE(head->loop_options().isDefault());
  ASSERT_TRUE(tail_tail->loop_options().isDefault());
}

// 测试案例：ExprSliceHeadWhenFactorEqualsSize
TEST(LoopNest, ExprSliceHeadWhenFactorEqualsSize) {
  // 当因子等于 For 循环的原始大小时，保持使用原始 For 循环。
  auto func = [](const ExprHandle& x) {
  // 返回一个表达式处理器对象，该对象将浮点数1.0和x转换为浮点数类型后相加
  return ExprHandle(1.0f) + cast<float>(x);
};
// 定义一个张量对象，名称为"f"，形状为{10}，计算函数为func
Tensor tensor = Compute("f", {10}, func);
// 创建一个循环嵌套对象，其中包含张量对象
LoopNest l({tensor});
// 禁止LINT检查，不对下一行进行初始化变量的规则进行检查
// 创建循环头指针和尾指针，并初始化为空指针
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ForPtr head;
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ForPtr tail;
// 获取所有写入张量缓冲区的循环嵌套，并选择第一个
std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
// 对第一个循环进行切片，从头部切片10次，并将切片后的头指针和尾指针赋给head和tail
LoopNest::sliceHead(loops[0], 10, &head, &tail);

// 断言头指针等于循环嵌套中的第一个循环
ASSERT_EQ(head, loops[0]);
// 断言尾指针为空指针
ASSERT_EQ(tail, nullptr);

// 获取简化后的循环体块
BlockPtr body = getSimplifiedBody(l);
// 对简化后的循环体块进行断言，检查其范围是否为{{0, 10}}
assertForRanges(body, {{0, 10}});
}

// 测试用例: LoopNest 类中的 ExprSliceHeadWhenFactorLargerThanSize 方法
TEST(LoopNest, ExprSliceHeadWhenFactorLargerThanSize) {
  // 定义一个 lambda 函数 func，对输入的表达式 x 执行操作：1.0f + x 转换为 float
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建一个张量 tensor，名为 "f"，形状为 {10}，使用 func 函数计算
  Tensor tensor = Compute("f", {10}, func);
  // 创建一个 LoopNest 对象 l，传入张量 tensor
  LoopNest l({tensor});
  // 声明 ForPtr 类型的指针 head 和 tail，用于存储循环头和尾
  ForPtr head;
  ForPtr tail;
  // 获取所有写入到 tensor.buf() 的循环嵌套列表中的第一个列表
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops[0] 进行头部切片，切片长度为 100，结果保存在 head 和 tail 中
  LoopNest::sliceHead(loops[0], 100, &head, &tail);

  // 断言 head 等于 loops[0]
  ASSERT_EQ(head, loops[0]);
  // 断言 tail 为 nullptr
  ASSERT_EQ(tail, nullptr);

  // 获取简化后的循环体 body
  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期：{{0, 10}}
  assertForRanges(body, {{0, 10}});
}

// 测试用例: LoopNest 类中的 ExprSliceHead 方法
TEST(LoopNest, ExprSliceHead) {
  // 同上，定义 func、创建 tensor、创建 l
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  ForPtr head;
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops[0] 进行头部切片，切片长度为 4，结果保存在 head 和 tail 中
  LoopNest::sliceHead(loops[0], 4, &head, &tail);

  // 断言 head 不为 nullptr 且不等于 loops[0]
  ASSERT_NE(head, nullptr);
  ASSERT_NE(head, loops[0]);
  // 断言 tail 不为 nullptr 且等于 loops[0]
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期：{{0, 4}, {4, 10}}
  assertForRanges(body, {{0, 4}, {4, 10}});
}

// 测试用例: LoopNest 类中的 ExprSliceHeadWithNonZeroStart 方法
TEST(LoopNest, ExprSliceHeadWithNonZeroStart) {
  // 同上，定义 func、创建 tensor、创建 l
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  ForPtr head;
  ForPtr tail;
  // 对 loops[0] 进行尾部切片，切片长度为 4，结果保存在 head 和 tail 中
  LoopNest::sliceTail(loops[0], 4, &head, &tail);
  // 对 tail 进行头部切片，切片长度为 2
  LoopNest::sliceHead(tail, 2);

  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期：{{0, 6}, {6, 8}, {8, 10}}
  assertForRanges(body, {{0, 6}, {6, 8}, {8, 10}});
}

// 测试用例: LoopNest 类中的 ExprSliceTailWhenFactorEqualsSize 方法
TEST(LoopNest, ExprSliceTailWhenFactorEqualsSize) {
  // 同上，定义 func、创建 tensor、创建 l
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  ForPtr head;
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops[0] 进行尾部切片，切片长度为 10，结果保存在 head 和 tail 中
  LoopNest::sliceTail(loops[0], 10, &head, &tail);

  // 断言 head 为 nullptr
  ASSERT_EQ(head, nullptr);
  // 断言 tail 等于 loops[0]
  ASSERT_EQ(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期：{{0, 10}}
  assertForRanges(body, {{0, 10}});
}

// 测试用例: LoopNest 类中的 ExprSliceTailWhenFactorLargerThanSize 方法
TEST(LoopNest, ExprSliceTailWhenFactorLargerThanSize) {
  // 同上，定义 func、创建 tensor、创建 l
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  ForPtr head;
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops[0] 进行尾部切片，切片长度为 10，结果保存在 head 和 tail 中
  LoopNest::sliceTail(loops[0], 10, &head, &tail);

  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期：{{0, 10}}
  assertForRanges(body, {{0, 10}});
}

// 当 factor 等于原始循环大小时，保持使用原始的 For 循环
auto func = [](const ExprHandle& x) {
  // 创建一个表达式处理器，计算结果为浮点数1.0加上x的浮点数强制转换结果
  return ExprHandle(1.0f) + cast<float>(x);
};
// 定义一个名为"f"的张量，形状为{10}，计算定义为func
Tensor tensor = Compute("f", {10}, func);
// 创建一个循环嵌套对象l，包含张量tensor
LoopNest l({tensor});
// 禁用lint检查未初始化的变量（head和tail）
// 初始化指针head和tail
ForPtr head;
ForPtr tail;
// 获取所有写入tensor.buf()的循环嵌套对象并存储在loops中，取第一个元素
std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
// 对loops[0]进行尾部分割，分割点为100，head和tail指向分割结果的头部和尾部
LoopNest::sliceTail(loops[0], 100, &head, &tail);

// 断言head应为nullptr
ASSERT_EQ(head, nullptr);
// 断言tail应指向loops[0]
ASSERT_EQ(tail, loops[0]);

// 获取循环嵌套对象l的简化后的主体块
BlockPtr body = getSimplifiedBody(l);
// 断言body中的循环范围为{{0, 10}}
assertForRanges(body, {{0, 10}});
TEST(LoopNest, ExprSliceTail) {
  // 定义一个 lambda 函数 func，将 ExprHandle 对象 x 转换为 float 类型并加上 1.0 返回
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建一个名为 tensor 的 Tensor 对象，计算结果存储在名为 "f" 的 tensor 中，维度为 {10}，使用 func 函数
  Tensor tensor = Compute("f", {10}, func);
  // 创建 LoopNest 对象 l，初始化时传入 tensor 对象的列表
  LoopNest l({tensor});
  
  // 获取写入 tensor.buf() 的所有循环嵌套列表的第一个元素
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 在 loops[0] 中对第 4 个循环进行切片操作，将切片头部和尾部存储在 head 和 tail 指针中
  LoopNest::sliceTail(loops[0], 4, &head, &tail);

  // 断言 head 指针不为空
  ASSERT_NE(head, nullptr);
  // 断言 head 指针指向的对象与 loops[0] 指向的对象相同
  ASSERT_EQ(head, loops[0]);
  // 断言 tail 指针不为空
  ASSERT_NE(tail, nullptr);
  // 断言 tail 指针指向的对象与 loops[0] 指向的对象不相同
  ASSERT_NE(tail, loops[0]);

  // 获取简化后的循环体 BlockPtr
  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期 {{0, 6}, {6, 10}}
  assertForRanges(body, {{0, 6}, {6, 10}});
}

TEST(LoopNest, ExprSplitAndSlice) {
  // 定义一个 lambda 函数 func，将 ExprHandle 对象 x 转换为 float 类型并加上 1.0 返回
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建一个名为 tensor 的 Tensor 对象，计算结果存储在名为 "f" 的 tensor 中，维度为 {100}，使用 func 函数
  Tensor tensor = Compute("f", {100}, func);
  // 创建 LoopNest 对象 l，初始化时传入 tensor 对象的列表
  LoopNest l({tensor});

  // 获取写入 tensor.buf() 的所有循环嵌套列表的第一个元素
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 在 loops[0] 中将循环切分为两部分，inner 循环为 [0, 21)，tail 循环为 [84, 100)
  LoopNest::splitWithTail(loops[0], 21, &inner, &tail);
  // 在 inner 循环中对前 2 个元素进行尾部切片操作
  LoopNest::sliceTail(inner, 2);
  // 在 loops[0] 中对前 2 个元素进行头部切片操作
  LoopNest::sliceHead(loops[0], 2);

  // 获取简化后的循环体 BlockPtr
  BlockPtr body = getSimplifiedBody(l);
  // 断言 body 的循环范围符合预期 {{0, 2}, {2, 4}, {0, 16}}
  assertForRanges(body, {{0, 2}, {2, 4}, {0, 16}});

  // 迭代简化后的循环体的第一个 For 循环
  auto biter = body->begin();
  ForPtr loop = to<For>(*biter++);
  // 断言 loop 的循环范围符合预期 {{0, 19}, {19, 21}}
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});

  // 迭代简化后的循环体的第二个 For 循环
  loop = to<For>(*biter);
  // 断言 loop 的循环范围符合预期 {{0, 19}, {19, 21}}
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});
}

TEST(LoopNest, ExprSliceAndNormalize) {
  // 定义一个 lambda 函数 func，将 ExprHandle 对象 x 转换为 float 类型并加上 1.0 返回
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建一个名为 tensor 的 Tensor 对象，计算结果存储在名为 "f" 的 tensor 中，维度为 {100}，使用 func 函数
  Tensor tensor = Compute("f", {100}, func);
  // 创建 LoopNest 对象 l，初始化时传入 tensor 对象的列表
  LoopNest l({tensor});

  // 前两个循环切片头部，第三个循环标准化尾部
  LoopNest::sliceHead(loops[0], 2);

  // 以下是具体的循环结构，暂不在此添加注释
}
    // 构造一个表达式：1.0f + 将 x 强制转换为 float 类型的表达式
    return ExprHandle(1.0f) + cast<float>(x);
  };

  // 创建一个名为 "f" 的计算张量 tensor，形状为 {10}，使用 func 函数定义
  Tensor tensor = Compute("f", {10}, func);

  // 创建循环嵌套对象 l，其中包含计算张量 tensor
  LoopNest l({tensor});

  // 获取所有写入 tensor.buf() 的循环嵌套，选取第一个嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 创建两个 ForPtr 指针 head 和 tail，用于表示循环范围
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;

  // 对 loops[0] 进行切片操作，将切片头部和尾部保存到 head 和 tail 中
  LoopNest::sliceHead(loops[0], 2, &head, &tail);
  // head: [0, 2)
  // tail: [2, 10)

  // 将 tail 的范围进行归一化，即调整为从 0 开始的范围
  LoopNest::normalize(tail);
  // normalized_tail: [0, 8)

  // 获取简化后的循环体 body
  BlockPtr body = getSimplifiedBody(l);

  // 使用断言验证循环体 body 的各个循环范围是否与预期相符
  assertForRanges(body, {{0, 2}, {0, 8}});
// 定义一个模板函数 evalExpr，用于评估表达式的值，给定一个变量和初始值。
template <typename T>
T evalExpr(const ExprHandle& expr, const VarHandle& var, T value) {
  // 使用 SimpleIREvaluator 创建表达式评估器对象
  ExprEval<SimpleIREvaluator> eval(expr, {var});
  // 返回根据给定初始值计算得到的表达式的值
  return eval.value<T>(value);
}

// 定义 LoopNest 类的测试用例 ExprSliceWithVariableDimension
TEST(LoopNest, ExprSliceWithVariableDimension) {
  // 定义 lambda 函数 testWithDimension，用于测试不同维度的情况
  auto testWithDimension =
      [](int dimension,
         const std::vector<std::pair<int, int>>& expected_for_ranges) {
        // 创建一个名为 dim 的整数变量
        VarHandle dim("dim", kInt);
        // 创建一个简单的张量，内容为自身维度
        Tensor tensor =
            Compute("f", {dim}, [](const ExprHandle& x) { return x; });
        // 创建 LoopNest 对象，包含上述张量
        LoopNest l({tensor});
        // 获取循环嵌套写入到 tensor.buf() 的所有循环
        std::vector<ForPtr> loops =
            l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

        // 获取循环嵌套的头部和尾部，切割头部循环
        ForPtr head;
        ForPtr tail;
        LoopNest::sliceHead(loops[0], 2, &head, &tail);

        // 切割尾部循环
        LoopNest::sliceTail(tail, 2);

        // 获取简化后的循环体
        BlockPtr body = getSimplifiedBody(l);
        // 断言简化后的循环体大小与期望的范围数相等
        ASSERT_EQ(expected_for_ranges.size(), 3);
        auto it = body->begin();
        // 遍历期望的每个循环范围，检查起始和结束值是否符合预期
        for (auto& start_stop : expected_for_ranges) {
          ForPtr loop = to<For>(*it++);
          // 使用 evalExpr 函数评估起始和结束表达式的值
          int start = evalExpr<int>(ExprHandle(loop->start()), dim, dimension);
          int stop = evalExpr<int>(ExprHandle(loop->stop()), dim, dimension);
          ASSERT_EQ(start, start_stop.first);
          ASSERT_EQ(stop, start_stop.second);
        }
      };

  // 对 testWithDimension 函数进行多次测试，每次使用不同的维度和预期范围
  testWithDimension(1, {{0, 1}, {1, 1}, {1, 1}});
  testWithDimension(2, {{0, 2}, {2, 2}, {2, 2}});
  testWithDimension(3, {{0, 2}, {2, 2}, {2, 3}});
  testWithDimension(4, {{0, 2}, {2, 2}, {2, 4}});
  testWithDimension(5, {{0, 2}, {2, 3}, {3, 5}});
  testWithDimension(10, {{0, 2}, {2, 8}, {8, 10}});
}

// 定义 LoopNest 类的测试用例 ExprSplitWithTail
TEST(LoopNest, ExprSplitWithTail) {
  // 定义 lambda 函数 func，返回输入加 1.0 的浮点数
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  // 创建一个张量，使用 func 定义
  Tensor tensor = Compute("f", {199}, func);
  // 创建 LoopNest 对象，包含上述张量
  LoopNest l({tensor});
  // 获取循环嵌套写入到 tensor.buf() 的所有循环
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 将 loops[0] 循环按照 17 进行切割
  LoopNest::splitWithTail(loops[0], 17);
  // 再次将 loops[0] 循环按照 7 进行切割
  LoopNest::splitWithTail(loops[0], 7);

  // 获取根语句的简化版本
  StmtPtr stmt = l.root_stmt();
  StmtPtr simplified = IRSimplifier::simplify(stmt);
  // 将简化版本转换为块
  BlockPtr body = to<Block>(simplified);
  // 断言块中的语句数为 3
  ASSERT_EQ(body->nstmts(), 3);
  auto biter = body->begin();

  // 验证分割后的循环顺序是否正确
  ForPtr loop = to<For>(*biter++);
  assertForRange(loop, 0, 7);

  loop = to<For>(*biter++);
  assertForRange(loop, 0, 4);

  loop = to<For>(*biter);
  assertForRange(loop, 0, 12);
}

// 定义 LoopNest 类的测试用例 ExprSplitWithTailNone
TEST(LoopNest, ExprSplitWithTailNone) {
  // 定义 lambda 函数 func，返回两个输入参数之和的浮点数
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return x + y;
  };
    {
      // 定义一个计算函数，计算表达式 1.0 + float(x) * x + float(y) * y
      return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
    };
    
    // 创建一个名为 "f" 的张量，大小为 {24, 5}，使用上述计算函数 func
    Tensor tensor = Compute("f", {24, 5}, func);
    
    // 创建一个循环嵌套对象，包含该张量
    LoopNest l({tensor});
    
    // 获取所有写入 tensor.buf() 的循环嵌套
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
    
    // 将 loops[0] 的循环拆分为大小为 4 的块
    LoopNest::splitWithTail(loops[0], 4);
    
    // 获取根语句（即整个循环嵌套的根）
    StmtPtr stmt = l.root_stmt();
    
    // 创建一个字符串输出流对象
    std::ostringstream oss;
    
    // 将 stmt 的内容写入 oss 中
    oss << *stmt;
    
    // 断言：oss 的字符数应大于 200
    ASSERT_GT(oss.str().size(), 200);
    
    // 断言：oss 的字符数应小于 600
    ASSERT_LT(oss.str().size(), 600);
    
    {
      // 比较与参考循环结构的结构
      VarHandle x_outer("i_outer", kInt);  // 外部循环变量 x_outer
      VarHandle x_inner("i_inner", kInt);  // 内部循环变量 x_inner
      VarHandle y("i", kInt);              // 循环变量 y
      VarHandle x_tail("i_tail", kInt);    // 尾部循环变量 x_tail
    
      // 创建一个名为 "f" 的缓冲区，大小为 {24, 5}，类型为浮点数
      BufHandle f("f", {24, 5}, kFloat);
    
      // 定义表达式 x_outer * 4 + x_inner
      ExprHandle x_1 = x_outer * 4 + x_inner;
    
      // 计算外部循环的结束条件
      ExprHandle x_outer_end = (ExprHandle(24) - 0) / 4;
    
      // 创建循环嵌套语句
      StmtPtr stmt = alloc<Block>(std::vector<StmtPtr>({For::make(
          x_outer,
          0,
          x_outer_end,
          For::make(
              x_inner,
              0,
              4,
              For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y)))))}));
    
      // 创建一个字符串输出流对象，用于参考循环结构的输出
      std::ostringstream oss_ref;
      oss_ref << *stmt;
    
      // 断言：oss 和 oss_ref 的内容应相等
      ASSERT_EQ(oss.str(), oss_ref.str());
    }
    
    {
      // 创建大小为 {24, 5} 的 PaddedBuffer<float> 对象 f_v 和 f_ref
      PaddedBuffer<float> f_v(24, 5, "f_v");
      PaddedBuffer<float> f_ref(24, 5, "f_res");
    
      // 使用 SimpleIREvaluator 运行 stmt，并传入 tensor
      SimpleIREvaluator ir_eval(stmt, {tensor});
      ir_eval(f_v);
    
      // 用双重循环计算 f_ref 的期望值
      for (int x = 0; x < 24; x++) {
        for (int y = 0; y < 5; y++) {
          f_ref(x, y) = 1 + x * x + y * y;
        }
      }
    
      // 断言：f_v 和 f_ref 应近似相等，精度为 1e-5
      ExpectAllNear(f_v, f_ref, 1e-5);
    }
// LoopNest 类的单元测试，测试在表达式拆分过程中是否正确处理无需插入掩码的情况
TEST(LoopNest, ExprSplitWithMaskRepeatedNoMask) {
  // 定义常量 M 为 64
  const int M = 64;
  // 创建名为 a 的缓冲区句柄，形状为 {M}，数据类型为 kFloat
  BufHandle a_buf("a", {M}, kFloat);
  // 创建名为 b 的缓冲区句柄，形状为 {M}，数据类型为 kFloat
  BufHandle b_buf("b", {M}, kFloat);
  // 定义张量 f，计算结果为 a_buf(m) + b_buf(m) + 1.0f，其中 m 为表达式句柄
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });

  // 创建 LoopNest 对象并传入张量列表
  LoopNest l({tensor});
  // 获取写入到张量缓冲区的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 在第一个循环中进行拆分，步长为 4
  LoopNest::splitWithMask(loops[0], 4);
  // 再次在同一个循环中进行拆分，步长为 4
  LoopNest::splitWithMask(loops[0], 4);

  // 对根语句进行简化
  StmtPtr stmt1 = IRSimplifier::simplify(l.root_stmt());

  // 验证简化后的 IR，确保两次拆分导致生成三个循环，但不应需要插入掩码
  checkIR(stmt1, R"IR(
# CHECK: for (
# CHECK-NOT: if (
# CHECK:   for (
# CHECK-NOT: if (
# CHECK:     for (
# CHECK-NOT: if (
# CHECK:       f[)IR");
}
TEST(LoopNest, getLoopAt) {
  // Input IR:
  //  for (int i = 0; i < 100; i++) {
  //    for (int j = 0; j < 100; j++) {
  //      A[i, j] = sin(i * j);
  //      for (int k1 = 0; k1 < 200; k1++) {
  //        B[i, j, k1] = (A[i, j]) / (k1 + 1);
  //      }
  //      for (int k2 = 0; k2 < 300; k2++) {
  //        C[i, j, k2] = (A[i, j]) * (k2 + 1);
  //      }
  //    }
  //  }
  // 分配内存给 A, B, C，分别为三维数组，大小为 [100, 100], [100, 100, 200], [100, 100, 300]
  BufPtr A = alloc<Buf>(
      "A",
      std::vector<ExprPtr>({alloc<IntImm>(100), alloc<IntImm>(100)}),
      kInt);
  BufPtr B = alloc<Buf>(
      "B",
      std::vector<ExprPtr>(
          {alloc<IntImm>(100), alloc<IntImm>(100), alloc<IntImm>(200)}),
      kInt);
  BufPtr C = alloc<Buf>(
      "C",
      std::vector<ExprPtr>(
          {alloc<IntImm>(100), alloc<IntImm>(100), alloc<IntImm>(300)}),
      kInt);
  // 将 BufPtr 转换为 BufHandle，便于处理
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  BufHandle c_buf(C);
  // 创建循环变量 i, j, k1, k2 的 VarHandle
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k1("k1", kInt);
  VarHandle k2("k2", kInt);
  // 创建三个不同的存储操作，存储计算结果到对应的缓冲区
  auto store1 = Store::make(a_buf, {i, j}, sin(i * j));
  auto store2 = Store::make(
      b_buf, {i, j, k1}, Div::make(Load::make(a_buf, {i, j}), (k1 + 1)));
  auto store3 = Store::make(
      c_buf, {i, j, k2}, Mul::make(Load::make(a_buf, {i, j}), (k2 + 1)));
  // 创建嵌套循环结构 for_k2, for_k1, for_j, for_i，将存储操作组织起来
  auto for_k2 = For::make(k2, 0, 300, Block::make({store3}));
  auto for_k1 = For::make(k1, 0, 200, Block::make({store2}));
  auto for_j = For::make(j, 0, 100, Block::make({store1, for_k1, for_k2}));
  auto for_i = For::make(i, 0, 100, for_j);
  // 创建循环嵌套结构 LoopNest
  LoopNest l(Block::make({for_i}), {B, C});
  // 获取在指定位置 [0, 2] 的循环结构，即第一层嵌套中的第三层循环 for_k2
  auto ret_k2 = l.getLoopAt(for_i, {0, 2});
  // 使用 FileCheck 运行 IR 格式检查，验证生成的 IR 是否符合预期模式
  std::ostringstream oss;
  oss << *ret_k2;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int k2
# CHECK-NEXT: C[i, j, k2] =
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, TileSimple) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 64, N = 64;
  // 创建两个缓冲区 a_buf 和 b_buf，大小为 [64, 64]，数据类型为 kFloat
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {M, N}, kFloat);
  // 定义一个 Tensor 对象 tensor，使用 a_buf 和 b_buf 的值进行计算
  Tensor tensor =
      Compute("f", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load({m, n}) + b_buf.load({m, n}) + 1.0f;
      });
  // 创建 LoopNest 对象 l，将 tensor 添加到其中
  LoopNest l({tensor});
  // 获取写入到 tensor.buf() 的所有循环结构，取第一个元素作为 loops
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops[0] 和 loops[1] 进行 4x8 的划分
  l.tile(loops[0], loops[1], 4, 8);

  // IR check
  // 对最终的 IR 进行检查，确保生成的 IR 符合预期格式
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i_outer
# CHECK:   for (int i_outer_1
# CHECK:     for (int i_inner
# CHECK:       for (int i_inner_1
# CHECK:         f[
# CHECK-NOT:     for (int i_tail
# CHECK-NOT: for (int i_tail)IR");

  // Correctness check
  // 对生成的代码进行正确性验证，比较计算结果
  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    // 使用循环遍历二维数组中的每个元素
    for (int n = 0; n < N; n++) {
      // 计算数组 a_v 中第 m 行、第 n 列的值为 2 * m
      a_v(m, n) = 2 * m;
      // 计算数组 b_v 中第 m 行、第 n 列的值为 3 * n
      b_v(m, n) = 3 * n;
      // 计算数组 c_ref 中第 m 行、第 n 列的值为 a_v(m, n) + b_v(m, n) + 1.0f
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  // 使用 SimpleIREvaluator 对给定的语句进行评估，并传入 a_v、b_v、c_v 作为参数
  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // 检查 c_v 和 c_ref 之间的所有元素是否在给定的容差范围内接近
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(c_v, c_ref, 1e-5);
TEST(LoopNest, TileWithTails) {
  // 定义常量 M 和 N，表示矩阵的维度为 64x64
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 64, N = 64;
  // 创建名为 a_buf 的缓冲区对象，表示矩阵 a，大小为 MxN，数据类型为 float
  BufHandle a_buf("a", {M, N}, kFloat);
  // 创建名为 b_buf 的缓冲区对象，表示矩阵 b，大小为 MxN，数据类型为 float
  BufHandle b_buf("b", {M, N}, kFloat);
  // 定义张量 tensor，计算表达式为 a_buf 加载(m, n)位置的值加上 b_buf 加载(m, n)位置的值再加上 1.0f
  Tensor tensor =
      Compute("f", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load({m, n}) + b_buf.load({m, n}) + 1.0f;
      });

  // 创建循环嵌套对象 l，传入张量 tensor
  LoopNest l({tensor});
  // 获取写入张量 tensor 的所有循环嵌套，存入 loops 中
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对第一个和第二个循环进行 tiling 操作，块大小为 (5, 9)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  l.tile(loops[0], loops[1], 5, 9);

  // IR 检查
  // 对简化后的 IR 语句进行检查，确保生成的 IR 符合预期格式
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i_outer
# CHECK:   for (int i_outer_1
# CHECK:     for (int i_inner
# CHECK:       for (int i_inner_1
# CHECK:         f[
# CHECK:   for (int i_inner
# CHECK:     f[
# CHECK: for (int i_tail)IR");

  // 正确性检查
  // 创建带填充的缓冲区对象 a_v, b_v, c_v, c_ref，用于存储计算结果和参考结果
  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  // 初始化 a_v, b_v 和 c_ref 中的数据
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  // 使用简化后的 IR 语句执行计算，将结果存入 c_v
  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 检查 c_v 和 c_ref 的近似值是否在给定的误差范围内
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LoopNest, TileInMiddle) {
  // 定义常量 M, N, L, K，表示矩阵的维度为 8x8x8x8
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 8, N = 8, L = 8, K = 8;
  // 创建名为 a_buf 的缓冲区对象，表示矩阵 a，大小为 MxNxLxK，数据类型为 float
  BufHandle a_buf("a", {M, N, L, K}, kFloat);
  // 创建名为 b_buf 的缓冲区对象，表示矩阵 b，大小为 MxNxLxK，数据类型为 float
  BufHandle b_buf("b", {M, N, L, K}, kFloat);
  // 定义张量 tensor，计算表达式为 a_buf 加载(m, n, l, k)位置的值加上 b_buf 加载(m, n, l, k)位置的值再加上 1.0f
  Tensor tensor = Compute(
      "f",
      {M, N, L, K},
      [&](const ExprHandle& m,
          const ExprHandle& n,
          const ExprHandle& l,
          const ExprHandle& k) {
        return a_buf.load({m, n, l, k}) + b_buf.load({m, n, l, k}) + 1.0f;
      });

  // 创建循环嵌套对象 nest，传入张量 tensor
  LoopNest nest({tensor});
  // 获取写入张量 tensor 的所有循环嵌套，存入 loops 中
  std::vector<ForPtr> loops =
      nest.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对第二个和第三个循环进行 tiling 操作，块大小为 (3, 3)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  nest.tile(loops[1], loops[2], 3, 3);

  // IR 检查
  // 对简化后的 IR 语句进行检查，确保生成的 IR 符合预期格式
  StmtPtr stmt = IRSimplifier::simplify(nest.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i
# CHECK:   for (int i_outer
# CHECK:     for (int i_outer_1
# CHECK:       for (int i_inner
# CHECK:         for (int i_inner_1
# CHECK:           for (int i_1
# CHECK:             f[
# CHECK:     for (int i_tail_1
# CHECK:       for (int i_inner_1
# CHECK:         for (int i_1
# CHECK:           f[
# CHECK:   for (int i_tail)IR");

  // 正确性检查
  // 创建带填充的缓冲区对象 a_v, b_v, c_v, c_ref，用于存储计算结果和参考结果
  PaddedBuffer<float> a_v(M, N, L, K, "a");
  PaddedBuffer<float> b_v(M, N, L, K, "b");
  PaddedBuffer<float> c_v(M, N, L, K, "c");
  PaddedBuffer<float> c_ref(M, N, L, K, "c_ref");
  // 初始化 a_v, b_v 和 c_ref 中的数据
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
          a_v(m, n, l, k) = 2 * (m + n + l + k);
          b_v(m, n, l, k) = 3 * (m + n + l + k);
          c_ref(m, n, l, k) = a_v(m, n, l, k) + b_v(m, n, l, k) + 1.0f;
        }
      }
    }
  }
    // 循环遍历三维数组，分别为 a_v, b_v, c_ref 赋值
    for (int n = 0; n < N; n++) {
      for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
          // 计算 a_v(m, n, l, k) 的值，根据公式 2 * (m + l)
          a_v(m, n, l, k) = 2 * (m + l);
          // 计算 b_v(m, n, l, k) 的值，根据公式 3 * (n + k)
          b_v(m, n, l, k) = 3 * (n + k);
          // 计算 c_ref(m, n, l, k) 的值，根据公式 a_v(m, n, l, k) + b_v(m, n, l, k) + 1.0f
          c_ref(m, n, l, k) = a_v(m, n, l, k) + b_v(m, n, l, k) + 1.0f;
        }
      }
    }
  }

  // 使用 SimpleIREvaluator 评估 stmt 语句，并验证结果是否在误差范围内
  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // 使用 ExpectAllNear 函数检查 c_v 和 c_ref 数组的所有元素是否在误差范围内
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(c_v, c_ref, 1e-5);
TEST(LoopNest, SplitWithTailWithLoopOptions) {
  // 定义常量 M 为 21
  const int M = 21;
  // 创建名为 a_buf 的缓冲区句柄，大小为 {M}，数据类型为 kFloat
  BufHandle a_buf("a", {M}, kFloat);
  // 创建名为 b_buf 的缓冲区句柄，大小为 {M}，数据类型为 kFloat
  BufHandle b_buf("b", {M}, kFloat);
  // 创建张量 tensor，计算每个元素为 a_buf[m] + b_buf[m] + 1.0f
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;

  // 创建循环嵌套对象 l，包含张量 tensor
  LoopNest l({tensor});
  // 查找循环语句
  auto loops = NodeFinder<For>::find(l.root_stmt());
  // 断言找到的循环数大于 0
  ASSERT_GT(loops.size(), 0);
  // 设置第一个找到的循环的 GPU 块索引为 IDX_Y
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  // 使用 LoopNest 的方法将第一个循环分裂为 4 个块，inner 和 tail 分别指向内部循环和尾部循环
  LoopNest::splitWithTail(loops[0], 4, &inner, &tail);
  // 断言 inner 和 tail 非空
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(tail, nullptr);
  // outer 指向原始循环
  ForPtr outer = loops[0];

  // 外部循环带有循环轴绑定
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // 内部循环无特殊选项
  ASSERT_TRUE(inner->loop_options().isDefault());

  // 尾部循环无特殊选项
  ASSERT_TRUE(tail->loop_options().isDefault());
}

TEST(LoopNest, SplitWithMaskWithLoopOptions) {
  // 定义常量 M 为 21
  const int M = 21;
  // 创建名为 a_buf 的缓冲区句柄，大小为 {M}，数据类型为 kFloat
  BufHandle a_buf("a", {M}, kFloat);
  // 创建名为 b_buf 的缓冲区句柄，大小为 {M}，数据类型为 kFloat
  BufHandle b_buf("b", {M}, kFloat);
  // 创建张量 tensor，计算每个元素为 a_buf[m] + b_buf[m] + 1.0f
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;

  // 创建循环嵌套对象 l，包含张量 tensor
  LoopNest l({tensor});
  // 查找循环语句
  auto loops = NodeFinder<For>::find(l.root_stmt());
  // 设置第一个找到的循环的 GPU 块索引为 IDX_Y
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  // 使用 LoopNest 的方法将第一个循环根据 mask 分裂为 4 个块，inner 指向内部循环
  LoopNest::splitWithMask(loops[0], 4, &inner);
  ForPtr outer = loops[0];

  // 外部循环带有循环轴绑定
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // 内部循环无特殊选项
  ASSERT_TRUE(inner->loop_options().isDefault());
}

TEST(LoopNest, ScheduleBroadcastAddBuffer) {
  // 定义常量 M、N、K 分别为 4、5、6
  const int M = 4;
  const int N = 5;
  const int K = 6;
  // 创建名为 a_buf 的缓冲区句柄，大小为 {M, N}，数据类型为 kFloat
  BufHandle a_buf("a", {M, N}, kFloat);
  // 创建名为 b_buf 的缓冲区句柄，大小为 {N, K}，数据类型为 kFloat
  BufHandle b_buf("b", {N, K}, kFloat);
  // 创建张量 c，计算每个元素为 a_buf[m, n] + b_buf[n, k]
  Tensor c = Compute(
      "broadcast_add",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  // 创建循环嵌套对象 l，包含张量 c
  LoopNest l({c});
  // 获取根语句对象
  StmtPtr stmt = l.root_stmt();

  // 创建填充缓冲区 a_v，大小为 {M, N}，名为 "a_v"
  PaddedBuffer<float> a_v(M, N, "a_v");
  // 填充缓冲区 a_v 的值
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 7 * m * n;
    }
  }
  // 备份缓冲区 a_v
  a_v.Backup();

  // 创建填充缓冲区 b_v，大小为 {N, K}，名为 "b_v"
  PaddedBuffer<float> b_v(N, K, "b_v");
  // 填充缓冲区 b_v 的值
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b_v(n, k) = 11 * n * k;
    }
  }
  // 备份缓冲区 b_v
  b_v.Backup();

  // 创建填充缓冲区 c_v，大小为 {M, N, K}，名为 "c_buf"
  PaddedBuffer<float> c_v(M, N, K, "c_buf");
  // 创建简单的 IR 评估器 ir_eval，评估 stmt 在给定缓冲区 a_buf、b_buf 和 c 下的结果
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c});
  ir_eval(a_v, b_v, c_v);

  // 检查缓冲区 a_v 和 b_v 的备份
  a_v.CheckBackup();
  b_v.CheckBackup();

  // 创建填充缓冲区 c_ref，大小为 {M, N, K}，名为 "c_ref"
  PaddedBuffer<float> c_ref(M, N, K, "c_ref");
  // 填充缓冲区 c_ref 的参考值
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        c_ref(m, n, k) = 7 * m * n + 11 * n * k;
      }
    }
  }
  // 断言 c_v 和 c_ref 的所有元素近似相等，精度为 1e-5
  ExpectAllNear(c_v, c_ref, 1e-5);
}
TEST(LoopNest, ScheduleFunctionCall01) {
  // 定义三个常量 M、N、K 分别表示维度大小
  const int M = 4;
  const int N = 5;
  const int K = 6;
  
  // 创建两个缓冲区对象 a_buf 和 b_buf，分别表示 MxN 和 NxK 的浮点数缓冲区
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  
  // 定义张量 c，表示大小为 MxNxK，通过 lambda 函数计算 a_buf 和 b_buf 的加载和
  Tensor c = Compute(
      "broadcast_add",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  
  // 定义张量 d，表示大小为 MxNxK，通过 lambda 函数计算 c 加 1 的结果
  Tensor d = Compute(
      "d",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c.load(m, n, k) + 1;
      });
  
  // 创建 LoopNest 对象 l，初始化为处理 d 张量
  LoopNest l({d}, {c, d});
  
  // 为代码生成做准备
  l.prepareForCodegen();
  
  // 获取根语句的指针并转换为字符串
  StmtPtr stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  
  // 断言根语句字符串的长度大于 100
  ASSERT_GT(oss.str().size(), 100);
  
  // 创建四个 PaddedBuffer 对象，分别用于存储 a、b、c、d 张量的值
  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N, K);
  PaddedBuffer<float> d_v(M, N, K);
  PaddedBuffer<float> d_ref(M, N, K);
  
  // 初始化 a_v，根据 i 和 j 的值设置 a_v(i, j) 的值
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  
  // 初始化 b_v，根据 i 和 j 的值设置 b_v(i, j) 的值
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  
  // 计算参考值 d_ref，根据 a_v、b_v 的值计算 d_ref(i, j, k) 的值
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        d_ref(i, j, k) = a_v(i, j) + b_v(j, k) + 1;
      }
    }
  }
  
  // 创建 SimpleIREvaluator 对象 eval，用于计算 d 张量的值
  SimpleIREvaluator eval(stmt, {a_buf, b_buf, d});
  eval(a_v, b_v, d_v);
  
  // 断言 d_v 与 d_ref 在给定精度下接近
  ExpectAllNear(d_v, d_ref, 1e-5);
}

TEST(LoopNest, ScheduleInlineSimple) {
  // 定义三个常量 M、N、K 分别表示维度大小
  const int M = 4;
  const int N = 5;
  const int K = 6;
  
  // 创建四个缓冲区对象 a_buf、b_buf、c_buf、d_buf，分别表示 MxN、NxK、MxN、MxK 的浮点数缓冲区
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);
  
  // 定义张量 x，表示大小为 MxNxK，通过 lambda 函数计算 a_buf 和 b_buf 的元素积
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  
  // 定义张量 y，表示大小为 MxNxK，通过 lambda 函数计算 c_buf、d_buf 和 x 的加载和
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });
  
  // 创建 LoopNest 对象 l1，初始化为处理 y 张量
  LoopNest l1({y}, {x, y});
  
  // 创建 LoopNest 对象 l2，复制 l1 并在 l2 中内联计算 x 缓冲区
  LoopNest l2(l1);
  l2.computeInline(x.buf());
  
  // 为代码生成做准备
  l1.prepareForCodegen();
  l2.prepareForCodegen();
  
  // 对 l1 和 l2 的根语句进行简化
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());
  
  // 创建两个 SimpleIREvaluator 对象 eval1 和 eval2，用于计算 y 张量的值
  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, c_buf, d_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, c_buf, d_buf, y});
  
  // 创建四个 PaddedBuffer 对象，分别用于存储 a、b、c、d 缓冲区的值
  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N);
  PaddedBuffer<float> d_v(M, K);
  
  // 初始化 a_v，根据 i 和 j 的值设置 a_v(i, j) 的值
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  
  // 初始化 b_v，根据 i 和 j 的值设置 b_v(i, j) 的值
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  
  // 初始化 c_v，根据 i 和 j 的值设置 c_v(i, j) 的值
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      c_v(i, j) = i + j;
    }
  }
  
  // 初始化 d_v，根据 i 和 j 的值设置 d_v(i, j) 的值
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      d_v(i, j) = i * j;
    }
  }
    }
  }

  # 创建大小为 M x N x K 的填充缓冲区 y_1 和 y_2，存储 float 类型的数据
  PaddedBuffer<float> y_1(M, N, K);
  PaddedBuffer<float> y_2(M, N, K);

  # 使用 eval1 函数计算结果并存储在 y_1 中
  eval1(a_v, b_v, c_v, d_v, y_1);
  # 使用 eval2 函数计算结果并存储在 y_2 中
  eval2(a_v, b_v, c_v, d_v, y_2);
  # 检查 y_1 和 y_2 的所有元素是否在给定的误差范围内接近
  ExpectAllNear(y_1, y_2, 1e-5);

  # 创建两个 ostringstream 对象 oss1 和 oss2
  std::ostringstream oss1, oss2;
  # 将 stmt1 的内容写入 oss1
  oss1 << *stmt1;
  # 将 stmt2 的内容写入 oss2
  oss2 << *stmt2;
  # 断言 oss1 中的字符串长度大于 oss2 中的字符串长度
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

static std::string remove_space(const std::string& str) {
  // 创建一个新的字符串副本，去除其中所有的空白字符
  std::string str_new = str;
  str_new.erase(
      remove_if(str_new.begin(), str_new.end(), isspace), str_new.end());
  return str_new;
}

void InlineFunc01Helper(const std::vector<std::string>& inline_order) {
  // 定义常量维度
  const int M = 4;
  const int N = 5;
  const int K = 6;
  // 创建四个缓冲区处理对象，分别对应张量操作中的四个数据数组
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);

  // 定义三个张量对象，每个对象对应一个计算操作
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });
  Tensor z = Compute(
      "z",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + y.load(m, n, k);
      });

  // 创建循环嵌套对象，并将计算的张量加入其中
  LoopNest l({z}, {x, y, z});
  // 根据指定的顺序对计算对象进行内联处理
  for (const std::string& order : inline_order) {
    if (order == "x") {
      l.computeInline(x.buf());
    } else if (order == "y") {
      l.computeInline(y.buf());
    } else {
      throw std::runtime_error("Invalid order: " + order);
    }
  }
  // 准备好进行代码生成的准备工作
  l.prepareForCodegen();
  // 获取整个语句的根节点
  StmtPtr stmt = l.root_stmt();

  // 创建字符串流对象，用于获取和处理语句的字符串表示形式
  std::ostringstream oss;
  oss << *stmt;
  // 移除字符串中的所有空白字符
  std::string str1 = remove_space(oss.str());

  {
    // 创建四个带填充的浮点数缓冲区对象
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    // 对第一个缓冲区进行初始化，填充值为 i * i
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    // 对第二个缓冲区进行初始化，填充值为 j * j
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b_v(i, j) = j * j;
      }
    }
    // 对第三个缓冲区进行初始化，填充值为 i + j
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    // 对第四个缓冲区进行初始化，填充值为 i * j
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    // 创建用于存储计算结果的浮点数填充缓冲区
    PaddedBuffer<float> z_v(M, N, K);
    // 创建用于存储参考结果的浮点数填充缓冲区
    PaddedBuffer<float> z_ref(M, N, K);
    // 计算参考结果并填充到 z_ref 中
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    // 创建简单的 IR 评估器对象，用于评估计算结果
    SimpleIREvaluator eval(stmt, {a_buf, b_buf, c_buf, d_buf, z});
    // 对填充缓冲区进行评估，结果存储在 z_v 中
    eval(a_v, b_v, c_v, d_v, z_v);
    // 检查 z_v 和 z_ref 的所有元素是否在指定的精度范围内接近
    ExpectAllNear(z_v, z_ref, 1e-5);
  }

  // 如果指定的内联顺序长度为 2，则进行额外的计算和代码生成准备
  if (inline_order.size() == 2) {
    Tensor z2 = Compute(
        "z",
        {M, N, K},
        [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
          return a_buf.load(m, n) * b_buf.load(n, k) +
              (c_buf.load(m, n) * d_buf.load(m, k) +
               a_buf.load(m, n) * b_buf.load(n, k));
        });
    // 创建另一个循环嵌套对象，并将额外的计算张量加入其中
    LoopNest l2({z2});
    // 准备 l2 对象进行代码生成
    l2.prepareForCodegen();
    // 获取 l2 对象的根节点语句
    StmtPtr stmt2 = l2.root_stmt();

    // 创建字符串流对象，用于获取和处理语句的字符串表示形式
    std::ostringstream oss2;
    oss2 << *stmt2;
    // 调用 remove_space 函数去除字符串中的空格，并将结果存储在 str2 中
    std::string str2 = remove_space(oss2.str());

    // 使用断言 ASSERT_EQ 检查 str1 和 str2 是否相等
    ASSERT_EQ(str1, str2);

    // 使用断言 ASSERT_GT 检查 str1 的长度是否大于 100
    ASSERT_GT(str1.size(), 100);
  }
// 定义一个测试用例，用于测试循环嵌套中内联函数的调度，第一种情况
TEST(LoopNest, ScheduleInlineFunc01) {
  // 调用内联函数助手，参数为{"x", "y"}，执行函数内联
  InlineFunc01Helper({"x", "y"});
  // 调用内联函数助手，参数为{"y", "x"}，执行函数内联
  InlineFunc01Helper({"y", "x"});
  // 调用内联函数助手，参数为{"x"}，执行函数内联
  InlineFunc01Helper({"x"});
  // 调用内联函数助手，参数为{"y"}，执行函数内联
  InlineFunc01Helper({"y"});
  // 调用内联函数助手，参数为空集合{}，执行函数内联
  InlineFunc01Helper({});
}

// 确保在需要时缓存随机变量
TEST(LoopNest, ScheduleInlineRandom) {
  // 定义常量 M, N, K 分别为 4, 5, 6
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 定义张量 x，使用 Compute 函数创建，生成一个 MxNxK 的张量，通过 Lambda 表达式计算
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 返回 kRand 函数与 kInt 的模运算结果
        return Mod::make(Intrinsics::make(kRand, kInt), 5);
      });
  
  // 定义张量 y，使用 Compute 函数创建，生成一个 MxNxK 的张量，通过 Lambda 表达式计算
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 返回张量 x 在指定位置 (m, n, k) 加载的值的两倍之和
        return x.load(m, n, k) + x.load(m, n, k);
      });

  // 创建 LoopNest 对象 l1，将张量 y 作为输出，张量 x, y 作为输入
  LoopNest l1({y}, {x, y});
  // 对 x 的缓存执行内联操作
  l1.computeInline(x.buf());

  // 对生成的 IR 进行简化处理
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // 检查生成的 IR 是否符合预期模式
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       int x = rand();
# CHECK:       y[i, i_1, i_2] = 2 * (x % 5);)IR");
}

// 确保不缓存未被内联的随机变量
TEST(LoopNest, ScheduleInlineRandomUnrelated) {
  // 定义常量 M, N, K 分别为 4, 5, 6
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 定义张量 x，使用 Compute 函数创建，生成一个 MxNxK 的张量，通过 Lambda 表达式计算
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 返回 m * n * k 的计算结果
        return m * n * k;
      });
  
  // 定义张量 y，使用 Compute 函数创建，生成一个 MxNxK 的张量，通过 Lambda 表达式计算
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 返回张量 x 在指定位置 (m, n, k) 加载的值、两次随机数操作的和
        return x.load(m, n, k) + Intrinsics::make(kRand, kInt) +
            Intrinsics::make(kRand, kInt);
      });

  // 创建 LoopNest 对象 l1，将张量 y 作为输出，张量 x, y 作为输入
  LoopNest l1({y}, {x, y});
  // 对 x 的缓存执行内联操作
  l1.computeInline(x.buf());

  // 对生成的 IR 进行简化处理
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // 检查生成的 IR 是否符合预期模式
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       y[i, i_1, i_2] = ((i * i_1) * i_2 + (rand())) + (rand());)IR");
}

// 确保生成正确数量的随机值，与生成张量的维度相匹配
TEST(LoopNest, ScheduleInlineRandomLowerDimensions) {
  // 定义常量 M, N, K 分别为 4, 5, 6
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 定义张量 x，使用 Compute 函数创建，生成一个 M 维度的张量，通过 Lambda 表达式计算
  Tensor x = Compute("x", {M}, [&](const VarHandle& m) {
    // 使用 Mod::make 方法创建一个表达式节点，该节点包含 Intrinsics::make 方法生成的随机数和整数常量 5
    return Mod::make(Intrinsics::make(kRand, kInt), 5);
  });

  // 创建名为 y 的张量，其计算定义如下，取自 x 的加载值并加上 x 的加载值
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m) + x.load(m);
      });

  // 创建 LoopNest 对象 l1，包含张量 y，并指定依赖的张量为 x 和 y
  LoopNest l1({y}, {x, y});

  // 将 x 的缓冲区计算内联到 l1 中
  l1.computeInline(x.buf());

  // 注释指出，由于 SimpleIREvaluator 中未实现 Rand，无法对结果进行比较，即使能够进行种子初始化。
  // 生成 l1 的根语句的简化形式，并将其赋给 stmt1
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // 检查生成的 IR
  // 使用 checkIR 方法比较 stmt1 与给定的 IR 字符串 R"IR( ...
  checkIR(stmt1, R"IR(
// Split a Compute operation into smaller parts and then inline it into another computation.

TEST(LoopNest, ScheduleInlineIntrinsics) {
  // 定义三个维度的大小
  const int M = 4;
  const int N = 5;
  const int K = 6;
  
  // 创建缓冲区对象，分别表示大小为 MxN 和 NxK 的浮点数矩阵
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);

  // 计算操作 x，返回 MxNxK 大小的张量，使用 a_buf 和 b_buf 进行加载和乘法运算
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });

  // 计算操作 y，返回 MxNxK 大小的张量，对 x 进行内联平方根运算
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x.load(m, n, k));
      });

  // 创建 MxN 和 NxK 大小的浮点数填充缓冲区对象
  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);

  // 填充缓冲区对象 a_v，其中 a_v(i, j) = i^2
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }

  // 填充缓冲区对象 b_v，其中 b_v(i, j) = j^2
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }

  // 创建 LoopNest 对象 l1 和 l2，并将计算操作 y 添加到其中
  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);

  // 在 l2 中进行 x 的内联处理
  l2.computeInline(x.buf());

  // 为代码生成准备 l1 和 l2
  l1.prepareForCodegen();
  l2.prepareForCodegen();

  // 简化 l1 和 l2 的根语句，并获取简化后的语句指针
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());

  // 创建 SimpleIREvaluator 对象 eval1 和 eval2，用于评估简化后的语句
  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, y});

  // 创建 MxNxK 大小的浮点数填充缓冲区对象 y_1 和 y_2
  PaddedBuffer<float> y_1(M, N, K);
  PaddedBuffer<float> y_2(M, N, K);

  // 使用 eval1 和 eval2 对 a_v, b_v 和 y_1, y_2 进行评估操作
  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);

  // 验证 y_1 和 y_2 是否接近，精度为 1e-5
  ExpectAllNear(y_1, y_2, 1e-5);

  // 创建 ostringstream 对象 oss1 和 oss2，用于存储简化后语句的字符串表示
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;

  // 断言简化后语句 oss1 的大小大于 oss2 的大小
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

// 确保能够处理 rand 和非 rand 内在函数。
TEST(LoopNest, ScheduleInlineRandWithIntrinsics) {
  // 定义三个维度的大小
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 创建计算操作 x，返回 MxNxK 大小的张量，使用 kRand 内在函数生成随机浮点数
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kRand, kFloat);
      });

  // 创建计算操作 y，返回 MxNxK 大小的张量，对 x 的加载值应用 kSqrt 内在函数
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x.load(m, n, k));
      });

  // 创建 LoopNest 对象 l1，并将计算操作 y 添加到其中
  LoopNest l1({y}, {x, y});

  // 在 l1 中进行 x 的内联处理
  l1.computeInline(x.buf());

  // 简化 l1 的根语句，并获取简化后的语句指针
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // 检查我们生成的 IR
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       float x = rand();
# CHECK:       y[i, i_1, i_2] = sqrt(x);)IR");
}
// 在 LoopNest 类中测试 ScheduleSplitAThenInline 测试用例
TEST(LoopNest, ScheduleSplitAThenInline) {
  // 创建张量 a，其计算表达式为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 创建张量 b，其计算表达式为 a.load(j + ExprHandle(8))，其中 j 是变量
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  // 创建 LoopNest 对象 l，并传入张量 b 和 a, b 作为构造函数参数
  LoopNest l({b}, {a, b});
  // 获取所有写入 a.buf() 的循环嵌套，并取第一个循环的列表
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 使用 LoopNest::splitWithMask 将 loops[0] 的循环分裂为大小为 4 的块
  LoopNest::splitWithMask(loops[0], 4);
  // 断言在 l 中 a.buf() 的计算没有被内联
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// 在 LoopNest 类中测试 ScheduleSplitBThenInline 测试用例
TEST(LoopNest, ScheduleSplitBThenInline) {
  // 创建张量 a，其计算表达式为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 创建张量 b，其计算表达式为 a.load(j + ExprHandle(8))，其中 j 是变量
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  // 创建 LoopNest 对象 l，并传入张量 b 和 a, b 作为构造函数参数
  LoopNest l({b}, {a, b});
  // 获取所有写入 b.buf() 的循环嵌套，并取第一个循环的列表
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(b.buf()).at(0);
  // 使用 LoopNest::splitWithMask 将 loops[0] 的循环分裂为大小为 3 的块
  LoopNest::splitWithMask(loops[0], 3);
  // 在 l 中内联 a.buf() 的计算
  l.computeInline(a.buf());
  // 为代码生成做准备
  l.prepareForCodegen();
  // 简化 l 的根语句并存入 s
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());

  // 创建一个大小为 6 的整数向量 output，所有元素初始化为 0
  std::vector<int> output(6, 0);
  // 创建 SimpleIREvaluator 对象 eval，并传入简化后的语句 s 和张量 b
  SimpleIREvaluator eval(s, {b});
  // 使用 eval 评估并存储结果到 output
  eval(output);

  // 断言 output 中的每个元素都等于 (i + 8) * (i + 8)，其中 i 是索引
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// 在 LoopNest 类中测试 ScheduleSplitTwiceThenInline 测试用例
TEST(LoopNest, ScheduleSplitTwiceThenInline) {
  // 创建张量 a，其计算表达式为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 创建张量 b，其计算表达式为 a.load(j + ExprHandle(8))，其中 j 是变量
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // 声明一个 ForPtr 类型的指针 i_inner
  ForPtr i_inner;

  // 创建 LoopNest 对象 l，并传入张量 b 和 a, b 作为构造函数参数
  LoopNest l({b}, {a, b});
  // 获取所有写入 a.buf() 的循环嵌套，并取第一个循环的列表
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 使用 LoopNest::splitWithMask 将 loops[0] 的循环分裂为大小为 4 的块，并将结果存入 i_inner
  LoopNest::splitWithMask(loops[0], 4, &i_inner);
  // 使用 LoopNest::splitWithMask 将 i_inner 的循环分裂为大小为 2 的块
  LoopNest::splitWithMask(i_inner, 2);
  // 断言在 l 中 a.buf() 的计算没有被内联
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// 在 LoopNest 类中测试 ScheduleInlineThenSplit 测试用例
TEST(LoopNest, ScheduleInlineThenSplit) {
  // 创建张量 a，其计算表达式为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 创建张量 b，其计算表达式为 a.load(j + ExprHandle(8))，其中 j 是变量
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  // 创建 LoopNest 对象 l，并传入张量 b 和 a, b 作为构造函数参数
  LoopNest l({b}, {a, b});
  // 在 l 中内联 a.buf() 的计算
  l.computeInline(a.buf());

  // 查找 l 的根语句中的所有 For 循环，并将结果存入 loops
  std::vector<ForPtr> loops = NodeFinder<For>::find(l.root_stmt());
  // 使用 LoopNest::splitWithMask 将 loops 中最后一个循环分裂为大小为 3 的块
  LoopNest::splitWithMask(loops.back(), 3);
  // 为代码生成做准备
  l.prepareForCodegen();
  // 简化 l 的根语句并存入 s
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建一个大小为 6 的整数向量 output，所有元素初始化为 0
  std::vector<int> output(6, 0);
  // 创建 SimpleIREvaluator 对象 eval，并传入简化后的语句 s 和张量 b
  SimpleIREvaluator eval(s, {b});
  // 使用 eval 评估并存储结果到 output
  eval(output);

  // 断言 output 中的每个元素都等于 (i + 8) * (i + 8)，其中 i 是索引
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}
// 定义一个名为 LoopNest 的测试类，测试循环嵌套和调度相关功能
TEST(LoopNest, ScheduleSplitInlineThenSplit) {
  // 定义一个长度为 18 的 Tensor a，使用 lambda 表达式计算元素 i * i
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 定义一个长度为 16 的 Tensor b，使用 lambda 表达式计算元素 a.load(j + ExprHandle(8))
  Tensor b = Compute(
      "b", {16}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  // 创建 LoopNest 对象 l，包含张量 b 和 a
  LoopNest l({b}, {a, b});
  // 查找所有的 For 循环节点
  auto loops = NodeFinder<For>::find(l.root_stmt());
  // 对 loops 中最后一个循环进行 splitWithMask 分割操作，参数为 2
  LoopNest::splitWithMask(loops.back(), 2);
  // 将张量 a 内联
  l.computeInline(a.buf());

  // 重新查找所有的 For 循环节点
  loops = NodeFinder<For>::find(l.root_stmt());
  // 对 loops 中第一个循环进行 splitWithMask 分割操作，参数为 2
  LoopNest::splitWithMask(loops.front(), 2);
  // 准备代码生成前的准备工作
  l.prepareForCodegen();
  // 对根语句 l.root_stmt() 进行简化操作，返回简化后的语句 s
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建一个长度为 16，初始值为 0 的整数向量 output
  std::vector<int> output(16, 0);
  // 创建 SimpleIREvaluator 对象 eval，使用简化后的语句 s 和张量 b 进行评估
  SimpleIREvaluator eval(s, {b});
  // 将评估结果存入 output
  eval(output);

  // 遍历 output，验证每个元素是否符合预期的值
  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// 测试在内联后简化掉的循环进行过度分割
TEST(LoopNest, ScheduleSplitInlineSimplify) {
  // 定义一个长度为 18 的 Tensor a，使用 lambda 表达式计算元素 4 * i - 2 * i
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) {
    return ExprHandle(4) * i - ExprHandle(2) * i;
  });
  // 定义一个长度为 2 的 Tensor b，使用 lambda 表达式计算元素 a.load(j) - ExprHandle(1)
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j) - ExprHandle(1); });

  // 创建 LoopNest 对象 l，包含张量 b 和 a
  LoopNest l({b}, {a, b});
  // 获取写入到 a.buf() 的所有循环嵌套，取第一个循环
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 对 loops 中的第一个循环进行 splitWithMask 分割操作，参数为 4
  LoopNest::splitWithMask(loops[0], 4);
  // 断言 a.buf() 是否可以内联，预期返回 false
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// 内联一个具有两个消费者的 Compute
TEST(LoopNest, ScheduleInlineThreeMixedOnce) {
  // 定义一个长度为 18 的 Tensor a，使用 lambda 表达式计算元素 i * i
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 定义一个长度为 6 的 Tensor b，使用 lambda 表达式计算元素 a.load(j + ExprHandle(8))
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // 定义一个大小为 4x3 的 Tensor c，使用 lambda 表达式计算元素 a.load(k) * b.load(l)
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  // 创建 LoopNest 对象 l，包含张量 c, a, b
  LoopNest l({c}, {a, b, c});
  // 获取写入到 a.buf() 的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 内联 a.buf()
  l.computeInline(a.buf());
  // 准备代码生成前的准备工作
  l.prepareForCodegen();

  // 对根语句 l.root_stmt() 进行简化操作，返回简化后的语句 s
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建一个长度为 4*3，初始值为 0 的整数向量 output
  std::vector<int> output(4 * 3, 0);
  // 创建 SimpleIREvaluator 对象 eval，使用简化后的语句 s 和张量 c 进行评估
  SimpleIREvaluator eval(s, {c});
  // 将评估结果存入 output
  eval(output);

  // 遍历 output，验证每个元素是否符合预期的值
  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l + 8));
    }
  }
}

// 内联一个 Compute 到 B，然后再内联 B 到 C
TEST(LoopNest, ScheduleInlineThreeMixedTwice) {
  // 定义一个长度为 18 的 Tensor a，使用 lambda 表达式计算元素 i * i
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 定义一个长度为 6 的 Tensor b，使用 lambda 表达式计算元素 a.load(j + ExprHandle(8))
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // 定义一个大小为 4x3 的 Tensor c，使用 lambda 表达式计算元素 a.load(k) * b.load(l)
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  // 创建 LoopNest 对象 l，包含张量 c, a, b
  LoopNest l({c}, {a, b, c});
  // 获取写入到 a.buf() 的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 内联 a.buf()
  l.computeInline(a.buf());
  // 内联 b.buf()
  l.computeInline(b.buf());
  // 准备代码生成前的准备工作
  l.prepareForCodegen();

  // 对根语句 l.root_stmt() 进行简化操作，返回简化后的语句 s
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建一个长度为 4*3，初始值为 0 的整数向量 output
  std::vector<int> output(4 * 3, 0);
  // 创建 SimpleIREvaluator 对象 eval，使用简化后的语句 s 和张量 c 进行评估
  SimpleIREvaluator eval(s, {c});
  // 将评估结果存入 output
  eval(output);

  // 遍历 output，验证每个元素是否符合预期的值
  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l +
// 定义一个测试用例，用于测试循环嵌套的调度和内联操作
TEST(LoopNest, ScheduleInlineThreeMixedInner) {
  // 定义张量a，其计算为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 定义张量b，其计算为 a[j + 8]，其中 j 是变量
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // 定义张量c，其计算为 a[k] * b[l]，其中 k 和 l 是变量
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  // 创建一个循环嵌套对象，传入计算的张量列表
  LoopNest l({c}, {a, b, c});
  // 获取所有写入张量 a 的循环嵌套并返回
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 将张量 b 进行内联
  l.computeInline(b.buf());
  // 为了进行代码生成做准备
  l.prepareForCodegen();

  // 简化生成的 IR 树并返回简化后的语句
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建一个大小为 4*3 的整数向量
  std::vector<int> output(4 * 3, 0);
  // 创建一个简单的 IR 评估器，并执行评估
  SimpleIREvaluator eval(s, {c});
  eval(output);

  // 验证输出结果是否符合预期
  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l + 8));
    }
  }
}

// 分离三个计算操作，然后内联第一个和第二个到第三个计算中
TEST(LoopNest, ScheduleInlineThreeMixedSplit) {
  // 定义张量 a，其计算为 i * i，其中 i 是变量
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  // 定义张量 b，其计算为 a[j + 8]，其中 j 是变量
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // 定义张量 c，其计算为 a[k] * b[l]，其中 k 和 l 是变量
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  // 创建一个循环嵌套对象，传入计算的张量列表
  LoopNest l({c}, {a, b, c});
  // 获取所有写入张量 a 的循环嵌套并返回
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  // 使用掩码分离循环
  LoopNest::splitWithMask(loops[0], 4);
  // 获取所有写入张量 b 的循环嵌套并返回
  loops = l.getAllLoopNestsWritingToBuf(b.buf()).at(0);
  // 使用掩码分离循环
  LoopNest::splitWithMask(loops[0], 3);
  // 获取所有写入张量 c 的循环嵌套并返回
  loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  // 使用掩码分离循环
  LoopNest::splitWithMask(loops[0], 2);

  // 断言内联操作不会成功
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// 检查内联操作是否适用于输出张量
TEST(LoopNest, ScheduleInlineOutputTensors) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  // 定义张量 x，其计算为 m * n * k，其中 m、n、k 是变量
  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return m * n * k;
      });
  // 定义张量 y，其计算为 x[m, n, k] + m，其中 m、n、k 是变量
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + m;
      });

  // 创建一个循环嵌套对象，传入计算的张量列表
  LoopNest l1({x, y});
  // 将张量 x 进行内联
  l1.computeInline(x.buf());

  // 简化生成的 IR 树并返回简化后的语句
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // 检查我们生成的 IR
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       x[i, i_1, i_2] = (i * i_1) * i_2;
# CHECK: for (int i_3 = 0; i_3 < 4; i_3++)
# CHECK:   for (int i_4 = 0; i_4 < 5; i_4++)
# CHECK:     for (int i_5 = 0; i_5 < 6; i_5++)
# CHECK:       y[i_3, i_4, i_5] = i_3 + (i_3 * i_4) * i_5;)IR");
}
TEST(LoopNest, ScheduleInlineWithCompoundIndices) {
  // 输入的IR（中间表示）：
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[i*2,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[0, j] + j * 100ll;
  //     }

  // 创建名为 A 的缓冲区，大小为 {20, 100}，元素类型为 kLong
  BufHandle a_buf("A", {20, 100}, kLong);
  // 创建名为 B 的缓冲区，大小为 {20, 100}，元素类型为 kLong
  BufHandle b_buf("B", {20, 100}, kLong);
  // 创建名为 i 的循环变量，类型为 kLong
  VarHandle i("i", kLong);
  // 创建名为 j 的循环变量，类型为 kLong
  VarHandle j("j", kLong);

  // 创建循环 i，范围是从 0 到 99，生成表达式 i * 500ll 并将结果存储到缓冲区 a_buf 中
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(a_buf, {i * 2, i}, Mul::make(i, static_cast<int64_t>(500))));
  
  // 创建循环 j，范围是从 0 到 99，生成表达式 Load::make(a_buf, {0, j}) + j * 100ll 并将结果存储到缓冲区 b_buf 中
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {static_cast<int64_t>(0), j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  
  // 创建一个代码块，包含上述两个循环
  auto par = Block::make({forI, forJ});

  // 创建一个 LoopNest 对象，以 par 作为根语句，b_buf.node() 作为输入节点
  LoopNest l(par, {b_buf.node()});
  
  // 验证尝试内联 a_buf 节点是否失败
  ASSERT_FALSE(l.computeInline(a_buf.node()));

  // 验证生成的IR保持不变
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t i = 0;
    # CHECK-NEXT:   A[
    # CHECK: for (int64_t j = 0;
    # CHECK-NEXT:   B[)IR");
}

TEST(LoopNest, ScheduleInlineConsumerIndicesWithCast) {
  // 输入的IR（中间表示）：
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[0ll,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[(int64_t)0, j] + j * 100ll;
  //     }

  // 创建名为 A 的缓冲区，大小为 {20, 100}，元素类型为 kLong
  BufHandle a_buf("A", {20, 100}, kLong);
  // 创建名为 B 的缓冲区，大小为 {20, 100}，元素类型为 kLong
  BufHandle b_buf("B", {20, 100}, kLong);
  // 创建名为 i 的循环变量，类型为 kLong
  VarHandle i("i", kLong);
  // 创建名为 j 的循环变量，类型为 kLong
  VarHandle j("j", kLong);

  // 创建循环 i，范围是从 0 到 99，生成表达式 i * 500ll 并将结果存储到缓冲区 a_buf 中
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(
          a_buf,
          {static_cast<int64_t>(0), i},
          Mul::make(i, static_cast<int64_t>(500))));
  
  // 创建循环 j，范围是从 0 到 99，生成表达式 Load::make(a_buf, {0, j}) + j * 100ll 并将结果存储到缓冲区 b_buf 中
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {0, j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  
  // 创建一个代码块，包含上述两个循环
  auto par = Block::make({forI, forJ});

  // 创建一个 LoopNest 对象，以 par 作为根语句，b_buf.node() 作为输入节点
  LoopNest l(par, {b_buf.node()});
  
  // 验证尝试内联 a_buf 节点是否成功
  ASSERT_TRUE(l.computeInline(a_buf.node()));

  // 验证生成的IR
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t j = 0; j < 100; j++) {
    # CHECK:   B[0ll, j] = j * 500ll + j * 100ll;
    # CHECK: })IR");
}
TEST(LoopNest, ScheduleInlineProducerIndicesWithCast) {
  // Input IR:
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[(int64_t)0,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[0ll, j] + j * 100ll;
  //     }
  // 创建名为 "A" 的缓冲区，形状为 {20, 100}，元素类型为长整型
  BufHandle a_buf("A", {20, 100}, kLong);
  // 创建名为 "B" 的缓冲区，形状为 {20, 100}，元素类型为长整型
  BufHandle b_buf("B", {20, 100}, kLong);
  // 创建名为 "i" 的变量，类型为长整型
  VarHandle i("i", kLong);
  // 创建名为 "j" 的变量，类型为长整型
  VarHandle j("j", kLong);
  // 创建用于循环的表达式，对应输入 IR 中的第一个循环
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(a_buf, {0, i}, Mul::make(i, static_cast<int64_t>(500))));
  // 创建用于循环的表达式，对应输入 IR 中的第二个循环
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {static_cast<int64_t>(0), j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  // 创建代码块，包含上述两个循环
  auto par = Block::make({forI, forJ});

  // 创建循环嵌套对象，初始化为 par，使用 b_buf.node() 作为输出缓冲区节点
  LoopNest l(par, {b_buf.node()});
  // 断言能够内联计算 a_buf 的节点
  ASSERT_TRUE(l.computeInline(a_buf.node()));

  // 检查生成的 IR 是否符合预期格式
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t j = 0; j < 100; j++) {
    # CHECK:   B[0ll, j] = j * 500ll + j * 100ll;
    # CHECK: })IR");
}

TEST(LoopNest, ScheduleFuserStyle) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  // 创建名为 "A" 的缓冲区，形状为 {ExprHandle(kTotalSize)}，元素类型为单精度浮点数
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);

  // 创建计算表达式 b，对应于输入 IR 中的第一个计算
  Tensor b =
      Compute("f", {kTotalSize}, [&](const std::vector<VarHandle>& axes) {
        return a_buf.load(axes[0]) + 11.0f;
      });

  // 创建计算表达式 c，对应于输入 IR 中的第二个计算
  Tensor c =
      Compute("g", {kTotalSize}, [&](const std::vector<VarHandle>& axes) {
        return b.load(axes[0]) + 1.0f;
      });

  // 创建循环嵌套对象，初始化为包含 b 和 c
  LoopNest l({b, c});
  // 准备用于代码生成的循环嵌套对象
  l.prepareForCodegen();
  // 获取根语句节点
  StmtPtr s = l.root_stmt();

  // 初始化用于测试的数据向量
  std::vector<float> a_data(kTotalSize, 7.0f);
  std::vector<float> b_data(kTotalSize, 0.0f);
  std::vector<float> c_data(kTotalSize, 0.0f);
  // 使用 SimpleIREvaluator 运行生成的 IR
  SimpleIREvaluator(s, {a_buf, b, c})(a_data, b_data, c_data);

  // 断言生成的 b_data 和 c_data 是否符合预期值
  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(b_data[i], 18.0f);
    ASSERT_EQ(c_data[i], 19.0f);
  }
}

TEST(LoopNest, ScheduleFuserThreeArg) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  // 创建名为 "A" 的缓冲区，形状为 {ExprHandle(kTotalSize)}，元素类型为单精度浮点数
  BufHandle a("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 "B" 的缓冲区，形状为 {ExprHandle(kTotalSize)}，元素类型为单精度浮点数
  BufHandle b("B", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 "C" 的缓冲区，形状为 {ExprHandle(kTotalSize)}，元素类型为单精度浮点数
  BufHandle c("C", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 "D" 的缓冲区，形状为 {ExprHandle(kTotalSize)}，元素类型为单精度浮点数
  BufHandle d("D", {ExprHandle(kTotalSize)}, kFloat);

  // 创建计算表达式 e，对应于输入 IR 中的第一个计算
  Tensor e = Compute("e", {kTotalSize}, [&](const VarHandle& i) {
    return a.load(i) + b.load(i);
  });
  // 创建计算表达式 f，对应于输入 IR 中的第二个计算
  Tensor f = Compute("f", {kTotalSize}, [&](const VarHandle& i) {
    return e.load(i) + c.load(i);
  });
  // 创建计算表达式 g，对应于输入 IR 中的第三个计算
  Tensor g = Compute("g", {kTotalSize}, [&](const VarHandle& i) {
    // 此处省略的代码行
  return f.load(i) + d.load(i);
});

LoopNest l({g}, {e, f, g});
l.computeInline(l.getLoopBodyFor(e));
l.computeInline(l.getLoopBodyFor(f));
l.prepareForCodegen();
StmtPtr s = l.root_stmt();

std::vector<float> a_data(kTotalSize, 1.0f);
std::vector<float> b_data(kTotalSize, 2.0f);
std::vector<float> c_data(kTotalSize, 3.0f);
std::vector<float> d_data(kTotalSize, 4.0f);
std::vector<float> g_data(kTotalSize, 0.0f);

// 创建一个简单的IR求值器，用于执行生成的语句s，并传入数据向量a_data, b_data, c_data, d_data, g_data
SimpleIREvaluator(s, {a, b, c, d, g})(a_data, b_data, c_data, d_data, g_data);

// 遍历计算后的g_data向量，检查每个元素是否等于10.0f
for (int i = 0; i < kTotalSize; i++) {
  ASSERT_EQ(g_data[i], 10.0f);
}
TEST(LoopNest, ScheduleDynamicShape2D) {
  auto testWithSize = [](int32_t M, int32_t N) {
    // 定义变量 m 和 n
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    // 创建缓冲区 a 和 b，形状为 {m, n}，数据类型为 kFloat
    BufHandle a("a", {m, n}, kFloat);
    BufHandle b("b", {m, n}, kFloat);
    // 定义张量 c，形状为 {m, n}，通过 lambda 函数计算 a[i, j] + b[i, j]
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    // 创建 LoopNest 对象，传入张量 c
    LoopNest l({c});
    // 获取根语句
    StmtPtr s = l.root_stmt();
    // 创建 SimpleIREvaluator 对象，用于计算 IR
    SimpleIREvaluator cg(s, {a, b, c, m, n});
    // 初始化测试数据向量 aData, bData, cData
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    // 调用 SimpleIREvaluator 进行计算
    cg.call({aData, bData, cData, M, N});
    // 验证 cData 是否与期望值相近
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  // 分别以不同的大小调用 testWithSize 函数进行测试
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

TEST(LoopNest, LoopNestComputeAt_1) {
  // 验证 compute_at 方法在以下示例上的工作：
  //
  // for (int i_a = 0; i_a < N; i_a++) {
  //   A[i_a] = i_a * i_a
  // }
  // for (int i_b = 0; i_b < N; i_b++) {
  //   B[i_b] = A[i_b]
  // }
  //
  // 转换后，i_b 循环应该有一个临时缓冲区的分配，并且该缓冲区应该在计算 B 时使用。
  // 转换后该循环中不应再使用 A，A 的计算也不应内联到 B 中，而应计算到临时变量中，
  // 并且临时变量应在 B 中使用。
  VarHandle N("N", kInt);
  // 定义张量 A，形状为 {N}，通过 lambda 函数计算 i_a * i_a
  Tensor A = Compute("A", {N}, [&](const VarHandle& i_a) { return i_a * i_a; });
  // 定义张量 B，形状为 {N}，通过 lambda 函数从 A 中加载数据
  Tensor B =
      Compute("B", {N}, [&](const VarHandle& i_b) { return A.load(i_b); });
  // 创建 LoopNest 对象，传入张量 B 和 A
  LoopNest l({B}, {A, B});
  // 获取写入张量 B 的所有循环嵌套，并选择第一个循环
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(B.buf()).at(0);
  // 在 A 的循环体中将 computeAt 应用到选择的循环
  LoopNest::computeAt(l.getLoopBodyFor(A), loops[0]);
  // 为代码生成做准备
  l.prepareForCodegen();
  // 创建 SimpleIREvaluator 对象，用于计算 IR
  SimpleIREvaluator cg(l.root_stmt(), {B, N});
  // 获取语句 s
  StmtPtr s = cg.stmt();

  // 检查生成的 IR 是否符合预期
  checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1]
# CHECK: for (int i = 0; i < N; i++)
# CHECK:   temp[
# CHECK-NOT: A[
# CHECK:   B[i_1] = temp[0]
# CHECK:   Free(temp))IR");

  // 现在检查循环是否仍然产生正确的结果
  std::vector<int> b_data(100, 0);
  // 调用 SimpleIREvaluator 进行计算
  cg.call({b_data, 100});

  // 创建预期结果向量 b_ref
  std::vector<int> b_ref(100, 0);
  for (int i = 0; i < 100; i++) {
    b_ref[i] = i * i;
  }
  // 断言 b_data 和 b_ref 是否全部相等
  assertAllEqual(b_data, b_ref);
}
TEST(LoopNest, LoopNestComputeAt_2) {
  // Verify that compute_at works on the following example:
  //
  // for (int py = 0; py < H+1; py++) {
  //   for (int px = 0; px < W+1; px++) {
  //     p[py, px] = py*px
  //   }
  // }
  // for (int cy = 0; cy < H; cy++) {
  //   for (int cx = 0; cx < W; cx++) {
  //     c[py, px] = p[cy,cx]   + p[cy+1,cx] +
  //                 p[cy,cx+1] + p[cy+1,cx+1]
  //   }
  // }

  // 定义常量 kW 和 kH，表示宽度和高度为 16
  const int kW = 16, kH = 16;
  // 定义变量 W 和 H，类型为整型，用于表示宽度和高度
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  // 创建张量 p，表示计算 px * py 的结果，维度为 H+1 x W+1
  Tensor p = Compute(
      "prod", {H + 1, W + 1}, [&](const VarHandle& py, const VarHandle& px) {
        return px * py;
      });
  // 创建张量 c，表示计算四个相邻元素的和，维度为 H x W
  Tensor c =
      Compute("cons", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
        return p.load(y, x) + p.load(y + 1, x) + p.load(y, x + 1) +
            p.load(y + 1, x + 1);
      });

  // 创建一个长度为 kW*kH 的整型向量 c_ref，用于存储参考结果
  std::vector<int> c_ref(kW * kH, 0);
  // 计算参考结果，填充到 c_ref 中
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }
  // 创建原始的循环嵌套对象 orig_loopnest，包含张量 c 和 p
  LoopNest orig_loopnest({c}, {p, c});

  {
    // 首先尝试在外部循环 cy 上计算 P
    LoopNest l(orig_loopnest);
    // 获取写入 c.buf() 的所有循环嵌套
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    // 在 P 的循环体中，将计算位置指定为 loops[0] 所在的位置
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[0]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    StmtPtr s = cg.stmt();

    // 检查生成的 IR
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK: for (int i_2 = 0; i_2 < H; i_2++)
# CHECK:   for
# CHECK:     for
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++)
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK: Free(temp))IR");

    // 确保循环仍然产生正确的结果
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // 现在尝试在内部循环 cx 上计算 P
    LoopNest l(orig_loopnest);
    // 获取写入 c.buf() 的所有循环嵌套
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    // 在 P 的循环体中，将计算位置指定为 loops[1] 所在的位置
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[1]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    StmtPtr s = cg.stmt();

    // 检查生成的 IR
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, 2]
# CHECK: for (int i_2 = 0; i_2 < H; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++)
# CHECK:     for
# CHECK:       for
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK: Free(temp))IR");

    // 确保循环仍然产生正确的结果
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}
TEST(LoopNest, LoopNestComputeAt_3) {
  // 验证 compute_at 在以下示例中的工作：
  //
  // A(x,y) = x*y
  // B(x,y) = A(x, y)
  // C(x,y) = B(x+1, y)
  // D(x,y) = A(x, y+1) + C(x, y)
  //
  // 即 'A' 直接和间接通过 'C' 影响 'D'。

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);  // 创建表示宽度的变量句柄 W
  VarHandle H("H", kInt);  // 创建表示高度的变量句柄 H
  Tensor A = Compute(
      "A", {H + 1, W + 1}, [&](const VarHandle& ay, const VarHandle& ax) {
        return ax * ay;  // 计算张量 A 的值，为 ax * ay
      });
  Tensor B = Compute(
      "B", {H + 1, W + 1}, [&](const VarHandle& by, const VarHandle& bx) {
        return A.load(by, bx);  // 计算张量 B 的值，为 A(by, bx)
      });
  Tensor C =
      Compute("C", {H, W}, [&](const VarHandle& cy, const VarHandle& cx) {
        return B.load(cy, cx + 1);  // 计算张量 C 的值，为 B(cy, cx + 1)
      });
  Tensor D =
      Compute("D", {H, W}, [&](const VarHandle& dy, const VarHandle& dx) {
        return A.load(dy + 1, dx) + C.load(dy, dx);  // 计算张量 D 的值，为 A(dy + 1, dx) + C(dy, dx)
      });

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = (y + 1) * x + y * (x + 1);  // 计算参考结果 c_ref
    }
  }

  LoopNest orig_loopnest({D}, {A, B, C, D});
  {
    // 首先尝试在轴 dy 上将 A 计算
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(D.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(A), loops[0]);  // 在 l 中的 A 循环体上进行 computeAt 操作
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {D, W, H});
    StmtPtr s = cg.stmt();

    // 检查生成的 IR
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1, W]
# CHECK: for (int i = 0; i < H + 1; i++)
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++)
# CHECK:     A[
# CHECK: for (int i_2 = 0; i_2 < H + 1; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W + 1; i_3++)
# CHECK:     B[
# CHECK: for (int i_4 = 0; i_4 < H; i_4++)
# CHECK:   for (int i_5 = 0; i_5 < W; i_5++)
# CHECK:     C[
# CHECK: for (int i_6 = 0; i_6 < H; i_6++)
# CHECK-NOT: A[)IR");

    // 现在检查循环是否仍然产生正确的结果。
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);  // 断言生成的结果与参考结果相等
  }
  {
    // 现在尝试在轴 dx 上将 A 计算
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(D.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(A), loops[1]);  // 在 l 中的 A 循环体的内部循环上进行 computeAt 操作
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {D, W, H});
    StmtPtr s = cg.stmt();

    // 检查生成的 IR
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1, 1]
# CHECK: for (int i = 0; i < H + 1; i++)
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++)
# CHECK:     A[
# CHECK: for (int i_2 = 0; i_2 < H + 1; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W + 1; i_3++)
# CHECK:     B[
# CHECK: for (int i_4 = 0; i_4 < H; i_4++)
# CHECK:   for (int i_5 = 0; i_5 < W; i_5++)
# CHECK:     C[
# CHECK: for (int i_6 = 0; i_6 < H; i_6++)
{
  // 定义常量 kW 和 kH，并声明变量 W 和 H 作为循环的边界
  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);  // 声明 W 为整数类型的变量句柄
  VarHandle H("H", kInt);  // 声明 H 为整数类型的变量句柄

  // 创建名为 p 的计算张量，其维度为 {H + 1, W + 1}，并使用 lambda 表达式定义计算逻辑
  Tensor p = Compute(
      "prod", {H + 1, W + 1}, [&](Axis py, Axis px) { return px * py; });

  // 创建名为 c 的约简张量，其维度为 {H, W}，使用 Sum 运算符，并使用 lambda 表达式定义约简逻辑
  Tensor c = Reduce(
      "cons",
      {H, W},
      Sum(),
      [&](Axis y, Axis x, Axis r, Axis s) { return p.load(y + r, x + s); },
      {2, 2});

  // 创建参考结果的向量 c_ref，初始化为 kW * kH 个零
  std::vector<int> c_ref(kW * kH, 0);

  // 计算 c_ref 中每个元素的期望值，并存储在 c_ref 中
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }

  // 创建原始循环嵌套对象 orig_loopnest，用于进一步的分析和操作
  LoopNest orig_loopnest({c}, {p, c});

  // 检查原始循环嵌套对象的中间表示是否符合预期
  checkIR(orig_loopnest.root_stmt(), R"IR(
# CHECK: for (int i = 0; i < H + 1; i++) {
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++) {
# CHECK:     prod[i, i_1] = i_1 * i;
# CHECK:   }
# CHECK: }
# CHECK: for (int i_2 = 0; i_2 < H; i_2++) {
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++) {
# CHECK:     cons[i_2, i_3] = int(0);
# CHECK:     for (int i_4 = 0; i_4 < 2; i_4++) {
# CHECK:       for (int i_5 = 0; i_5 < 2; i_5++) {
# CHECK:         cons[i_2, i_3] = ReduceOp((cons[i_2, i_3]) + (prod[i_2 + i_4, i_3 + i_5]), reduce_args={i_4, i_5});
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
)IR");

  {
    // 创建循环嵌套对象 l，并复制 orig_loopnest
    LoopNest l(orig_loopnest);

    // 获取写入到 c.buf() 的所有循环嵌套，并选择第一个循环嵌套作为关键循环
    auto loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);

    // 在 p 的循环体上使用循环 l 的关键循环
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[0]);

    // FIXME: 在此处调用 simplify 会破坏中间表示
    // MALFORMED INPUT: could not find base node in Load - temp[...]
    // l.simplify();

    // 消除无效存储
    l.eliminateDeadStores();

    // 为代码生成做准备
    l.prepareForCodegen();

    // 创建简单的中间表示求值器 cg，使用 l 的根语句和 {c, W, H} 作为参数
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});

    // 检查中间表示是否符合预期
    checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK: for (int i = 0; i < H; i++) {
# CHECK:   for (int idx0 = 0; idx0 < 2; idx0++) {
# CHECK:     for (int idx1 = 0; idx1 < W + 1; idx1++) {
# CHECK:       temp[(0 + idx0 * (1 * (W + 1))) + idx1 * 1] = (idx0 + i) * (idx1 + 0);
# CHECK:     }
# CHECK:   }
# CHECK:   for (int i_1 = 0; i_1 < W; i_1++) {
# CHECK:     cons[(0 + i * (1 * W)) + i_1 * 1] = int(0);
# CHECK:     for (int i_2 = 0; i_2 < 2; i_2++) {
# CHECK:       for (int i_3 = 0; i_3 < 2; i_3++) {
# CHECK:         cons[(0 + i * (1 * W)) + i_1 * 1] = (cons[(0 + i * (1 * W)) + i_1 * 1]) + (temp[(0 + i_2 * (1 * (W + 1))) + (i_1 + i_3) * 1]);
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
# CHECK: Free(temp);
)IR");

    // 确保循环计算产生了正确的结果
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});
    assertAllEqual(c_data, c_ref);
  }

  {
    // 现在尝试在内部循环 cx (最内层循环) 上计算 P
    # 使用原始循环嵌套(orig_loopnest)创建一个新的循环嵌套对象l
    LoopNest l(orig_loopnest);
    
    # 获取所有写入到缓冲区(c.buf())的循环嵌套，并将其放入loops向量中的第一个元素
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    
    # 将循环嵌套l中的p所属的循环体计算位置设置为loops向量中索引为1的循环
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[1]);
    
    # 简化循环嵌套l，优化其结构
    l.simplify();
    
    # 消除循环嵌套l中的死存储（即不会再被使用的存储）
    l.eliminateDeadStores();
    
    # 为生成代码做准备，可能会进行进一步的优化或准备工作
    l.prepareForCodegen();
    
    # 使用简单的IR评估器创建cg对象，评估循环嵌套l的根语句，并传入c、W、H等参数
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    
    # 检查IR代码cg.stmt()，并使用R"IR(...)语法定义其原始字符串内容
    checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(temp); // 分配临时数组 temp
# CHECK: for (int i = 0; i < H; i++) { // 外层循环，遍历 H 次
# CHECK:   for (int i_1 = 0; i_1 < W; i_1++) { // 内层循环，遍历 W 次
# CHECK:     for (int idx0 = 0; idx0 < 2; idx0++) { // 内层循环，遍历 2 次
# CHECK:       for (int idx1 = 0; idx1 < 2; idx1++) { // 内层循环，遍历 2 次
# CHECK:         temp[(0 + idx0 * (1 * 2)) + idx1 * 1] = (i + idx0) * (i_1 + idx1); // 计算并存储值到 temp 数组中
# CHECK:       }
# CHECK:     }
# CHECK:     cons[(0 + i * (1 * W)) + i_1 * 1] = 0; // 初始化 cons 数组的值为 0
# CHECK:     for (int i_2 = 0; i_2 < 2; i_2++) { // 内层循环，遍历 2 次
# CHECK:       for (int i_3 = 0; i_3 < 2; i_3++) { // 内层循环，遍历 2 次
# CHECK:         cons[(0 + i * (1 * W)) + i_1 * 1] = (cons[(0 + i * (1 * W)) + i_1 * 1]) + (temp[(0 + i_2 * (1 * 2)) + i_3 * 1]); // 根据 temp 数组的值更新 cons 数组
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
# CHECK: Free(temp); // 释放临时数组 temp 的内存空间
// 检查循环嵌套，应用于以下代码块
// CHECK:     for (int r = 0; r < 3; r++) {
// CHECK:       B[n, h] = ReduceOp((B[n, h]) + (temp[0, r + h]), reduce_args={r});
// CHECK:     }
// CHECK:   }
// CHECK: }

)IR");

// 对循环结构进行简化
l.simplify();

// 准备进行代码生成前的准备工作
l.prepareForCodegen();

// 获取根语句指针
StmtPtr s = l.root_stmt();

// 创建简单的 IR 评估器，传入根语句和数据集 {IP, B}
SimpleIREvaluator cg(s, {IP, B});

// 创建一个大小为 N * H 的全一张量 At，数据类型为 kFloat
auto At = at::arange(N * H, at::kFloat).reshape({N, H});

// 使用 conv1d 函数进行一维卷积操作 Rt，At 是输入张量，卷积核为大小为 1x1x3 的全一张量
// 参数 stride 设置为 1，padding 设置为 3
auto Rt = at::conv1d(
    At, at::ones({1, 1, 3}), at::Tensor(), /*stride=*/1, /*padding=*/3);

// 创建一个与 Rt 相同大小的空张量 Bt
auto Bt = at::empty_like(Rt);

// 调用 cg 对象，传入 At 和 Bt 的数据指针进行计算
cg.call({At.data_ptr<float>(), Bt.data_ptr<float>()});

// 断言 Rt 和 Bt 张量内容近似相等
ASSERT_TRUE(at::allclose(Rt, Bt));
}

// LoopOrderHelper 类，继承自 IRVisitor，用于辅助获取循环顺序信息
class LoopOrderHelper : public IRVisitor {
  std::stringstream ordering;

 public:
  // 获取给定语句 s 的循环顺序信息的方法
  std::string getOrder(StmtPtr s) {
    ordering.str("");
    s->accept(this);
    return ordering.str();
  }

  // 重写 visit 方法，用于访问 For 循环节点
  void visit(ForPtr v) final {
    ordering << v->var()->name_hint() << ",";
    IRVisitor::visit(v);
  }
};

// LoopNest 类的单元测试 LoopNestReorderAxis1
TEST(LoopNest, LoopNestReorderAxis1) {
  // 创建一个张量对象 tensor，计算表达式 f
  Tensor tensor =
      Compute("f", {2, 3}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });

  // 创建 LoopNest 对象 l，包含张量 tensor
  LoopNest l({tensor});

  // 复制根语句并准备名称，返回新的语句指针 stmt1
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 创建一个包含 6 个零的整数向量 stmt1_output
  std::vector<int> stmt1_output(6, 0);

  // 创建 SimpleIREvaluator 对象 cg，传入 stmt1 和张量 tensor
  SimpleIREvaluator cg(stmt1, {tensor});

  // 调用 cg 对象，传入 stmt1_output 的数据指针进行计算
  cg.call({stmt1_output});

  // 获取写入到 tensor.buf() 的所有循环嵌套，并选择第一个
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 重新排序 loops 中的轴，将第一个和第二个轴交换
  LoopNest::reorderAxis(loops[0], loops[1]);

  // 复制根语句并准备名称，返回新的语句指针 stmt2
  StmtPtr stmt2 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 断言 stmt1 和 stmt2 不相等
  ASSERT_NE(stmt1, stmt2);

  // 创建 LoopOrderHelper 对象 loopOrderHelper
  LoopOrderHelper loopOrderHelper;

  // 获取 stmt1 的循环顺序信息并赋值给 order1
  std::string order1 = loopOrderHelper.getOrder(stmt1);

  // 获取 stmt2 的循环顺序信息并赋值给 order2
  std::string order2 = loopOrderHelper.getOrder(stmt2);

  // 断言 order1 等于 "j,i,"
  ASSERT_EQ(order1, "j,i,");

  // 断言 order2 等于 "i,j,"
  ASSERT_EQ(order2, "i,j,");

  // 创建一个包含 6 个零的整数向量 stmt2_output
  std::vector<int> stmt2_output(6, 0);

  // 创建 SimpleIREvaluator 对象 cg2，传入 stmt2 和张量 tensor
  SimpleIREvaluator cg2(stmt2, {tensor});

  // 调用 cg2 对象，传入 stmt2_output 的数据指针进行计算
  cg.call({stmt2_output});

  // 遍历比较 stmt1_output 和 stmt2_output 的每个元素
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  // 将 loops 还原回原来的顺序
  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[1]);

  // 获取 l 的根语句并赋值给 stmt3
  StmtPtr stmt3 = l.root_stmt();

  // 获取 stmt3 的循环顺序信息并赋值给 order3
  std::string order3 = loopOrderHelper.getOrder(stmt3);

  // 断言 order3 等于 order1
  ASSERT_EQ(order3, order1);

  // 创建两个字符串流 oss1 和 oss2
  std::ostringstream oss1, oss2;

  // 将 stmt1 和 stmt3 分别输出到 oss1 和 oss2 中
  oss1 << *stmt1;
  oss2 << *stmt3;

  // 断言 oss1.str() 等于 oss2.str()
  // 即 stmt1 和 stmt3 序列化后的字符串应该完全相同
  ASSERT_EQ(oss1.str(), oss2.str());
}
// 定义一个名为 LoopNest 的测试类，用于测试循环嵌套的轴重排序功能
TEST(LoopNest, LoopNestReorderPartialAxes) {
  // 创建一个张量对象 tensor，其维度为 {2, 3, 4}，并定义计算表达式
  Tensor tensor = Compute(
      "f",
      {2, 3, 4},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  // 创建 LoopNest 对象 l，用于管理 tensor 的循环嵌套
  LoopNest l({tensor});

  // 创建 LoopOrderHelper 对象，用于帮助管理循环顺序
  LoopOrderHelper loopOrderHelper;
  // 克隆根语句并清理名称，得到 stmt1
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));
  // 断言根据 stmt1 的循环顺序，其顺序为 "i,j,k,"
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "i,j,k,");

  // 创建一个大小为 24 的整数向量 stmt1_output，并利用 stmt1 进行简单的IR求值
  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  // 获取写入 tensor 缓冲区的所有循环嵌套，取第一个
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops 的第一个和第二个循环进行轴重排序
  LoopNest::reorderAxis(loops[0], loops[1]);
  // 断言根据当前 l 的根语句，其顺序为 "j,i,k,"
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "j,i,k,");

  // 克隆根语句得到 stmt2
  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  // 创建一个大小为 24 的整数向量 stmt2_output，并利用 stmt2 进行简单的IR求值
  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  // 验证 stmt1_output 和 stmt2_output 是否一致
  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  // 再次获取写入 tensor 缓冲区的所有循环嵌套
  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops 的第二个和第三个循环进行轴重排序
  LoopNest::reorderAxis(loops[1], loops[2]);
  // 断言根据当前 l 的根语句，其顺序为 "j,k,i,"
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "j,k,i,");

  // 克隆根语句得到 stmt3
  StmtPtr stmt3 = Stmt::clone(l.root_stmt());

  // 创建一个大小为 24 的整数向量 stmt3_output，并利用 stmt3 进行简单的IR求值
  std::vector<int> stmt3_output(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor});
  cg3.call({stmt3_output});

  // 验证 stmt1_output 和 stmt3_output 是否一致
  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt3_output[i]);
  }
}

// 定义另一个测试类 LoopNest，用于测试循环嵌套的内部轴重排序功能
TEST(LoopNest, LoopNestReorderInternalAxis) {
  // 创建一个张量对象 tensor，其维度为 {1, 2, 3, 4}，并定义计算表达式
  Tensor tensor = Compute(
      "f",
      {1, 2, 3, 4},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  // 创建 LoopNest 对象 l，用于管理 tensor 的循环嵌套
  LoopNest l({tensor});

  // 创建 LoopOrderHelper 对象，用于帮助管理循环顺序
  LoopOrderHelper loopOrderHelper;
  // 克隆根语句并清理名称，得到 stmt1
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));
  // 断言根据 stmt1 的循环顺序，其顺序为 "i,j,k,l,"
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "i,j,k,l,");

  // 创建一个大小为 24 的整数向量 stmt1_output，并利用 stmt1 进行简单的IR求值
  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  // 获取写入 tensor 缓冲区的所有循环嵌套，取第一个
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 对 loops 的第三个和第二个循环进行轴重排序
  LoopNest::reorderAxis(loops[2], loops[1]);
  // 断言根据当前 l 的根语句，其顺序为 "i,k,j,l,"
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "i,k,j,l,");

  // 克隆根语句得到 stmt2
  StmtPtr stmt2 = l.root_stmt();

  // 创建一个大小为 24 的整数向量 stmt2_output，并利用 stmt2 进行简单的IR求值
  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  // 验证 stmt1_output 和 stmt2_output 是否一致
  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}
TEST(LoopNest, LoopNestReorderEnclosingAxis) {
  // 创建一个张量对象，表示一个计算操作 "f"，输入维度为{1, 2, 3, 4}，计算函数为 lambda 表达式
  Tensor tensor = Compute(
      "f",
      {1, 2, 3, 4},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  
  // 创建一个 LoopNest 对象，并将 tensor 加入其中
  LoopNest l({tensor});

  // 创建一个 LoopOrderHelper 对象
  LoopOrderHelper loopOrderHelper;

  // 克隆并重命名根语句
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 准备一个用于存储输出的向量，初始值全为 0，大小为 24
  std::vector<int> stmt1_output(24, 0);

  // 创建一个 SimpleIREvaluator 对象，用于评估 stmt1
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  // 获取所有写入 tensor 缓冲区的循环嵌套，并取第一个嵌套
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 重新排序 loops 中的第一个和第四个循环轴
  LoopNest::reorderAxis(loops[0], loops[3]);

  // 断言循环顺序符合预期 "l,j,k,i,"
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "l,j,k,i,");

  // 获取当前的根语句，并命名为 stmt2
  StmtPtr stmt2 = l.root_stmt();

  // 准备一个用于存储输出的向量，初始值全为 0，大小为 24
  std::vector<int> stmt2_output(24, 0);

  // 创建一个 SimpleIREvaluator 对象，用于评估 stmt2
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  // 检查 stmt1_output 和 stmt2_output 是否一致
  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

TEST(LoopNest, LoopNestReorderSameAxis) {
  // 创建一个张量对象，表示一个计算操作 "f"，输入维度为 {2, 3}，计算函数为 lambda 表达式
  Tensor tensor =
      Compute("f", {2, 3}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  
  // 创建一个 LoopNest 对象，并将 tensor 加入其中
  LoopNest l({tensor});

  // 克隆当前根语句，并命名为 stmt1
  StmtPtr stmt1 = Stmt::clone(l.root_stmt());

  // 获取所有写入 tensor 缓冲区的循环嵌套，并取第一个嵌套
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 尝试重新排序同一个循环轴，不应对根语句产生影响
  LoopNest::reorderAxis(loops[1], loops[1]);

  // 获取当前的根语句，并命名为 stmt2
  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  // 创建两个字符串流，用于比较 stmt1 和 stmt2 的字符串表示是否相同
  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;

  // 断言两者的字符串表示应该一致
  ASSERT_EQ(oss.str(), oss2.str());
}
TEST(LoopNest, LoopNestReorderExtraStatements) {
  /* We're going for a structure like this:
   * for i in ...
   *   Stmt 1
   *   for j in ...
   *     Stmt 2
   *     for k in ...
   *       Stmt 3
   *     Stmt 4
   */

  // 创建一个 Tensor 对象，使用 Compute 函数定义其计算表达式
  Tensor tensor = Compute(
      "f",
      {2, 3, 4},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  // 创建 LoopNest 对象，并将 Tensor 对象作为参数传入
  LoopNest l({tensor});

  // 创建一个名为 extra 的 BufHandle 对象，表示额外的缓冲区
  BufHandle extra("res", {6, 3}, kFloat);

  // 获取循环嵌套结构中所有写入 tensor.buf() 的循环嵌套对象列表，并选取第一个
  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // 获取 loops[0] 循环的循环变量 i
  VarHandle i = VarHandle(loops[0]->var());

  // 创建三个 Store::make 的语句，用于将值存储到 extra 缓冲区中
  StmtPtr store_1 = Store::make(extra, {i, 0}, 1.f);
  StmtPtr store_2 = Store::make(extra, {i, 1}, 2.f);
  StmtPtr store_3 = Store::make(extra, {i, 2}, 4.f);

  // 将 store_1 添加到 loops[0] 的主体前面
  loops[0]->body()->prepend_stmt(store_1);
  // 将 store_2 添加到 loops[1] 的主体前面
  loops[1]->body()->prepend_stmt(store_2);
  // 将 store_3 添加到 loops[1] 的主体末尾
  loops[1]->body()->append_stmt(store_3);

  // 克隆并清理循环嵌套的根语句，并赋值给 stmt1
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 创建两个整数向量，用于测试计算结果
  std::vector<int> extra1(6, 0);
  std::vector<int> res1(24, 0);

  // 使用 SimpleIREvaluator 执行 stmt1，计算结果存储在 res1 和 extra1 中
  SimpleIREvaluator cg(stmt1, {tensor, extra});
  cg.call({res1, extra1});

  /* Then we reorder loop y and z, we want it to look like:
   *
   * for i in ...
   *   Stmt 1
   *   for j in ...
   *     Stmt 2
   *   for j_1 in ...
   *    for k in ...
   *       Stmt 3
   *   for j_2 in ...
   *     Stmt 4
   *
   * We need extra loops because we don't have dependency info about stmt 3
   * and 4.
   *
   */

  // 重新排序 loops[1] 和 loops[2] 的轴
  LoopNest::reorderAxis(loops[1], loops[2]);
  // 克隆并清理循环嵌套的根语句，并赋值给 stmt2
  StmtPtr stmt2 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 检查生成的 IR 是否符合预期
  checkIR(stmt2, R"IR(
# CHECK: for
# CHECK:   res[i, 0] = 1
# CHECK:   for
# CHECK:     res[i, 1] = 2
# CHECK:   for
# CHECK:     for
# CHECK:       f[
# CHECK:   for
# CHECK:     res[i, 2] = 4
)IR");

  // 创建两个整数向量，用于测试计算结果
  std::vector<int> extra2(6, 0);
  std::vector<int> res2(24, 0);

  // 使用 SimpleIREvaluator 执行 stmt2，计算结果存储在 res2 和 extra2 中
  SimpleIREvaluator cg2(stmt2, {tensor, extra});
  cg2.call({res2, extra2});

  // 比较 res1 和 res2，确保它们相等
  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(res1[i], res2[i]);
  }
  // 比较 extra1 和 extra2，确保它们相等
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(extra1[i], extra2[i]);
  }

  /* Now reorder x and the y above stmt 3:
   *
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *
   * for y in ...
   *   for z in ...
   *    for x in ...
   *       Stmt 3
   *
   * for x in ...
   *   for y in ...
   *     Stmt 4
   *
   *
   */

  // 再次获取所有写入 tensor.buf() 的循环嵌套对象列表，并选取第一个
  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // 重新排序 loops[0] 和 loops[2] 的轴
  LoopNest::reorderAxis(loops[0], loops[2]);
  // 克隆并清理循环嵌套的根语句，并赋值给 stmt3
  StmtPtr stmt3 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // 检查生成的 IR 是否符合预期
  checkIR(stmt3, R"IR(
# CHECK: for
# CHECK:   res[i, 0] = 1
# CHECK:   for
# CHECK:     res[i, 1] = 2
# CHECK: for
# CHECK:   for
# CHECK:     for
# CHECK:       f[
# CHECK: for
# CHECK:   for
# CHECK:     res[i_2, 2] = 4
)IR");
}
TEST(LoopNest, LoopNestReorderLongStringOfPreOrphans) {
  // 循环遍历 i，范围为 [0, 5)
  for (int i = 0; i < 5; ++i) {
    // 循环遍历 j，范围为 [0, 5)
    for (int j = 0; j < 5; ++j) {
      // 如果 i 不等于 j，则执行 LoopNestReorderTestHelper 函数，前置增加操作，不添加后置增加操作
      if (i != j) {
        LoopNestReorderTestHelper(true, false, i, j);
      }
    }
  }
}

TEST(LoopNest, LoopNestReorderLongStringOfPostOrphans) {
  // 循环遍历 i，范围为 [0, 5)
  for (int i = 0; i < 5; ++i) {
    // 循环遍历 j，范围为 [0, 5)
    for (int j = 0; j < 5; ++j) {
      // 如果 i 不等于 j，则执行 LoopNestReorderTestHelper 函数，不前置增加操作，后置增加操作
      if (i != j) {
        LoopNestReorderTestHelper(false, true, i, j);
      }
    }
  }
}



void LoopNestReorderTestHelper(
    bool prepend,
    bool append,
    int index1,
    int index2) {
  // 创建一个张量 c，形状为 [2, 3, 2, 3, 2]，每个元素初始化为 -1
  Tensor c = Compute(
      "5d", {2, 3, 2, 3, 2}, [](const std::vector<VarHandle>&) { return -1; });
  // 创建 LoopNest 对象 l，包含张量 c
  LoopNest l({c});

  // 创建名为 "extra" 的缓存，形状为 [5]，类型为整数
  BufHandle extra("extra", {5}, kInt);

  // 获取写入到缓存 c.buf() 的所有循环嵌套，并选择第一个
  auto loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  int j = 0;
  // 遍历循环嵌套中的每个循环
  for (auto l : loops) {
    // 在每个循环层级添加一个计数循环执行次数的增量
    LoadPtr load =
        alloc<Load>(extra.node(), std::vector<ExprPtr>({alloc<IntImm>(j)}));
    AddPtr add = alloc<Add>(load, alloc<IntImm>(1));
    StmtPtr store = alloc<Store>(
        extra.node(), std::vector<ExprPtr>({alloc<IntImm>(j)}), add);
    // 如果 prepend 为真，则在循环体前加入 store 语句
    if (prepend) {
      l->body()->prepend_stmt(store);
    }
    // 如果 append 为真，则在循环体后加入 store 语句
    if (append) {
      l->body()->append_stmt(Stmt::clone(store));
    }

    j++;
  }

  // 复制当前循环嵌套的根语句，并保存为 stmt1
  StmtPtr stmt1 = Stmt::clone(l.root_stmt());

  // 初始化长度为 5 的额外向量 extra1，元素值为 0
  std::vector<int> extra1(5, 0);
  // 初始化长度为 2 * 3 * 2 * 3 * 2 的结果向量 res1，元素值为 0
  std::vector<int> res1(2 * 3 * 2 * 3 * 2, 0);
  // 创建 SimpleIREvaluator 对象 cg，传入 stmt1 和 {c, extra}，执行计算
  SimpleIREvaluator cg(stmt1, {c, extra});
  cg.call({res1, extra1});

  // 初始化循环维度数组 loopExtents
  std::vector<int> loopExtents = {2, 3, 2, 3, 2};

  // 初始化预期的循环次数为 0
  int expected_loops = 0;
  // 如果 prepend 为真，则增加预期循环次数
  if (prepend) {
    expected_loops++;
  }
  // 如果 append 为真，则增加预期循环次数
  if (append) {
    expected_loops++;
  }
  // 遍历 loopExtents 数组
  for (int i = 0; i < 5; ++i) {
    // 计算当前维度的预期循环次数
    expected_loops *= loopExtents[i];
    // 断言 extra1[i] 等于预期循环次数
    ASSERT_EQ(extra1[i], expected_loops);
  }

  // 重新获取写入到缓存 c.buf() 的所有循环嵌套
  loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  // 重新排序 loops[index1] 和 loops[index2] 的轴
  LoopNest::reorderAxis(loops[index1], loops[index2]);
  // 复制当前循环嵌套的根语句，并保存为 stmt2
  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  // 创建输出字符串流 oss 和 oss2，分别存储 stmt1 和 stmt2 的字符串表示
  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  // 断言 oss.str() 和 oss2.str() 不相等，即 stmt1 和 stmt2 不同
  ASSERT_NE(oss.str(), oss2.str());

  // 初始化长度为 5 的额外向量 extra2，元素值为 0
  std::vector<int> extra2(5, 0);
  // 初始化长度为 2 * 3 * 2 * 3 * 2 的结果向量 res2，元素值为 0
  std::vector<int> res2(2 * 3 * 2 * 3 * 2, 0);
  // 创建 SimpleIREvaluator 对象 cg2，传入 stmt2 和 {c, extra}，执行计算
  SimpleIREvaluator cg2(stmt2, {c, extra});
  cg2.call({res2, extra2});

  // 重新初始化预期的循环次数为 0
  expected_loops = 0;
  // 如果 prepend 为真，则增加预期循环次数
  if (prepend) {
    expected_loops++;
  }
  // 如果 append 为真，则增加预期循环次数
  if (append) {
    expected_loops++;
  }
  // 遍历 loopExtents 数组
  for (int i = 0; i < 5; ++i) {
    // 计算当前维度的预期循环次数
    expected_loops *= loopExtents[i];
    // 断言 extra2[i] 等于预期循环次数
    ASSERT_EQ(extra2[i], expected_loops);
  }

  // 遍历长度为 2 * 3 * 2 * 3 * 2 的结果向量，断言 res2[i] 等于 res1[i]
  for (int i = 0; i < 2 * 3 * 2 * 3 * 2; ++i) {
    ASSERT_EQ(res2[i], res1[i]);
  }
}
TEST(LoopNest, LoopNestReorderLongStringFull) {
  // 循环嵌套测试：重排长字符串的完整循环嵌套
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // 跳过无操作语句（noops），因为我们检查重新排序后循环不相同。
      if (i != j) {
        // 调用辅助函数测试循环嵌套的重排
        LoopNestReorderTestHelper(true, true, i, j);
      }
    }
  }
}

TEST(LoopNest, LoopNestReorderInternalLoopNest) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });
  Tensor z = Compute(
      "z",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + y.load(m, n, k);
      });

  // 创建循环嵌套对象
  LoopNest l({z}, {x, y, z});
  // 获取写入 y 缓冲区的所有循环嵌套，并选择特定的索引
  ForPtr a = l.getAllLoopNestsWritingToBuf(y.buf())[0][2];
  ForPtr b = l.getAllLoopNestsWritingToBuf(y.buf())[0][0];
  // 重排指定的轴
  LoopNest::reorderAxis(a, b);

  // 为代码生成做准备
  l.prepareForCodegen();
  // 简化中间表示（IR）
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());

  // 检查生成的 IR 是否按正确顺序包含了三个嵌套，但在中间交换了 k 和 m
  checkIR(stmt, R"IR(
# CHECK: < 4
# CHECK: < 5
# CHECK: < 6
# CHECK: < 6
# CHECK: < 5
# CHECK: < 4
# CHECK: < 4
# CHECK: < 5
# CHECK: < 6)IR");

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    // 初始化缓冲区 a_v
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    // 初始化缓冲区 b_v
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b_v(i, j) = j * j;
      }
    }
    // 初始化缓冲区 c_v
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    // 初始化缓冲区 d_v
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);

    // 计算参考结果 z_ref
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    // 使用简单的 IR 评估器执行计算
    SimpleIREvaluator eval(stmt, {a_buf, b_buf, c_buf, d_buf, z});
    eval(a_v, b_v, c_v, d_v, z_v);
    // 检查计算结果 z_v 是否接近参考结果 z_ref
    ExpectAllNear(z_v, z_ref, 1e-5);
  }
}
TEST(LoopNest, OuterLoopVectorization) {
  // 创建一个大小为8x8的张量，并定义其内容为一个函数的计算结果
  Tensor tensor =
      Compute("f", {8, 8}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  // 创建一个循环嵌套对象，并传入张量作为参数
  LoopNest l({tensor});

  // 对循环嵌套对象中写入张量缓冲区的所有循环嵌套进行向量化处理，并验证处理成功
  ASSERT_TRUE(
      LoopNest::vectorize(l.getAllLoopNestsWritingToBuf(tensor.buf())[0][0]));

  // 获取根语句，并验证其为块类型
  StmtPtr root_stmt = l.root_stmt();
  BlockPtr outer_block = to<Block>(root_stmt);
  ASSERT_NE(outer_block, nullptr);

  // 找到最外层的块
  while (BlockPtr inner_block = to<Block>(outer_block->front())) {
    outer_block = inner_block;
  }

  // 验证向量化处理后，只剩下一个循环层级
  ASSERT_EQ(outer_block->nstmts(), 1);
  ForPtr for_loop = to<For>(outer_block->front());
  ASSERT_NE(for_loop, nullptr);
  BlockPtr for_body = for_loop->body();
  ASSERT_EQ(for_body->nstmts(), 1);
  ASSERT_EQ(to<For>(for_body->front()), nullptr);
}

TEST(LoopNest, VectorizeLoopNotNormalized) {
  // 创建一个包含嵌套循环的块，用于计算数组 A 的元素值
  // 输入的 IR 示例:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 1; j < 5; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 1, 5, for_body);
  auto outer_for = For::make(i, 0, 10, inner_for);
  auto block = Block::make({outer_for});
  LoopNest l(block, {a_buf.node()});

  // 对内部循环进行向量化处理，并验证处理成功
  ASSERT_TRUE(LoopNest::vectorize(inner_for));
  ASSERT_EQ(outer_for->body()->nstmts(), 1);
  ASSERT_EQ(to<For>(outer_for->body()->front()), nullptr);
}

namespace {

// 根据给定的上界值生成一个常量上界循环的 IR 字符串表示
std::string constantUpperBoundLoopIR(int upper_bound_val) {
  ExprHandle upper_bound(upper_bound_val);
  // 创建一个张量 A，其大小为 {upper_bound}，内容为每个元素乘以2
  Tensor A =
      Compute("A", {upper_bound}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  // 获取写入到 A 缓冲区的所有循环嵌套，并对第一个进行完全展开
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(loops[0], &unrolled);
  // 将展开后的 IR 字符串输出到流中
  std::ostringstream oss;
  oss << *unrolled;
  return oss.str();
}

} // namespace

TEST(LoopNest, Unroll) {
  // 验证常量上界循环展开的 IR 字符串
  const std::string actual = constantUpperBoundLoopIR(3);
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0] = 0;
# CHECK: A[1] = 2;
# CHECK: A[2] = 4)IR";

  // 使用 FileCheck 验证生成的 IR 字符串
  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

TEST(LoopNest, UnrollOuter) {
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  // 创建一个大小为 {outer_bound} x {inner_bound} 的张量 A，并定义其内容
  Tensor A = Compute(
      "A",
      {outer_bound, inner_bound},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  // 获取写入到 A 缓冲区的所有循环嵌套，并对第一个进行完全展开
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(loops[0], &unrolled);
  // 使用 checkIR 函数验证展开后的 IR 字符串格式
  checkIR(unrolled, R"IR(
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[0, i] = i;
# CHECK: }
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[1, i] = i + 1;
# CHECK: }
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[2, i] = i + 2;
# CHECK: })IR");
}
TEST(LoopNest, UnrollInner) {
  // 定义外部循环的边界为常量表达式 3
  ExprHandle outer_bound(3);
  // 定义内部循环的边界为常量表达式 4
  ExprHandle inner_bound(4);
  // 创建张量 A，大小为 outer_bound x inner_bound，元素为 x + y 的计算结果
  Tensor A = Compute(
      "A",
      {outer_bound, inner_bound},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  // 将张量 A 加入到循环嵌套中
  LoopNest l({A});
  // 获取写入到 A.buf() 的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  // 初始化未展开的语句指针为 nullptr
  StmtPtr unrolled = nullptr;
  // 对循环嵌套中的第一个循环进行完全展开
  LoopNest::fullUnroll(
      static_to<For>(loops[0]->body()->stmts().front()), &unrolled);
  // 检查展开后的中间表示与预期的中间表示是否匹配
  checkIR(loops[0], R"IR(
# CHECK: for (int i = 0; i < 3; i++) {
# CHECK: A[i, 0] = i;
# CHECK: A[i, 1] = i + 1;
# CHECK: A[i, 2] = i + 2;
# CHECK: A[i, 3] = i + 3;
# CHECK: })IR");
}

TEST(LoopNest, UnrollMultipleStatements) {
  // 定义常量 kTotalSize 为 3
  const int kTotalSize = 3;
  // 定义缓冲区 A 和 B，大小为 kTotalSize，元素类型为 kInt
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  // 定义变量 x
  VarHandle x("x", kInt);
  // 创建循环 f，迭代范围为 [0, kTotalSize)，包含两个存储操作
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make(
          {Store::make(a_buf, {x}, x * 2),
           Store::make(b_buf, {x}, Load::make(a_buf, {x}))}));
  // 创建父级块包含循环 f
  auto parent_block = Block::make({f});
  // 初始化未展开的语句指针为 nullptr
  StmtPtr unrolled = nullptr;
  // 对循环 f 进行完全展开
  LoopNest::fullUnroll(f, &unrolled);
  // 检查展开后的中间表示与预期的中间表示是否匹配
  checkIR(unrolled, R"IR(
# CHECK: A[0] = 0;
# CHECK: B[0] = A[0];
# CHECK: A[1] = 2;
# CHECK: B[1] = A[1];
# CHECK: A[2] = 4
# CHECK: B[2] = A[2];)IR");
}

TEST(LoopNest, UnrollNonLiteralConstantBounds) {
  // 定义缓冲区 A，大小为 3x4，元素类型为 kInt
  BufHandle a_buf("A", {3, 4}, kInt);
  // 定义变量 i 和 j
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建内部循环体，包含存储操作 A[i, j] = i * j
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 j，迭代范围为 [0, 4)
  auto inner_for = For::make(j, 0, 4, for_body);
  // 创建外部循环 i，迭代范围为 [2-1, 12/3)，包含内部循环 j
  auto outer_for = For::make(
      i,
      IntImm::make(2) - IntImm::make(1),
      IntImm::make(12) / IntImm::make(3),
      inner_for);
  // 创建块 b，包含外部循环 i
  auto b = Block::make({outer_for});

  // 将外部循环 i 和内部循环 j 添加到 loops 向量中
  std::vector<ForPtr> loops = {outer_for, inner_for};
  // 初始化未展开的语句指针为 nullptr
  StmtPtr unrolled = nullptr;
  // 对外部循环进行完全展开
  LoopNest::fullUnroll(loops[0], &unrolled);
  // 检查展开后的中间表示与预期的中间表示是否匹配
  checkIR(unrolled, R"IR(
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[1, j] = j;
# CHECK: }
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[2, j] = 2 * j;
# CHECK: }
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[3, j] = 3 * j;
# CHECK: })IR");
}

TEST(LoopNest, UnrollNonConstantBounds) {
  // 定义变量 M 和 N
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  // 定义缓冲区 A，大小为 M x N，元素类型为 kInt
  BufHandle a_buf("A", {M, N}, kInt);
  // 定义变量 i 和 j
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建内部循环体，包含存储操作 A[i, j] = i * j
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 j，迭代范围为 [0, N)
  auto inner_for = For::make(j, 0, N, for_body);
  // 创建外部循环 i，迭代范围为 [0, M)，包含内部循环 j
  auto outer_for = For::make(i, 0, M, inner_for);
  // 创建块 block，包含外部循环 i
  auto block = Block::make({outer_for});
  // 创建循环嵌套 l，根语句为 block，写入的缓冲区为 a_buf.node()
  LoopNest l(block, {a_buf.node()});

  // 对内部循环进行展开，展开因子为 8
  LoopNest::unroll(inner_for, 8);
  // 简化循环嵌套 l
  l.simplify();
  // 检查展开后的中间表示与预期的中间表示是否匹配
  checkIR(l.root_stmt(), R"IR(
    // 对矩阵 A 进行初始化，外层循环控制行数 i，范围为 0 到 M
    // 内层循环控制每行的处理，每次处理8列，j_outer 范围为 0 到 N/8
    // 以下语句为对矩阵 A 的赋值操作，依次为每行的每个位置赋值
    // A[i, 8 * j_outer] =
    // A[i, 8 * j_outer + 1] =
    // A[i, 2 * (4 * j_outer + 1)] =
    // A[i, 8 * j_outer + 3] =
    // A[i, 4 * (2 * j_outer + 1)] =
    // A[i, 8 * j_outer + 5] =
    // A[i, 8 * j_outer + 6] =
    // A[i, 8 * j_outer + 7] =
    // 内层循环结束后，处理剩余不足一组的列数（N % 8）
    // A[i, 8 * (N / 8) + j_tail] =
    // 外层循环结束后完成对矩阵 A 的初始化操作
    )IR");
TEST(LoopNest, UnrollByFactorsLessThan2) {
  // 定义 M 和 N 作为整型变量
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  // 创建一个名为 A 的缓冲区，维度为 {M, N}，数据类型为整型
  BufHandle a_buf("A", {M, N}, kInt);
  // 定义整型变量 i 和 j 用于循环索引
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建一个块，包含一个将 i*j 存储到 a_buf 的语句
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 for 循环，循环变量为 j，范围是 [0, N)，循环体为 for_body
  auto inner_for = For::make(j, 0, N, for_body);
  // 创建外部循环 for 循环，循环变量为 i，范围是 [0, M)，循环体为 inner_for
  auto outer_for = For::make(i, 0, M, inner_for);
  // 创建一个包含外部循环的块
  auto block = Block::make({outer_for});
  // 创建一个 LoopNest 对象 l，包含块 block，处理的缓冲区为 a_buf
  LoopNest l(block, {a_buf.node()});

  // 使用因子为 1 进行展开应该不做任何操作
  LoopNest::unroll(inner_for, 1);
  // 检查生成的 IR 是否符合指定模式
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");

  // 使用因子为 0 进行展开应该不做任何操作
  LoopNest::unroll(inner_for, 0);
  // 再次检查生成的 IR 是否符合指定模式
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");

  // 使用负因子进行展开应该不做任何操作
  LoopNest::unroll(inner_for, -2);
  // 第三次检查生成的 IR 是否符合指定模式
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");
}

TEST(LoopNest, UnrollByFactorEqualToIters) {
  // 输入的 IR:
  //   for (int i = 0; i < 5; i++) {
  //     A[i] = i * i;
  //   }
  // 创建一个名为 A 的缓冲区，维度为 {5}，数据类型为整型
  BufHandle a_buf("A", {5}, kInt);
  // 定义整型变量 i 用于循环索引
  VarHandle i("i", kInt);
  // 创建一个块，包含将 i*i 存储到 a_buf 的语句
  auto for_body = Block::make({Store::make(a_buf, {i}, i * i)});
  // 创建 for 循环，循环变量为 i，范围是 [0, 5)，循环体为 for_body
  auto for_loop = For::make(i, 0, 5, for_body);
  // 创建一个包含 for_loop 的块
  auto block = Block::make({for_loop});
  // 创建一个 LoopNest 对象 l，包含块 block，处理的缓冲区为 a_buf
  LoopNest l(block, {a_buf.node()});

  // 使用因子为 5 进行展开
  LoopNest::unroll(for_loop, 5);
  // 检查生成的 IR 是否符合指定模式
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i_outer = 0; i_outer < (5 - 0) / 5; i_outer++)
    # CHECK:   A[5 * i_outer]
    # CHECK:   A[5 * i_outer + 1]
    # CHECK:   A[5 * i_outer + 2]
    # CHECK:   A[5 * i_outer + 3]
    # CHECK:   A[5 * i_outer + 4]
  )IR");
}

TEST(LoopNest, UnrollEmpty) {
  // 调用 constantUpperBoundLoopIR 函数并将结果存储在 actual 中
  const std::string actual = constantUpperBoundLoopIR(0);
  // 设置用于验证的模式字符串
  const std::string& verification_pattern = R"IR(
# CHECK-NOT: A[
  )IR";

  // 运行 FileCheck 对象来验证 actual 是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

TEST(LoopNest, NoUnroll) {
  // 定义名为 N 的整型变量 upper_bound
  VarHandle upper_bound("N", kInt);
  // 定义张量 A，使用 lambda 表达式计算，返回 x*2
  Tensor A = Compute("A", {upper_bound}, [&](const VarHandle& x) { return x * 2; });
  // 创建 LoopNest 对象 l，处理的张量为 A
  LoopNest l({A});
  // 获取写入 A.buf() 的所有循环嵌套
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  // 声明一个空指针 unrolled
  StmtPtr unrolled = nullptr;
  // 调用 fullUnroll 方法，期望抛出异常并检查异常消息是否包含 "non-constant loop"
  ASSERT_THROWS_WITH(
      LoopNest::fullUnroll(loops[0], &unrolled), "non-constant loop");
}
TEST(LoopNest, UnrollWithLet) {
  // 定义常量 kTotalSize 为 3
  const int kTotalSize = 3;
  // 创建名为 A 的缓冲区对象，大小为 kTotalSize，类型为整型
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 kTotalSize，类型为整型
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  // 创建整型变量 e 和 x
  VarHandle e("e", kInt);
  VarHandle x("x", kInt);
  
  // 创建循环对象 f，初始化 x 从 0 到 kTotalSize
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make(
          // 在循环体中创建块，包含 Let 语句设置 e 为 7，以及两个 Store 操作存储到 a_buf 和 b_buf
          {Let::make(e, 7),
           Store::make(a_buf, {x}, e),
           Store::make(b_buf, {x}, e + 1)}));
  
  // 创建父块 parent_block，包含循环对象 f
  auto parent_block = Block::make({f});
  
  // 初始化 unrolled 为空指针
  StmtPtr unrolled = nullptr;
  
  // 对循环进行完全展开，结果存储在 unrolled 中
  LoopNest::fullUnroll(f, &unrolled);
  
  // 创建字符串流 oss，用于存储 unrolled 的输出内容
  std::ostringstream oss;
  oss << *unrolled;
  
  // 验证字符串的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int e = 7;
# CHECK: A[0] = e;
# CHECK: B[0] = e + 1;
# CHECK: A[1] = e;
# CHECK: B[1] = e + 1;
# CHECK: A[2] = e;
# CHECK: B[2] = e + 1;)IR";
  
  // 使用 FileCheck 进行验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 创建大小为 kTotalSize 的整型数组 a_v 和 b_v，初始化为 0
  std::vector<int> a_v(kTotalSize, 0);
  std::vector<int> b_v(kTotalSize, 0);
  
  // 创建 SimpleIREvaluator 对象 eval，用于评估 unrolled 对应的表达式
  SimpleIREvaluator eval(unrolled, {a_buf, b_buf});
  
  // 执行评估，结果存储在 a_v 和 b_v 中
  eval(a_v, b_v);
  
  // 遍历结果数组，进行断言
  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v[i], 7);
    ASSERT_EQ(b_v[i], 8);
  }
}

TEST(LoopNest, IsNormalized) {
  // 创建大小为 100 的 A 和 B 缓冲区对象
  BufHandle a_buf("A", {ExprHandle(100)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  
  // 创建整型变量 i
  VarHandle i("i", kInt);
  
  // 创建 for_stmt 循环，初始化 i 从 50 到 100，存储 B[i] 到 A[i] 的值
  auto for_stmt =
      For::make(i, 50, 100, Store::make(a_buf, {i}, Load::make(b_buf, {i})));
  
  // 创建包含 for_stmt 的块
  Block::make({for_stmt});
  
  // 断言 for_stmt 是否是归一化的（即起始值是否为 0）
  ASSERT_FALSE(LoopNest::isNormalized(for_stmt));
  
  // 设置 for_stmt 的起始值为 0，再次断言是否归一化
  for_stmt->set_start(alloc<IntImm>(0));
  ASSERT_TRUE(LoopNest::isNormalized(for_stmt));
  
  // 创建整型变量 N
  VarHandle N("N", kInt);
  
  // 设置 for_stmt 的起始值为 N.node()，再次断言是否归一化
  for_stmt->set_start(N.node());
  ASSERT_FALSE(LoopNest::isNormalized(for_stmt));
}

TEST(LoopNest, NormalizeStartPositive) {
  // 创建大小为 50 的 A 和 B 缓冲区对象
  const int kTotalSize = 50;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  
  // 创建整型变量 x
  VarHandle x("x", kInt);
  
  // 创建 for_body 块，包含 A[x] = B[x] 和 B[x] = x * 2 的存储操作
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  
  // 创建 for_stmt 循环，初始化 x 从 50 到 100，使用 for_body 块作为循环体
  auto for_stmt = For::make(x, 50, 100, for_body);
  
  // 创建包含 for_stmt 的块
  Block::make({for_stmt});
  
  // 对 for_stmt 进行归一化处理
  LoopNest::normalize(for_stmt);
  
  // 简化处理后的结果
  auto result = IRSimplifier::simplify(for_stmt);
  
  // 创建字符串流 oss，用于存储 result 的输出内容
  std::ostringstream oss;
  oss << *result;
  
  // 验证字符串的模式
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   A[x + 50] = B[x + 50];
        # CHECK:   B[x + 50] = 2 * (x + 50);
      )IR";
  
  // 使用 FileCheck 进行验证
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}
TEST(LoopNest, NormalizeStartNegative) {
  // 定义一个测试用例，测试循环嵌套的正规化处理，起始索引为负数的情况

  // 声明总大小常量
  const int kTotalSize = 150;
  // 创建名为 A 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  // 声明循环变量 x
  VarHandle x("x", kInt);
  // 定义循环体语句块
  auto for_body = Block::make(
      {Store::make(a_buf, {x + 50}, Load::make(kInt, b_buf, {x + 50})),
       Store::make(b_buf, {x + 50}, x * 2)});
  // 创建 for 循环语句，循环变量 x 初始值为 -50，终止条件为 x < 100，循环体为 for_body
  auto for_stmt = For::make(x, -50, 100, for_body);
  // 创建包含 for_stmt 的语句块
  Block::make({for_stmt});

  // 对循环进行正规化处理
  LoopNest::normalize(for_stmt);

  // 对正规化后的 IR 进行简化
  auto result = IRSimplifier::simplify(for_stmt);
  // 创建字符串流对象
  std::ostringstream oss;
  // 将简化后的 IR 输出到字符串流中
  oss << *result;
  // 期望的 IR 字符串
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 150; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * (x - 50);
      )IR";
  // 使用 FileCheck 检查实际输出的 IR 是否符合期望的格式
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeStartZero) {
  // 定义一个测试用例，测试循环嵌套的正规化处理，起始索引为零的情况

  // 声明总大小常量
  const int kTotalSize = 100;
  // 创建名为 A 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  // 声明循环变量 x
  VarHandle x("x", kInt);
  // 定义循环体语句块
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  // 创建 for 循环语句，循环变量 x 初始值为 0，终止条件为 x < 100，循环体为 for_body
  auto for_stmt = For::make(x, 0, 100, for_body);
  // 创建包含 for_stmt 的语句块
  Block::make({for_stmt});

  // 对循环进行正规化处理
  LoopNest::normalize(for_stmt);

  // 对正规化后的 IR 进行简化
  auto result = IRSimplifier::simplify(for_stmt);
  // 创建字符串流对象
  std::ostringstream oss;
  // 将简化后的 IR 输出到字符串流中
  oss << *result;
  // 期望的 IR 字符串
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * x;
      )IR";
  // 使用 FileCheck 检查实际输出的 IR 是否符合期望的格式
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeStartVariable) {
  // 定义一个测试用例，测试循环嵌套的正规化处理，起始索引为变量的情况

  // 声明总大小常量
  const int kTotalSize = 100;
  // 创建名为 A 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 kTotalSize，类型为整数
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  // 声明循环变量 x 和起始索引变量 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 定义循环体语句块
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  // 创建 for 循环语句，循环变量 x 初始值为 y，终止条件为 x < 100，循环体为 for_body
  auto for_stmt = For::make(x, y, 100, for_body);
  // 创建包含 for_stmt 的语句块
  auto parent_block = Block::make({for_stmt});

  // 对循环进行正规化处理
  LoopNest::normalize(for_stmt);

  // 对正规化后的 IR 进行简化
  auto result = IRSimplifier::simplify(for_stmt);
  // 创建字符串流对象
  std::ostringstream oss;
  // 将简化后的 IR 输出到字符串流中
  oss << *result;
  // 期望的 IR 字符串
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100 - y; x++) {
        # CHECK:   A[x + y] = B[x + y];
        # CHECK:   B[x + y] = 2 * (x + y);
      )IR";
  // 使用 FileCheck 检查实际输出的 IR 是否符合期望的格式
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}
TEST(LoopNest, NormalizeOnNestedOuterLoop) {
  // 定义一个测试用例，测试在嵌套的外部循环上进行归一化处理

  // 构造缓冲区对象 a_buf，代表数组 A，起始索引为 50
  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  // 构造缓冲区对象 b_buf，代表数组 B，起始索引为 100
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  // 定义整数变量 x，用于循环迭代
  VarHandle x("x", kInt);
  // 定义整数变量 y，用于内层循环迭代
  VarHandle y("y", kInt);
  // 定义内层循环体，包含对数组 A 和 B 的访问以及计算
  auto inner_for_body = Store::make(
      a_buf, {x}, Load::make(a_buf, {x}) + Load::make(b_buf, {y}) + y * 2);
  // 定义内层循环，y 的范围为 [10, 100)，循环体为 inner_for_body
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  // 定义外层循环，x 的范围为 [50, 100)，循环体为 inner_for
  auto for_stmt = For::make(x, 50, 100, inner_for);
  // 构造一个包含 for_stmt 的代码块
  Block::make({for_stmt});

  // 调用 LoopNest 类的 normalize 方法，对外层循环 for_stmt 进行归一化处理
  LoopNest::normalize(for_stmt);

  // 对归一化后的 IR 进行简化
  auto result = IRSimplifier::simplify(for_stmt);
  // 创建一个字符串流 oss，将简化后的 IR 输出到 oss 中
  std::ostringstream oss;
  oss << *result;
  // 定义预期的 IR 字符串 expected_ir
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   for (int y = 10; y < 100; y++) {
        # CHECK:     A[x + 50] = ((A[x + 50]) + (B[y])) + 2 * y;
      )IR";
  // 使用 FileCheck 工具验证 oss 中的 IR 是否符合 expected_ir 的格式要求
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeOnNestedInnerLoop) {
  // 定义一个测试用例，测试在嵌套的内部循环上进行归一化处理

  // 构造缓冲区对象 a_buf，代表数组 A，起始索引为 50
  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  // 构造缓冲区对象 b_buf，代表数组 B，起始索引为 100
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  // 定义整数变量 x，用于循环迭代
  VarHandle x("x", kInt);
  // 定义整数变量 y，用于内层循环迭代
  VarHandle y("y", kInt);
  // 定义内层循环体，包含对数组 A 和 B 的访问以及计算
  auto inner_for_body = Store::make(
      a_buf, {x}, Load::make(a_buf, {x}) + Load::make(b_buf, {y}) + y * 2);
  // 定义内层循环，y 的范围为 [10, 100)，循环体为 inner_for_body
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  // 定义外层循环，x 的范围为 [50, 100)，循环体为 inner_for
  auto for_stmt = For::make(x, 50, 100, inner_for);
  // 构造一个包含 for_stmt 的代码块
  Block::make({for_stmt});

  // 调用 LoopNest 类的 normalize 方法，对内层循环 inner_for 进行归一化处理
  LoopNest::normalize(inner_for);

  // 对归一化后的 IR 进行简化
  auto result = IRSimplifier::simplify(for_stmt);
  // 创建一个字符串流 oss，将简化后的 IR 输出到 oss 中
  std::ostringstream oss;
  oss << *result;
  // 定义预期的 IR 字符串 expected_ir
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 50; x < 100; x++) {
        # CHECK:   for (int y = 0; y < 90; y++) {
        # CHECK:     A[x] = (((A[x]) + (B[y + 10])) + 2 * y) + 20;
      )IR";
  // 使用 FileCheck 工具验证 oss 中的 IR 是否符合 expected_ir 的格式要求
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}
TEST(LoopNest, NormalizeAndSplitWithTail) {
  // 创建一个虚拟张量以构建 LoopNest。
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);  // 创建一个缓冲区句柄 'a'，大小为 {n}，元素类型为 kFloat
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });  // 定义张量 b，通过 a.load(i) 计算每个元素
  LoopNest l({b});  // 创建 LoopNest 对象，并传入张量列表 {b}

  // 创建输入的 IR:
  //   for (int x = 5; x < 10; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 5;  // 定义总大小 kTotalSize 为 5
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);  // 创建缓冲区句柄 'A'，大小为 {ExprHandle(kTotalSize)}，元素类型为 kInt
  VarHandle x("x", kInt);  // 创建整型变量 'x'
  auto for_stmt = For::make(x, 5, 10, Store::make(a_buf, {x}, x * 2));  // 创建 for 循环语句，将 x * 2 存储到 a_buf 的 {x} 处
  auto parent_block = Block::make({for_stmt});  // 创建包含 for_stmt 的代码块

  LoopNest::normalize(for_stmt);  // 对 for_stmt 进行归一化处理

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_inner;  // 内部循环的指针 x_inner
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_tail;  // 尾部循环的指针 x_tail
  LoopNest::splitWithTail(for_stmt, 10, &x_inner, &x_tail);  // 使用 splitWithTail 方法对 for_stmt 进行分裂，分别得到内部循环 x_inner 和尾部循环 x_tail

  auto x_outer_result = IRSimplifier::simplify(for_stmt);  // 简化外部循环结果为 x_outer_result
  std::ostringstream oss_outer;  // 创建一个输出流 oss_outer
  oss_outer << *x_outer_result;  // 将 x_outer_result 输出到 oss_outer 中
  const std::string& expected_outer_ir =
      R"IR(
        # CHECK: {
        # CHECK: }
      )IR";  // 期望的外部 IR 字符串，用于对比检查
  torch::jit::testing::FileCheck().run(expected_outer_ir, oss_outer.str());  // 使用 FileCheck 检查外部 IR 结果

  auto x_tail_result = IRSimplifier::simplify(x_tail);  // 简化尾部循环结果为 x_tail_result
  std::ostringstream oss_tail;  // 创建一个输出流 oss_tail
  oss_tail << *x_tail_result;  // 将 x_tail_result 输出到 oss_tail 中
  const std::string& expected_tail_ir =
      R"IR(
        # CHECK: for (int x_tail = 0; x_tail < 5; x_tail++) {
        # CHECK:   A[x_tail + 5] = 2 * (x_tail + 5);
      )IR";  // 期望的尾部 IR 字符串，用于对比检查
  torch::jit::testing::FileCheck().run(expected_tail_ir, oss_tail.str());  // 使用 FileCheck 检查尾部 IR 结果
}

TEST(LoopNest, NotNormalizeAndSplitWithTail) {
  // 创建一个虚拟张量以构建 LoopNest。
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);  // 创建一个缓冲区句柄 'a'，大小为 {n}，元素类型为 kFloat
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });  // 定义张量 b，通过 a.load(i) 计算每个元素
  LoopNest l({b});  // 创建 LoopNest 对象，并传入张量列表 {b}

  // 创建输入的 IR:
  //   for (int x = 5; x < 15; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 10;  // 定义总大小 kTotalSize 为 10
  BufHandle a_buf("A", {kTotalSize}, kInt);  // 创建缓冲区句柄 'A'，大小为 {kTotalSize}，元素类型为 kInt
  VarHandle x("x", kInt);  // 创建整型变量 'x'
  auto for_stmt = For::make(x, 5, 15, Store::make(a_buf, {x}, x * 2));  // 创建 for 循环语句，将 x * 2 存储到 a_buf 的 {x} 处
  auto parent_block = Block::make({for_stmt});  // 创建包含 for_stmt 的代码块

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_inner;  // 内部循环的指针 x_inner
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_tail;  // 尾部循环的指针 x_tail
  LoopNest::splitWithTail(for_stmt, 8, &x_inner, &x_tail);  // 使用 splitWithTail 方法对 for_stmt 进行分裂，分别得到内部循环 x_inner 和尾部循环 x_tail

  auto x_outer_result = IRSimplifier::simplify(for_stmt);  // 简化外部循环结果为 x_outer_result
  std::ostringstream oss_outer;  // 创建一个输出流 oss_outer
  oss_outer << *x_outer_result;  // 将 x_outer_result 输出到 oss_outer 中
  const std::string& expected_outer_ir =
      R"IR(
        # CHECK: {
        # CHECK: }
      )IR";  // 期望的外部 IR 字符串，用于对比检查
  torch::jit::testing::FileCheck().run(expected_outer_ir, oss_outer.str());  // 使用 FileCheck 检查外部 IR 结果

  auto x_tail_result = IRSimplifier::simplify(x_tail);  // 简化尾部循环结果为 x_tail_result
  std::ostringstream oss_tail;  // 创建一个输出流 oss_tail
  oss_tail << *x_tail_result;  // 将 x_tail_result 输出到 oss_tail 中
  const std::string& expected_tail_ir =
      R"IR(
        # CHECK: for (int x_tail = 0; x_tail < 2; x_tail++) {
        # CHECK:   A[x_tail + 13] = 2 * (x_tail + 13);
      )IR";  // 期望的尾部 IR 字符串，用于对比检查
  torch::jit::testing::FileCheck().run(expected_tail_ir, oss_tail.str());  // 使用 FileCheck 检查尾部 IR 结果
}
TEST(LoopNest, FlattenSimpleLoopNest2D) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 5; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  // 定义一个名为 A 的缓冲区，维度为 {10, 5}，元素类型为整型
  BufHandle a_buf("A", {10, 5}, kInt);
  // 定义一个名为 i 的循环变量，类型为整型
  VarHandle i("i", kInt);
  // 定义一个名为 j 的循环变量，类型为整型
  VarHandle j("j", kInt);
  // 创建内层循环体，对缓冲区 a_buf 执行 A[i,j] = i * j 的存储操作
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内层循环，变量 j 在范围 [0, 5) 上迭代
  auto inner_for = For::make(j, 0, 5, for_body);
  // 创建外层循环，变量 i 在范围 [0, 10) 上迭代，循环体为 inner_for
  auto outer_for = For::make(i, 0, 10, inner_for);
  // 创建包含外层循环的父级块
  auto parent_block = Block::make({outer_for});

  // 将外层循环和内层循环放入 loops 向量
  std::vector<ForPtr> loops = {outer_for, inner_for};
  // 初始化一个指向被扁平化后循环的指针
  ForPtr flattened = nullptr;
  // 调用 LoopNest::flatten 函数扁平化循环，结果存储在 flattened 中，并断言扁平化成功
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  // 断言 flattened 指向的是 loops 中的第一个元素
  ASSERT_EQ(flattened, loops.front());

  // 对扁平化后的循环进行简化操作，结果存储在 result 中
  auto result = IRSimplifier::simplify(flattened);
  // 创建一个 ostringstream 对象 oss，用于将 result 的内容写入字符串流
  std::ostringstream oss;
  oss << *result;
  // 定义期望的内部表示字符串 expected_ir
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 50; i_flat++) {
        # CHECK:   A[i_flat / 5, i_flat % 5] =
      )IR";
  // 使用 FileCheck 运行 expected_ir 对 oss.str() 的结果进行检查
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  // 创建 SimpleIREvaluator 对象 eval1，用于评估原始嵌套循环的结果
  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    // 创建 PaddedBuffer<int> 对象 inp1，用于存储评估结果
    PaddedBuffer<int> inp1(10, 5);
    // 对原始嵌套循环进行评估，将结果存储在 inp1 中
    eval1(inp1);
    // 创建 SimpleIREvaluator 对象 eval2，用于评估扁平化后循环的结果
    SimpleIREvaluator eval2(flattened, {a_buf});
    // 创建 PaddedBuffer<int> 对象 inp2，用于存储扁平化后循环的评估结果
    PaddedBuffer<int> inp2(10, 5);
    // 对扁平化后的循环进行评估，将结果存储在 inp2 中
    eval2(inp2);
    // 断言 inp1 和 inp2 的所有元素在给定精度下近似相等
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenSimpleLoopNest3D) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 5; j++) {
  //       for (int k = 0; k < 7; k++) {
  //         A[i,j,k] = i + j * k;
  //       }
  //     }
  //   }
  // 定义一个名为 A 的缓冲区，维度为 {10, 5, 7}，元素类型为整型
  BufHandle a_buf("A", {10, 5, 7}, kInt);
  // 定义一个名为 i 的循环变量，类型为整型
  VarHandle i("i", kInt);
  // 定义一个名为 j 的循环变量，类型为整型
  VarHandle j("j", kInt);
  // 定义一个名为 k 的循环变量，类型为整型
  VarHandle k("k", kInt);
  // 创建内层循环体，对缓冲区 a_buf 执行 A[i,j,k] = i + j * k 的存储操作
  auto for_body = Block::make({Store::make(a_buf, {i, j, k}, i + j * k)});
  // 创建内层循环，变量 k 在范围 [0, 7) 上迭代，循环体为 for_body
  auto for1 = For::make(k, 0, 7, for_body);
  // 创建中间循环，变量 j 在范围 [0, 5) 上迭代，循环体为 for1
  auto for2 = For::make(j, 0, 5, for1);
  // 创建外层循环，变量 i 在范围 [0, 10) 上迭代，循环体为 for2
  auto for3 = For::make(i, 0, 10, for2);
  // 创建包含外层循环的父级块
  auto parent_block = Block::make({for3});

  // 将三个循环放入 loops 向量
  std::vector<ForPtr> loops = {for3, for2, for1};
  // 初始化一个指向被扁平化后循环的指针
  ForPtr flattened = nullptr;
  // 调用 LoopNest::flatten 函数扁平化循环，结果存储在 flattened 中，并断言扁平化成功
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  // 断言 flattened 指向的是 loops 中的第一个元素
  ASSERT_EQ(flattened, loops.front());

  // 对扁平化后的循环进行简化操作，结果存储在 result 中
  auto result = IRSimplifier::simplify(flattened);
  // 创建一个 ostringstream 对象 oss，用于将 result 的内容写入字符串流
  std::ostringstream oss;
  oss << *result;
  // 定义期望的内部表示字符串 expected_ir
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 350; i_flat++) {
        # CHECK:   A[i_flat / 35, (i_flat / 7) % 5, i_flat % 7] =
      )IR";
  // 使用 FileCheck 运行 expected_ir 对 oss.str() 的结果进行检查
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  // 创建 SimpleIREvaluator 对象 eval1，用于评估原始嵌套循环的结果
  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    // 创建 PaddedBuffer<int> 对象 inp1，用于存储评估结果
    PaddedBuffer<int> inp1(10, 5, 7);
    // 对原始嵌套循环进行评估，将结果存储在 inp1 中
    eval1(inp1);
    // 创建 SimpleIREvaluator 对象 eval2，用于评估扁平化后循环的结果
    SimpleIREvaluator eval2(flattened, {a_buf});
    // 创建 PaddedBuffer<int> 对象 inp2，用于存
TEST(LoopNest, FlattenLoopNestAfterNormalize) {
  // Input IR:
  //   for (int i = 2; i < 10; i++) {
  //     for (int j = 3; j < 15; j++) {
  //       A[i - 2,j - 3] = i * j;
  //     }
  //   }
  // 定义一个名为 a_buf 的缓冲区，大小为 8x12，数据类型为整型
  BufHandle a_buf("A", {8, 12}, kInt);
  // 定义一个整型变量 i
  VarHandle i("i", kInt);
  // 定义一个整型变量 j
  VarHandle j("j", kInt);
  // 创建内部循环体，存储操作：将 i*j 存入 a_buf 的位置 (i-2, j-3)
  auto for_body = Block::make({Store::make(a_buf, {i - 2, j - 3}, i * j)});
  // 创建内部循环：变量 j，范围从 3 到 15，执行 for_body
  auto inner_for = For::make(j, 3, 15, for_body);
  // 创建外部循环：变量 i，范围从 2 到 10，执行 inner_for
  auto outer_for = For::make(i, 2, 10, inner_for);
  // 创建包含外部循环的父级块
  auto parent_block = Block::make({outer_for});

  // 创建循环列表，包括外部和内部循环
  std::vector<ForPtr> loops = {outer_for, inner_for};
  // 初始化一个指向被展平后循环的指针
  ForPtr flattened = nullptr;
  // 调用 flatten 函数，将 loops 中的循环展平，并将结果存入 flattened
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  // 断言展平后的循环与原始外部循环相同
  ASSERT_EQ(flattened, loops.front());

  // 简化展平后的 IR
  auto result = IRSimplifier::simplify(flattened);
  // 创建输出流 oss，用于生成字符串形式的 IR
  std::ostringstream oss;
  oss << *result;
  // 预期的 IR 字符串
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 96; i_flat++) {
        # CHECK:   A[i_flat / 12, i_flat % 12] =
      )IR";
  // 使用 FileCheck 检查生成的 IR 是否符合预期
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  {
    // 创建简单 IR 评估器 eval1，用于执行 loops[0] 的 IR
    SimpleIREvaluator eval1(loops[0], {a_buf});
    // 创建大小为 8x12 的填充整型缓冲区 inp1
    PaddedBuffer<int> inp1(8, 12);
    // 评估 loops[0] 的 IR，结果存入 inp1
    eval1(inp1);
    // 创建简单 IR 评估器 eval2，用于执行展平后的 IR
    SimpleIREvaluator eval2(flattened, {a_buf});
    // 创建大小为 8x12 的填充整型缓冲区 inp2
    PaddedBuffer<int> inp2(8, 12);
    // 评估展平后的 IR，结果存入 inp2
    eval2(inp2);
    // 检查 inp1 和 inp2 是否在指定精度内相等
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenLoopNestWithNonLiteralConstantBounds) {
  // Input IR:
  //   for (int i = 0; i < 15-5; i++) {
  //     for (int j = 0; j < 20/4; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  // 定义一个名为 a_buf 的缓冲区，大小为 10x5，数据类型为整型
  BufHandle a_buf("A", {10, 5}, kInt);
  // 定义一个整型变量 i
  VarHandle i("i", kInt);
  // 定义一个整型变量 j
  VarHandle j("j", kInt);
  // 创建内部循环体，存储操作：将 i*j 存入 a_buf 的位置 (i, j)
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环：变量 j，范围从 0 到 20/4，执行 for_body
  auto inner_for =
      For::make(j, 0, IntImm::make(20) / IntImm::make(4), for_body);
  // 创建外部循环：变量 i，范围从 0 到 15-5，执行 inner_for
  auto outer_for =
      For::make(i, 0, IntImm::make(15) - IntImm::make(5), inner_for);
  // 创建包含外部循环的块 b
  auto b = Block::make({outer_for});

  // 创建循环列表，包括外部和内部循环
  std::vector<ForPtr> loops = {outer_for, inner_for};
  // 初始化一个指向被展平后循环的指针
  ForPtr flattened = nullptr;
  // 调用 flatten 函数，将 loops 中的循环展平，并将结果存入 flattened
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  // 断言展平后的循环与原始外部循环相同
  ASSERT_EQ(flattened, loops.front());

  // 简化展平后的 IR
  auto result = IRSimplifier::simplify(flattened);
  // 检查生成的 IR 是否符合预期
  checkIR(result, R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 50; i_flat++) {
        # CHECK:   A[i_flat / 5, i_flat % 5] =
      )IR");

  {
    // 创建简单 IR 评估器 eval1，用于执行 loops[0] 的 IR
    SimpleIREvaluator eval1(loops[0], {a_buf});
    // 创建大小为 10x5 的填充整型缓冲区 inp1
    PaddedBuffer<int> inp1(10, 5);
    // 评估 loops[0] 的 IR，结果存入 inp1
    eval1(inp1);
    // 创建简单 IR 评估器 eval2，用于执行展平后的 IR
    SimpleIREvaluator eval2(flattened, {a_buf});
    // 创建大小为 10x5 的填充整型缓冲区 inp2
    PaddedBuffer<int> inp2(10, 5);
    // 评估展平后的 IR，结果存入 inp2
    eval2(inp2);
    // 检查 inp1 和 inp2 是否在指定精度内相等
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}
TEST(LoopNest, FlattenImperfectLoopNest) {
  // 定义一个测试用例，测试不完美的嵌套循环结构的扁平化
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     A[i, i] = 0;
  //     for (int j = 0; j < 15; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  // 不进行扁平化操作

  // 创建一个名为 A 的缓存对象，维度为 {10, 15}，数据类型为整型
  BufHandle a_buf("A", {10, 15}, kInt);
  // 定义变量 i 和 j，数据类型为整型
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建内部循环体，将 i * j 存储到缓存 a_buf 的相应位置
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环，遍历 j 从 0 到 14
  auto inner_for = For::make(j, 0, 15, for_body);
  // 创建外部循环，遍历 i 从 0 到 9，内部嵌套内部循环和初始化 A[i, i] = 0
  auto outer_for = For::make(
      i, 0, 10, Block::make({Store::make(a_buf, {i, i}, 0), inner_for}));
  // 创建包含外部循环的代码块 par
  auto par = Block::make({outer_for});
  // 创建哈希提供器对象 hasher
  HashProvider hasher;
  // 计算扁平化前的哈希值
  auto hash_before = hasher.hash(par);

  // 准备进行扁平化的循环列表，包括外部循环和内部循环
  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  // 调用 LoopNest 的 flatten 方法尝试扁平化循环
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  // 断言扁平化后的结果为 nullptr
  ASSERT_EQ(flattened, nullptr);
  // 计算扁平化后的哈希值
  auto hash_after = hasher.hash(par);
  // 断言扁平化前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, FlattenReductionLoopNest) {
  // 定义一个测试用例，测试带有归约操作的嵌套循环结构的扁平化
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     S[i] = 0;
  //     for (int j = 0; j < 15; j++) {
  //       S[i] = S[i] + A[i,j];
  //     }
  //   }
  // 不进行扁平化操作

  // 创建名为 A 和 S 的缓存对象，维度分别为 {10, 15} 和 {10}，数据类型为整型
  BufHandle a_buf("A", {10, 15}, kInt);
  BufHandle s_buf("S", {10}, kInt);
  // 定义变量 i 和 j，数据类型为整型
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建内部循环体，实现 S[i] = S[i] + A[i,j]
  auto for_body = Block::make({Store::make(
      s_buf, {i}, Load::make(s_buf, {i}) + Load::make(a_buf, {i, j}))});
  // 创建内部循环，遍历 j 从 0 到 14
  auto inner_for = For::make(j, 0, 15, for_body);
  // 创建外部循环，遍历 i 从 0 到 9，内部嵌套内部循环和初始化 S[i] = 0
  auto outer_for =
      For::make(i, 0, 10, Block::make({Store::make(s_buf, {i}, 0), inner_for}));
  // 创建包含外部循环的代码块 par
  auto par = Block::make({outer_for});
  // 创建哈希提供器对象 hasher
  HashProvider hasher;
  // 计算扁平化前的哈希值
  auto hash_before = hasher.hash(par);

  // 准备进行扁平化的循环列表，包括外部循环和内部循环
  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  // 调用 LoopNest 的 flatten 方法尝试扁平化循环
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  // 断言扁平化后的结果为 nullptr
  ASSERT_EQ(flattened, nullptr);
  // 计算扁平化后的哈希值
  auto hash_after = hasher.hash(par);
  // 断言扁平化前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, FlattenReductionLoopNestFromTensor) {
  // 定义一个测试用例，测试从张量生成的带有归约操作的嵌套循环结构的扁平化
  const int M = 3;
  const int N = 7;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  // 创建名为 b 的缓存对象，维度为 {m, n}，数据类型为浮点型
  BufHandle b("b", {m, n}, kFloat);
  // 创建张量对象 c，通过归约操作从 b 生成，维度为 {M}，操作为求和
  Tensor c = Reduce("sum", {M}, Sum(), b, {N});
  // 创建 LoopNest 对象 loop，传入张量 c
  LoopNest loop({c});
  // 创建哈希提供器对象 hasher
  HashProvider hasher;
  // 计算扁平化前的哈希值
  auto hash_before = hasher.hash(loop.root_stmt());

  // 获取写入到缓存 c.buf() 的所有循环嵌套
  auto loops = loop.getAllLoopNestsWritingToBuf(c.buf())[1];
  ForPtr flattened = nullptr;
  // 调用 LoopNest 的 flatten 方法尝试扁平化循环
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  // 断言扁平化后的结果为 nullptr
  ASSERT_EQ(flattened, nullptr);
  // 计算扁平化后的哈希值
  auto hash_after = hasher.hash(loop.root_stmt());
  // 断言扁平化前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}
TEST(LoopNest, FlattenIncorrectLoopsAsInput) {
  // 定义一个测试用例，验证循环嵌套的展开处理是否正确

  // 定义数组 A 的缓冲区，形状为 10x5 的整数数组
  BufHandle a_buf("A", {10, 5}, kInt);
  
  // 定义循环变量 i 和 j，均为整数类型
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  
  // 定义第一个内部循环的执行体，将 i * j 存入数组 A 的缓冲区
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 j，范围为 0 到 5，执行体为 for_body1
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  // 创建外部循环 i，范围为 0 到 10，执行体为 inner_for1
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  
  // 定义第二个内部循环的执行体，将 A[x,y] 加上 x 和 y 存入数组 A 的缓冲区
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  // 创建内部循环 y，范围为 0 到 5，执行体为 for_body2
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  // 创建外部循环 x，范围为 0 到 10，执行体为 inner_for2
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  
  // 创建并行块 par，包含外部循环 outer_for1 和 outer_for2
  auto par = Block::make({outer_for1, outer_for2});
  
  // 创建 HashProvider 对象 hasher，计算并记录块 par 的哈希值
  HashProvider hasher;
  auto hash_before = hasher.hash(par);
  
  // 将外部循环 outer_for1 和内部循环 inner_for2 放入 loops 向量
  std::vector<ForPtr> loops = {outer_for1, inner_for2};
  // 初始化指针 flattened
  ForPtr flattened = nullptr;
  // 调用 LoopNest::flatten 函数尝试展开循环，断言展开失败
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  // 断言展开后的 flattened 为空指针
  ASSERT_EQ(flattened, nullptr);
  
  // 再次计算并记录块 par 的哈希值
  auto hash_after = hasher.hash(par);
  // 断言展开前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, DetectInlineRankMismatch) {
  // 定义一个测试用例，验证内联计算时的秩不匹配检测

  // 定义常量 kTotalSize 为 8
  const int kTotalSize = 8;
  
  // 定义数组 A 的缓冲区，形状为 [kTotalSize] 的浮点数数组
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  
  // 定义张量 a，用于加载数组 A 中的元素
  Tensor a = Compute(
      "a", {kTotalSize}, [&](const VarHandle& i) { return a_buf.load(i); });
  
  // 定义张量 reshape，形状为 [kTotalSize / 2, 2]
  Tensor reshape = Compute(
      "reshape",
      {kTotalSize / 2, 2},
      [&](const VarHandle& i, const VarHandle& j) { return a.load(i, j); });
  
  // 创建 LoopNest 对象 l，传入张量 reshape 和 a 作为参数
  LoopNest l({reshape}, {a, reshape});
  // 获取张量 a 的循环体并尝试进行内联计算，断言内联不成功
  ASSERT_FALSE(l.computeInline(l.getLoopBodyFor(a)));
}

TEST(LoopNest, CacheReadsSimple) {
  // 定义一个测试用例，验证简单的缓存读取操作

  // 创建张量 A，形状为 [64, 64]，初始化为 i * j
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  
  // 创建张量 B，形状为 [20, 10]，定义为从数组 A 中读取偏移后的值
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 3);
      });
  
  // 创建张量 C，形状为 [20, 10]，定义为两个 A 元素的和
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });
  
  // 创建 LoopNest 对象 l，传入张量 B 和 C，以及 A 作为参数
  LoopNest l({B, C}, {A, B, C});
  
  // 获取写入张量 B 的所有循环嵌套中的第二个内部循环 j_loop
  StmtPtr j_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][1];
  
  // 在循环 j_loop 中缓存对数组 A 的访问，存入 A_local
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);
  
  // 准备进行代码生成前的准备工作
  l.prepareForCodegen();
  
  // 简化并重命名语句树，得到简化后的结果
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  
  // 使用 SimpleIREvaluator 对象 cg 对简化后的结果进行代码生成
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();
  
  // 验证生成的代码是否符合预期
  checkIR(result, R"IR(
#CHECK: Allocate(A); // dtype=int, dims=[64, 64]
#CHECK: Allocate(A_local); // dtype=int, dims=[1, 10]
#CHECK: for (int i
#CHECK:  for (int j
#CHECK:   A[
#CHECK:  }
#CHECK: }
#CHECK: for (int i_1
#CHECK:  for (int j_1
#CHECK:   A_local[j_1] = A[
#CHECK:  }
#CHECK:  for (int j_2
#CHECK:   B[j_2 + 10 * i_1] = A_local[j_2];
#CHECK:  }
#CHECK: }
#CHECK: for (int i_2
#CHECK:  for (int j_3
#CHECK:   C[
#CHECK:  }
#CHECK: }
)IR");
}
  // 创建一个大小为200的整数向量，所有元素初始化为0，用作数据存储
  std::vector<int> b_data(200, 0);
  // 创建一个大小为200的整数向量，所有元素初始化为0，用作数据存储
  std::vector<int> c_data(200, 0);
  // 调用计算图cg，并传入b_data和c_data作为参数
  cg.call({b_data, c_data});

  // 创建一个大小为200的整数向量，用作预期的参考数据
  std::vector<int> b_ref(200, 0);
  // 创建一个大小为200的整数向量，用作预期的参考数据
  std::vector<int> c_ref(200, 0);

  // 循环填充预期的参考数据b_ref和c_ref
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      // 计算预期的B数据元素并存储到b_ref中
      b_ref[i * 10 + j] = (i + 30) * (j + 40) + (i + 31) * (j + 41);
      // 计算预期的C数据元素并存储到c_ref中
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  // 断言b_data与b_ref相等
  assertAllEqual(b_data, b_ref);
  // 断言c_data与c_ref相等
  assertAllEqual(c_data, c_ref);
}



TEST(LoopNest, CacheReadsOuter) {
  // 创建一个64x64的Tensor A，通过lambda函数初始化为i*j
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  // 创建一个20x10的Tensor B，通过lambda函数使用A的load函数计算元素值
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 40) + A.load(i + 31, j + 41);
      });
  // 创建一个20x10的Tensor C，通过lambda函数使用A的load函数计算元素值
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  // 创建LoopNest对象l，传入B、C作为写入缓冲区的张量，A、B、C作为所有张量
  LoopNest l({B, C}, {A, B, C});
  // 获取写入B缓冲区的所有循环嵌套，并选择第一个循环嵌套的第一个循环
  StmtPtr i_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][0];
  // 在指定的循环嵌套中缓存对A的访问，命名为"A_local"
  LoopNest::cacheAccesses(A.buf(), "A_local", i_loop);

  // 为代码生成做准备
  l.prepareForCodegen();
  // 简化IR并重新命名变量名
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  // 创建SimpleIREvaluator对象cg，传入简化后的IR和B、C张量作为参数
  SimpleIREvaluator cg(result, {B, C});
  // 获取最终的IR语句
  result = cg.stmt();

  // 检查IR结果是否与预期的匹配
  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[21, 11]
#CHECK: A_local[j_1 + 11 * i_1] =
#CHECK: B[j_2 + 10 * i_2] = (A_local[j_2 + 11 * i_2]) + (A_local[(j_2 + 11 * i_2) + 12]);
      )IR");

  // 创建一个大小为200的整数向量，所有元素初始化为0，用作数据存储
  std::vector<int> b_data(200, 0);
  // 创建一个大小为200的整数向量，所有元素初始化为0，用作数据存储
  std::vector<int> c_data(200, 0);
  // 调用计算图cg，并传入b_data和c_data作为参数
  cg.call({b_data, c_data});

  // 创建一个大小为200的整数向量，用作预期的参考数据
  std::vector<int> b_ref(200, 0);
  // 创建一个大小为200的整数向量，用作预期的参考数据
  std::vector<int> c_ref(200, 0);

  // 循环填充预期的参考数据b_ref和c_ref
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      // 计算预期的B数据元素并存储到b_ref中
      b_ref[i * 10 + j] = (i + 30) * (j + 40) + (i + 31) * (j + 41);
      // 计算预期的C数据元素并存储到c_ref中
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  // 断言b_data与b_ref相等
  assertAllEqual(b_data, b_ref);
  // 断言c_data与c_ref相等
  assertAllEqual(c_data, c_ref);
}



TEST(LoopNest, CacheReadsInternal) {
  // 创建一个64x64的Tensor A，通过lambda函数初始化为i*j
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  // 创建一个20x10的Tensor B，通过lambda函数使用A的load函数计算元素值
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 40) + A.load(i + 31, j + 41);
      });
  // 创建一个20x10的Tensor C，通过lambda函数使用A的load函数计算元素值
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  // 创建LoopNest对象l，传入B、C作为写入缓冲区的张量，A、B、C作为所有张量
  LoopNest l({B, C}, {A, B, C});
  // 获取写入B缓冲区的所有循环嵌套，并选择第一个循环嵌套的第二个循环
  StmtPtr j_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][1];
  // 在指定的循环嵌套中缓存对A的访问，命名为"A_local"
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);

  // 为代码生成做准备
  l.prepareForCodegen();
  // 简化IR并重新命名变量名
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  // 创建SimpleIREvaluator对象cg，传入简化后的IR和B、C张量作为参数
  SimpleIREvaluator cg(result, {B, C});
  // 获取最终的IR语句
  result = cg.stmt();

  // 检查IR结果是否与预期的匹配
  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[2, 11]
#CHECK: A_local[k + 11 * j_1] =
      )IR");
}
#CHECK: Allocate(A_local); // 分配 A_local 数组，数据类型为整数，维度为[1, 64]
#CHECK: for (int j = 0; j < 64
#CHECK:   A_local[j] = i * j;
#CHECK: for (int j_1 = 0; j_1 < 64
TEST(LoopNest, DeadStoreElimination) {
  VarHandle y("y", kInt);  // 定义名为 y 的整数变量
  VarHandle x("x_tail", kInt);  // 定义名为 x_tail 的整数变量
  BufHandle f("f", {26, 5}, kInt);  // 定义名为 f 的整数缓冲区，大小为 26x5
  BufHandle g("g", {26, 5}, kInt);  // 定义名为 g 的整数缓冲区，大小为 26x5
  ExprHandle x_outer_end = 5;  // 定义名为 x_outer_end 的表达式，赋值为 5
  ExprHandle x_2 = x + x_outer_end * 4;  // 定义名为 x_2 的表达式，计算 x + x_outer_end * 4
  ForPtr stmt1 = For::make(
      x,
      0,
      5,
      For::make(
          y,
          0,
          5,
          Block::make({
              Store::make(f, {x_2, y}, (x_2 + y)),  // 在 f 中存储值 (x_2 + y) 到位置 (x_2, y)
              Store::make(g, {x_2, y}, (x_2 * y)),  // 在 g 中存储值 (x_2 * y) 到位置 (x_2, y)
          })));
  StmtPtr stmt = Block::make({stmt1});

  // 如果没有被输出使用，则会消除该存储操作
  LoopNest loop(Stmt::clone(stmt), {f.node()});
  loop.eliminateDeadStores();

  checkIR(loop.root_stmt(), R"IR(
#CHECK:     f[x_tail + 5 * 4, y] = x_tail + 5 * 4 + y;
#CHECK-NOT: g[x_tail + 5 * 4, y]
      )IR");

  // 如果被不同的输出使用，则不会消除该存储操作
  LoopNest loop2(stmt, {f.node(), g.node()});
  loop2.eliminateDeadStores();

  checkIR(loop2.root_stmt(), R"IR(
#CHECK:     f[x_tail + 5 * 4, y] = x_tail + 5 * 4 + y;
#CHECK:     g[x_tail + 5 * 4, y] = x_tail + 5 * 4 * y;
      )IR");
}

TEST(LoopNest, DeadStoreEliminationWithIntermediates) {
  VarHandle x("x", kInt);  // 定义名为 x 的整数变量
  VarHandle y("y", kInt);  // 定义名为 y 的整数变量
  VarHandle z("z", kInt);  // 定义名为 z 的整数变量
  BufHandle f("f", {26 * 5}, kInt);  // 定义名为 f 的整数缓冲区，大小为 26 * 5
  BufHandle g("g", {26 * 5}, kInt);  // 定义名为 g 的整数缓冲区，大小为 26 * 5
  BufHandle h("h", {26, 5}, kInt);  // 定义名为 h 的整数缓冲区，大小为 26x5
  ExprHandle x_outer_end = 5;  // 定义名为 x_outer_end 的表达式，赋值为 5
  ExprHandle x_2 = x + x_outer_end * 4;  // 定义名为 x_2 的表达式，计算 x + x_outer_end * 4
  ForPtr stmt1 = For::make(x, 0, 26 * 5, Store::make(f, {x}, x));  // 循环写入 f 中每个位置 x 的值为 x
  ForPtr stmt2 = For::make(z, 0, 26 * 5, Store::make(g, {z}, z + 1));  // 循环写入 g 中每个位置 z 的值为 z + 1
  ForPtr stmt3 = For::make(
      x,
      0,
      5,
      For::make(
          y,
          0,
          5,
          Block::make({
              Store::make(h, {x, y}, Load::make(f, {x * y})),  // 在 h 中存储 f[x * y] 的加载值到位置 (x, y)
          })));
  StmtPtr stmt = Block::make({stmt1, stmt2, stmt3});

  // 如果没有被 h 的生产者使用，则会消除对 g 的写入，但不会消除对 f 的写入
  LoopNest loop(Stmt::clone(stmt), {h.node()});
  loop.eliminateDeadStores();

  checkIR(loop.root_stmt(), R"IR(
  #CHECK:     f[x] = x;
  #CHECK-NOT: g[z] =
  #CHECK:     h[x, y] = f[x * y];
      )IR");

  // 检查不会消除对 g 的写入，因为 g 是输出
  LoopNest loop2(stmt, {h.node(), g.node()});
  loop2.eliminateDeadStores();

  checkIR(loop2.root_stmt(), R"IR(
  #CHECK:     f[x] = x;
  #CHECK:     g[z] = z + 1;
  #CHECK:     h[x, y] = f[x * y];
      )IR");
}
TEST(LoopNest, CompoundTensorSimple) {
  // 创建一个名为 "A" 的缓冲区，大小为 {10, 5}，数据类型为整型
  BufHandle a_buf("A", {10, 5}, kInt);
  // 创建整型变量 i, j, x, y
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建第一个循环体，存储 a_buf[i, j] = i * j
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 j: 0 到 4
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  // 创建外部循环 i: 0 到 9，包含内部循环
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  // 创建第二个循环体，存储 a_buf[x, y] = a_buf[x, y] + x + y
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  // 创建内部循环 y: 0 到 4
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  // 创建外部循环 x: 0 到 9，包含内部循环
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  // 创建整体块，包含两个外部循环
  BlockPtr body = Block::make({outer_for1, outer_for2});

  // 创建 Tensor 对象 A，关联缓冲区 a_buf 和整体块 body
  Tensor A = Tensor(a_buf.node(), body);

  // 创建 LoopNest 对象 l，传入 Tensor 对象列表
  LoopNest l({A});
  // 为代码生成做准备
  l.prepareForCodegen();

  // 初始化大小为 50 的整型数组 a_data，元素值为 0
  std::vector<int> a_data(50, 0);

  // 对根语句进行简化
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建 SimpleIREvaluator 对象 cg，传入简化后的根语句和 Tensor 对象列表
  SimpleIREvaluator cg(s, {A});

  // 初始化大小为 50 的整型数组 a_ref，用于存储预期结果
  std::vector<int> a_ref(50, 0);

  // 使用嵌套循环计算预期结果并存入 a_ref 中
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      a_ref[i * 5 + j] = (i * j) + i + j;
    }
  }

  // 调用 SimpleIREvaluator 对象 cg 的 call 方法，计算结果存入 a_data
  cg.call({a_data});

  // 断言 a_data 和 a_ref 中的所有元素相等
  assertAllEqual(a_data, a_ref);
}

TEST(LoopNest, InlineConstantIndex) {
  // 定义常量 N = 10
  const int N = 10;
  // 创建名为 "a" 的缓冲区，大小为 {1, N, 1}，数据类型为浮点型
  BufHandle x_buf("a", {1, N, 1}, kFloat);
  // 创建 Tensor 对象 y，使用 Compute 函数定义
  Tensor y = Compute(
      "f",
      {1, N, 1},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return x_buf.load(m, n, o);
      });
  // 创建 Tensor 对象 z，使用 Compute 函数定义，依赖于 Tensor 对象 y
  Tensor z = Compute(
      "f",
      {1, N, 1},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return y.load(m, n, o);
      });

  // 创建 LoopNest 对象 l，传入 Tensor 对象列表
  LoopNest l({z}, {y, z});
  // 简化循环嵌套
  l.simplify();
  // 断言成功将 y 内联
  ASSERT_TRUE(l.computeInline(y.buf()));
}

TEST(LoopNest, CompoundTensorUsed) {
  // 创建名为 "A" 的缓冲区，大小为 {10, 5}，数据类型为整型
  BufHandle a_buf("A", {10, 5}, kInt);
  // 创建整型变量 i, j, x, y
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建第一个循环体，存储 a_buf[i, j] = i * j
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  // 创建内部循环 j: 0 到 4
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  // 创建外部循环 i: 0 到 9，包含内部循环
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  // 创建第二个循环体，存储 a_buf[x, y] = a_buf[x, y] + x + y
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  // 创建内部循环 y: 0 到 4
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  // 创建外部循环 x: 0 到 9，包含内部循环
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  // 创建整体块，包含两个外部循环
  BlockPtr body = Block::make({outer_for1, outer_for2});

  // 创建 Tensor 对象 A，关联缓冲区 a_buf 和整体块 body
  Tensor A = Tensor(a_buf.node(), body);
  // 创建 Tensor 对象 B，使用 Compute 函数定义，依赖于 Tensor 对象 A
  Tensor B = Compute("B", {10, 3}, [&](const VarHandle& i, const VarHandle& j) {
    return A.load(i, j + 1) + A.load(i, j + 2);
  });

  // 创建 LoopNest 对象 l，传入 Tensor 对象列表
  LoopNest l({B}, {A, B});
  // 断言无法将 A 内联
  ASSERT_FALSE(l.computeInline(A.buf()));
  // 为代码生成做准备
  l.prepareForCodegen();

  // 初始化大小为 50 的整型数组 a_data 和 b_data，元素值为 0
  std::vector<int> a_data(50, 0);
  std::vector<int> b_data(50, 0);

  // 对根语句进行简化
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  // 创建 SimpleIREvaluator 对象 cg，传入简化后的根语句和 Tensor 对象列表
  SimpleIREvaluator cg(s, {B});

  // 初始化大小为 50 的整型数组 b_ref，用于存储预期结果
  std::vector<int> b_ref(50, 0);

  // 定义计算函数 AT(i, j) = i * j + i + j
  auto AT = [](int i, int j) { return i * j + i + j; };
  // 使用嵌套循环计算预期结果并存入 b_ref 中
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 3; ++j) {
      b_ref[i * 3 + j] = AT(i, j + 1) + AT(i, j + 2);
    }
  }

  // 调用 SimpleIREvaluator 对象 cg 的 call 方法，计算结果存入 b_data
  cg.call({b_data});

  // 断言 b_data 和 b_ref 中的所有
TEST(LoopNest, InlineFromLoad) {
  // 定义常量 N 为 1024
  constexpr int N = 1024;
  // 创建名为 a 的缓冲区，大小为 {N}，数据类型为 kInt
  BufHandle a("A", {N}, kInt);
  // 创建名为 b 的缓冲区，大小为 {N}，数据类型为 kInt
  BufHandle b("B", {N}, kInt);
  // 创建名为 i 的变量，数据类型为 kInt
  VarHandle i("i", kInt);
  // 创建名为 j 的变量，数据类型为 kInt
  VarHandle j("j", kInt);
  // 创建存储语句 store_a: for 循环，将 i 从 0 到 N-1，将 i 存储到缓冲区 a 中
  auto store_a = For::make(i, 0, N, Store::make(a, {i}, i));
  // 创建存储语句 store_b: for 循环，将 j 从 0 到 N-1，将 a[j] 加载并存储到缓冲区 b 中
  auto store_b = For::make(j, 0, N, Store::make(b, {j}, Load::make(a, {j})));
  // 创建循环嵌套 l，包含 store_a 和 store_b 语句块，并依赖于 b 缓冲区的节点
  LoopNest l(Block::make({store_a, store_b}), {b.node()});

  // 对缓冲区 a 进行内联计算
  l.computeInline(a.node());

  // 检查内联后是否将 A[j] 替换为 j
  std::ostringstream oss;
  oss << *l.root_stmt();
  // 使用 FileCheck 进行验证，检查输出 IR 是否符合指定模式
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: for (int j
# CHECK-NOT: B[j] = A[j]
# CHECK-NEXT: B[j] = j
)IR",
      oss.str());
}

TEST(LoopNest, OptimizeConditionalsSimple) {
  // 输入的 IR 为：
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }

  // 创建名为 a_buf 的缓冲区，大小为 {20}，数据类型为 kInt
  BufHandle a_buf("A", {20}, kInt);
  // 创建名为 b_buf 的缓冲区，大小为 {5}，数据类型为 kInt
  BufHandle b_buf("B", {5}, kInt);
  // 创建名为 c_buf 的缓冲区，大小为 {15}，数据类型为 kInt
  BufHandle c_buf("C", {15}, kInt);
  // 创建名为 i 的变量，数据类型为 kInt
  VarHandle i("i", kInt);
  // 创建存储语句 store，根据条件选择加载 b_buf[i] 或 c_buf[i-5] 存储到 a_buf[i] 中
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  // 创建 for 循环语句 forI，将 store 语句放入循环体中，i 从 0 到 19
  auto forI = For::make(i, 0, 20, store);
  // 创建并行语句 par，包含 for 循环 forI
  auto par = Block::make({forI});

  // 创建循环嵌套 nest，包含 par 语句块，并依赖于 a_buf 缓冲区的节点
  LoopNest nest(par, {a_buf.node()});
  // 优化条件语句的计算
  nest.optimizeConditionals();

  std::ostringstream oss;
  oss << *nest.root_stmt();
  // 定义验证模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 15
# CHECK-NEXT: A[i + 5] = C[i]
      )IR";
  // 使用 FileCheck 进行验证，检查输出 IR 是否符合指定模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// 定义测试用例 `LoopNest` 中的 `OptimizeConditionalsNestedConditions` 函数
TEST(LoopNest, OptimizeConditionalsNestedConditions) {
  // 输入的中间表示(IR)如下：
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }

  // 定义缓冲区 `A`，大小为 20，数据类型为整型
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // 定义缓冲区 `B`，大小为 5，数据类型为整型
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // 定义缓冲区 `C`，大小为 5，数据类型为整型
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // 定义缓冲区 `D`，大小为 10，数据类型为整型
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  // 定义变量 `i`，数据类型为整型
  VarHandle i("i", kInt);
  // 创建存储操作，存储结果到缓冲区 `A`
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      // 根据条件选择语句，嵌套使用 IfThenElse 表达式
      IfThenElse::make(
          // 判断 `i < 10` 条件
          CompareSelect::make(i, 10, kLT),
          // 如果条件成立，进一步判断 `i < 5`
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),
              // 如果条件成立，加载缓冲区 `B` 中的数据
              Load::make(b_buf, {i}),
              // 否则加载缓冲区 `C` 中偏移后的数据
              Load::make(c_buf, {i - 5})),
          // 如果第一个条件不成立，加载缓冲区 `D` 中偏移后的数据
          Load::make(d_buf, {i - 10})));
  // 创建循环操作 `forI`，遍历 `i` 从 0 到 19 的范围
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  // 创建代码块 `par`，包含 `forI` 循环操作
  auto par = Block::make({forI});

  // 创建 `LoopNest` 对象 `nest`，以及其关联的缓冲区 `A`
  LoopNest nest(par, {a_buf.node()});
  // 对循环嵌套进行条件优化
  nest.optimizeConditionals();

  // 创建字符串流 `oss`，用于存储优化后的循环嵌套表达式
  std::ostringstream oss;
  oss << *nest.root_stmt();
  // 定义字符串 `verification_pattern`，用于验证优化后的表达式格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK: for (int i = 0; i < 10
# CHECK-NEXT: A[i + 10] = D[i]
      )IR";
  // 使用 FileCheck 工具验证优化后的表达式格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// 定义测试用例 `LoopNest.OptimizeConditionalsMultipleStores`，测试循环嵌套和条件语句优化
TEST(LoopNest, OptimizeConditionalsMultipleStores) {
  // 输入的中间表示(IR)代码注释:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     B[j] = IfThenElse(j<30 ? 1 : 0, C[j], D[j])
  //   }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建缓冲区 `A`，长度为 20，数据类型为整数
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建缓冲区 `B`，长度为 5，数据类型为整数
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建缓冲区 `C`，长度为 100，数据类型为整数
  BufHandle c_buf("C", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建缓冲区 `D`，长度为 100，数据类型为整数
  BufHandle d_buf("D", {100}, kInt);
  // 创建整型变量 `i`
  VarHandle i("i", kInt);
  // 创建整型变量 `j`
  VarHandle j("j", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建存储操作 `storeA`，存储操作结果到 `A` 中
  auto storeA = Store::make(
      a_buf,
      {i},
      // 创建条件语句，根据 `i` 的值选择从 `B` 或 `C` 中加载数据到 `A` 中
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),  // 如果 `i < 5` 则为真
          Load::make(b_buf, {i}),          // 加载 `B[i]`
          Load::make(c_buf, {i - 5})));    // 加载 `C[i-5]`
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建 `for` 循环，循环 `i` 从 0 到 19，并执行 `storeA` 操作
  auto forI = For::make(i, 0, 20, storeA);
  // 创建存储操作 `storeB`，存储操作结果到 `B` 中
  auto storeB = Store::make(
      b_buf,
      {j},
      // 创建条件语句，根据 `j` 的值选择从 `C` 或 `D` 中加载数据到 `B` 中
      IfThenElse::make(
          CompareSelect::make(j, 30, kLT),  // 如果 `j < 30` 则为真
          Load::make(c_buf, {j}),           // 加载 `C[j]`
          Load::make(d_buf, {j})));         // 加载 `D[j]`
  // 创建 `for` 循环，循环 `j` 从 0 到 99，并执行 `storeB` 操作
  auto forJ = For::make(j, 0, 100, storeB);
  // 创建并行块，包含 `forI` 和 `forJ` 循环
  auto par = Block::make({forI, forJ});

  // 创建循环嵌套 `nest`，以 `a_buf` 作为根语句
  LoopNest nest(par, {a_buf.node()});
  // 对循环嵌套进行条件语句优化
  nest.optimizeConditionals();

  // 创建输出字符串流 `oss`
  std::ostringstream oss;
  // 将优化后的根语句 `nest.root_stmt()` 输出到 `oss`
  oss << *nest.root_stmt();
  // 期望的验证模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 15
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK: for (int j = 0; j < 30
# CHECK-NEXT: B[j] = C[j]
# CHECK: for (int j = 0; j < 70
# CHECK-NEXT: B[j + 30] = D[j + 30]
      )IR";
  // 运行 FileCheck 进行验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// 定义一个名为 LoopNest 的测试用例，用于优化多个存储操作中的条件语句
TEST(LoopNest, OptimizeConditionalsMultipleStoresInOneLoop) {
  // 输入的中间表示(IR)：
  //   for (int i = 0; i < 50; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //     B[j] = IfThenElse(j<30 ? 1 : 0, C[j], D[j])
  //   }
  // 只有在写入 A 的第一个条件语句会被优化。

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建一个名为 a_buf 的缓冲区对象，大小为 100，类型为整型
  BufHandle a_buf("A", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建一个名为 b_buf 的缓冲区对象，大小为 100，类型为整型
  BufHandle b_buf("B", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建一个名为 c_buf 的缓冲区对象，大小为 100，类型为整型
  BufHandle c_buf("C", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建一个名为 d_buf 的缓冲区对象，大小为 100，类型为整型
  BufHandle d_buf("D", {100}, kInt);
  // 创建一个名为 i 的整型变量
  VarHandle i("i", kInt);
  // 创建存储 A 的操作，根据条件选择写入 B[i] 或 C[i-5]
  auto storeA = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  // 创建存储 B 的操作，根据条件选择写入 C[i] 或 D[i]
  auto storeB = Store::make(
      b_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 30, kLT),
          Load::make(c_buf, {i}),
          Load::make(d_buf, {i})));
  // 创建循环语句 forI，对 i 从 0 到 50 执行 storeA 和 storeB 操作
  auto forI = For::make(i, 0, 50, Block::make({storeA, storeB}));
  // 创建一个包含 forI 的块 par
  auto par = Block::make({forI});

  // 创建 LoopNest 对象 nest，对 a_buf 节点进行嵌套循环优化
  LoopNest nest(par, {a_buf.node()});
  // 执行条件语句优化
  nest.optimizeConditionals();

  // 创建字符串流 oss，将 nest 的根语句输出到 oss 中
  std::ostringstream oss;
  oss << *nest.root_stmt();
  // 用于验证的字符串模式，检查输出结果是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK-NEXT: B[i] = C[i]
# CHECK: for (int i = 0; i < 45
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK-NEXT: B[i + 5] = IfThenElse(i + 5<30 ? 1 : 0, C[i + 5], D[i + 5])
      )IR";
  // 使用 FileCheck 运行验证模式，检查 oss 的输出是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(LoopNest, OptimizeConditionalsOuterLoopVar) {
  // 输入的IR：
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = IfThenElse(i<10, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //     }
  //   }
  // 当前情况下，条件变量 `i` 不是最内层循环变量，因此没有被优化。

  // 创建缓冲区对象 a_buf，表示数组 A，大小为 {20}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // 创建缓冲区对象 b_buf，表示数组 B，大小为 {5}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // 创建缓冲区对象 c_buf，表示数组 C，大小为 {5}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // 创建缓冲区对象 d_buf，表示数组 D，大小为 {10}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  // 创建整型变量对象 i，表示循环变量 i
  VarHandle i("i", kInt);
  // 创建整型变量对象 j，表示循环变量 j
  VarHandle j("j", kInt);
  // 创建存储操作 store 对象，用于将条件表达式结果存储到数组 A 中
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),  // 比较 i 是否小于 10
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),  // 比较 i 是否小于 5
              Load::make(b_buf, {i}),  // 若满足条件则加载数组 B 的值
              Load::make(c_buf, {i - 5})),  // 否则加载数组 C 的值
          Load::make(d_buf, {i - 10})));  // 如果 i 不小于 10，则加载数组 D 的值
  // 创建循环对象 forI，表示嵌套循环，其中 i 的范围是 [0, 20)，j 的范围是 [0, 100)，执行 store 操作
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, For::make(j, 0, 100, store));
  // 创建代码块对象 par，将嵌套循环 forI 作为其唯一成员
  auto par = Block::make({forI});
  // 创建 LoopNest 对象 nest，表示循环嵌套结构，使用数组 A 的节点作为依赖节点
  LoopNest nest(par, {a_buf.node()});

  // 创建哈希提供者对象 hasher
  HashProvider hasher;
  // 计算优化前的循环嵌套结构的哈希值
  auto hash_before = hasher.hash(nest.root_stmt());
  // 执行条件优化操作
  nest.optimizeConditionals();
  // 计算优化后的循环嵌套结构的哈希值
  auto hash_after = hasher.hash(nest.root_stmt());
  // 断言优化前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsCompValuesNotOrdered) {
  // 输入的IR：
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5, IfThenElse(i<10, B[i], C[i-5]), D[i-10])
  //   }
  // 此处不应进行优化，因为其中一个条件使用了 '>'。

  // 创建缓冲区对象 a_buf，表示数组 A，大小为 {20}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // 创建缓冲区对象 b_buf，表示数组 B，大小为 {5}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // 创建缓冲区对象 c_buf，表示数组 C，大小为 {5}，数据类型为 kInt
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // 创建缓冲区对象 d_buf，表示数组 D，大小为 {10}，数据类型为 kInt
  VarHandle i("i", kInt);
  // 创建存储操作 store 对象，用于将条件表达式结果存储到数组 A 中
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),  // 比较 i 是否小于 5
          IfThenElse::make(
              CompareSelect::make(i, 10, kLT),  // 比较 i 是否小于 10
              Load::make(b_buf, {i}),  // 若满足条件则加载数组 B 的值
              Load::make(c_buf, {i - 5})),  // 否则加载数组 C 的值
          Load::make(d_buf, {i - 10})));  // 如果 i 不小于 5，则加载数组 D 的值
  // 创建循环对象 forI，表示循环，其中 i 的范围是 [0, 20)，执行 store 操作
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  // 创建代码块对象 par，将循环 forI 作为其唯一成员
  auto par = Block::make({forI});
  // 创建 LoopNest 对象 nest，表示循环嵌套结构，使用数组 A 的节点作为依赖节点
  LoopNest nest(par, {a_buf.node()});

  // 创建哈希提供者对象 hasher
  HashProvider hasher;
  // 计算优化前的循环嵌套结构的哈希值
  auto hash_before = hasher.hash(nest.root_stmt());
  // 执行条件优化操作
  nest.optimizeConditionals();
  // 计算优化后的循环嵌套结构的哈希值
  auto hash_after = hasher.hash(nest.root_stmt());
  // 断言优化前后的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}
TEST(LoopNest, OptimizeConditionalsCompValuesNotConstants) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<N, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }
  // 在这里不应进行优化，因为其中一个条件使用了'>'。

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'A' 的缓冲区，大小为 {20}，数据类型为整数
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'B' 的缓冲区，大小为 {5}，数据类型为整数
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'C' 的缓冲区，大小为 {5}，数据类型为整数
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'D' 的缓冲区，大小为 {10}，数据类型为整数
  BufHandle d_buf("D", {10}, kInt);
  // 创建名为 'i' 的变量，数据类型为整数
  VarHandle i("i", kInt);
  // 创建名为 'N' 的变量，数据类型为整数
  VarHandle N("N", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建 Store 节点，将条件赋值语句嵌套在其中
  auto store = Store::make(
      a_buf,
      {i},
      // 创建 IfThenElse 节点，根据条件选择赋值语句
      IfThenElse::make(
          CompareSelect::make(i, N, kLT),  // 比较 i 是否小于 N
          // 如果 i < N，继续判断
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),  // 比较 i 是否小于 5
              Load::make(b_buf, {i}),  // 如果 i < 5，加载 b_buf[i]
              Load::make(c_buf, {i - 5})),  // 否则加载 c_buf[i-5]
          Load::make(d_buf, {i - 10})));  // 如果 i >= N，加载 d_buf[i-10]
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建 For 循环节点，循环范围从 0 到 20，执行 store 节点
  auto forI = For::make(i, 0, 20, store);
  // 创建 Block 节点，包含上述的 For 循环节点
  auto par = Block::make({forI});
  // 创建 LoopNest 对象，将 Block 节点作为根节点，a_buf.node() 表示使用 a_buf
  LoopNest nest(par, {a_buf.node()});

  // 创建 HashProvider 对象，计算优化前的语句哈希值
  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  // 执行条件优化
  nest.optimizeConditionals();
  // 计算优化后的语句哈希值
  auto hash_after = hasher.hash(nest.root_stmt());
  // 断言优化前后语句的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(i>5, B[i], C[i-5]), D[i-10])
  //   }
  // 在这里不应进行优化，因为其中一个条件使用了 '>'。

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'A' 的缓冲区，大小为 {20}，数据类型为整数
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'B' 的缓冲区，大小为 {5}，数据类型为整数
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'C' 的缓冲区，大小为 {5}，数据类型为整数
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建名为 'D' 的缓冲区，大小为 {10}，数据类型为整数
  BufHandle d_buf("D", {10}, kInt);
  // 创建名为 'i' 的变量，数据类型为整数
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建 Store 节点，将条件赋值语句嵌套在其中
  auto store = Store::make(
      a_buf,
      {i},
      // 创建 IfThenElse 节点，根据条件选择赋值语句
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),  // 比较 i 是否小于 10
          // 如果 i < 10，继续判断
          IfThenElse::make(
              CompareSelect::make(i, 5, kGT),  // 比较 i 是否大于 5
              Load::make(b_buf, {i}),  // 如果 i > 5，加载 b_buf[i]
              Load::make(c_buf, {i - 5})),  // 否则加载 c_buf[i-5]
          Load::make(d_buf, {i - 10})));  // 如果 i >= 10，加载 d_buf[i-10]
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  // 创建 For 循环节点，循环范围从 0 到 20，执行 store 节点
  auto forI = For::make(i, 0, 20, store);
  // 创建 Block 节点，包含上述的 For 循环节点
  auto par = Block::make({forI});
  // 创建 LoopNest 对象，将 Block 节点作为根节点，a_buf.node() 表示使用 a_buf
  LoopNest nest(par, {a_buf.node()});

  // 创建 HashProvider 对象，计算优化前的语句哈希值
  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  // 执行条件优化
  nest.optimizeConditionals();
  // 计算优化后的语句哈希值
  auto hash_after = hasher.hash(nest.root_stmt());
  // 断言优化前后语句的哈希值相等
  ASSERT_EQ(hash_before, hash_after);
}
TEST(LoopNest, OptimizeConditionalsInvalidCondition2) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(10<i, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because of the invalid condition:
  //    "10 < i".

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);  // 创建名为 A 的缓冲区，大小为 20，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);   // 创建名为 B 的缓冲区，大小为 5，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);   // 创建名为 C 的缓冲区，大小为 5，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);  // 创建名为 D 的缓冲区，大小为 10，类型为整数
  VarHandle i("i", kInt);            // 创建名为 i 的整数变量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(10, i, kLT),  // 创建一个比较选择节点，判断是否 10 < i
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),  // 在条件为真时，再创建一个比较选择节点，判断是否 i < 5
              Load::make(b_buf, {i}),   // 如果条件为真，则加载 B[i]
              Load::make(c_buf, {i - 5})),  // 如果条件为假，则加载 C[i-5]
          Load::make(d_buf, {i - 10})));  // 如果最初的条件为假，则加载 D[i-10]
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);  // 创建一个循环，遍历 i 从 0 到 19，并执行 store 操作
  auto par = Block::make({forI});  // 创建一个代码块，包含上述循环
  LoopNest nest(par, {a_buf.node()});  // 创建一个循环嵌套对象，根据 par 和 a_buf 节点初始化

  HashProvider hasher;  // 创建一个哈希提供者对象
  auto hash_before = hasher.hash(nest.root_stmt());  // 计算优化前的循环嵌套根语句的哈希值
  nest.optimizeConditionals();  // 对循环嵌套对象进行条件优化
  auto hash_after = hasher.hash(nest.root_stmt());  // 计算优化后的循环嵌套根语句的哈希值
  ASSERT_EQ(hash_before, hash_after);  // 断言优化前后的哈希值相等
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition3) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(k<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because the conditions use different
  // variables: "i < 10" and "k < 5"

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);  // 创建名为 A 的缓冲区，大小为 20，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);   // 创建名为 B 的缓冲区，大小为 5，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);   // 创建名为 C 的缓冲区，大小为 5，类型为整数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);  // 创建名为 D 的缓冲区，大小为 10，类型为整数
  VarHandle i("i", kInt);            // 创建名为 i 的整数变量
  VarHandle k("k", kInt);            // 创建名为 k 的整数变量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),  // 创建一个比较选择节点，判断是否 i < 10
          IfThenElse::make(
              CompareSelect::make(k, 5, kLT),  // 在条件为真时，再创建一个比较选择节点，判断是否 k < 5
              Load::make(b_buf, {i}),   // 如果条件为真，则加载 B[i]
              Load::make(c_buf, {i - 5})),  // 如果条件为假，则加载 C[i-5]
          Load::make(d_buf, {i - 10})));  // 如果最初的条件为假，则加载 D[i-10]
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);  // 创建一个循环，遍历 i 从 0 到 19，并执行 store 操作
  auto par = Block::make({forI});  // 创建一个代码块，包含上述循环
  LoopNest nest(par, {a_buf.node()});  // 创建一个循环嵌套对象，根据 par 和 a_buf 节点初始化

  HashProvider hasher;  // 创建一个哈希提供者对象
  auto hash_before = hasher.hash(nest.root_stmt());  // 计算优化前的循环嵌套根语句的哈希值
  nest.optimizeConditionals();  // 对循环嵌套对象进行条件优化
  auto hash_after = hasher.hash(nest.root_stmt());  // 计算优化后的循环嵌套根语句的哈希值
  ASSERT_EQ(hash_before, hash_after);  // 断言优化前后的哈希值相等
}
TEST(LoopNest, OptimizeConditionalsInvalidCondition4) {
  // 测试用例：循环嵌套优化条件语句的无效条件情况（第四个测试）

  // 输入的 IR（中间表示）：
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(k<10, IfThenElse(k<5, B[i], C[i-5]), D[i-10])
  //   }
  // 这里不应进行优化，因为条件使用了变量 'k'，它不是循环变量。

  // 定义缓冲区和变量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle k("k", kInt);

  // 定义存储操作
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(k, 10, kLT),  // 如果 k < 10
          IfThenElse::make(
              CompareSelect::make(k, 5, kLT),  // 如果 k < 5
              Load::make(b_buf, {i}),  // 则加载 B[i]
              Load::make(c_buf, {i - 5})),  // 否则加载 C[i-5]
          Load::make(d_buf, {i - 10})));  // 否则加载 D[i-10]

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);  // 创建循环结构，范围是 i 从 0 到 19
  auto par = Block::make({forI});  // 创建块结构，包含上述循环
  LoopNest nest(par, {a_buf.node()});  // 创建循环嵌套对象，以及使用的缓冲区节点

  // 计算和比较哈希值以确保优化前后结果一致
  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();  // 执行条件语句优化
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);  // 断言优化前后哈希值一致
}

TEST(LoopNest, OptimizeConditionalsNotNormalized) {
  // 测试用例：循环嵌套优化条件语句的非规范化情况

  // 输入的 IR（中间表示）：
  //   for (int i = 2; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }

  // 定义缓冲区和变量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {15}, kInt);
  VarHandle i("i", kInt);

  // 定义存储操作
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),  // 如果 i < 5
          Load::make(b_buf, {i}),  // 则加载 B[i]
          Load::make(c_buf, {i - 5})));  // 否则加载 C[i-5]

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 2, 20, store);  // 创建循环结构，范围是 i 从 2 到 19
  auto par = Block::make({forI});  // 创建块结构，包含上述循环
  LoopNest nest(par, {a_buf.node()});  // 创建循环嵌套对象，以及使用的缓冲区节点

  // 计算和比较哈希值以确保优化前后结果一致
  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();  // 执行条件语句优化
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);  // 断言优化前后哈希值一致
}

static std::pair<BufHandle, Tensor> colReduce(int M, int N) {
  // 列约简函数，用于生成缓冲区和张量

  BufHandle a("a", {M, N}, kFloat);  // 创建名为 "a" 的缓冲区，大小为 M x N，数据类型为 kFloat
  Tensor t = Reduce(
      "b",
      {N},
      Sum(),
      [&](const VarHandle& n, const VarHandle& m) { return a.load(m, n); },  // 定义列求和的操作
      {M});
  return {a, Tensor(t.buf(), LoopNest::sanitizeNames(t.stmt()))};  // 返回缓冲区和张量对
}
static StmtPtr splitTailReorder(Tensor b) {
  // 定义向量宽度常量为8
  constexpr int kVectorWidth = 8;
  // 创建一个循环嵌套对象，用于操作张量b
  LoopNest nest({b});
  // 获取所有写入缓冲区b的循环嵌套，选取第一个
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[0];
  // 将第一个循环嵌套在kVectorWidth的向量宽度下拆分，处理尾部
  nest.splitWithTail(loops[0], kVectorWidth);
  // 现在循环嵌套看起来如下：
  //
  // for (int i_outer = 0; ...
  //   for (int i_inner = 0; ...
  //     b[i_outer * 8 + i_inner] = float(0);
  //     for (int j = 0; ...
  //       b[i_outer * 8 + i_inner] = ReduceOp(...);
  //
  // for (int i_tail = 0; ...
  //   b[i_tail + ((100 - 0) / 8) * 8] = float(0);
  //   for (int j = 0; ...
  //     b[i_tail + ((100 - 0) / 8) * 8] = ReduceOp(...);
  //
  // 由于对b有4次写入操作，我们从下面的`getAllLoopNestsWritingToBuf`调用中获取4个循环嵌套。
  //
  // 写入 #2: "b[i_outer * 8 + i_inner] = ReduceOp(...)"
  // 循环嵌套 #2: {i_outer, i_inner, j};
  // 我们需要重新排序i_inner和j。
  auto loopnests = nest.getAllLoopNestsWritingToBuf(b.buf());
  // 对第二个循环嵌套中的i_inner和j进行重新排序
  LoopNest::reorderAxis(loopnests[1][1], loopnests[1][2]);
  // 为代码生成做准备
  nest.prepareForCodegen();
  // 返回根语句
  return nest.root_stmt();
}

static StmtPtr splitMaskReorder(Tensor b) {
  // 定义向量宽度常量为8
  constexpr int kVectorWidth = 8;
  // 创建一个循环嵌套对象，用于操作张量b
  LoopNest nest({b});
  // 获取所有写入缓冲区b的循环嵌套，选取第二个
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[1];
  // 使用掩码将第一个循环嵌套在kVectorWidth的向量宽度下拆分
  nest.splitWithMask(loops[0], kVectorWidth);
  // 重新排序第二个循环嵌套中的轴loops[1]和loops[2]
  LoopNest::reorderAxis(loops[1], loops[2]);
  // 为代码生成做准备
  nest.prepareForCodegen();
  // 返回根语句
  return nest.root_stmt();
}

static void checkColReduce(StmtPtr s, BufHandle p, Tensor t) {
  // 获取矩阵的维度M和N
  int M = immediateAs<int>(p.dim(0));
  int N = immediateAs<int>(p.dim(1));
  // 创建填充缓冲区a、b和ref
  PaddedBuffer<float> a(M, N);
  PaddedBuffer<float> b(N);
  PaddedBuffer<float> ref(N);
  // 初始化填充缓冲区a为1.0f
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a(i, j) = 1.0f;
    }
  }
  // 初始化填充缓冲区b为0.0f
  for (int i = 0; i < N; i++) {
    b(i) = 0.0f;
  }
  // 初始化填充缓冲区ref为76.0f
  for (int i = 0; i < N; i++) {
    ref(i) = 76.0f;
  }
  // 使用SimpleIREvaluator调用s，传入p和t作为参数
  SimpleIREvaluator(s, {p, t}).call({a, b});
  // 检查缓冲区b是否与ref在给定精度下相近
  ExpectAllNear(b, ref, 1e-5);
}

TEST(LoopNest, ColReduceSplitTailEvenReorder) {
  // 定义测试中的矩阵维度M和N
  constexpr int M = 76, N = 128;
  // 调用colReduce函数获取p
  auto p = colReduce(M, N);
  // 调用splitTailReorder函数获取s
  StmtPtr s = splitTailReorder(p.second);

  // 创建ostringstream对象oss，并将s的内容输出到oss中
  std::ostringstream oss;
  oss << *s;
  // 设置用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i_outer
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int j
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK-NOT: for (
      )IR";
  // 运行FileCheck验证oss中的输出是否符合verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 调用checkColReduce函数验证s的正确性
  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ColReduceSplitTailUnevenReorder) {
  // 定义测试中的矩阵维度M和N
  constexpr int M = 76, N = 100;
  // 调用colReduce函数获取p
  auto p = colReduce(M, N);
  // 调用splitTailReorder函数获取s
  StmtPtr s = splitTailReorder(p.second);

  // 创建ostringstream对象oss，并将s的内容输出到oss中
  std::ostringstream oss;
  oss << *s;
  // 设置用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i_outer
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int j
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int i_tail
# CHECK-NEXT: b[
# CHECK-NEXT: for (int j
      )IR";
  // 运行FileCheck验证oss中的输出是否符合verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}


这段代码是一个C++代码的测试案例，主要测试了对张量进行拆分、重新排序和检查缩减操作的功能。
TEST(LoopNest, ColReduceSplitMaskEvenReorder) {
  // 设置常量 M 和 N 分别为 76 和 128
  constexpr int M = 76, N = 128;
  // 调用 colReduce 函数获取结果对 p 进行解构
  auto p = colReduce(M, N);
  // 对 p.second 进行 splitMaskReorder 操作，返回结果为 StmtPtr 类型的 s
  StmtPtr s = splitMaskReorder(p.second);
  // 检查 colReduce 结果的正确性
  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ColReduceSplitMaskUnevenReorder) {
  // 设置常量 M 和 N 分别为 76 和 100
  constexpr int M = 76, N = 100;
  // 调用 colReduce 函数获取结果对 p 进行解构
  auto p = colReduce(M, N);
  // 对 p.second 进行 splitMaskReorder 操作，返回结果为 StmtPtr 类型的 s
  StmtPtr s = splitMaskReorder(p.second);
  // 检查 colReduce 结果的正确性
  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ReorderAxisWithMultipleConds) {
  // 定义数组 A 的缓冲区及其大小
  BufHandle a_buf("A", {20}, kInt);
  // 定义循环变量 i 和 j
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 生成内部循环体 forJ
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {i}, Mul::make(i, j)));
  // 生成内部条件 inner_cond
  auto inner_cond = Cond::make(CompareSelect::make(i, 10, kLT), forJ, nullptr);
  // 生成外部条件 outer_cond
  auto outer_cond =
      Cond::make(CompareSelect::make(i, 5, kGT), inner_cond, nullptr);
  // 生成外部循环体 forI
  auto forI = For::make(i, 0, 20, outer_cond);
  // 将 forI 放入 Block 中组成复合语句 par
  StmtPtr par = Block::make({forI});
  // 使用 par 和 a_buf.node() 构建 LoopNest 对象 l
  LoopNest l(par, {a_buf.node()});
  // 对循环进行轴重排
  LoopNest::reorderAxis(forI, forJ);
  // 确保重排后的结果与 par 相同
  ASSERT_EQ(par, l.root_stmt());
  // 简化 par
  par = IRSimplifier::simplify(par);

  // 定义用于验证的字符串模式 verification_pattern
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: for (int i
# CHECK-NEXT: if (i>5
# CHECK-NEXT: if (i<10
# CHECK-NEXT: A[i] = i * j
# CHECK-NOT: for (
      )IR";
  // 将 par 的内容输出到字符串流 oss
  std::ostringstream oss;
  oss << *par;
  // 使用 FileCheck 运行验证模式 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, VectorizeUse) {
  // 设置常量 N 为 8
  constexpr int N = 8;
  // 定义名为 a 的缓冲区
  BufHandle a("a", {N}, kFloat);
  // 定义张量 b，其计算方式为 a 加 1.0f
  Tensor b =
      Compute("b", {N}, [&](const VarHandle& n) { return a.load(n) + 1.0f; });
  // 定义张量 c，其计算方式为 b 加 2.0f
  Tensor c =
      Compute("c", {N}, [&](const VarHandle& n) { return b.load(n) + 2.0f; });
  // 使用 c 和 b, c 构建 LoopNest 对象 nest
  LoopNest nest({c}, {b, c});
  // 获取所有写入 b 缓冲区的循环嵌套
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[0];
  // 对第一个循环进行向量化
  ASSERT_TRUE(LoopNest::vectorize(loops[0]));
  // 获取所有写入 c 缓冲区的循环嵌套
  loops = nest.getAllLoopNestsWritingToBuf(c.buf())[0];
  // 对第一个循环进行向量化
  ASSERT_TRUE(LoopNest::vectorize(loops[0]));
  // 为代码生成做准备
  nest.prepareForCodegen();
  // 获取根语句并赋给变量 s
  StmtPtr s = nest.root_stmt();
  // 将根语句的内容输出到字符串流 oss
  std::ostringstream oss;
  oss << *nest.root_stmt();
  // 使用 FileCheck 运行验证模式
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: c[Ramp
      )IR",
      oss.str());
}

const char* int64Loop = R"IR(
# CHECK: for (int64_t i = 0ll; i < 12ll; i++) {
# CHECK:   b[i] = (a[i]) + 1ll;
# CHECK: }
)IR";

TEST(LoopNest, Int64Direct) {
  // 设置常量 N 为 12
  constexpr int64_t N = 12;
  // 定义名为 a 和 b 的缓冲区，类型为 kLong
  BufHandle a("a", {N}, kLong);
  BufHandle b("b", {N}, kLong);
  // 定义循环变量 n
  VarHandle n("i", kLong);
  // 构建 for 循环语句 s
  StmtPtr s = For::make(
      n, LongImm::make(0l), N, b.store({n}, a.load({n}) + LongImm::make(1l)));
  // 简化 for 循环语句 s
  s = IRSimplifier::simplify(s);
  // 将 s 的内容输出到字符串流 oss
  std::ostringstream oss;
  oss << *s;
  // 使用 FileCheck 运行验证模式 int64Loop
  torch::jit::testing::FileCheck().run(int64Loop, oss.str());
}
TEST(LoopNest, Int64Compute) {
  // 定义常量 N 为 12
  constexpr int64_t N = 12;
  // 创建一个名为 a 的缓冲区，包含 N 个元素，元素类型为长整型
  BufHandle a("a", {N}, kLong);
  // 定义张量 b，形状为 {N}，通过 lambda 函数定义计算逻辑
  Tensor b = Compute("b", {N}, [&](const VarHandle& n) {
    return a.load(n) + LongImm::make(1l);
  });
  // 创建循环嵌套对象 nest，包含张量 b
  LoopNest nest({b});
  // 准备嵌套循环以便进行代码生成
  nest.prepareForCodegen();
  // 简化嵌套循环结构
  nest.simplify();
  // 创建一个字符串流对象 oss
  std::ostringstream oss;
  // 将嵌套循环的根语句输出到字符串流 oss 中
  oss << *nest.root_stmt();
  // 使用 FileCheck 进行语法验证
  torch::jit::testing::FileCheck().run(int64Loop, oss.str());
}

TEST(LoopNest, DistributeLoopWithAllStmtsAsPivots) {
  // 输入的 IR 示例
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  // 定义缓冲区 a_buf 和 b_buf，分别包含 20 个整型元素
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  // 定义循环变量 i, j, k
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  // 创建初始化 A[i] 的语句
  auto initA = Store::make(a_buf, {i}, 0);
  // 创建内部循环 forJ
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  // 创建初始化 B[i] 的语句
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  // 创建内部循环 forK
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  // 创建外部循环 forI，包含所有子语句
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  // 创建并行块 par，包含外部循环 forI
  auto par = Block::make({forI});

  // 定义字符串形式的 IR 校验模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";

  // 创建循环嵌套对象 nest，包含缓冲区 a_buf 和 b_buf
  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  // 进行循环分发，返回新的循环对象列表 new_loops
  auto new_loops = LoopNest::distributeLoop(forI, {initA, forJ, initB});

  // 创建字符串流对象 oss
  std::ostringstream oss;
  // 将并行块 par 输出到字符串流 oss 中
  oss << *par;
  // 使用 FileCheck 进行语法验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 分发后的第一个循环必须与原始的 forI 循环相同
  ASSERT_EQ(new_loops.front(), forI);
}
TEST(LoopNest, DistributeLoopWithOneStmtAsPivot) {
  // 定义一个测试用例，用于测试循环嵌套的分布操作，其中包含一个语句作为枢轴

  // 输入的 IR 表示如下:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }

  // 创建两个缓冲区对象
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);

  // 创建三个整型变量对象
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  // 初始化 A[i] = 0 的赋值操作
  auto initA = Store::make(a_buf, {i}, 0);

  // 创建内部的循环 forJ，循环次数为 0 到 99，循环体内包含 A[i] 的更新操作
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));

  // 初始化 B[i] = A[i] 的赋值操作
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));

  // 创建内部的循环 forK，循环次数为 0 到 49，循环体内包含 B[i] 的更新操作
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));

  // 创建外部的循环 forI，循环次数为 0 到 19，循环体内包含上述所有初始化和循环操作
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));

  // 将 forI 包装成一个 Block 对象，作为整体的并行操作单元
  auto par = Block::make({forI});

  // 创建 LoopNest 对象，传入 par 和两个缓冲区节点的数组
  LoopNest nest(par, {a_buf.node(), b_buf.node()});

  // 对 forI 进行分布操作，分布其中的 forJ 循环
  auto new_loops = LoopNest::distributeLoop(forI, {forJ});

  // 创建一个 ostringstream 对象 oss，用于输出 par 的内容
  std::ostringstream oss;
  oss << *par;

  // 定义字符串 verification_pattern，用于验证 par 输出是否符合预期的 IR 格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  // 使用 FileCheck 进行验证，确保输出的 par 符合预期的 IR 格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：分布后的第一个循环应与原始的 forI 循环相同
  ASSERT_EQ(new_loops.front(), forI);
}

TEST(LoopNest, DistributeLoopWithoutAnyPivot) {
  // 定义一个测试用例，用于测试循环嵌套的分布操作，其中没有任何语句作为枢轴

  // 输入的 IR 表示如下:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }

  // 创建两个缓冲区对象
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);

  // 创建三个整型变量对象
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  // 初始化 A[i] = 0 的赋值操作
  auto initA = Store::make(a_buf, {i}, 0);

  // 创建内部的循环 forJ，循环次数为 0 到 99，循环体内包含 A[i] 的更新操作
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));

  // 初始化 B[i] = A[i] 的赋值操作
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));

  // 创建内部的循环 forK，循环次数为 0 到 49，循环体内包含 B[i] 的更新操作
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));

  // 创建外部的循环 forI，循环次数为 0 到 19，循环体内包含上述所有初始化和循环操作
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));

  // 将 forI 包装成一个 Block 对象，作为整体的并行操作单元
  auto par = Block::make({forI});

  // 定义字符串 verification_pattern，用于验证 par 输出是否符合预期的 IR 格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
      )IR";
  // 使用 FileCheck 进行验证，确保输出的 par 符合预期的 IR 格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(LoopNest, DistributeLoopOverInnerLoops) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  // 创建名为 A 的缓冲区，大小为 20，类型为整数
  BufHandle a_buf("A", {20}, kInt);
  // 创建名为 B 的缓冲区，大小为 20，类型为整数
  BufHandle b_buf("B", {20}, kInt);
  // 创建整数类型的变量 i
  VarHandle i("i", kInt);
  // 创建整数类型的变量 j
  VarHandle j("j", kInt);
  // 创建整数类型的变量 k
  VarHandle k("k", kInt);
  // 初始化 A[i] = 0 的表达式
  auto initA = Store::make(a_buf, {i}, 0);
  // 创建内部循环 for (int j = 0; j < 100; j++) 的表达式
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  // 初始化 B[i] = A[i] 的表达式
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  // 创建内部循环 for (int k = 0; k < 50; k++) 的表达式
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  // 创建外部循环 for (int i = 0; i < 20; i++) 的表达式
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  // 创建包含外部循环的块
  auto par = Block::make({forI});

  // 创建 LoopNest 对象，表示循环嵌套结构
  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  // 对外部循环进行内部循环分布操作
  auto new_loops = LoopNest::distributeLoopOverInnerLoops(forI);

  // 创建输出流对象 oss
  std::ostringstream oss;
  // 将 par 的内容写入输出流 oss
  oss << *par;
  // 设置字符串验证模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  // 运行字符串模式验证器进行验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：分布后的第一个循环必须与原始 For 循环相同
  ASSERT_EQ(new_loops.front(), forI);
}
{
    // 检查在没有任何枢轴的情况下分发循环和其父语句的案例。

    // 定义用于存储数组 A 和 B 的缓冲区对象，大小为 100x100，数据类型为整数
    BufHandle a_buf("A", {100, 100}, kInt);
    BufHandle b_buf("B", {100, 100}, kInt);

    // 定义循环变量 m, i, j, k，数据类型为整数
    VarHandle m("m", kInt);
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);
    VarHandle k("k", kInt);

    // 初始化数组 A[m,i] 的操作：将其值设为 0
    auto initA = Store::make(a_buf, {m, i}, 0);

    // 循环结构 j: 从 0 到 99，对数组 A[m,i] 进行更新操作
    auto forJ = For::make(
        j,
        0,
        100,
        Store::make(
            a_buf,
            {m, i},
            Add::make(Load::make(a_buf, {m, i}), Mul::make(i, j))
        )
    );

    // 初始化数组 B[m,i] 的操作：将其值设为数组 A[m,i] 的值
    auto initB = Store::make(b_buf, {m, i}, Load::make(a_buf, {m, i}));

    // 循环结构 k: 从 0 到 49，对数组 B[m,i] 进行更新操作
    auto forK = For::make(
        k,
        0,
        50,
        Store::make(
            b_buf,
            {m, i},
            Add::make(Load::make(b_buf, {m, i}), Mul::make(i, k))
        )
    );

    // 主循环结构 i: 从 0 到 19，包含上述所有的初始化和更新操作
    auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));

    {
        // 检查在所有语句中分发循环及其父语句的情况。

        // 验证模式，用于检查输出 IR 是否符合预期的语句顺序和格式
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: A[m, i] = 0
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m, i] =
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: B[m, i] = A[m, i]
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m, i] =
# CHECK-NOT: for (
            )IR";

        // 克隆主循环结构 forI，以备后续的修改操作
        auto newForI = to<For>(Stmt::clone(forI));

        // 创建循环结构 forM，循环变量 m 的范围为 0 到 49，将主循环 forI 作为其主体
        auto forM = For::make(m, 0, 50, newForI);

        // 将 forM 放入一个块中作为其父语句
        auto par = Block::make({forM});

        // 创建一个 LoopNest 对象，并传入数组 A 和 B 的节点信息
        LoopNest nest(par, {a_buf.node(), b_buf.node()});

        // 对主循环进行分发并获得新的循环结构列表
        auto newLoops = LoopNest::distributeLoopAndParents(newForI);

        // 使用流 oss 将 par 的内容转换为字符串，用于后续的验证
        std::ostringstream oss;
        oss << *par;

        // 使用 FileCheck 对象运行验证模式，检查输出是否符合预期
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        // 确保分发后的第一个循环与原始的 forM 循环相同
        ASSERT_EQ(newLoops.front(), forM);
    }

    {
        // 检查在所有内部循环中分发循环及其父语句的情况。

        // 验证模式，用于检查输出 IR 是否符合预期的语句顺序和格式
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: A[m, i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m, i] =
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: B[m, i] = A[m, i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m, i] =
# CHECK-NOT: for (
            )IR";

        // 克隆主循环结构 forI，以备后续的修改操作
        auto newForI = to<For>(Stmt::clone(forI));

        // 创建循环结构 forM，循环变量 m 的范围为 0 到 49，将主循环 forI 作为其主体
        auto forM = For::make(m, 0, 50, newForI);

        // 将 forM 放入一个块中作为其父语句
        auto par = Block::make({forM});

        // 创建一个 LoopNest 对象，并传入数组 A 和 B 的节点信息
        LoopNest nest(par, {a_buf.node(), b_buf.node()});

        // 对主循环进行分发并获得新的循环结构列表
        auto newLoops = LoopNest::distributeLoopAndParentsOverInnerLoops(newForI);

        // 使用流 oss 将 par 的内容转换为字符串，用于后续的验证
        std::ostringstream oss;
        oss << *par;

        // 使用 FileCheck 对象运行验证模式，检查输出是否符合预期
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
    }
}
    // 断言检查：确保新循环列表的第一个元素与变量 forM 相等
    ASSERT_EQ(newLoops.front(), forM);
TEST(LoopNest, fuseLoopsSimple) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建名为 A 的缓冲区对象，大小为 100，类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 100，类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建整数类型的变量 j
  VarHandle j("j", kInt);
  // 创建整数类型的变量 k
  VarHandle k("k", kInt);
  // 创建 for 循环节点，用于执行 A[j] = 10 * j 的存储操作
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建 for 循环节点，用于执行 B[k] = 20 * k 的存储操作
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {k}, Mul::make(20, k)));
  // 创建一个并行块，包含以上两个 for 循环节点
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义用于存放融合后循环的指针
  ForPtr fused_loop;
  // 断言融合给定的循环节点成功，并将结果存入 fused_loop
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  // 创建一个字符串流对象 oss，将并行块 par 的内容写入 oss
  std::ostringstream oss;
  oss << *par;
  // 定义用于验证模式的字符串，用于检查输出是否符合预期模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  // 运行文件检查工具，检查 oss 中的输出是否符合 verification_pattern 的模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环节点与第一个循环节点相同
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsMultiple) {
  // Input IR:
  //   for (int i = 0; i < 100; i++) {
  //     A[i+100] = 20 + i;
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建名为 A 的缓冲区对象，大小为 200，类型为整数
  BufHandle a_buf("A", {200}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 100，类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建整数类型的变量 i
  VarHandle i("i", kInt);
  // 创建整数类型的变量 j
  VarHandle j("j", kInt);
  // 创建整数类型的变量 k
  VarHandle k("k", kInt);
  // 创建 for 循环节点，用于执行 A[i+100] = 20 + i 的存储操作
  auto forI =
      For::make(i, 0, 100, Store::make(a_buf, {i + 100}, Add::make(20, i)));
  // 创建 for 循环节点，用于执行 A[j] = 10 * j 的存储操作
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建 for 循环节点，用于执行 B[k] = 20 * k 的存储操作
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {k}, Mul::make(20, k)));
  // 创建一个并行块，包含以上三个 for 循环节点
  auto par = Block::make({forI, forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义用于存放融合后循环的指针
  ForPtr fused_loop;
  // 断言融合给定的循环节点成功，并将结果存入 fused_loop
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forJ, forK}, &fused_loop));

  // 创建一个字符串流对象 oss，将并行块 par 的内容写入 oss
  std::ostringstream oss;
  oss << *par;
  // 定义用于验证模式的字符串，用于检查输出是否符合预期模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i + 100] =
# CHECK-NEXT: A[i] =
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  // 运行文件检查工具，检查 oss 中的输出是否符合 verification_pattern 的模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环节点与第一个循环节点相同
  ASSERT_EQ(fused_loop, forI);
}
TEST(LoopNest, fuseLoopsNested) {
  // Input IR:
  //   for (int m = 0; m < 20; m++) {
  //     A[m] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[m] = A[m] + m * j;
  //     }
  //   }
  //   for (int n = 0; n < 20; n++) {
  //     B[n] = A[n];
  //     for (int k = 0; k < 50; k++) {
  //       B[n] = B[n] + n * k;
  //     }
  //   }

  // 定义缓冲区A，其形状为{20, 100}，数据类型为整型
  BufHandle a_buf("A", {20, 100}, kInt);
  // 定义缓冲区B，其形状为{20, 100}，数据类型为整型
  BufHandle b_buf("B", {20, 100}, kInt);
  // 定义循环变量m，类型为整型
  VarHandle m("m", kInt);
  // 定义循环变量n，类型为整型
  VarHandle n("n", kInt);
  // 定义循环变量j，类型为整型
  VarHandle j("j", kInt);
  // 定义循环变量k，类型为整型
  VarHandle k("k", kInt);

  // 初始化A[m]为0的操作
  auto initA = Store::make(a_buf, {m}, 0);
  // 内部循环：对于每个m，执行 A[m] = A[m] + m * j 的操作
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {m}, Add::make(Load::make(a_buf, {m}), Mul::make(m, j))));
  // 初始化B[n]为A[n]的操作
  auto initB = Store::make(b_buf, {n}, Load::make(a_buf, {n}));
  // 内部循环：对于每个n，执行 B[n] = B[n] + n * k 的操作
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {n}, Add::make(Load::make(b_buf, {n}), Mul::make(n, k))));
  // 外部循环：对于每个m，执行初始化A[m]和内部循环forJ
  auto forM = For::make(m, 0, 20, Block::make({initA, forJ}));
  // 外部循环：对于每个n，执行初始化B[n]和内部循环forK
  auto forN = For::make(n, 0, 20, Block::make({initB, forK}));
  // 并行执行：将两个外部循环forM和forN组成块
  auto par = Block::make({forM, forN});
  // 禁止Lint警告：初始化未使用的变量
  // 定义指针fused_loop，用于存储融合后的循环
  ForPtr fused_loop;
  // 断言：对于给定的循环列表{forM, forN}，成功地进行循环融合，并将结果存储在fused_loop中
  ASSERT_TRUE(LoopNest::fuseLoops({forM, forN}, &fused_loop));

  // 创建ostringstream对象oss，将合并后的循环块par输出为字符串
  std::ostringstream oss;
  oss << *par;
  // 验证模式字符串，用于检查合并后的循环块par的输出格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int m
# CHECK-NEXT: A[m] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m] =
# CHECK: B[m] = A[m]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m] =
# CHECK-NOT: for (
      )IR";
  // 使用FileCheck类来运行verification_pattern检查oss.str()的输出格式是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：融合后的循环fused_loop必须与第一个循环forM相同
  ASSERT_EQ(fused_loop, forM);
}

TEST(LoopNest, fuseLoopsNested2D) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       B[m,n] = m + n * 100;
  //     }
  //   }

  // 定义缓冲区A，其形状为{20, 100}，数据类型为整型
  BufHandle a_buf("A", {20, 100}, kInt);
  // 定义缓冲区B，其形状为{20, 100}，数据类型为整型
  BufHandle b_buf("B", {20, 100}, kInt);
  // 定义循环变量i，类型为整型
  VarHandle i("i", kInt);
  // 定义循环变量j，类型为整型
  VarHandle j("j", kInt);
  // 定义循环变量m，类型为整型
  VarHandle m("m", kInt);
  // 定义循环变量n，类型为整型
  VarHandle n("n", kInt);

  // 内部循环：对于每个(i, j)，执行 A[i, j] = i * j * 500 的操作
  auto forI = For::make(
      i,
      0,
      20,
      For::make(
          j,
          0,
          100,
          Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500))));
  // 内部循环：对于每个(m, n)，执行 B[m, n] = m + n * 100 的操作
  auto forM = For::make(
      m,
      0,
      20,
      For::make(
          n,
          0,
          50,
          Store::make(b_buf, {m, n}, Add::make(m, Mul::make(n, 100)))));
  // 并行执行：将两个外部循环forI和forM组成块
  auto par = Block::make({forI, forM});
  // 禁止Lint警告：初始化未使用的变量
  // 定义指针fused_loop，用于存储融合后的循环
  ForPtr fused_loop;
  // 断言：对于给定的循环列表{forI, forM}，成功地进行循环融合，并将结果存储在fused_loop中
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  // 创建ostringstream对象oss，将合并后的循环块par输出为字符串
  std::ostringstream oss;
  oss << *par;
  // 验证模式字符串，用于检查合并后的循环块par的输出格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int n
# CHECK-NEXT: B[i, n] =
      )IR";
  // 使用FileCheck类来运行verification_pattern检查oss.str()的输出格式是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// 测试函数：将嵌套的2D循环中的内部循环进行融合
TEST(LoopNest, fuseLoopsNested2DInner) {
  // 输入的IR：
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //     for (int n = 0; n < 100; n++) {
  //       B[i,n] = m + n * 100;
  //     }
  //   }

  // 定义缓冲区和变量
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle n("n", kInt);

  // 创建内部循环 forJ，计算 A[i,j] = i * j * 500
  auto forJ = For::make(
      j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500)));
  
  // 创建内部循环 forN，计算 B[i,n] = m + n * 100
  auto forN = For::make(
      n, 0, 100, Store::make(b_buf, {i, n}, Add::make(i, Mul::make(n, 100))));
  
  // 创建外部循环 forI，包含 forJ 和 forN
  auto forI = For::make(i, 0, 20, Block::make({forJ, forN}));

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;

  // 调用函数尝试融合 forJ 和 forN 循环，期望获得融合后的循环对象
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forN}, &fused_loop));

  // 将 forI 循环对象输出为字符串
  std::ostringstream oss;
  oss << *forI;

  // 定义字符串验证模式，检查融合后的循环
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK-NEXT: B[i, j] =
# CHECK-NOT: for (
      )IR";

  // 运行文件检查工具来验证字符串输出是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环对象与原始的 forJ 循环对象相同
  ASSERT_EQ(fused_loop, forJ);
}
TEST(LoopNest, fuseLoopsNotContiguous) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   B[0] = 0;
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建名为 A 的缓冲区对象，大小为 100，元素类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 100，元素类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建整数类型的变量 j
  VarHandle j("j", kInt);
  // 创建整数类型的变量 k
  VarHandle k("k", kInt);
  // 创建循环结构体，对数组 A 执行赋值操作：A[j] = 10 * j
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 初始化数组 B 的第一个元素为 0
  auto initB = Store::make(b_buf, {0}, 0);
  // 创建循环结构体，对数组 B 执行赋值操作：B[k] = 20 * k
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // 创建包含多个语句的代码块
  auto par = Block::make({forJ, initB, forK});
  // 声明指向循环结构体的指针，用于接收融合后的循环
  ForPtr fused_loop;
  // 断言融合操作返回 false，表明循环不能被成功融合
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsWithDifferentParents) {
  // Input IR:
  //   for (int i = 0; i < 50; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  //   B[0] = 0;
  //   for (int k = 50; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建名为 A 的缓冲区对象，大小为 50x100，元素类型为整数
  BufHandle a_buf("A", {50, 100}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 100，元素类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建整数类型的变量 i 和 j
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建整数类型的变量 k
  VarHandle k("k", kInt);
  // 创建循环结构体，对数组 A 执行赋值操作：A[i,j] = i * j
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(i, j)));
  // 创建循环结构体，对数组 B 执行赋值操作：B[0] = 0
  auto forI = For::make(i, 0, 50, forJ);
  // 初始化数组 B 的第一个元素为 0
  auto initB = Store::make(b_buf, {0}, 0);
  // 创建循环结构体，对数组 B 执行赋值操作：B[k] = 20 * k
  auto forK = For::make(k, 50, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // 创建包含多个语句的代码块
  auto par = Block::make({forI, initB, forK});
  // 声明指向循环结构体的指针，用于接收融合后的循环
  ForPtr fused_loop;
  // 断言融合操作返回 false，表明循环不能被成功融合
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsWithVariableBounds) {
  // Input IR:
  //   for (int j = 0; j < N; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < N; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建名为 A 的缓冲区对象，大小为 20，元素类型为整数
  BufHandle a_buf("A", {20}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 20，元素类型为整数
  BufHandle b_buf("B", {20}, kInt);
  // 创建整数类型的变量 j 和 k，以及变量 N
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle N("N", kInt);
  // 创建循环结构体，对数组 A 执行赋值操作：A[j] = 10 * j
  auto forJ = For::make(j, 0, N, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环结构体，对数组 B 执行赋值操作：B[j] = 20 * k
  auto forK = For::make(k, 0, N, Store::make(b_buf, {j}, Mul::make(20, k)));
  // 创建包含多个语句的代码块
  auto par = Block::make({forJ, forK});
  // 声明指向循环结构体的指针，用于接收融合后的循环
  ForPtr fused_loop;
  // 断言融合操作返回 true，表明循环成功融合
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  // 创建字符串输出流对象 oss，并将融合后的循环结构体输出到 oss 中
  std::ostringstream oss;
  oss << *par;
  // 创建字符串 verification_pattern，用于验证输出结果
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  // 运行验证模式，检查 oss 中的输出结果是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环与原始循环相同
  ASSERT_EQ(fused_loop, forJ);
}
TEST(LoopNest, fuseLoopsWithExprBounds) {
  // 定义一个测试用例，用于测试循环融合功能，此处示例展示了两个简单的循环结构
  // 输入的 IR（中间表示）：
  //   for (int j = 0; j < M + N; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < M + N; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建缓冲区对象，表示数组 A 和 B
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  
  // 定义循环变量 j 和 k，以及常量 M 和 N
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  
  // 创建第一个循环结构 forJ，对数组 A 进行赋值操作
  auto forJ = For::make(j, 0, M + N, Store::make(a_buf, {j}, Mul::make(10, j)));
  
  // 创建第二个循环结构 forK，对数组 B 进行赋值操作
  auto forK = For::make(k, 0, M + N, Store::make(b_buf, {j}, Mul::make(20, k)));
  
  // 创建一个并行块 par，包含了 forJ 和 forK 两个循环
  auto par = Block::make({forJ, forK});
  
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义一个指针 fused_loop 用于接收融合后的循环对象
  ForPtr fused_loop;
  
  // 断言循环融合函数成功执行，将 forJ 和 forK 融合成一个循环，结果保存在 fused_loop 中
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  // 创建一个字符串流对象 oss，用于打印 par 块的内容到字符串
  std::ostringstream oss;
  oss << *par;
  
  // 定义验证模式字符串，用于验证生成的 IR 是否符合预期的格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  
  // 使用 FileCheck 工具验证 oss 中的 IR 是否符合 verification_pattern 的规则
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环对象 fused_loop 必须与原始的 forJ 循环相同
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithDifferentExprBounds) {
  // 定义另一个测试用例，测试具有不同表达式边界的循环融合功能
  // 输入的 IR（中间表示）：
  //   for (int j = M; j < N * 2; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = M; k < N + N; k++) {
  //     B[k] = 20 * k;
  //   }

  // 创建缓冲区对象，表示数组 A 和 B
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  
  // 定义循环变量 j 和 k，以及常量 M 和 N
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  
  // 创建第一个循环结构 forJ，对数组 A 进行赋值操作
  auto forJ = For::make(j, M, N * 2, Store::make(a_buf, {j}, Mul::make(10, j)));
  
  // 创建第二个循环结构 forK，对数组 B 进行赋值操作
  auto forK = For::make(k, M, N + N, Store::make(b_buf, {j}, Mul::make(20, k)));
  
  // 创建一个并行块 par，包含了 forJ 和 forK 两个循环
  auto par = Block::make({forJ, forK});
  
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义一个指针 fused_loop 用于接收融合后的循环对象
  ForPtr fused_loop;
  
  // 断言循环融合函数成功执行，将 forJ 和 forK 融合成一个循环，结果保存在 fused_loop 中
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  // 创建一个字符串流对象 oss，用于打印 par 块的内容到字符串
  std::ostringstream oss;
  oss << *par;
  
  // 定义验证模式字符串，用于验证生成的 IR 是否符合预期的格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  
  // 使用 FileCheck 工具验证 oss 中的 IR 是否符合 verification_pattern 的规则
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环对象 fused_loop 必须与原始的 forJ 循环相同
  ASSERT_EQ(fused_loop, forJ);
}
TEST(LoopNest, fuseLoopsWithNonOverlappingBufferAccesses) {
  // 定义一个测试用例，测试循环融合功能，确保融合的循环没有重叠的缓冲区访问

  // 输入的 IR（中间表示）:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k+100] = 30 * k
  //   }

  // 定义名为 A 的缓冲区对象，大小为 200，数据类型为 kInt
  BufHandle a_buf("A", {200}, kInt);
  // 定义整型变量 j
  VarHandle j("j", kInt);
  // 定义整型变量 k
  VarHandle k("k", kInt);
  // 创建循环 forJ，遍历 j 从 10 到 100，对缓冲区 a_buf 中的元素进行写操作，写入 10*j 的结果
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，遍历 k 从 10 到 100，对缓冲区 a_buf 中的元素进行写操作，写入 30*k 的结果
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 100}, Mul::make(30, k)));
  // 创建一个并行块，包含上述两个循环
  auto par = Block::make({forJ, forK});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 声明一个名为 fused_loop 的循环指针，用于接收融合后的循环
  ForPtr fused_loop;
  // 断言：调用 LoopNest 类的 fuseLoops 方法尝试融合 forJ 和 forK 循环，期望操作成功
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  // 创建一个字符串流对象 oss，将并行块 par 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *par;
  // 定义一个字符串，用于验证输出结果是否符合指定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: A[j + 100] =
# CHECK-NOT: for (
      )IR";
  // 使用 FileCheck 类检查 oss 中的输出是否匹配 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：融合后的循环 fused_loop 应该与第一个循环 forJ 相同
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithNonOverlapping2DBufferAccesses) {
  // 定义一个测试用例，测试循环融合功能，确保融合的循环没有重叠的二维缓冲区访问

  // 输入的 IR（中间表示）:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       A[m+20,n+100] = m + n * 100;
  //     }
  //   }

  // 定义名为 A 的二维缓冲区对象，大小为 {20, 100}，数据类型为 kInt
  BufHandle a_buf("A", {20, 100}, kInt);
  // 定义名为 B 的二维缓冲区对象，大小为 {20, 50}，数据类型为 kInt
  BufHandle b_buf("B", {20, 50}, kInt);
  // 定义整型变量 i
  VarHandle i("i", kInt);
  // 定义整型变量 j
  VarHandle j("j", kInt);
  // 定义整型变量 m
  VarHandle m("m", kInt);
  // 定义整型变量 n
  VarHandle n("n", kInt);
  // 创建存储操作 storeA1，将 i*j*500 的结果存储到缓冲区 a_buf 的 (i, j) 位置
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  // 创建循环 forJ，遍历 j 从 0 到 100，执行 storeA1 操作
  auto forJ = For::make(j, 0, 100, storeA1);
  // 创建循环 forI，遍历 i 从 0 到 20，执行 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);
  // 创建存储操作 storeA2，将 m+n*100 的结果存储到缓冲区 a_buf 的 (m+20, n+100) 位置
  auto storeA2 =
      Store::make(a_buf, {m + 20, n + 100}, Add::make(m, Mul::make(n, 100)));
  // 创建循环 forN，遍历 n 从 0 到 50，执行 storeA2 操作
  auto forN = For::make(n, 0, 50, storeA2);
  // 创建循环 forM，遍历 m 从 0 到 20，执行 forN 循环
  auto forM = For::make(m, 0, 20, forN);
  // 创建一个并行块，包含上述四个循环
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 声明一个名为 fused_loop 的循环指针，用于接收融合后的循环
  ForPtr fused_loop;
  // 断言：调用 LoopNest 类的 fuseLoops 方法尝试融合 forI 和 forM 循环，期望操作成功
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  // 创建一个字符串流对象 oss，将并行块 par 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *par;
  // 定义一个字符串，用于验证输出结果是否符合指定模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int n
# CHECK-NEXT: A[i + 20, n + 100] =
# CHECK-NOT: for (
      )IR";
  // 使用 FileCheck 类检查 oss 中的输出是否匹配 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：融合后的循环 fused_loop 应该与第一个循环 forI 相同
  ASSERT_EQ(fused_loop, forI);
}
TEST(LoopNest, fuseLoopsWith2DReductions) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 50; j++) {
  //       A[i,j] = 0
  //       for (int k = 0; k < 100; k++) {
  //         A[i,j] = A[i,j] + B[i,j,k];
  //       }
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 40; n++) {
  //       C[m,n] = A[m,n];
  //     }
  //   }
  // 创建名为 A 的缓冲区，维度为 {20, 50}，类型为整型
  BufHandle a_buf("A", {20, 50}, kInt);
  // 创建名为 B 的缓冲区，维度为 {20, 50, 100}，类型为整型
  BufHandle b_buf("B", {20, 50, 100}, kInt);
  // 创建名为 C 的缓冲区，维度为 {20, 40}，类型为整型
  BufHandle c_buf("C", {20, 40}, kInt);
  // 创建整型变量 i
  VarHandle i("i", kInt);
  // 创建整型变量 j
  VarHandle j("j", kInt);
  // 创建整型变量 k
  VarHandle k("k", kInt);
  // 创建整型变量 m
  VarHandle m("m", kInt);
  // 创建整型变量 n
  VarHandle n("n", kInt);
  // 初始化 A[i,j] 为 0 的存储操作
  auto initA = Store::make(a_buf, {i, j}, 0);
  // 计算 A[i,j] = A[i,j] + B[i,j,k] 的存储操作
  auto sumA = Store::make(
      a_buf,
      {i, j},
      Add::make(Load::make(a_buf, {i, j}), Load::make(b_buf, {i, j, k})));
  // 循环 k，将 sumA 放入循环体
  auto forK = For::make(k, 0, 100, sumA);
  // 循环 j，将 initA 和 forK 放入循环体
  auto forJ = For::make(j, 0, 50, Block::make({initA, forK}));
  // 循环 i，将 forJ 放入循环体
  auto forI = For::make(i, 0, 20, forJ);
  // 存储 A[m,n] = A[m,n] 的操作
  auto storeC = Store::make(c_buf, {m, n}, Load::make(a_buf, {m, n}));
  // 循环 n，将 storeC 放入循环体
  auto forM = For::make(m, 0, 20, For::make(n, 0, 40, storeC));
  // 创建代码块 par 包含 forI 和 forM
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  // 调用 LoopNest::fuseLoops 方法，期望合并 forI 和 forM，结果存储在 fused_loop 中
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  // 创建一个字符串流 oss，将 par 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *par;
  // 设置 IR 校验模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK-NEXT: for (int k
# CHECK-NEXT: A[i, j] = (A[i, j]) +
# CHECK: for (int n
# CHECK-NEXT: C[i, n] = A[i, n]
      )IR";
  // 使用 FileCheck 运行校验模式，检查 oss 中的字符串
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言：合并后的循环 fused_loop 应该与 forI 相同
  ASSERT_EQ(fused_loop, forI);
}
TEST(LoopNest, fuseLoopsWithComplexIndices) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 20; j++) {
  //       A[i,j*20+j+2] = i + j;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 20; n++) {
  //       B[m,n] = A[m,n*20+n+2];
  //     }
  //   }

  // 创建表示数组 A 和 B 的缓冲区对象
  BufHandle a_buf("A", {20, 400}, kInt);
  BufHandle b_buf("B", {20, 400}, kInt);

  // 定义循环变量 i, j, m, n
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  // 定义写入 A 数组的表达式
  auto writeA = Store::make(a_buf, {i, j * 20 + j + 2}, i + j);
  // 创建第一组嵌套循环 (i, j)
  auto forI = For::make(i, 0, 20, For::make(j, 0, 20, writeA));

  // 定义写入 B 数组的表达式
  auto storeB = Store::make(b_buf, {m, n}, Load::make(a_buf, {m, n * 20 + n + 2}));
  // 创建第二组嵌套循环 (m, n)
  auto forM = For::make(m, 0, 20, For::make(n, 0, 20, storeB));

  // 将两组嵌套循环放入一个并行块中
  auto par = Block::make({forI, forM});

  // 定义用于存储融合后循环的指针
  ForPtr fused_loop;

  // 调用函数尝试融合第一组和第二组嵌套循环
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  // 将并行块 par 转换为字符串流
  std::ostringstream oss;
  oss << *par;

  // 定义用于验证输出的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, (j * 20 + j) + 2] = i + j
# CHECK: for (int n
# CHECK-NEXT: B[i, n] = A[i, (n * 20 + n) + 2]
# CHECK-NOT: for (
      )IR";

  // 使用 FileCheck 运行模式字符串验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言融合后的循环与第一组循环 forI 相同
  ASSERT_EQ(fused_loop, forI);
}
TEST(LoopNest, fuseLoopsWithTranspose) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 20; j++) {
  //       A[i,j] = i + j;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 20; n++) {
  //       B[m,n] = A[n,m];  // Transpose
  //     }
  //   }

  // 定义名为 a_buf 的缓冲区，表示矩阵 A，大小为 20x20，元素类型为整数
  BufHandle a_buf("A", {20, 20}, kInt);
  // 定义名为 b_buf 的缓冲区，表示矩阵 B，大小为 20x20，元素类型为整数
  BufHandle b_buf("B", {20, 20}, kInt);
  // 定义变量 i 和 j，类型为整数，表示两层循环中的索引变量
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 创建写操作 writeA，将 i + j 的结果存储到矩阵 A 的位置 (i, j)
  auto writeA = Store::make(a_buf, {i, j}, i + j);
  // 创建外层循环 forI，范围是 i 从 0 到 19，内层循环是 j 从 0 到 19，执行写操作 writeA
  auto forI = For::make(i, 0, 20, For::make(j, 0, 20, writeA));
  // 创建写操作 storeB，从矩阵 A 中读取数据进行转置，并存储到矩阵 B 的位置 (m, n)
  auto storeB = Store::make(b_buf, {m, n}, Load::make(a_buf, {n, m}));
  // 创建外层循环 forM，范围是 m 从 0 到 19，内层循环是 n 从 0 到 19，执行写操作 storeB
  auto forM = For::make(m, 0, 20, For::make(n, 0, 20, storeB));
  // 创建代码块 par，包含两个嵌套的循环 forI 和 forM
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义一个指向循环的指针 fused_loop
  ForPtr fused_loop;
  // 调用 LoopNest::fuseLoops 函数，尝试融合 forI 和 forM 循环，期望返回 false 表示无法融合
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies1) {
  // Input IR:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k-1] = 20 * k;
  //   }

  // 定义名为 a_buf 的缓冲区，表示数组 A，大小为 100，元素类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 定义变量 j 和 k，类型为整数，表示两个独立的循环的索引变量
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  // 创建循环 forJ，范围是 j 从 10 到 99，执行 A[j] = 10 * j 的写操作
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，范围是 k 从 10 到 99，执行 A[k-1] = 20 * k 的写操作
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k - 1}, Mul::make(20, k)));
  // 创建代码块 par，包含两个独立的循环 forJ 和 forK
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义一个指向循环的指针 fused_loop
  ForPtr fused_loop;
  // 调用 LoopNest::fuseLoops 函数，尝试融合 forJ 和 forK 循环，期望返回 false 表示无法融合
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies2) {
  // Input IR:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k+50] = 20 * k;
  //   }

  // 定义名为 a_buf 的缓冲区，表示数组 A，大小为 150，元素类型为整数
  BufHandle a_buf("A", {150}, kInt);
  // 定义变量 j 和 k，类型为整数，表示两个独立的循环的索引变量
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  // 创建循环 forJ，范围是 j 从 10 到 99，执行 A[j] = 10 * j 的写操作
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，范围是 k 从 10 到 99，执行 A[k+50] = 20 * k 的写操作
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 50}, Mul::make(20, k)));
  // 创建代码块 par，包含两个独立的循环 forJ 和 forK
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义一个指向循环的指针 fused_loop
  ForPtr fused_loop;
  // 调用 LoopNest::fuseLoops 函数，尝试融合 forJ 和 forK 循环，期望返回 false 表示无法融合
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}


这些注释描述了三个测试用例中的每一行代码的具体功能和意图。
TEST(LoopNest, fuseLoopsThatViolateDependencies3) {
  // Input IR:
  //   for (int m = 0; m < 20; m++) {
  //     A[m] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[m] = A[m] + m * j;
  //     }
  //   }
  //   for (int n = 0; n < 20; n++) {
  //     B[n] = A[n+1];
  //     for (int k = 0; k < 50; k++) {
  //       B[n] = B[n] + n * k;
  //     }
  //   }
  // 定义数组 A 的缓冲区及其维度
  BufHandle a_buf("A", {25, 100}, kInt);
  // 定义数组 B 的缓冲区及其维度
  BufHandle b_buf("B", {20, 50}, kInt);
  // 定义循环变量 m
  VarHandle m("m", kInt);
  // 定义循环变量 n
  VarHandle n("n", kInt);
  // 定义循环变量 j
  VarHandle j("j", kInt);
  // 定义循环变量 k
  VarHandle k("k", kInt);
  // 初始化数组 A 的循环体
  auto initA = Store::make(a_buf, {m}, 0);
  // 循环体 forJ，对数组 A 进行计算
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {m}, Add::make(Load::make(a_buf, {m}), Mul::make(m, j))));
  // 初始化数组 B 的循环体
  auto initB = Store::make(b_buf, {n}, Load::make(a_buf, {n + 1}));
  // 循环体 forK，对数组 B 进行计算
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {n}, Add::make(Load::make(b_buf, {n}), Mul::make(n, k))));
  // 循环 m 的整体循环体
  auto forM = For::make(m, 0, 20, Block::make({initA, forJ}));
  // 循环 n 的整体循环体
  auto forN = For::make(n, 0, 20, Block::make({initB, forK}));
  // 将两个循环块组合成一个并行块
  auto par = Block::make({forM, forN});
  // 进行循环融合，预期不会成功，返回值存储在 fused_loop 中
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forM, forN}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies4) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       A[m+1,n] = m + n * 100;
  //     }
  //   }
  // 定义数组 A 的缓冲区及其维度
  BufHandle a_buf("A", {30, 100}, kInt);
  // 定义循环变量 i
  VarHandle i("i", kInt);
  // 定义循环变量 j
  VarHandle j("j", kInt);
  // 定义循环变量 m
  VarHandle m("m", kInt);
  // 定义循环变量 n
  VarHandle n("n", kInt);
  // 循环体 forI，对数组 A 进行计算
  auto forI = For::make(
      i,
      0,
      20,
      For::make(
          j,
          0,
          100,
          Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500))));
  // 循环体 forM，对数组 A 进行计算
  auto forM = For::make(
      m,
      0,
      20,
      For::make(
          n,
          0,
          50,
          Store::make(a_buf, {m + 1, n}, Add::make(m, Mul::make(n, 100)))));
  // 将两个循环块组合成一个并行块
  auto par = Block::make({forI, forM});
  // 进行循环融合，预期不会成功，返回值存储在 fused_loop 中
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}
TEST(LoopNest, fuseLoopsThatViolateDependencies5) {
  // 输入的IR（Intermediate Representation）：
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //     for (int n = 0; n < 100; n++) {
  //       A[i,n+1] = m + n * 100;
  //     }
  //   }
  
  // 创建名为 A 的缓冲区对象，大小为 {20, 200}，数据类型为整型
  BufHandle a_buf("A", {20, 200}, kInt);
  // 创建变量 i，数据类型为整型
  VarHandle i("i", kInt);
  // 创建变量 j，数据类型为整型
  VarHandle j("j", kInt);
  // 创建变量 n，数据类型为整型
  VarHandle n("n", kInt);
  // 创建循环 forJ，循环变量 j 从 0 到 99，循环体为将 i * j * 500 存储到 A[i,j] 中
  auto forJ = For::make(
      j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500)));
  // 创建循环 forN，循环变量 n 从 0 到 99，循环体为将 m + n * 100 存储到 A[i,n+1] 中
  auto forN = For::make(
      n,
      0,
      100,
      Store::make(a_buf, {i, n + 1}, Add::make(i, Mul::make(n, 100))));
  // 创建循环 forI，循环变量 i 从 0 到 19，循环体为包含 forJ 和 forN 的块
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  auto forI = For::make(i, 0, 20, Block::make({forJ, forN}));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  // 断言尝试将 forJ 和 forN 循环融合，并将结果存储在 fused_loop 中
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forN}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies6) {
  // 输入的IR（Intermediate Representation）：
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * A[99-k];
  //   }
  
  // 创建名为 A 的缓冲区对象，大小为 {100}，数据类型为整型
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 {100}，数据类型为整型
  BufHandle b_buf("B", {100}, kInt);
  // 创建变量 j，数据类型为整型
  VarHandle j("j", kInt);
  // 创建变量 k，数据类型为整型
  VarHandle k("k", kInt);
  // 创建循环 forJ，循环变量 j 从 0 到 99，循环体为将 10 * j 存储到 A[j] 中
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，循环变量 k 从 0 到 99，循环体为将 20 * A[99-k] 存储到 B[k] 中
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  // 断言尝试将 forJ 和 forK 循环融合，并将结果存储在 fused_loop 中
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies7) {
  // 输入的IR（Intermediate Representation）：
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * A[99-k];
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  
  // 创建名为 A 的缓冲区对象，大小为 {100}，数据类型为整型
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为 B 的缓冲区对象，大小为 {100}，数据类型为整型
  BufHandle b_buf("B", {100}, kInt);
  // 创建变量 j，数据类型为整型
  VarHandle j("j", kInt);
  // 创建变量 k，数据类型为整型
  VarHandle k("k", kInt);
  // 创建循环 forK，循环变量 k 从 0 到 99，循环体为将 20 * A[99-k] 存储到 B[k] 中
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  // 创建循环 forJ，循环变量 j 从 0 到 99，循环体为将 10 * j 存储到 A[j] 中
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  auto par = Block::make({forK, forJ});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  // 断言尝试将 forK 和 forJ 循环融合，并将结果存储在 fused_loop 中
  ASSERT_FALSE(LoopNest::fuseLoops({forK, forJ}, &fused_loop));
}
TEST(LoopNest, areLoopsPerfectlyNested) {
  // 定义测试用例，验证嵌套循环是否完美嵌套

  // 创建名为 A 的缓冲区，维度为 {20, 30, 40}，元素类型为整型
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  // 定义循环变量 i，类型为整型
  VarHandle i("i", kInt);
  // 定义循环变量 j，类型为整型
  VarHandle j("j", kInt);
  // 定义循环变量 k，类型为整型
  VarHandle k("k", kInt);
  // 创建存储操作，将 i * j * k 的结果存储到缓冲区 a_buf 中
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  // 创建 k 的循环，范围是从 0 到 40，内部包含存储操作 store
  auto forK = For::make(k, 0, 40, store);
  // 创建 j 的循环，范围是从 0 到 30，内部包含 forK 循环
  auto forJ = For::make(j, 0, 30, forK);
  // 创建 i 的循环，范围是从 0 到 20，内部包含 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);
  // 创建包含 forI 循环的块
  auto par = Block::make({forI});
  // 断言三层嵌套的循环 forI, forJ, forK 是否完美嵌套
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // 验证以任何其他顺序指定循环会失败
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forJ, forI, forK}));
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forK, forJ}));
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forK, forJ, forI}));

  // 向 forK 的循环体添加一个语句应该是可以的
  auto init = Store::make(a_buf, {i, j}, 0);
  forK->body()->insert_stmt_before(init, store);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // 向 forJ 的循环体添加一个语句应该导致测试失败
  forK->body()->remove_stmt(init);
  forJ->body()->insert_stmt_before(init, forK);
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // 类似地，向 forI 的循环体添加一个语句应该导致测试失败
  forJ->body()->remove_stmt(init);
  forI->body()->insert_stmt_before(init, forJ);
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));
}

TEST(LoopNest, reorderNestedLoops2D) {
  // 定义测试用例，验证二维嵌套循环重新排序

  // 创建名为 A 的缓冲区，维度为 {20, 30, 40}，元素类型为整型
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  // 定义循环变量 i，类型为整型
  VarHandle i("i", kInt);
  // 定义循环变量 j，类型为整型
  VarHandle j("j", kInt);
  // 创建存储操作，将 i * j 的结果存储到缓冲区 a_buf 中
  auto store = Store::make(a_buf, {i, j}, Mul::make(i, j));
  // 创建 j 的循环，范围是从 0 到 30，内部包含存储操作 store
  auto forJ = For::make(j, 0, 30, store);
  // 创建 i 的循环，范围是从 0 到 20，内部包含 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);
  // 创建包含 forI 循环的块
  auto par = Block::make({forI});

  // 对循环 forI 和 forJ 进行重新排序，将 j 放在 i 前面
  auto reordered = LoopNest::reorder({forI, forJ}, {1, 0});

  // 断言重新排序后的循环顺序
  ASSERT_EQ(reordered[0], forJ);
  ASSERT_EQ(reordered[1], forI);
  // 验证 forJ 和 forI 是否完美嵌套
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forJ, forI}));
  // 验证 store 是否在 forI 的主体内
  ASSERT_EQ(store->get_parent(), forI->body());
}
TEST(LoopNest, reorderNestedLoops3D) {
  // 定义测试用例函数，用于测试三重嵌套循环的重新排序功能

  // 创建名为 A 的缓冲区对象，维度为 {20, 30, 40}，元素类型为整型
  BufHandle a_buf("A", {20, 30, 40}, kInt);

  // 创建整型变量 i、j、k 对应的变量句柄
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  // 创建存储操作，将乘法表达式 i * j * k 存储到缓冲区 a_buf 的 {i, j, k} 位置
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));

  // 创建 k 循环，范围从 0 到 40，循环体为 store 操作
  auto forK = For::make(k, 0, 40, store);

  // 创建 j 循环，范围从 0 到 30，循环体为 forK 循环
  auto forJ = For::make(j, 0, 30, forK);

  // 创建 i 循环，范围从 0 到 20，循环体为 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);

  // 创建包含 forI 循环的块
  auto par = Block::make({forI});

  // 对循环嵌套进行重新排序，顺序为 {2, 0, 1}
  auto reordered = LoopNest::reorder({forI, forJ, forK}, {2, 0, 1});

  // 断言重新排序后的循环顺序
  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forI);
  ASSERT_EQ(reordered[2], forJ);

  // 断言重新排序后的循环是否完全嵌套
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forI, forJ}));

  // 断言 forK 循环的父节点为 par
  ASSERT_EQ(forK->get_parent(), par);

  // 断言 store 操作的父节点为 forJ 循环的主体
  ASSERT_EQ(store->get_parent(), forJ->body());
}

TEST(LoopNest, reorderNestedLoops4D) {
  // 定义测试用例函数，用于测试四重嵌套循环的重新排序功能

  // 创建名为 A 的缓冲区对象，维度为 {20, 30, 40, 50}，元素类型为整型
  BufHandle a_buf("A", {20, 30, 40, 50}, kInt);

  // 创建整型变量 i、j、k、l 对应的变量句柄
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle l("l", kInt);

  // 创建存储操作，将乘法表达式 i * j * k * l * 500 存储到缓冲区 a_buf 的 {i, j, k, l} 位置
  auto store = Store::make(
      a_buf,
      {i, j, k, l},
      Mul::make(Mul::make(Mul::make(Mul::make(i, j), k), l), 500));

  // 创建 l 循环，范围从 0 到 50，循环体为 store 操作
  auto forL = For::make(l, 0, 50, store);

  // 创建 k 循环，范围从 0 到 40，循环体为 forL 循环
  auto forK = For::make(k, 0, 40, forL);

  // 创建 j 循环，范围从 0 到 30，循环体为 forK 循环
  auto forJ = For::make(j, 0, 30, forK);

  // 创建 i 循环，范围从 0 到 20，循环体为 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);

  // 创建包含 forI 循环的块
  auto par = Block::make({forI});

  // 对循环嵌套进行重新排序，顺序为 {2, 0, 3, 1}
  auto reordered = LoopNest::reorder({forI, forJ, forK, forL}, {2, 0, 3, 1});

  // 断言重新排序后的循环顺序
  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forI);
  ASSERT_EQ(reordered[2], forL);
  ASSERT_EQ(reordered[3], forJ);

  // 断言重新排序后的循环是否完全嵌套
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forI, forL, forJ}));

  // 断言 forK 循环的父节点为 par
  ASSERT_EQ(forK->get_parent(), par);

  // 断言 store 操作的父节点为 forJ 循环的主体
  ASSERT_EQ(store->get_parent(), forJ->body());
}
TEST(LoopNest, reorderTrivialPermutation) {
  // 定义一个测试用例，验证简单的循环重排操作

  // Input IR: 输入的中间表示（IR）：
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }

  // 创建名为 A 的缓冲区，维度为 {20, 30, 40}，元素类型为 kInt
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  
  // 定义三个循环变量 i, j, k，类型均为 kInt
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  
  // 创建一个存储节点，将 i * j * k 的结果存储到缓冲区 a_buf 中的索引 {i, j, k} 处
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  
  // 创建循环节点 forK，循环变量 k 从 0 到 39，循环体为 store
  auto forK = For::make(k, 0, 40, store);
  
  // 创建循环节点 forJ，循环变量 j 从 0 到 29，循环体为 forK
  auto forJ = For::make(j, 0, 30, forK);
  
  // 创建循环节点 forI，循环变量 i 从 0 到 19，循环体为 forJ
  auto forI = For::make(i, 0, 20, forJ);
  
  // 创建一个块节点 par，包含循环节点 forI
  auto par = Block::make({forI});
  
  // 调用 LoopNest::reorder 函数，对 {forI, forJ, forK} 中的循环以 {0, 1, 2} 的顺序进行重排
  auto reordered = LoopNest::reorder({forI, forJ, forK}, {0, 1, 2});
  
  // 断言重排后的循环顺序与预期相符
  ASSERT_EQ(reordered[0], forI);
  ASSERT_EQ(reordered[1], forJ);
  ASSERT_EQ(reordered[2], forK);
  
  // 断言 forI, forJ, forK 三个循环是否严格嵌套
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));
  
  // 断言 store 的父节点为 forK 的循环体
  ASSERT_EQ(store->get_parent(), forK->body());
}

TEST(LoopNest, reorderInvalidPermutations) {
  // 定义一个测试用例，验证无效的循环重排操作

  // Input IR: 输入的中间表示（IR）：
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }

  // 创建名为 A 的缓冲区，维度为 {20, 30, 40}，元素类型为 kInt
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  
  // 定义三个循环变量 i, j, k，类型均为 kInt
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  
  // 创建一个存储节点，将 i * j * k 的结果存储到缓冲区 a_buf 中的索引 {i, j, k} 处
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  
  // 创建循环节点 forK，循环变量 k 从 0 到 39，循环体为 store
  auto forK = For::make(k, 0, 40, store);
  
  // 创建循环节点 forJ，循环变量 j 从 0 到 29，循环体为 forK
  auto forJ = For::make(j, 0, 30, forK);
  
  // 创建循环节点 forI，循环变量 i 从 0 到 19，循环体为 forJ
  auto forI = For::make(i, 0, 20, forJ);
  
  // 以下为一系列对 LoopNest::reorder 的调用，预期抛出特定的异常消息
  
  // 断言调用 LoopNest::reorder({forI, forJ, forK}, {0, 1, 2, 3}) 时抛出 "invalid permutation size" 异常
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {0, 1, 2, 3}),
      "invalid permutation size");
  
  // 断言调用 LoopNest::reorder({forI, forJ, forK}, {1, 2}) 时抛出 "invalid permutation size" 异常
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 2}),
      "invalid permutation size");
  
  // 断言调用 LoopNest::reorder({forI, forJ, forK}, {2, 1, 3}) 时抛出 "invalid permutation for reorder" 异常
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {2, 1, 3}),
      "invalid permutation for reorder");
  
  // 断言调用 LoopNest::reorder({forI, forJ, forK}, {1, 1, 0}) 时抛出 "invalid permutation for reorder" 异常
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 1, 0}),
      "invalid permutation for reorder");
  
  // 断言调用 LoopNest::reorder({forI, forJ, forK}, {0, 0, 0}) 时抛出 "invalid permutation for reorder" 异常
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {0, 0, 0}),
      "invalid permutation for reorder");
}
TEST(LoopNest, reorderInvalidLoopNest) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       A[i,j] = 0
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }

  // 创建名为 `A` 的缓冲区，维度为 {20, 30, 40}，数据类型为整型
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  // 创建整型变量 `i`
  VarHandle i("i", kInt);
  // 创建整型变量 `j`
  VarHandle j("j", kInt);
  // 创建整型变量 `k`
  VarHandle k("k", kInt);
  // 创建存储节点，存储表达式 `i * j * k` 到 `A[i,j,k]`
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  // 创建 `k` 的循环，范围为 [0, 40)，循环体为上述存储节点
  auto forK = For::make(k, 0, 40, store);
  // 创建 `j` 的循环，范围为 [0, 30)，循环体为上述 `forK`
  auto forJ = For::make(j, 0, 30, forK);
  // 创建 `i` 的循环，范围为 [0, 20)，循环体为上述 `forJ`
  auto forI = For::make(i, 0, 20, forJ);
  // 创建包含 `forI` 的块
  auto par = Block::make({forI});

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 对于 `forK`，这行代码是无效的，不会影响程序运行
  auto par = Block::make({forI});

  // 指定错误顺序的循环会失败
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forK, forI, forJ}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");

  // 将语句添加到 `forJ` 循环会失败
  auto init = Store::make(a_buf, {i}, 0);
  forJ->body()->insert_stmt_before(init, forK);
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");

  // 将语句从 `forJ` 移动到 `forI` 也会失败
  forJ->body()->remove_stmt(init);
  forI->body()->insert_stmt_before(init, forJ);
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");
}

TEST(LoopNest, compressBufferSimple) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }

  // 创建名为 `A` 的缓冲区，维度为 {100, 200}，数据类型为整型
  BufHandle aBuf("A", {100, 200}, kInt);
  // 创建名为 `B` 的缓冲区，维度为 {100, 200}，数据类型为整型
  BufHandle bBuf("B", {100, 200}, kInt);
  // 创建整型变量 `i`
  VarHandle i("i", kInt);
  // 创建整型变量 `j`
  VarHandle j("j", kInt);
  // 创建 `j` 的循环，范围为 [0, 200)，循环体为将 `sin(i * j)` 存储到 `A[i,j]`
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));
  // 创建 `j` 的循环，范围为 [0, 199)，循环体为将 `A[i,j] + A[i,j+1]` 存储到 `B[i,j]`
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));
  // 创建 `i` 的循环，范围为 [0, 100)，循环体包含 `forJ1` 和 `forJ2`
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  // 创建包含 `forI` 的块
  auto par = Block::make({forI});
  // 压缩缓冲区 `A`，以便于后续优化
  LoopNest::compressBuffer(aBuf.node(), par);

  // 创建输出流对象
  std::ostringstream oss;
  // 将 `par` 的内容输出到流中
  oss << *par;
  // 验证输出流内容是否符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[0, j]) + (A[0, j + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言缓冲区 `A` 的维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);
  // 断言缓冲区 `A` 的第一个维度值为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  // 断言缓冲区 `A` 的第二个维度值为 200
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}
TEST(LoopNest, compressBufferMultipleDims) {
  // 定义测试用例 `compressBufferMultipleDims`，测试多维缓冲区压缩

  // 定义多维缓冲区 A 和 B，分别为 100x200 和 100x200 的整型数组
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);

  // 定义循环变量 i 和 j
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  // 创建第一个存储操作，将 sin(i * j) 存入 A[i, j]
  auto store1 = Store::make(aBuf, {i, j}, sin(i * j));

  // 创建第二个存储操作，将 A[i, j] + A[i, j] 存入 B[i, j]
  auto store2 = Store::make(
      bBuf,
      {i, j},
      Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j})));

  // 创建内部循环 forJ，对 j 进行迭代，执行 store1 和 store2
  auto forJ = For::make(j, 0, 200, Block::make({store1, store2}));

  // 创建外部循环 forI，对 i 进行迭代，执行 forJ
  auto forI = For::make(i, 0, 100, forJ);

  // 创建并行块 par，包含 forI
  auto par = Block::make({forI});

  // 使用 LoopNest 类中的 compressBuffer 方法，压缩 aBuf 在 par 中的使用
  LoopNest::compressBuffer(aBuf.node(), par);

  // 将 par 转换为字符串
  std::ostringstream oss;
  oss << *par;

  // 定义 IR 校验模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, 0] =
# CHECK-NEXT: B[i, j] = (A[0, 0]) + (A[0, 0])
      )IR";

  // 使用 FileCheck 进行字符串校验
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);

  // 断言 aBuf 的第一个维度大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);

  // 断言 aBuf 的第二个维度大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);
}

TEST(LoopNest, compressBufferMultipleDims2) {
  // 定义测试用例 `compressBufferMultipleDims2`，测试多维缓冲区压缩（第二个例子）

  // 定义多维缓冲区 A 和 B，分别为 100x200x300 的整型数组
  BufHandle aBuf("A", {100, 200, 300}, kInt);
  BufHandle bBuf("B", {100, 200, 300}, kInt);

  // 定义循环变量 i、j 和 k
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  // 创建第一个存储操作，将 sin(i * j * k) 存入 A[i, j, k]
  auto store1 = Store::make(aBuf, {i, j, k}, sin(i * j * k));

  // 创建内部循环 forK1，对 k 进行迭代，执行 store1
  auto forK1 = For::make(k, 0, 300, store1);

  // 创建第二个存储操作，将 A[i, j, k] + A[i, j, k+1] 存入 B[i, j, k]
  auto store2 = Store::make(
      bBuf,
      {i, j, k},
      Add::make(Load::make(aBuf, {i, j, k}), Load::make(aBuf, {i, j, k + 1})));

  // 创建内部循环 forK2，对 k 进行迭代，执行 store2
  auto forK2 = For::make(k, 0, 299, store2);

  // 创建中间循环 forJ，对 j 进行迭代，执行 forK1 和 forK2
  auto forJ = For::make(j, 0, 200, Block::make({forK1, forK2}));

  // 创建外部循环 forI，对 i 进行迭代，执行 forJ
  auto forI = For::make(i, 0, 100, forJ);

  // 创建并行块 par，包含 forI
  auto par = Block::make({forI});

  // 使用 LoopNest 类中的 compressBuffer 方法，压缩 aBuf 在 par 中的使用
  LoopNest::compressBuffer(aBuf.node(), par);

  // 将 par 转换为字符串
  std::ostringstream oss;
  oss << *par;

  // 定义 IR 校验模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: for (int k
# CHECK-NEXT: A[0, 0, k] =
# CHECK: for (int k
# CHECK-NEXT: B[i, j, k] = (A[0, 0, k]) + (A[0, 0, k + 1])
      )IR";

  // 使用 FileCheck 进行字符串校验
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的维度为 3
  ASSERT_EQ(aBuf.node()->ndim(), 3);

  // 断言 aBuf 的第一个维度大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);

  // 断言 aBuf 的第二个维度大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);

  // 断言 aBuf 的第三个维度大小为 300
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(2), 300);
}
TEST(LoopNest, compressBufferDifferentOrderIndices) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[j, i] = sin(i*j)
  //   }
  //   for (int j = 0; j < 99; ++j) {
  //     B[i, j] = A[j, i] + A[j+1, 0]
  //   }
  // }

  // 创建名为 aBuf 的缓冲区对象，表示大小为 {100, 200} 的二维整数缓冲区
  BufHandle aBuf("A", {100, 200}, kInt);
  // 创建名为 bBuf 的缓冲区对象，表示大小为 {100, 200} 的二维整数缓冲区
  BufHandle bBuf("B", {100, 200}, kInt);
  // 创建整数变量 i 的句柄对象
  VarHandle i("i", kInt);
  // 创建整数变量 j 的句柄对象
  VarHandle j("j", kInt);
  // 生成第一个内部循环的 IR 表示，存储 sin(i * j) 到缓冲区 aBuf 的 {j, i} 位置
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {j, i}, sin(i * j)));
  // 生成第二个内部循环的 IR 表示，计算并存储 A[j, i] + A[j + 1, i] 到缓冲区 bBuf 的 {i, j} 位置
  auto forJ2 = For::make(
      j,
      0,
      99,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {j, i}), Load::make(aBuf, {j + 1, i}))));
  // 生成外部循环的 IR 表示，包含前面两个内部循环
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  // 创建包含外部循环的 IR 表示的块对象
  auto par = Block::make({forI});
  // 调用 LoopNest 类的 compressBuffer 方法，压缩缓冲区 aBuf 的节点，使用 par 作为上下文
  LoopNest::compressBuffer(aBuf.node(), par);

  // 创建输出流 oss
  std::ostringstream oss;
  // 将 par 的内容转换为字符串输出到 oss 中
  oss << *par;
  // 定义 IR 字符串模式用于验证输出结果
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[j, 0] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[j, 0]) + (A[j + 1, 0])
      )IR";
  // 使用 FileCheck 工具验证 oss 中的输出是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);
  // 断言 aBuf 的第一维大小为 100
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 100);
  // 断言 aBuf 的第二维大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);
}

TEST(LoopNest, compressBufferVariableBounds) {
  // Input IR:
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < N-1; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }

  // 创建名为 aBuf 的缓冲区对象，表示大小为 {100, 200} 的二维整数缓冲区
  BufHandle aBuf("A", {100, 200}, kInt);
  // 创建名为 bBuf 的缓冲区对象，表示大小为 {100, 200} 的二维整数缓冲区
  BufHandle bBuf("B", {100, 200}, kInt);
  // 创建整数变量 i 的句柄对象
  VarHandle i("i", kInt);
  // 创建整数变量 j 的句柄对象
  VarHandle j("j", kInt);
  // 创建整数变量 M 的句柄对象
  VarHandle M("M", kInt);
  // 创建整数变量 N 的句柄对象
  VarHandle N("N", kInt);
  // 生成第一个内部循环的 IR 表示，存储 sin(i * j) 到缓冲区 aBuf 的 {i, j} 位置
  auto forJ1 = For::make(j, 0, N, Store::make(aBuf, {i, j}, sin(i * j)));
  // 生成第二个内部循环的 IR 表示，计算并存储 A[i, j] + A[i, j + 1] 到缓冲区 bBuf 的 {i, j} 位置
  auto forJ2 = For::make(
      j,
      0,
      N - 1,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));
  // 生成外部循环的 IR 表示，包含前面两个内部循环
  auto forI = For::make(i, 0, M, Block::make({forJ1, forJ2}));
  // 创建包含外部循环的 IR 表示的块对象
  auto par = Block::make({forI});
  // 调用 LoopNest 类的 compressBuffer 方法，压缩缓冲区 aBuf 的节点，使用 par 作为上下文
  LoopNest::compressBuffer(aBuf.node(), par);

  // 创建输出流 oss
  std::ostringstream oss;
  // 将 par 的内容转换为字符串输出到 oss 中
  oss << *par;
  // 定义 IR 字符串模式用于验证输出结果
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[0, j]) + (A[0, j + 1])
      )IR";
  // 使用 FileCheck 工具验证 oss 中的输出是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);
  // 断言 aBuf 的第一维大小为 1
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  // 断言 aBuf 的第二维大小为 200
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}
TEST(LoopNest, compressBufferNoCommonParentLoops) {
  // 声明名为 compressBufferNoCommonParentLoops 的测试用例

  // 定义一个二维缓冲区 aBuf，名为 "A"，大小为 {100, 200}，元素类型为整数
  BufHandle aBuf("A", {100, 200}, kInt);

  // 定义一个二维缓冲区 bBuf，名为 "B"，大小为 {100, 200}，元素类型为整数
  BufHandle bBuf("B", {100, 200}, kInt);

  // 声明一个名为 i 的整数变量
  VarHandle i("i", kInt);

  // 声明一个名为 j 的整数变量
  VarHandle j("j", kInt);

  // 创建一个循环，对变量 j 进行迭代，范围是 [0, 200)，并在每次迭代中将 sin(i * j) 存储到 aBuf 的对应位置
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));

  // 创建一个循环，对变量 j 进行迭代，范围是 [0, 199)，并在每次迭代中将 A[i,j] + A[i,j+1] 存储到 bBuf 的对应位置
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));

  // 创建一个循环，对变量 i 进行迭代，范围是 [0, 100)，在每次迭代中执行 forJ1 的循环体
  auto forI1 = For::make(i, 0, 100, forJ1);

  // 创建一个循环，对变量 i 进行迭代，范围是 [0, 100)，在每次迭代中执行 forJ2 的循环体
  auto forI2 = For::make(i, 0, 100, forJ2);

  // 创建一个代码块，包含 forI1 和 forI2 循环
  auto par = Block::make({forI1, forI2});

  // 调用 LoopNest::compressBuffer 方法，压缩缓冲区 aBuf 的节点，传入 par 作为压缩的范围
  LoopNest::compressBuffer(aBuf.node(), par);

  // 创建一个字符串流 oss，将 par 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *par;

  // 定义一个字符串，包含了 par 输出内容的验证模式，用于检查输出是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: B[i, j] = (A[i, j]) + (A[i, j + 1])
      )IR";

  // 使用 FileCheck 进行模式匹配，验证输出是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的节点维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);

  // 断言 aBuf 的节点第一维大小为 100
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 100);

  // 断言 aBuf 的节点第二维大小为 200
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}

TEST(LoopNest, compressBufferIndicesMixed) {
  // 声明名为 compressBufferIndicesMixed 的测试用例

  // 定义一个二维缓冲区 aBuf，名为 "A"，大小为 {300, 200}，元素类型为整数
  BufHandle aBuf("A", {300, 200}, kInt);

  // 定义一个二维缓冲区 bBuf，名为 "B"，大小为 {100, 200}，元素类型为整数
  BufHandle bBuf("B", {100, 200}, kInt);

  // 声明一个名为 i 的整数变量
  VarHandle i("i", kInt);

  // 声明一个名为 j 的整数变量
  VarHandle j("j", kInt);

  // 创建一个循环，对变量 j 进行迭代，范围是 [0, 200)，并在每次迭代中将 sin(i * j) 存储到 aBuf 的对应位置
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i + j, j}, sin(i * j)));

  // 创建一个循环，对变量 j 进行迭代，范围是 [0, 199)，并在每次迭代中将 A[i+j,j] + A[i+j,j+1] 存储到 bBuf 的对应位置
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(
              Load::make(aBuf, {i + j, j}), Load::make(aBuf, {i + j, j + 1}))));

  // 创建一个循环，对变量 i 进行迭代，范围是 [0, 100)，在每次迭代中执行 forJ1 和 forJ2 的循环体
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));

  // 创建一个代码块，包含 forI 循环
  auto par = Block::make({forI});

  // 调用 LoopNest::compressBuffer 方法，压缩缓冲区 aBuf 的节点，传入 par 作为压缩的范围
  LoopNest::compressBuffer(aBuf.node(), par);

  // 创建一个字符串流 oss，将 par 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *par;

  // 定义一个字符串，包含了 par 输出内容的验证模式，用于检查输出是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i + j, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[i + j, j]) + (A[i + j, j + 1])
      )IR";

  // 使用 FileCheck 进行模式匹配，验证输出是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言 aBuf 的节点维度为 2
  ASSERT_EQ(aBuf.node()->ndim(), 2);

  // 断言 aBuf 的节点第一维大小为 300
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 300);

  // 断言 aBuf 的节点第二维大小为 200
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}
TEST(LoopNest, compressMultipleBuffers) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int k = 0; k < 199; ++k) {
  //     B[i,k] = A[i,k] + A[i, k+1]
  //   }
  //   for (int m = 0; m < 50; ++m) {
  //     C[i,m] = B[i,m]
  //   }
  // }

  // 创建三个缓冲区对象，分别表示数组 A, B, C，每个数组大小为 100x200 的整数类型
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  BufHandle cBuf("C", {100, 200}, kInt);

  // 定义四个循环变量 i, j, k, m，都是整数类型
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);

  // 创建表示内部循环的表达式，分别用于数组 A, B, C 的填充
  auto forJ = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));
  auto forK = For::make(
      k,
      0,
      199,
      Store::make(
          bBuf,
          {i, k},
          Add::make(Load::make(aBuf, {i, k}), Load::make(aBuf, {i, k + 1}))));
  auto forM =
      For::make(m, 0, 50, Store::make(cBuf, {i, m}, Load::make(bBuf, {i, m})));

  // 创建表示外部循环的表达式，将内部循环组合在一起
  auto forI = For::make(i, 0, 100, Block::make({forJ, forK, forM}));
  auto par = Block::make({forI});

  // 压缩所有缓冲区 A, B, C 的维度
  LoopNest::compressAllBuffers(par);

  // 将 par 对象的内容转换为字符串
  std::ostringstream oss;
  oss << *par;

  // 验证字符串 oss 是否符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int k
# CHECK-NEXT: B[0, k] = (A[0, k]) + (A[0, k + 1])
# CHECK: for (int m
# CHECK-NEXT: C[0, 0] = B[0, m]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 断言每个缓冲区的维度是否符合预期
  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
  ASSERT_EQ(bBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, bBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, bBuf.node()->dim(1), 200);
  ASSERT_EQ(cBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, cBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, cBuf.node()->dim(1), 1);
}

TEST(LoopNest, sanitizeNames) {
  std::vector<ExprHandle> dim_args;

  // 创建表达式向量 dim_args，包含两个带有特殊命名的变量
  // 第一个变量名为 "i"，第二个变量名为 "N:2"
  dim_args.emplace_back(ExprHandle(alloc<Var>("i", kInt)));
  dim_args.emplace_back(ExprHandle(alloc<Var>("N:2", kInt)));

  // 创建十个维度为 "N" 的变量并添加到 dim_args 中
  for (int i = 0; i < 10; i++) {
    dim_args.emplace_back(ExprHandle(alloc<Var>("N", kInt)));
  }

  // 创建一个 Tensor X，并使用 Lambda 表达式定义其维度及内容
  Tensor X = Compute("$X:!", dim_args, [&](const std::vector<VarHandle>& v) {
    // 返回计算结果，将向量 v 中的第一个、第二个和第九个元素相加，并加一
    return v[0] + v[1] + v[9] + 1;
  });
  // 使用 Reduce 函数对张量 X 进行降维操作，按照 Sum 函数计算每个维度的总和
  Tensor Y = Reduce(
      "%X\"+",
      {}, // 空的维度参数列表，表示对所有维度进行操作
      Sum(), // 使用 Sum() 函数进行求和操作
      [&](const std::vector<VarHandle>& v) { return X.load(v); }, // 从张量 X 中加载对应维度的数据
      dim_args); // 维度参数

  // 最后，验证经过处理后的结果：
  LoopNest l({X, Y}); // 创建 LoopNest 对象，包含张量 X 和 Y
  StmtPtr s = l.root_stmt(); // 获取 LoopNest 树结构的根语句
  LoopNest::sanitizeNames(s); // 对树结构中的变量名进行清理和重命名

  std::ostringstream oss; // 创建字符串输出流
  oss << *s; // 将根语句 s 输出到字符串流中
  const std::string& verification_pattern =
      R"IR(
// CHECK:  for (int i = 0; i < i_1; i++) {
// CHECK-NEXT:    for (int j = 0; j < N_2_1; j++) {
// CHECK-NEXT:      for (int k = 0; k < N_9; k++) {
// CHECK-NEXT:        for (int l = 0; l < N_8; l++) {
// CHECK-NEXT:          for (int m = 0; m < N_7; m++) {
// CHECK-NEXT:            for (int n = 0; n < N_6; n++) {
// CHECK-NEXT:              for (int o = 0; o < N_5; o++) {
// CHECK-NEXT:                for (int p = 0; p < N_4; p++) {
// CHECK-NEXT:                  for (int i1 = 0; i1 < N_3; i1++) {
// CHECK-NEXT:                    for (int j1 = 0; j1 < N_2; j1++) {
// CHECK-NEXT:                      for (int k1 = 0; k1 < N_1; k1++) {
// CHECK-NEXT:                        for (int l1 = 0; l1 < N; l1++) {
// CHECK-NEXT:                          // 在多重循环中，计算并填充 v_X__ 数组的元素，按照给定的索引和算术运算。
// CHECK-NEXT:                          v_X__[i, j, k, l, m, n, o, p, i1, j1, k1, l1] = ((i + j) + j1) + 1;
// CHECK:  v_X___1 = int(0);
// CHECK-NEXT:  for (int i_2 = 0; i_2 < i_1; i_2++) {
// CHECK-NEXT:    for (int j_1 = 0; j_1 < N_2_1; j_1++) {
// CHECK-NEXT:      for (int k_1 = 0; k_1 < N_9; k_1++) {
// CHECK-NEXT:        for (int l_1 = 0; l_1 < N_8; l_1++) {
// CHECK-NEXT:          for (int m_1 = 0; m_1 < N_7; m_1++) {
// CHECK-NEXT:            for (int n_1 = 0; n_1 < N_6; n_1++) {
// CHECK-NEXT:              for (int o_1 = 0; o_1 < N_5; o_1++) {
// CHECK-NEXT:                for (int p_1 = 0; p_1 < N_4; p_1++) {
// CHECK-NEXT:                  for (int i1_1 = 0; i1_1 < N_3; i1_1++) {
// CHECK-NEXT:                    for (int j1_1 = 0; j1_1 < N_2; j1_1++) {
// CHECK-NEXT:                      for (int k1_1 = 0; k1_1 < N_1; k1_1++) {
// CHECK-NEXT:                        for (int l1_1 = 0; l1_1 < N; l1_1++) {
// CHECK-NEXT:                          // 在多重循环中，使用 ReduceOp 函数对 v_X__ 中的元素进行归约操作。
// CHECK-NEXT:                          v_X___1 = ReduceOp((v_X___1) + (v_X__[i_2, j_1, k_1, l_1, m_1, n_1, o_1, p_1, i1_1, j1_1, k1_1, l1_1]), reduce_args={i_2, j_1, k_1, l_1, m_1, n_1, o_1, p_1, i1_1, j1_1, k1_1, l1_1});
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
```