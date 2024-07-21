# `.\pytorch\test\cpp\tensorexpr\test_boundsinference.cpp`

```py
// 包含必要的头文件来进行测试和运行时支持
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// 包含 Google Test 框架的头文件，用于编写和运行测试用例
#include <gtest/gtest.h>

// 包含与 TensorExpr 库相关的自定义头文件
#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// 静态函数：验证常量边界信息是否与参考值匹配
static void verifyConstBounds(
    const TensorAccessBoundsInfo& access_info,
    const std::vector<std::pair<int, int>>& ref) {
  // 获取参考值的维度大小
  size_t ndim = ref.size();
  // 断言实际起始点和停止点的大小与参考值的维度大小一致
  ASSERT_EQ(access_info.start.size(), ndim);
  ASSERT_EQ(access_info.stop.size(), ndim);
  // 遍历每个维度的边界信息
  for (const auto i : c10::irange(ndim)) {
    // 如果参考值中的起始点为非负数，则检查起始点是否为常量
    if (ref[i].first >= 0) {
      ASSERT_TRUE(access_info.start[i]->isConstant());
      // 获取起始点的具体值并与参考值进行比较
      int start_i = immediateAs<int>(access_info.start[i]);
      ASSERT_EQ(start_i, ref[i].first);
    }
    // 如果参考值中的停止点为非负数，则检查停止点是否为常量
    if (ref[i].second >= 0) {
      ASSERT_TRUE(access_info.stop[i]->isConstant());
      // 获取停止点的具体值并与参考值进行比较
      int stop_i = immediateAs<int>(access_info.stop[i]);
      ASSERT_EQ(stop_i, ref[i].second);
    }
  }
}

// 定义 BoundsInference 测试用例，用于测试边界推断功能
TEST(BoundsInference, _1) {
  // 验证边界推断是否能正确处理以下示例：
  // for i in 0..100:
  //   b[i] = a[i]
  // 对于此循环，边界推断应该得出以下结果：
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 99}}

  // 创建表示循环上界的表达式
  ExprHandle n(100);
  // 创建表示缓冲区 a 的句柄，并指定数据类型为 kFloat
  BufHandle a("a", {n}, kFloat);
  // 使用 Compute 函数创建张量 b，并定义其计算逻辑
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  // 创建 LoopNest 对象，并将张量 b 添加到其中
  LoopNest l({b});
  // 执行边界推断并获取结果
  auto bounds_info = inferBounds(l.root_stmt());

  // 预期结果应包含两个条目：一个针对 'b' 和一个针对 'a'
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  // 验证缓冲区 a 的边界信息是否与预期匹配
  verifyConstBounds(bounds_info.at(a.node())[0], {{0, 99}});

  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
  // 验证张量 b 的边界信息是否与预期匹配
  verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 99}});
}

} // namespace jit
} // namespace torch
TEST(BoundsInference, _4) {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..200:
  //   for x in 0..320:
  //     c[y,x] = a[y,x] * b[y,x]
  ExprHandle W(320);  // 定义常量 W = 320，表示矩阵的宽度
  ExprHandle H(200);  // 定义常量 H = 200，表示矩阵的高度
  BufHandle a("a", {H, W}, kFloat);  // 创建缓冲区 a，大小为 H x W，存储浮点数
  Tensor b = Compute("b", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return x * y;  // 计算 b[y,x] = x * y
  });
  Tensor c = Compute("c", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return a.load(y, x) * b.load(y, x);  // 计算 c[y,x] = a[y,x] * b[y,x]
  });
  LoopNest l({c});  // 创建一个循环嵌套对象，包含计算张量 c 的循环结构
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);  // 获取计算张量 c 的循环语句列表
  StmtPtr body = l.getLoopBodyFor(c);  // 获取计算张量 c 的循环体

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);  // 推断顶层循环作用域的边界信息
    ASSERT_EQ(bounds_info.size(), 3);  // 断言边界信息包含三个条目

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);  // 断言缓冲区 a 的边界信息条目数量为1
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);  // 断言 a 的第一个边界信息类型为加载
    verifyConstBounds(bounds_info.at(a.node())[0], {{0, 199}, {0, 319}});  // 验证 a 的边界为 {{0, 199}, {0, 319}}

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);  // 断言张量 b 的边界信息条目数量为1
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);  // 断言 b 的第一个边界信息类型为加载
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 199}, {0, 319}});  // 验证 b 的边界为 {{0, 199}, {0, 319}}


这段代码用于测试边界推断功能，验证对于给定的循环嵌套结构，能够正确推断出每个缓冲区或张量的访问边界范围。
  // 断言检查缓冲区 c 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
  // 断言检查缓冲区 c 的第一个边界信息的类型是否为 kStore
  ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
  // 验证常量边界，确保缓冲区 c 的第一个边界信息符合期望的边界范围 {{0, 199}, {0, 319}}
  verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 199}, {0, 319}});
}
{
  // 推断内部循环作用域的边界
  auto bounds_info = inferBounds(loops[1]);
  // 断言检查 bounds_info 的大小是否为 3
  ASSERT_EQ(bounds_info.size(), 3);

  // 断言检查节点 a 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  // 断言检查节点 a 的第一个边界信息的类型是否为 kLoad
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  // 验证常量边界，确保节点 a 的第一个边界信息符合期望的边界范围 {{-1, -1}, {0, 319}}
  verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {0, 319}});

  // 断言检查缓冲区 b 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  // 断言检查缓冲区 b 的第一个边界信息的类型是否为 kLoad
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
  // 验证常量边界，确保缓冲区 b 的第一个边界信息符合期望的边界范围 {{-1, -1}, {0, 319}}
  verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {0, 319}});

  // 断言检查缓冲区 c 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
  // 断言检查缓冲区 c 的第一个边界信息的类型是否为 kStore
  ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
  // 验证常量边界，确保缓冲区 c 的第一个边界信息符合期望的边界范围 {{-1, -1}, {0, 319}}
  verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {0, 319}});
}
{
  // 推断内部循环体作用域的边界
  auto bounds_info = inferBounds(body);
  // 断言检查 bounds_info 的大小是否为 3
  ASSERT_EQ(bounds_info.size(), 3);

  // 断言检查节点 a 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  // 断言检查节点 a 的第一个边界信息的类型是否为 kLoad
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  // 验证常量边界，确保节点 a 的第一个边界信息符合期望的边界范围 {{-1, -1}, {-1, -1}}
  verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {-1, -1}});

  // 断言检查缓冲区 b 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  // 断言检查缓冲区 b 的第一个边界信息的类型是否为 kLoad
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
  // 验证常量边界，确保缓冲区 b 的第一个边界信息符合期望的边界范围 {{-1, -1}, {-1, -1}}
  verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {-1, -1}});

  // 断言检查缓冲区 c 的边界信息集合的大小是否为 1
  ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
  // 断言检查缓冲区 c 的第一个边界信息的类型是否为 kStore
  ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
  // 验证常量边界，确保缓冲区 c 的第一个边界信息符合期望的边界范围 {{-1, -1}, {-1, -1}}
  verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {-1, -1}});
}
TEST(BoundsInference, _6) {
  // 验证边界推断是否适用于以下示例：
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..20:
  //   for x in 0..32:
  //     c[y,x] = a[y+100,x+100] * b[y*2,x*5]

  // 定义表达式常量
  ExprHandle W(320);
  ExprHandle H(200);
  ExprHandle CW(32);
  ExprHandle CH(20);

  // 定义缓冲区变量 a，表示一个二维数组
  BufHandle a("a", {H, W}, kFloat);

  // 定义张量 b，根据二重循环计算结果
  Tensor b = Compute("b", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return x * y;
  });

  // 定义张量 c，根据复杂的索引和计算公式生成结果
  Tensor c = Compute("c", {CH, CW}, [&](const VarHandle& y, const VarHandle& x) {
    return a.load(y + 100, x + 100) * b.load(y * 2, x * 5);
  });

  // 创建循环嵌套对象，并传入张量 c
  LoopNest l({c});

  // 获取张量 c 对应的循环语句
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);

  // 获取张量 c 的循环体语句
  StmtPtr body = l.getLoopBodyFor(c);

  {
    // 推断顶层循环作用域的边界信息
    auto bounds_info = inferBounds(loops[0]);

    // 断言：边界信息的大小为 3
    ASSERT_EQ(bounds_info.size(), 3);

    // 断言：对于缓冲区变量 a 的边界信息
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    // 验证边界：{{100, 119}, {100, 131}} 表示 y 的范围是 [100, 119]，x 的范围是 [100, 131]
    verifyConstBounds(bounds_info.at(a.node())[0], {{100, 119}, {100, 131}});

    // 断言：对于张量 b 的边界信息
    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    // 验证边界：张量 b 的边界与 a 保持一致，但是这里并未完全展示
  }
}
    // 验证常数边界，使用 bounds_info 中 b.buf() 的信息作为索引，验证其第一个元素是否在 {0, 38} 和 {0, 155} 的范围内
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 38}, {0, 155}});

    // 断言确保 bounds_info 中 c.buf() 的大小为 1
    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    // 断言确保 bounds_info 中 c.buf() 的第一个元素的类型为 kStore
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    // 验证常数边界，使用 bounds_info 中 c.buf() 的信息作为索引，验证其第一个元素是否在 {0, 19} 和 {0, 31} 的范围内
    verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 19}, {0, 31}});
  }
  {
    // 在内部循环作用域推断边界
    auto bounds_info = inferBounds(loops[1]);
    // 断言确保 bounds_info 的大小为 3
    ASSERT_EQ(bounds_info.size(), 3);

    // 断言确保 bounds_info 中 a.node() 的大小为 1
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    // 断言确保 bounds_info 中 a.node() 的第一个元素的类型为 kLoad
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    // 验证常数边界，使用 bounds_info 中 a.node() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 和 {100, 131} 的范围内
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {100, 131}});

    // 断言确保 bounds_info 中 b.buf() 的大小为 1
    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    // 断言确保 bounds_info 中 b.buf() 的第一个元素的类型为 kLoad
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    // 验证常数边界，使用 bounds_info 中 b.buf() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 和 {0, 155} 的范围内
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {0, 155}});

    // 断言确保 bounds_info 中 c.buf() 的大小为 1
    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    // 断言确保 bounds_info 中 c.buf() 的第一个元素的类型为 kStore
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    // 验证常数边界，使用 bounds_info 中 c.buf() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 和 {0, 31} 的范围内
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {0, 31}});
  }
  {
    // 在内部循环体的作用域推断边界
    auto bounds_info = inferBounds(body);
    // 断言确保 bounds_info 的大小为 3
    ASSERT_EQ(bounds_info.size(), 3);

    // 断言确保 bounds_info 中 a.node() 的大小为 1
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    // 断言确保 bounds_info 中 a.node() 的第一个元素的类型为 kLoad
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    // 验证常数边界，使用 bounds_info 中 a.node() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 的范围内
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {-1, -1}});

    // 断言确保 bounds_info 中 b.buf() 的大小为 1
    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    // 断言确保 bounds_info 中 b.buf() 的第一个元素的类型为 kLoad
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    // 验证常数边界，使用 bounds_info 中 b.buf() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 的范围内
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {-1, -1}});

    // 断言确保 bounds_info 中 c.buf() 的大小为 1
    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    // 断言确保 bounds_info 中 c.buf() 的第一个元素的类型为 kStore
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    // 验证常数边界，使用 bounds_info 中 c.buf() 的信息作为索引，验证其第一个元素是否在 {-1, -1} 的范围内
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {-1, -1}});
  }
TEST(BoundsInference, MultipleTopLoopLoad) {
  // 创建一个名为 a 的缓冲区，大小为 100，存储类型为 kFloat
  BufHandle a("a", {100}, kFloat);
  // 创建张量 b，它从缓冲区 a 中加载数据，形状为 {64}
  Tensor b = Compute("b", {64}, [&](const VarHandle& x) { return a.load(x); });
  // 创建张量 c，它从缓冲区 a 中加载偏移后的数据，形状为 {32}
  Tensor c =
      Compute("c", {32}, [&](const VarHandle& x) { return a.load(x + 10); });
  // 创建张量 d，它从缓冲区 a 中加载另一个偏移后的数据，形状为 {96}
  Tensor d =
      Compute("d", {96}, [&](const VarHandle& x) { return a.load(x + 2); });
  // 创建循环嵌套 l，包含张量 b、c、d
  LoopNest l({b, c, d});

  // 推断根语句的边界信息
  auto bounds_info = inferBounds(l.root_stmt());

  // 断言边界信息的大小为 4
  ASSERT_EQ(bounds_info.size(), 4);

  // 对缓冲区 a 进行读取操作的断言
  {
    auto bounds = bounds_info[a.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kLoad);
    // 边界断言：
    // 起始：三个加载边界的最小值 = 循环起始的最小值 + 偏移量 = 0+0 (b)。
    // 结束：三个加载边界的最大值 = 循环结束的最大值 + 偏移量 - 1 = 96 + 2 - 1 (d)。
  // 验证边界常数是否符合预期，这里验证的是变量 bound，期望其在范围 [0, 97] 内。
  verifyConstBounds(bound, {{0, 97}});
}

// 只写入 b, c, d。
{
  // 获取 b 的边界信息
  auto bounds = bounds_info[b.buf()];
  // 断言 b 的边界信息列表长度为 1
  ASSERT_EQ(bounds.size(), 1);
  // 获取 b 的第一个边界信息
  auto bound = bounds[0];
  // 断言 b 的第一个边界信息类型为 TensorAccessKind::kStore
  ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
  // 只验证变量 b 的循环范围
  verifyConstBounds(bound, {{0, 63}});
}
{
  // 获取 c 的边界信息
  auto bounds = bounds_info[c.buf()];
  // 断言 c 的边界信息列表长度为 1
  ASSERT_EQ(bounds.size(), 1);
  // 获取 c 的第一个边界信息
  auto bound = bounds[0];
  // 断言 c 的第一个边界信息类型为 TensorAccessKind::kStore
  ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
  // 只验证变量 c 的循环范围
  verifyConstBounds(bound, {{0, 31}});
}
{
  // 获取 d 的边界信息
  auto bounds = bounds_info[d.buf()];
  // 断言 d 的边界信息列表长度为 1
  ASSERT_EQ(bounds.size(), 1);
  // 获取 d 的第一个边界信息
  auto bound = bounds[0];
  // 断言 d 的第一个边界信息类型为 TensorAccessKind::kStore
  ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
  // 只验证变量 d 的循环范围
  verifyConstBounds(bound, {{0, 95}});
}
}

TEST(BoundsInference, MultipleTopLoopStore) {
  // 创建四个缓冲区对象，各自表示名为a, b, c, d的缓冲区，每个大小为100个元素，元素类型为float
  BufHandle a("a", {100}, kFloat);
  BufHandle b("b", {100}, kFloat);
  BufHandle c("c", {100}, kFloat);
  BufHandle d("d", {100}, kFloat);
  // 创建一个名为x的整数变量
  VarHandle x("x", kInt);

  // 生成一个语句对象，包含三个嵌套的For循环，每个循环内部有一个Store操作
  StmtPtr stmt = Block::make(
      {For::make(x, 0, 64, Store::make(b, {x}, Load::make(a, {x}))),
       For::make(x, 0, 32, Store::make(c, {x + 10}, Load::make(a, {x}))),
       For::make(x, 0, 96, Store::make(d, {x + 2}, Load::make(a, {x})))});

  // 推断该语句中的边界信息
  auto bounds_info = inferBounds(stmt);

  // 断言边界信息的大小为4
  ASSERT_EQ(bounds_info.size(), 4);

  // 对缓冲区a进行边界信息验证
  // a只被读取
  {
    auto bounds = bounds_info[a.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kLoad);
    // 边界：没有偏移，因此这只是最大循环边界
    verifyConstBounds(bound, {{0, 95}});
  }

  // 对缓冲区b进行边界信息验证
  // b, c, d只被写入
  {
    auto bounds = bounds_info[b.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // 这应等同于b循环的{偏移，偏移 + 范围}。
    // b循环没有偏移，因此仅为循环范围。
    verifyConstBounds(bound, {{0, 63}});
  }
  {
    auto bounds = bounds_info[c.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // 这应等同于c循环的{偏移，偏移 + 范围}。
    // 偏移为10，范围为32-1。
    verifyConstBounds(bound, {{10, 41}});
  }
  {
    auto bounds = bounds_info[d.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // 这应等同于d循环的{偏移，偏移 + 范围}。
    // 偏移为2，范围为96-1。
    verifyConstBounds(bound, {{2, 97}});
  }
}

TEST(BoundsInference, CacheReads) {
  // 创建一个64x64的Tensor A，通过lambda表达式初始化
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  // 创建一个20x10的Tensor B，通过lambda表达式初始化，依赖于Tensor A的加载操作
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 3);
      });
  // 创建一个20x10的Tensor C，通过lambda表达式初始化，依赖于两个Tensor A的加载操作
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  // 将Tensor B和C放入循环嵌套对象中
  LoopNest l({B, C});
  // 在Tensor A的特定循环中缓存访问
  StmtPtr j_loop = l.getLoopStmtsFor(B)[1];
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);

  // 推断循环嵌套对象中的边界信息，缓存访问后
  auto bounds_info_after = inferBounds(l.root_stmt());

  // 缓存访问不应改变现有边界，但应为缓存添加一个新的边界信息
  for (auto& pair : bounds_info_after) {
    auto beforeIt = bounds_info_before.find(pair.first);
    if (beforeIt != bounds_info_before.end()) {
      // 如果 `beforeIt` 不等于 `bounds_info_before` 的末尾迭代器，则执行以下代码块
      // 断言：TensorAccessBoundInfos 的数量相同
      ASSERT_EQ(pair.second.size(), beforeIt->second.size());

      // 遍历 `pair.second` 中的每个元素
      for (const auto i : c10::irange(pair.second.size())) {
        // 获取当前迭代中的 TensorAccessBoundsInfo 对象 `after` 和 `before`
        TensorAccessBoundsInfo& after = pair.second[i];
        TensorAccessBoundsInfo& before = beforeIt->second[i];

        // 断言：维度数量相同
        ASSERT_EQ(before.start.size(), after.start.size());

        // 比较边界是否相等
        for (const auto j : c10::irange(before.start.size())) {
          ASSERT_TRUE(exprEquals(before.start[j], after.start[j]));
          ASSERT_TRUE(exprEquals(before.stop[j], after.stop[j]));
        }
      }
    } else {
      // 如果 `beforeIt` 等于 `bounds_info_before` 的末尾迭代器，则执行以下代码块
      // 断言：pair 的第一个元素的名字提示为 "A_local"
      ASSERT_EQ(pair.first->name_hint(), "A_local");

      // 断言：pair.second 的大小为 2，即应该同时有加载和存储操作
      ASSERT_EQ(pair.second.size(), 2);
      TensorAccessBoundsInfo& first = pair.second[0];
      TensorAccessBoundsInfo& second = pair.second[1];

      // 断言：第一个和第二个 TensorAccessBoundsInfo 的类型不同
      ASSERT_NE(first.kind, second.kind);

      // 断言：维度数量为 2
      ASSERT_EQ(first.start.size(), second.start.size());
      ASSERT_EQ(first.start.size(), 2);

      // 比较加载和存储的边界是否相等
      for (const auto j : c10::irange(first.start.size())) {
        ASSERT_TRUE(exprEquals(first.start[j], second.start[j]));
        ASSERT_TRUE(exprEquals(first.stop[j], second.stop[j]));
      }
    }
  }
TEST(BoundsInference, Flattened) {
  // 创建一个张量 b，维度为 {3, 4, 5}，计算方式为 x * y + z
  Tensor b = Compute(
      "b",
      {3, 4, 5},
      [&](const VarHandle& z, const VarHandle& y, const VarHandle& x) {
        return x * y + z;
      });

  // 创建一个循环嵌套对象 l，包含张量 b
  LoopNest l({b});
  // 准备进行代码生成的前期工作
  l.prepareForCodegen();
  // 推断根语句的边界信息
  auto bounds_info = inferBounds(l.root_stmt());

  // 只有一个缓冲区
  ASSERT_EQ(bounds_info.size(), 1);
  // 获取缓冲区的边界信息
  auto& TABI = bounds_info[b.buf()][0];
  // 断言是存储类型的张量访问
  ASSERT_EQ(TABI.kind, TensorAccessKind::kStore);
  // 断言扁平化的边界应该只有一个维度
  ASSERT_EQ(TABI.start.size(), 1);
  ASSERT_EQ(TABI.stop.size(), 1);

  // 断言边界应该是 0 -> (3*4*5)-1
  ASSERT_TRUE(exprEquals(TABI.start[0], alloc<IntImm>(0)));
  ASSERT_TRUE(exprEquals(TABI.stop[0], alloc<IntImm>(3 * 4 * 5 - 1)));
}

TEST(BoundsInference, GetPotentialHazards) {
  // 创建缓冲区和变量
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;

  {
    /*
     * A[0] = B[0];
     * B[0] = 3;      B 上的 WAR
     * A[0] = B[0];   A 上的 WAW，B 上的 RAW
     * C[0] = 5;
     */

    // 创建存储操作
    StorePtr store1 = Store::make(a, {0}, Load::make(b, {0}));
    StorePtr store2 = Store::make(b, {0}, 3);
    StorePtr store3 = Store::make(a, {0}, Load::make(b, {0}));
    StorePtr store4 = Store::make(c, {0}, 5);
    // 创建语句块包含这些存储操作
    StmtPtr stmt = Block::make({store1, store2, store3, store4});

    // 创建内存依赖分析器
    MemDependencyChecker analyzer;
    // 接受语句块并分析内存依赖
    stmt->accept(&analyzer);

    // 断言各种潜在冲突类型
    ASSERT_EQ(
        HazardKind::WriteAfterRead,
        getPotentialHazards(analyzer, store1, store2));

    ASSERT_EQ(
        HazardKind::ReadAfterWrite,
        getPotentialHazards(analyzer, store2, store3));

    ASSERT_EQ(
        HazardKind::WriteAfterWrite,
        getPotentialHazards(analyzer, store1, store3));

    // 第四个存储没有依赖关系
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store1, store4));
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store2, store4));
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store3, store4));
  }
}

TEST(BoundsInference, GetPotentialHazardsLoopNoHazard) {
  // 创建张量 A 和 B，使用 Lambda 表达式定义计算
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B = Compute("B", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return (i + 1) * (j + 1);
  });

  // 创建循环嵌套对象 l，包含张量 A 和 B
  LoopNest l({A, B});

  using namespace analysis;

  // 创建内存依赖分析器
  MemDependencyChecker analyzer;
  // 接受根语句并分析内存依赖
  l.root_stmt()->accept(&analyzer);

  // 获取 A 和 B 的循环根语句
  ForPtr loopRootA = l.getLoopStmtsFor(A)[0];
  ForPtr loopRootB = l.getLoopStmtsFor(B)[0];

  // 断言循环之间没有依赖关系
  ASSERT_EQ(
      HazardKind::NoDependency,
      getPotentialHazards(analyzer, loopRootA, loopRootB));
}

TEST(BoundsInference, GetPotentialHazardsLoopCall) {
  // 创建张量 A，使用 Lambda 表达式定义计算
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {

    return i * j;
  });

  // 创建循环嵌套对象 l，包含张量 A
  LoopNest l({A});

  using namespace analysis;

  // 创建内存依赖分析器
  MemDependencyChecker analyzer;
  // 接受根语句并分析内存依赖
  l.root_stmt()->accept(&analyzer);

  // 获取张量 A 的循环根语句
  ForPtr loopRootA = l.getLoopStmtsFor(A)[0];

  // 断言循环之间没有依赖关系
  ASSERT_EQ(
      HazardKind::NoDependency,
      getPotentialHazards(analyzer, loopRootA, loopRootB));
}
    // 返回 i * j 的乘积作为匿名函数的结果
    return i * j;
  });
  // 创建一个大小为 64x64 的张量 B，使用匿名函数计算 A 的元素加上常数 5
  Tensor B =
      Compute("B", {64, 64}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i, j) + 5;
      });

  // 创建一个循环嵌套对象 l，包含张量 A 和 B
  LoopNest l({A, B});

  // 使用分析命名空间中的内存依赖检查器进行内存依赖分析
  MemDependencyChecker analyzer;
  // 分析循环嵌套的根语句，传递依赖分析器对象给它
  l.root_stmt()->accept(&analyzer);

  // 获取张量 A 和 B 的循环语句块的根节点
  ForPtr loopRootA = l.getLoopStmtsFor(A)[0];
  ForPtr loopRootB = l.getLoopStmtsFor(B)[0];

  // 断言两个循环的内存访问存在读后写的潜在危险
  ASSERT_EQ(
      HazardKind::ReadAfterWrite,
      getPotentialHazards(analyzer, loopRootA, loopRootB));
TEST(BoundsInference, GetPotentialHazardsLoopSplit) {
  // 创建一个大小为 64x64 的张量 A，用 lambda 表达式计算每个元素的值
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });

  // 创建循环嵌套对象 l，包含张量 A
  LoopNest l({A});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;

  // 获取张量 A 的外层循环作为 outer
  ForPtr outer = l.getLoopStmtsFor(A)[0];
  // 使用偏移量 5 对 outer 进行拆分，并创建内部循环 inner 和尾部循环 tail
  // 这会生成一个尾部循环，该循环也会写入张量 A
  LoopNest::splitWithTail(outer, 5, &inner, &tail);

  // 使用分析命名空间
  using namespace analysis;

  // 创建内存依赖检查器对象 analyzer
  MemDependencyChecker analyzer;
  // 分析 l 的根语句，检查内存依赖关系
  l.root_stmt()->accept(&analyzer);

  // 断言 outer 和 tail 之间存在写后写的危险
  ASSERT_EQ(
      HazardKind::WriteAfterWrite, getPotentialHazards(analyzer, outer, tail));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferWithPartialOverlap) {
  // 输入的 IR：
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k-1] = 20 * k;
  //   }
  
  // 创建名为 A 的缓冲区，大小为 {200}，类型为 kInt
  BufHandle a_buf("A", {200}, kInt);
  // 创建变量 j 和 k，类型为 kInt
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  // 创建循环 forJ，范围为 [10, 100)，对缓冲区 A 的索引 j 进行写入操作
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，范围为 [10, 100)，对缓冲区 A 的索引 k-1 进行写入操作
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k - 1}, Mul::make(20, k)));
  // 创建顺序块 par，包含循环 forJ 和 forK
  auto par = Block::make({forJ, forK});

  // 创建内存依赖检查器对象 analyzer
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 分析 par 的语句，检查内存依赖关系
  par->accept(&analyzer);
  // 断言 forJ 和 forK 存在冲突的重叠部分
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言 forK 和 forJ 存在冲突的重叠部分
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferWithFullOverlap) {
  // 输入的 IR：
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k] = 20 * k;
  //   }
  
  // 创建名为 A 的缓冲区，大小为 {200}，类型为 kInt
  BufHandle a_buf("A", {200}, kInt);
  // 创建变量 j 和 k，类型为 kInt
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  // 创建循环 forJ，范围为 [10, 100)，对缓冲区 A 的索引 j 进行写入操作
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 创建循环 forK，范围为 [10, 100)，对缓冲区 A 的索引 k 进行写入操作
  auto forK = For::make(k, 10, 100, Store::make(a_buf, {k}, Mul::make(20, k)));
  // 创建顺序块 par，包含循环 forJ 和 forK
  auto par = Block::make({forJ, forK});

  // 创建内存依赖检查器对象 analyzer
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 分析 par 的语句，检查内存依赖关系
  par->accept(&analyzer);
  // 断言 forJ 和 forK 存在冲突的重叠部分
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言 forK 和 forJ 存在冲突的重叠部分
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}
TEST(BoundsInference, HasConflictingOverlapSameBufferWithFullOverlapRAW) {
  // 定义一个名为 "BoundsInference" 的测试用例，检查在完全重叠的情况下是否存在冲突的内存访问
  // 输入的 IR 示例：
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     B[k] = A[k];
  //   }
  
  // 创建名为 "A" 的缓冲区，大小为 200，数据类型为整数
  BufHandle a_buf("A", {200}, kInt);
  // 创建名为 "B" 的缓冲区，大小为 200，数据类型为整数
  BufHandle b_buf("B", {200}, kInt);
  // 定义整型变量 j
  VarHandle j("j", kInt);
  // 定义整型变量 k
  VarHandle k("k", kInt);
  
  // 构建循环结构 forJ，遍历 j 从 10 到 100，将表达式 10 * j 存储到缓冲区 A 的索引 j 处
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 构建循环结构 forK，遍历 k 从 10 到 100，将缓冲区 A 中索引 k 处的值加载到缓冲区 B 的索引 k 处
  auto forK =
      For::make(k, 10, 100, Store::make(b_buf, {k}, Load::make(a_buf, {k})));
  // 创建并行块，包含 forJ 和 forK 两个循环
  auto par = Block::make({forJ, forK});

  // 创建内存依赖检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对并行块 par 进行内存依赖分析
  par->accept(&analyzer);
  
  // 断言在 forJ 和 forK 之间存在冲突的内存重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言在 forK 和 forJ 之间存在冲突的内存重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferNotOverlapping) {
  // 定义一个名为 "BoundsInference" 的测试用例，检查在没有重叠的情况下是否存在冲突的内存访问
  // 输入的 IR 示例：
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k+100] = 20 * k;
  //   }
  
  // 创建名为 "A" 的缓冲区，大小为 200，数据类型为整数
  BufHandle a_buf("A", {200}, kInt);
  // 定义整型变量 j
  VarHandle j("j", kInt);
  // 定义整型变量 k
  VarHandle k("k", kInt);
  
  // 构建循环结构 forJ，遍历 j 从 10 到 100，将表达式 10 * j 存储到缓冲区 A 的索引 j 处
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // 构建循环结构 forK，遍历 k 从 10 到 100，将表达式 20 * k 存储到缓冲区 A 的索引 k+100 处
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 100}, Mul::make(20, k)));
  // 创建并行块，包含 forJ 和 forK 两个循环
  auto par = Block::make({forJ, forK});

  // 创建内存依赖检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对并行块 par 进行内存依赖分析
  par->accept(&analyzer);
  
  // 断言在 forJ 和 forK 之间不存在冲突的内存重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言在 forK 和 forJ 之间不存在冲突的内存重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forK, forJ));
}
// 定义一个名为 `BoundsInference` 的测试用例，测试与内存依赖相关的边界推断功能

TEST(BoundsInference, HasConflictingOverlap2DBufferWithOverlap) {
  // 输入的 IR 代码段:
  //   for (const auto i : c10::irange(20)) {
  //     for (const auto j : c10::irange(100)) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (const auto m : c10::irange(20)) {
  //     for (const auto n : c10::irange(50)) {
  //       A[m+1,n] = m + n * 100;
  //     }
  //   }

  // 创建名为 `a_buf` 的缓冲区对象，表示二维数组 A，大小为 20x100，元素类型为整数
  BufHandle a_buf("A", {20, 100}, kInt);
  // 创建名为 `b_buf` 的缓冲区对象，表示二维数组 B，大小为 20x50，元素类型为整数
  BufHandle b_buf("B", {20, 50}, kInt);
  // 定义整数类型的变量 `i`
  VarHandle i("i", kInt);
  // 定义整数类型的变量 `j`
  VarHandle j("j", kInt);
  // 定义整数类型的变量 `m`
  VarHandle m("m", kInt);
  // 定义整数类型的变量 `n`
  VarHandle n("n", kInt);

  // 创建存储操作 `storeA1`，将 i * j * 500 存储到数组 A 中的位置 (i, j)
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  // 创建 j 的循环，范围是从 0 到 100，循环体是存储操作 `storeA1`
  auto forJ = For::make(j, 0, 100, storeA1);
  // 创建 i 的循环，范围是从 0 到 20，循环体是 j 的循环 `forJ`
  auto forI = For::make(i, 0, 20, forJ);

  // 创建存储操作 `storeA2`，将 m + n * 100 存储到数组 A 中的位置 (m + 1, n)
  auto storeA2 =
      Store::make(a_buf, {m + 1, n}, Add::make(m, Mul::make(n, 100)));
  // 创建 n 的循环，范围是从 0 到 50，循环体是存储操作 `storeA2`
  auto forN = For::make(n, 0, 50, storeA2);
  // 创建 m 的循环，范围是从 0 到 20，循环体是 n 的循环 `forN`
  auto forM = For::make(m, 0, 20, forN);

  // 创建并行块 `par`，包含两个并行的循环块 `forI` 和 `forM`
  auto par = Block::make({forI, forM});

  // 创建内存依赖检查器 `analyzer`
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对并行块 `par` 进行内存依赖分析
  par->accept(&analyzer);

  // 断言以下各种内存访问之间存在冲突的重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA1, forM));
}
TEST(BoundsInference, HasConflictingOverlap2DBufferWithNoOverlap) {
  // 定义一个测试用例，检验在没有重叠的情况下，二维缓冲区的边界推断

  // 创建名为 A 的缓冲区，大小为 20x100，数据类型为整型
  BufHandle a_buf("A", {20, 100}, kInt);
  // 创建名为 B 的缓冲区，大小为 20x50，数据类型为整型
  BufHandle b_buf("B", {20, 50}, kInt);
  // 定义整型变量 i
  VarHandle i("i", kInt);
  // 定义整型变量 j
  VarHandle j("j", kInt);
  // 定义整型变量 m
  VarHandle m("m", kInt);
  // 定义整型变量 n
  VarHandle n("n", kInt);

  // 创建第一个存储操作，将 i * j * 500 存储到 A[i, j]
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  // 创建内层循环，j 从 0 到 99，将 storeA1 应用于每个 j
  auto forJ = For::make(j, 0, 100, storeA1);
  // 创建外层循环，i 从 0 到 19，包含内层循环 forJ
  auto forI = For::make(i, 0, 20, forJ);

  // 创建第二个存储操作，将 m + n * 100 存储到 A[m+20, n+100]
  auto storeA2 = Store::make(a_buf, {m + 20, n + 100}, Add::make(m, Mul::make(n, 100)));
  // 创建内层循环，n 从 0 到 49，将 storeA2 应用于每个 n
  auto forN = For::make(n, 0, 50, storeA2);
  // 创建外层循环，m 从 0 到 19，包含内层循环 forN
  auto forM = For::make(m, 0, 20, forN);

  // 创建代码块 par，包含两个并行的循环结构 forI 和 forM
  auto par = Block::make({forI, forM});

  // 创建内存依赖性检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对代码块 par 进行内存依赖性分析
  par->accept(&analyzer);

  // 断言以下所有情况下不存在重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, forM));
}
TEST(BoundsInference, HasConflictingOverlapDifferentBuffers) {
  // 测试用例：检查不同缓冲区间的冲突重叠

  // 定义缓冲区 A，大小为 20x100，元素类型为整数
  BufHandle a_buf("A", {20, 100}, kInt);
  // 定义缓冲区 B，大小为 20x50，元素类型为整数
  BufHandle b_buf("B", {20, 50}, kInt);
  
  // 定义循环变量 i 和 j，类型为整数
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // 构建存储操作，将 i * j * 500 存储到 A 的索引为 {i, j} 的位置
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  // 构建循环结构，范围是 j 从 0 到 99，内部执行 storeA1 操作
  auto forJ = For::make(j, 0, 100, storeA1);
  // 外层循环，范围是 i 从 0 到 19，内部执行 forJ 循环
  auto forI = For::make(i, 0, 20, forJ);
  
  // 定义循环变量 m 和 n，类型为整数
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  // 构建存储操作，将 m + n * 100 存储到 B 的索引为 {m, n} 的位置
  auto storeA2 = Store::make(b_buf, {m, n}, Add::make(m, Mul::make(n, 100)));
  // 构建循环结构，范围是 n 从 0 到 49，内部执行 storeA2 操作
  auto forN = For::make(n, 0, 50, storeA2);
  // 外层循环，范围是 m 从 0 到 19，内部执行 forN 循环
  auto forM = For::make(m, 0, 20, forN);
  
  // 构建代码块，包含所有的循环结构
  auto par = Block::make({forI, forM});

  // 创建内存依赖性检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对 par 代码块执行内存依赖性分析
  par->accept(&analyzer);

  // 检查不同循环结构之间是否有冲突的重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, forM));
}

TEST(BoundsInference, HasConflictingOverlapDueToRAWDependence) {
  // 测试用例：检查由于 RAW 依赖引起的冲突重叠

  // 定义缓冲区 A，大小为 100，元素类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 定义缓冲区 B，大小为 100，元素类型为整数
  BufHandle b_buf("B", {100}, kInt);
  
  // 定义循环变量 j 和 k，类型为整数
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  
  // 构建存储操作，将 10 * j 存储到 A 的索引为 j 的位置
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  
  // 构建存储操作，将 20 * A[99-k] 存储到 B 的索引为 k 的位置
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  
  // 构建代码块，包含所有的循环结构
  auto par = Block::make({forJ, forK});

  // 创建内存依赖性检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 对 par 代码块执行内存依赖性分析
  par->accept(&analyzer);

  // 检查由于 RAW 依赖引起的冲突重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}
TEST(BoundsInference, HasConflictingOverlapDueToWARDependence) {
  // 定义一个测试用例，验证由于WAR依赖而导致的冲突重叠情况

  // 创建名为A的缓冲区，大小为100，数据类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为B的缓冲区，大小为100，数据类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 定义变量j，数据类型为整数
  VarHandle j("j", kInt);
  // 定义变量k，数据类型为整数
  VarHandle k("k", kInt);
  
  // 创建循环forK，迭代k从0到99，执行B[k] = 20 * A[99-k]的存储操作
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  
  // 创建循环forJ，迭代j从0到99，执行A[j] = 10 * j的存储操作
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  
  // 创建一个并行块，包含forK和forJ两个循环
  auto par = Block::make({forK, forJ});

  // 创建内存依赖检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 分析并行块par的内存依赖关系
  par->accept(&analyzer);
  
  // 断言forJ和forK之间存在冲突重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言forK和forJ之间存在冲突重叠
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapWithLoads) {
  // 定义一个测试用例，验证带加载操作时的冲突重叠情况

  // 创建名为A的缓冲区，大小为100，数据类型为整数
  BufHandle a_buf("A", {100}, kInt);
  // 创建名为B的缓冲区，大小为100，数据类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建名为C的缓冲区，大小为100，数据类型为整数
  BufHandle c_buf("C", {100}, kInt);
  // 定义变量j，数据类型为整数
  VarHandle j("j", kInt);
  // 定义变量k，数据类型为整数
  VarHandle k("k", kInt);
  
  // 创建循环forK，迭代k从10到99，执行B[k] = 20 * A[99-k]的存储操作
  auto forK = For::make(
      k,
      10,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  
  // 创建循环forJ，迭代j从10到99，执行C[j] = 10 * A[j]的存储操作
  auto forJ = For::make(
      j,
      10,
      100,
      Store::make(c_buf, {j}, Mul::make(10, Load::make(a_buf, {j}))));
  
  // 创建一个并行块，包含forK和forJ两个循环
  auto par = Block::make({forK, forJ});

  // 创建内存依赖检查器对象
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 分析并行块par的内存依赖关系
  par->accept(&analyzer);
  
  // 断言forJ和forK之间不存在冲突重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forK));
  // 断言forK和forJ之间不存在冲突重叠
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forK, forJ));
}
TEST(BoundsInference, IsOverlapping) {
  // 定义一个名为 BoundsInference 的测试用例，测试内存边界推断中的重叠检测功能

  // Input IR:
  //   for (const auto i : c10::irange(100)) {
  //     A[i] = i * 10;               // storeA1
  //     B[i] = A[99-i] * 20;         // loadA1
  //     C[i] = A[i + 100] * 10;      // loadA2
  //     A[i + 50] = i * 50;          // storeA2
  //     A[i + 150] = i * 150;        // storeA3
  //   }
  // 定义一个输入的 IR，包含了一些基本的数组操作说明

  // 创建名为 a_buf 的缓冲区句柄，表示数组 A，大小为 300，元素类型为整数
  BufHandle a_buf("A", {300}, kInt);
  // 创建名为 b_buf 的缓冲区句柄，表示数组 B，大小为 100，元素类型为整数
  BufHandle b_buf("B", {100}, kInt);
  // 创建名为 c_buf 的缓冲区句柄，表示数组 C，大小为 100，元素类型为整数
  BufHandle c_buf("C", {100}, kInt);
  // 创建名为 i 的变量句柄，表示循环变量 i，类型为整数
  VarHandle i("i", kInt);

  // 创建存储操作 storeA1，将 i * 10 存储到数组 A 中索引为 i 的位置
  auto storeA1 = Store::make(a_buf, {i}, i * 10);
  // 创建加载操作 loadA1，从数组 A 中加载索引为 99-i 的位置的值
  auto loadA1 = Load::make(a_buf, {ExprHandle(99) - i});
  // 创建存储操作 storeB，将 loadA1 * 20 存储到数组 B 中索引为 i 的位置
  auto storeB = Store::make(b_buf, {i}, Mul::make(loadA1, 20));
  // 创建加载操作 loadA2，从数组 A 中加载索引为 i+100 的位置的值
  auto loadA2 = Load::make(a_buf, {i + 100});
  // 创建存储操作 storeC，将 loadA2 * 10 存储到数组 C 中索引为 i 的位置
  auto storeC = Store::make(c_buf, {i}, Mul::make(loadA2, 10));
  // 创建存储操作 storeA2，将 i * 50 存储到数组 A 中索引为 i+50 的位置
  auto storeA2 = Store::make(a_buf, {i + 50}, i * 50);
  // 创建存储操作 storeA3，将 i * 150 存储到数组 A 中索引为 i+150 的位置
  auto storeA3 = Store::make(a_buf, {i + 150}, i * 150);

  // 创建 for 循环结构 forI，循环变量为 i，范围从 0 到 99，
  // 循环体内包含 storeA1、storeB、storeC、storeA2、storeA3 这些操作
  auto forI = For::make(
      i, 0, 100, Block::make({storeA1, storeB, storeC, storeA2, storeA3}));

  // 创建内存依赖分析器对象 analyzer
  tensorexpr::analysis::MemDependencyChecker analyzer;
  // 使用分析器分析 forI 循环体的内存依赖关系
  forI->accept(&analyzer);

  // 断言：检查 storeA1 和 loadA1 是否存在重叠
  ASSERT_TRUE(isOverlapping(analyzer, storeA1, to<Load>(loadA1.node())));
  // 断言：检查 storeA1 和 loadA2 是否存在重叠
  ASSERT_FALSE(isOverlapping(analyzer, storeA1, to<Load>(loadA2.node())));
  // 断言：检查 storeA1 和 storeA2 是否存在重叠
  ASSERT_TRUE(isOverlapping(analyzer, storeA1, storeA2));
  // 断言：检查 storeA1 和 storeA3 是否存在重叠
  ASSERT_FALSE(isOverlapping(analyzer, storeA1, storeA3));
}

} // namespace jit
} // namespace torch
```