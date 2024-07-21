# `.\pytorch\test\cpp\tensorexpr\test_memdependency.cpp`

```
// 包含 Google Test 框架中的头文件
#include <gtest/gtest.h>
// 包含 TensorExpr 的测试基础类的头文件
#include <test/cpp/tensorexpr/test_base.h>

// 包含 TensorExpr 中的不同模块和功能的头文件
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// 定义命名空间为 torch::jit，测试帮助函数用于确定两个缓冲区的区域是否重叠
// No Overlap & partial overlap 是显而易见的。Contains 意味着 A 区域更大且完全包含 B 区域，
// 而 ContainedOrEqual 是反向的。相等的区域属于 ContainedOrEqual。
TEST(MemDependency, BoundOverlap) {
  // 使用 analysis 命名空间
  using namespace analysis;

  // 定义 CB lambda 表达式，接受两个整数参数 s 和 e
  auto CB = [](int s, int e) {
    // 返回一个 Bound 对象，其起始和结束位置为 s 和 e
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };

  // 进行三种重叠情况的合理性检查。
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(0, 0), CB(0, 0)));
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(0, 3), CB(2, 5)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(0, 0), CB(1, 1)));

  // 部分重叠可以以任意顺序工作。
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(0, 10), CB(7, 14)));
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(7, 14), CB(0, 10)));

  // 当一个 Bound 包含另一个 Bound 时，返回 Contains。
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(2, 15), CB(7, 9)));
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(2, 15), CB(0, 16)));

  // 当两个 Bound 完全重叠时，返回 ContainedOrEqual。
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(2, 15), CB(2, 15)));

  // 当只有一个 Bound 的一端匹配时，返回 Contains。
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(2, 15), CB(2, 10)));
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(2, 15), CB(3, 15)));
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(0, 10), CB(0, 9)));
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(2, 10), CB(2, 15)));
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(3, 15), CB(2, 15)));

  // 当 a < b 时，返回 NoOverlap。
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(0, 2), CB(5, 10)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(2, 2), CB(3, 3)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(100, 120), CB(130, 130)));

  // 当 a > b 时，返回 NoOverlap。
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(5, 10), CB(0, 2)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(3, 3), CB(2, 2)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(130, 130), CB(100, 120)));

  // 当相邻时，返回 NoOverlap。
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(0, 100), CB(101, 120)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(2, 3), CB(0, 1)));

  // 当中间边界匹配时，返回 PartialOverlap。
  ASSERT_EQ(
      OverlapKind::PartialOverlap, boundOverlap(CB(0, 100), CB(100, 120)));
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(0, 2), CB(2, 4)));
  ASSERT_EQ(
      OverlapKind::PartialOverlap, boundOverlap(CB(100, 120), CB(0, 100)));
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(2, 3), CB(1, 2)));

  // 当一个 Bound 的一个端点与另一个 Bound 的一端重叠时，返回 Contains 或 ContainedOrEqual。
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(2, 15), CB(15, 15)));
  ASSERT_EQ(OverlapKind::Contains, boundOverlap(CB(2, 15), CB(2, 2)));
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(2, 2), CB(2, 15)));
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(15, 15), CB(2, 15)));
}

// 定义一个测试用例，验证内存依赖的边界比较
TEST(MemDependency, BoundComparison) {
  // 使用 analysis 命名空间
  using namespace analysis;

  // 定义一个 lambda 表达式 CB，接受两个整数参数 s 和 e
  auto CB = [](int s, int e) {
}

// 定义一个测试用例，验证符号边界重叠
TEST(MemDependency, BoundOverlapSymbolic) {
  // 创建四个变量 x, y, z, w，类型为 kInt 的 VarHandle
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  // 使用 analysis 命名空间
  using namespace analysis;

  // 定义一个 lambda 表达式 CB，接受 ExprHandle 类型的参数 s 和 e
  auto CB = [](ExprHandle s, ExprHandle e) {
    // 返回从 s 到 e 的边界对象 Bound
    return Bound(s.node(), e.node());
  };

  // 进行一些基本的健全性检查，其中起始和结束都是符号的情况，但差值是常量。
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  ASSERT_EQ(OverlapKind::ContainedOrEqual, boundOverlap(CB(x, x), CB(x, x)));
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      boundOverlap(CB(x, x + 3), CB(x + 2, x + 5)));
  ASSERT_EQ(OverlapKind::NoOverlap, boundOverlap(CB(x, x), CB(x + 1, x + 1)));

  // 无法推断 y 的符号，因此无法确定添加 y 是否大于或小于 y/2。
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      boundOverlap(CB(x, x + y), CB(x, x + y / 2)));

  // 对于这个边界没有足够的信息，必须采取最保守的选项：可能存在重叠。
  ASSERT_EQ(OverlapKind::PartialOverlap, boundOverlap(CB(x, y), CB(z, w)));

  // 对不透明术语的数学运算是有效的。
  ASSERT_EQ(
      OverlapKind::ContainedOrEqual,
      boundOverlap(CB(x + w, y - z), CB(x + w, y - z)));
  // 即使需要简化也是如此。
  ASSERT_EQ(
      OverlapKind::ContainedOrEqual,
      boundOverlap(CB(x - w - w, y), CB(x - w * 2, y)));
}

// 测试多维索引边界重叠的辅助函数。
// 这使用 boundOverlap 在每个维度上进行边界检查，并返回最低级别的重叠。
TEST(MemDependency, BoundOverlapMultiDim) {
  // 使用 analysis 命名空间
  using namespace analysis;

  // 定义一个 lambda 表达式 CB，接受两个整数参数 s 和 e
  auto CB = [](int s, int e) {
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };

  // Sanity check one dimensional cases.
  ASSERT_EQ(OverlapKind::ContainedOrEqual, overlaps({CB(0, 0)}, {CB(0, 0)}));
  // 检查一维情况，期望结果为完全包含或相等

  ASSERT_EQ(OverlapKind::NoOverlap, overlaps({CB(0, 2)}, {CB(5, 10)}));
  // 检查一维情况，期望结果为完全无重叠

  ASSERT_EQ(
      OverlapKind::PartialOverlap, overlaps({CB(0, 100)}, {CB(100, 120)}));
  // 检查一维情况，期望结果为部分重叠

  // Total overlap in 3 dims.
  ASSERT_EQ(
      OverlapKind::ContainedOrEqual,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(0, 4)}));
  // 检查三维情况，期望结果为完全包含或相等

  ASSERT_EQ(
      OverlapKind::ContainedOrEqual,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(0, 10)}));
  // 检查三维情况，期望结果为完全包含或相等

  // Total overlap in 2 dims, no overlap in another.
  ASSERT_EQ(
      OverlapKind::NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(5, 10)}));
  // 检查三维情况，期望结果为完全无重叠

  // Total overlap in 2 dims, partial overlap in another.
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 2), CB(0, 5), CB(5, 10)}));
  // 检查三维情况，期望结果为部分重叠

  // This case is most important, so verify the overlap in any dim. (dim 2)
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 2), CB(2, 6), CB(0, 5)}));
  // 检查三维情况，特别验证在任何一个维度上的重叠情况（第二维度）

  // Dim 1.
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(1, 3), CB(0, 5), CB(0, 5)}));
  // 检查三维情况，验证在第一维度上的部分重叠情况

  // Total overlap in 1 dim, partial in 2.
  ASSERT_EQ(
      OverlapKind::PartialOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(2, 6), CB(0, 5), CB(5, 10)}));
  // 检查三维情况，期望结果为在一个维度上完全重叠，在另外两个维度上部分重叠

  // Total overlap, partial overlap, no overlap.
  ASSERT_EQ(
      OverlapKind::NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(2, 6), CB(11, 15), CB(0, 5)}));
  // 检查三维情况，期望结果为完全无重叠

  // Total overlap (B) in 2 dims, total overlap (A) in another.
  ASSERT_EQ(
      OverlapKind::Contains,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 3), CB(0, 4)}));
  // 检查三维情况，期望结果为完全包含

  // Total overlap (A) in 2 dims, total overlap (B) in another.
  ASSERT_EQ(
      OverlapKind::Contains,
      overlaps(
          {CB(0, 12), CB(0, 15), CB(0, 4)}, {CB(0, 2), CB(0, 3), CB(0, 14)}));
  // 检查三维情况，期望结果为完全包含

  // Total (B), No Overlap, Total (A).
  ASSERT_EQ(
      OverlapKind::NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 6), CB(11, 15), CB(1, 2)}));
  // 检查三维情况，期望结果为完全无重叠
// Test the helper we use to subtract bounds: returns the regions(s) of A which
// remain after removing the region of B.
TEST(MemDependency, BoundSubtract) {
  using namespace analysis;

  // Lambda function CB constructs a Bound object from start `s` to end `e`
  auto CB = [](int s, int e) {
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };

  // Lambda function EQ checks if two IndexBounds objects are equal
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  // Tests for subtractBound function:

  // Test case: subtracting identical bounds results in an empty set
  ASSERT_EQ(subtractBound(CB(0, 0), CB(0, 0)).size(), 0);
  ASSERT_EQ(subtractBound(CB(5, 5), CB(5, 5)).size(), 0);

  // Test case: no overlap between bounds
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(2, 2)), {CB(5, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(0, 4)), {CB(5, 5)}));

  // Test case: overlap on one side of the bounds
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(4, 7)), {CB(1, 3)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 5), CB(5, 7)), {CB(0, 4)}));
  ASSERT_TRUE(EQ(subtractBound(CB(4, 5), CB(1, 4)), {CB(5, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(0, 4)), {CB(5, 5)}));

  // Test case: overlap on both sides of the bounds
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(0, 7)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(5, 7)), {}));

  // Test case: internal overlap within the bounds
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(2, 3)), {CB(1, 1), CB(4, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 5), CB(2, 4)), {CB(0, 1), CB(5, 5)}));
}

// Test case for symbolic bounds subtraction
TEST(MemDependency, BoundSubtractSymbolic) {
  // Define symbolic variables
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace analysis;

  // Lambda function CB constructs a Bound object from start `s` to end `e` using symbolic expressions
  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  // Lambda function EQ checks if two IndexBounds objects are equal
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
  // 返回函数调用结果
  return indexBoundsEquals(x, y);
};

// 单个元素的减法边界测试
// NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
ASSERT_TRUE(EQ(subtractBound(CB(x, x), CB(x, x)), {}));

// 单个元素的减法边界测试，包含加法表达式
ASSERT_TRUE(EQ(subtractBound(CB(x + 1, x + 1), CB(x + 1, x + 1)), {}));

// 单个元素的减法边界测试，包含乘法表达式
ASSERT_TRUE(EQ(subtractBound(CB(x * 2, x * 2), CB(x * 2, x * 2)), {}));

// 减去常量范围的低部分
ASSERT_TRUE(
    EQ(subtractBound(CB(x, x + 10), CB(x, x + 4)), {CB(x + 5, x + 10)}));

// 减去常量范围的高部分
ASSERT_TRUE(
    EQ(subtractBound(CB(x, x + 10), CB(x + 6, x + 12)), {CB(x, x + 5)}));

// 减去完全重叠的常量范围
ASSERT_TRUE(EQ(subtractBound(CB(x, x + 10), CB(x, x + 10)), {}));
ASSERT_TRUE(EQ(subtractBound(CB(x + 2, x + 10), CB(x, x + 12)), {}));

// 减去内部的常量范围
ASSERT_TRUE(
    EQ(subtractBound(CB(x, x + 10), CB(x + 3, x + 7)),
       {CB(x, x + 2), CB(x + 8, x + 10)}));

// 大小可推断但不是常量，仅适用于单个变量的情况
ASSERT_TRUE(EQ(subtractBound(CB(0, x), CB(0, x * 2)), {}));
ASSERT_TRUE(EQ(subtractBound(CB(0, x * 2), CB(0, x - 1)), {CB(x, x * 2)}));

// 大小不可推断的情况
ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(z, w)), {CB(x, y)}));
ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(x, z)), {CB(x, y)}));
ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(0, x)), {CB(x, y)}));
ASSERT_TRUE(EQ(subtractBound(CB(x, x), CB(0, 0)), {CB(x, x)}));
// Tests the helper function that does subtraction, but for multi dimensional
// indices bounds.
TEST(MemDependency, BoundSubtractMultiDim) {
  // 引入命名空间 analysis
  using namespace analysis;

  // 定义 CB lambda 函数，用于创建 Bound 对象，接受整数参数
  auto CB = [](int s, int e) {
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };

  // 定义 EQ lambda 函数，用于比较两个 IndexBounds 向量是否相等
  auto EQ = [](std::vector<IndexBounds> x, std::vector<IndexBounds> y) {
    // 如果向量大小不相等，返回 false
    if (x.size() != y.size()) {
      return false;
    }
    // 遍历每个元素，比较它们是否相等
    for (auto i = 0U; i < x.size(); ++i) {
      if (!indexBoundsEquals(x[i], y[i])) {
        return false;
      }
    }
    // 如果所有元素相等，返回 true
    return true;
  };

  // 单维度边界相等的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 9)}, {CB(0, 9)}), {}));
  // 单维度部分重叠的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(3, 9)}, {CB(0, 12)}), {}));
  // 单维度部分重叠且有剩余的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 12)}, {CB(0, 9)}), {{CB(10, 12)}}));
  // 单维度重叠但不完全的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 12)}, {CB(3, 12)}), {{CB(0, 2)}}));
  // 单维度无重叠的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 9)}, {CB(1, 8)}), {{CB(0, 0)}, {CB(9, 9)}}));

  // 多维度完全重叠的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 9), CB(0, 2)}), {}));
  // 多维度完全重叠但不同长度的检查
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 10), CB(0, 20)}), {}));

  // 多维度在一个维度上部分重叠的检查
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 3), CB(0, 2)}),
         {{CB(4, 9), CB(0, 2)}}));

  // 多维度在一个维度上部分重叠的检查
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 9), CB(0, 20)}, {CB(0, 9), CB(0, 10)}),
         {{CB(0, 9), CB(11, 20)}}));

  // 两个维度上部分重叠的检查
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 5), CB(0, 5)}, {CB(2, 8), CB(2, 8)}),
         {{CB(0, 1), CB(0, 5)}, {CB(2, 5), CB(0, 1)}}));

  // 三个维度上部分重叠的检查
  ASSERT_TRUE(
      EQ(subtractIndicesBounds(
             {CB(0, 5), CB(0, 5), CB(0, 5)}, {CB(2, 8), CB(2, 8), CB(2, 8)}),
         {{CB(0, 1), CB(0, 5), CB(0, 5)},
          {CB(2, 5), CB(0, 1), CB(0, 5)},
          {CB(2, 5), CB(2, 5), CB(0, 1)}}));
}

// Tests the multi dimensional subtraction code for bounds that cannot be fully
// materialized.
TEST(MemDependency, BoundSubtractMultiDimSymbolic) {
  // 创建两个整型变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  // 引入命名空间 analysis
  using namespace analysis;

  // 定义 CB lambda 函数，用于创建 Bound 对象，接受表达式参数
  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  // 定义 EQ lambda 函数，用于比较两个 IndexBounds 向量是否相等
  auto EQ = [](std::vector<IndexBounds> x, std::vector<IndexBounds> y) {
    // 如果向量大小不相等，返回 false
    if (x.size() != y.size()) {
      return false;
    }
    // 遍历每个元素，比较它们是否相等
    for (auto i = 0U; i < x.size(); ++i) {
      if (!indexBoundsEquals(x[i], y[i])) {
        return false;
      }
    }
    // 如果所有元素相等，返回 true
    return true;
  };
    // 返回 true
      };
    
      // 无法确定重叠情况。
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      ASSERT_TRUE(EQ(subtractIndicesBounds({CB(x, x)}, {CB(0, 0)}), {{CB(x, x)}}));
    
      // 不同的完全重叠情况。
      ASSERT_TRUE(EQ(
          subtractIndicesBounds({CB(x, x), CB(x, x)}, {CB(x, x), CB(x, x)}), {}));
      ASSERT_TRUE(EQ(
          subtractIndicesBounds({CB(x, y), CB(x, y)}, {CB(x, y), CB(x, y)}), {}));
      ASSERT_TRUE(EQ(
          subtractIndicesBounds({CB(x, x), CB(y, y)}, {CB(x, x), CB(y, y)}), {}));
      ASSERT_TRUE(EQ(
          subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(0, y)}), {}));
    
      // 第一维度上的单向重叠。
      ASSERT_TRUE(
          EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x - 5), CB(0, y)}),
             {{CB(x - 4, x), CB(0, y)}}));
      // 第二维度上的重叠。
      ASSERT_TRUE(
          EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(5, y)}),
             {{CB(0, x), CB(0, 4)}}));
    
      // 第一维度上的内部重叠。
      ASSERT_TRUE(
          EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(2, x - 5), CB(0, y)}),
             {{CB(0, 1), CB(0, y)}, {CB(x - 4, x), CB(0, y)}}));
      // 第二维度上的重叠。
      ASSERT_TRUE(EQ(
          subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(10, y - 10)}),
          {{CB(0, x), CB(0, 9)}, {CB(0, x), CB(y - 9, y)}}));
    
      // 两个维度上都有重叠。
      ASSERT_TRUE(
          EQ(subtractIndicesBounds(
                 {CB(0, x), CB(0, y)}, {CB(5, x - 5), CB(10, y - 10)}),
             {
                 {CB(0, 4), CB(0, y)},
                 {CB(x - 4, x), CB(0, y)},
                 {CB(0, x), CB(0, 9)},
                 {CB(0, x), CB(y - 9, y)},
             }));
}

// 简单检查分析器是否有效...
TEST(MemDependency, MemDependencyCheckerSimple) {
  // 创建名为 "A" 的缓冲区句柄，包含一个整数类型的元素
  BufHandle a("A", {1}, kInt);
  // 创建名为 "B" 的缓冲区句柄，包含一个整数类型的元素
  BufHandle b("B", {1}, kInt);

  // 创建内存依赖检查器对象
  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * B[0] = A[0] + 1;
   */

  // 创建将值 3 存储到缓冲区 A 中索引为 0 的存储操作
  StorePtr aStore = Store::make(a, {0}, 3);
  // 创建将缓冲区 A 中索引为 0 的值加 1 后存储到缓冲区 B 中索引为 0 的存储操作
  StorePtr bStore = Store::make(b, {0}, Add::make(Load::make(a, {0}), 1));

  // 创建由上述存储操作组成的语句块
  StmtPtr stmt = Block::make({aStore, bStore});

  // 将该语句块接受内存依赖检查器进行分析
  stmt->accept(&analyzer);

  // 断言：B 存储操作直接依赖于 A 存储操作
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));
  // 断言：A 存储操作不会间接依赖于 B 存储操作
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, bStore));
  // 断言：B 存储操作间接依赖于 A 存储操作，但这是个健全性检查
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aStore));
}

// 检查直接和间接依赖之间的差异
TEST(MemDependency, MemDependencyCheckerMultiStmt) {
  // 创建名为 "A"、"B" 和 "C" 的缓冲区句柄，每个包含一个整数类型的元素
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  BufHandle c("C", {1}, kInt);

  // 创建内存依赖检查器对象
  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * B[0] = A[0];
   * C[0] = B[0] + 1;
   */

  // 创建将值 3 存储到缓冲区 A 中索引为 0 的存储操作
  StorePtr aStore = Store::make(a, {0}, 3);
  // 创建将缓冲区 A 中索引为 0 的值存储到缓冲区 B 中索引为 0 的存储操作
  StorePtr bStore = Store::make(b, {0}, Load::make(a, {0}));
  // 创建将缓冲区 B 中索引为 0 的值加 1 后存储到缓冲区 C 中索引为 0 的存储操作
  StorePtr cStore = Store::make(c, {0}, Add::make(Load::make(b, {0}), 1));

  // 创建由上述存储操作组成的语句块
  StmtPtr stmt = Block::make({aStore, bStore, cStore});

  // 将该语句块接受内存依赖检查器进行分析
  stmt->accept(&analyzer);

  // 断言：C 存储操作间接依赖于 A 存储操作
  ASSERT_FALSE(analyzer.dependsDirectly(cStore, aStore));
  ASSERT_TRUE(analyzer.dependsIndirectly(cStore, aStore));

  // 断言：C 存储操作直接依赖于 B 存储操作，而 B 直接依赖于 A 存储操作
  ASSERT_TRUE(analyzer.dependsDirectly(cStore, bStore));
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));

  // 断言：依赖关系只能从上到下传播
  ASSERT_FALSE(analyzer.dependsIndirectly(bStore, cStore));
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, bStore));
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, cStore));
}

// 验证我们确实过滤了完全被后续写入覆盖的写入操作。
TEST(MemDependency, MemDependencyCheckerOverlap) {
  // 创建名为 "A" 和 "B" 的缓冲区句柄，每个包含一个整数类型的元素
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);

  // 创建内存依赖检查器对象
  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * A[0] = 6;
   * B[0] = A[0] + 1;
   */

  // 创建将值 3 存储到缓冲区 A 中索引为 0 的存储操作
  StorePtr aStore = Store::make(a, {0}, 3);
  // 创建将值 6 存储到缓冲区 A 中索引为 0 的存储操作（完全覆盖前一个操作）
  StorePtr a2Store = Store::make(a, {0}, 6);
  // 创建将缓冲区 A 中索引为 0 的值加 1 后存储到缓冲区 B 中索引为 0 的存储操作
  StorePtr bStore = Store::make(b, {0}, Add::make(Load::make(a, {0}), 1));

  // 创建由上述存储操作组成的语句块
  StmtPtr stmt = Block::make({aStore, a2Store, bStore});

  // 将该语句块接受内存依赖检查器进行分析
  stmt->accept(&analyzer);

  // 断言：B 存储操作间接依赖于第二个 A 存储操作，但不依赖于第一个操作，因为它被完全覆盖了
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, a2Store));
  ASSERT_FALSE(analyzer.dependsIndirectly(bStore, aStore));

  // 断言：两个 A 存储操作之间没有依赖关系
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, a2Store));
  ASSERT_FALSE(analyzer.dependsIndirectly(a2Store, aStore));
}
TEST(MemDependency, MemDependencyCheckerLoop) {
  // 创建名为 "A" 的缓冲区，包含1个元素，类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建名为 "B" 的缓冲区，包含1个元素，类型为整数
  BufHandle b("B", {1}, kInt);
  // 创建名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);

  // 引入分析命名空间
  using namespace analysis;

  // 创建内存依赖检查器对象
  MemDependencyChecker analyzer;

  /*
   * for (int x = 0; x < 10; ++x) {
   *   A[x] = x;
   * }
   * B[0] = A[0] + 1;
   */

  // 创建存储操作，将 x 赋值给 A[x]
  StorePtr aStore = Store::make(a, {x}, x);
  // 创建循环语句，循环从 0 到 9，将 aStore 添加到循环体中
  StmtPtr loop = For::make(x, 0, 10, aStore);
  // 创建存储操作，将 A[0] + 1 的结果存储到 B[0]
  StorePtr bStore = Store::make(b, {0}, Add::make(Load::make(a, {4}), 1));

  // 创建语句块，包含上述的循环和存储操作
  StmtPtr stmt = Block::make({loop, bStore});

  // 使用依赖分析器分析该语句块
  stmt->accept(&analyzer);

  // 断言 B[0] 直接依赖于 A[x] 存储操作
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));

  // 断言 B[0] 依赖于循环语句
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));
  // 断言 A 在循环中，但不依赖于任何循环迭代
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, loop));

  // 获取 aStore 的访问信息
  auto aStoreAccess = analyzer.accessFor(aStore);
  ASSERT_NE(aStoreAccess, nullptr);

  // 确保其边界覆盖 x 的范围：0 <= x < 10
  ASSERT_TRUE(indexBoundsEquals(
      aStoreAccess->bounds(), {Bound(alloc<IntImm>(0), alloc<IntImm>(9))}));
}

// 缩减也应该促进依赖性。
TEST(MemDependency, MemDependencyCheckerLoopReduce) {
  // 创建名为 "A" 的缓冲区，包含10个元素，类型为整数
  BufHandle a("A", {10}, kInt);
  // 创建名为 "B" 的缓冲区，包含10个元素，类型为整数
  BufHandle b("B", {10}, kInt);
  // 创建名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);

  // 引入分析命名空间
  using namespace analysis;

  // 创建内存依赖检查器对象
  MemDependencyChecker analyzer;

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; ++x) {
   *   A[0] = A[x] + 1;
   * }
   * B[0] = A[0];
   */

  // 初始化 A[0] 为 0
  StorePtr aInit = Store::make(a, {0}, 0);
  // 创建求和表达式，对 A 执行累加操作
  ExprHandle reduce = Sum()(a, 1, {x}, {x});
  // 存储操作，将 reduce 的结果存储到 A[0]
  StorePtr aReduce = Store::make(a, {0}, reduce);
  // 创建循环语句，循环从 0 到 9，将 aReduce 添加到循环体中
  StmtPtr loop = For::make(x, 0, 10, aReduce);
  // 存储操作，将 A[0] 的值存储到 B[0]
  StorePtr bStore = Store::make(b, {0}, Load::make(a, {0}));

  // 创建语句块，包含上述的初始化、循环和存储操作
  StmtPtr stmt = Block::make({aInit, loop, bStore});

  // 使用依赖分析器分析该语句块
  stmt->accept(&analyzer);

  // 断言 B -> A 的依赖关系
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aReduce));

  // B 对 A 的初始化间接依赖，因为缩减操作依赖于初始化操作
  ASSERT_FALSE(analyzer.dependsDirectly(bStore, aInit));
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aInit));

  // 确保 A[0] 对 A[x] 的依赖
  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, aInit));

  // 断言 B 依赖于循环语句
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));
  // 断言 A 在循环中，并依赖于其他迭代
  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, loop));

  // 循环内容对初始化操作也有依赖
  ASSERT_TRUE(analyzer.dependsDirectly(loop, aInit));

  // 查找缩减中的加载项
  auto reduceLoads = NodeFinder<Load>::find(reduce.node());
  // 获取循环内加载项的访问信息
  for (auto load : reduceLoads) {
    auto loopLoad = analyzer.accessFor(load);
    // 确保其边界为 0 到 9
    ASSERT_TRUE(indexBoundsEquals(
        loopLoad->bounds(), {Bound(alloc<IntImm>(0), alloc<IntImm>(9))}));
  }
}

// 缩减操作不会影响依赖分析。
TEST(MemDependency, MemDependencyCheckerLoopReduceExpanded) {
  BufHandle a("A", {10}, kInt);  // 创建一个名为 A 的缓冲区，包含 10 个整数，类型为 kInt
  BufHandle b("B", {10}, kInt);  // 创建一个名为 B 的缓冲区，包含 10 个整数，类型为 kInt
  VarHandle x("x", kInt);  // 创建一个名为 x 的变量，类型为 kInt

  using namespace analysis;  // 引入分析命名空间

  MemDependencyChecker analyzer;  // 创建内存依赖检查器对象

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; ++x) {
   *   A[x] = A[x] + 1;
   * }
   * B[0] = A[0];
   */

  StorePtr aInit = Store::make(a, {0}, 0);  // 创建一个初始化 A[0] 的存储操作
  ExprHandle aLoad = Load::make(a, {x});  // 创建一个加载操作，从 A 中加载 A[x] 的值
  StorePtr aReduce = Store::make(a, {0}, Add::make(aLoad, 1));  // 创建一个存储操作，将 A[0] 更新为 A[x]+1 的结果
  StmtPtr loop = For::make(x, 0, 10, aReduce);  // 创建一个循环语句，循环范围是 x 从 0 到 9，循环体是 aReduce
  StorePtr bStore = Store::make(b, {0}, Load::make(a, {0}));  // 创建一个存储操作，将 B[0] 更新为 A[0] 的值

  StmtPtr stmt = Block::make({aInit, loop, bStore});  // 创建一个语句块，包含 aInit、loop 和 bStore 三个操作

  stmt->accept(&analyzer);  // 使用内存依赖检查器分析这个语句块

  // B -> A.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aReduce));  // 断言 bStore 直接依赖于 aReduce

  // B depends indirectly on the initializer of A, since the reduction depends
  // on it.
  ASSERT_FALSE(analyzer.dependsDirectly(bStore, aInit));  // 断言 bStore 不直接依赖于 aInit
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aInit));  // 断言 bStore 间接依赖于 aInit

  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, aInit));  // 断言 aReduce 直接依赖于 aInit

  // B depends on the loop.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));  // 断言 bStore 直接依赖于 loop
  // A is in the loop and depends on other iterations.
  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, loop));  // 断言 aReduce 直接依赖于 loop

  // The loop contents depend on the initializer too.
  ASSERT_TRUE(analyzer.dependsDirectly(loop, aInit));  // 断言 loop 直接依赖于 aInit

  // Pull out the access for the store inside the loop.
  auto loopLoad = analyzer.accessFor(aLoad.node());  // 从分析器中获取加载操作 aLoad 的访问信息
  // It should have 10 element long bounds.
  ASSERT_TRUE(indexBoundsEquals(
      loopLoad->bounds(), {Bound(alloc<IntImm>(0), alloc<IntImm>(9))}));  // 断言加载操作的边界为 [0, 9]
}

// Can determine dependencies of outputs, through to inputs.
TEST(MemDependency, MemDependencyCheckerInputsOutputs) {
  // 定义缓冲区和变量
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  // 使用输入和输出初始化分析器
  analysis::MemDependencyChecker analyzer({a}, {b});

  // 创建一个 Relu 示例
  /*
   * for (int x = 0; x < 10; ++x) {
   *   B[x] = Max(A[x], 0);
   * }
   */

  // 加载表达式 aLoad = A[x]
  ExprHandle aLoad = Load::make(a, {x});
  // 存储操作 bStore = B[x] = Max(A[x], 0)
  StorePtr bStore = Store::make(b, {x}, Max::make(aLoad, 0, true));
  // 创建循环语句
  StmtPtr loop = For::make(x, 0, 10, bStore);

  // 将循环语句添加到代码块中
  StmtPtr stmt = Block::make({loop});

  // 分析器处理整个语句块
  stmt->accept(&analyzer);

  // 输出依赖于输入（间接依赖）
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));
  // aLoad 直接依赖于输入 A
  ASSERT_TRUE(analyzer.dependsDirectly(aLoad.node(), a.node()));
  // bStore 直接依赖于输入 A
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, a.node()));
  // 输出直接依赖于存储操作 bStore
  ASSERT_TRUE(analyzer.dependsDirectly(b.node(), bStore));

  // 使用 AccessInfo 进行检查
  auto input = analyzer.input(a.node());
  auto output = analyzer.output(b.node());

  // 输出间接依赖于输入
  ASSERT_TRUE(analyzer.dependsIndirectly(output, input));
  // 不是直接依赖
  ASSERT_FALSE(analyzer.dependsDirectly(output, input));
  // 不是反向顺序依赖
  ASSERT_FALSE(analyzer.dependsIndirectly(input, output));

  // output -> bStore -> bLoad -> input.
  auto storeAccess = analyzer.accessFor(bStore);
  auto loadAccess = analyzer.accessFor(aLoad.node());

  // 输出直接依赖于存储操作 bStore
  ASSERT_TRUE(analyzer.dependsDirectly(output, storeAccess));
  // 输入直接依赖于加载操作 aLoad
  ASSERT_TRUE(analyzer.dependsDirectly(loadAccess, input));
}

// 检测输出是否不依赖于输入
TEST(MemDependency, MemDependencyCheckerOutputDoesntDepend) {
  // 定义缓冲区和变量
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  // 使用输入和输出初始化分析器
  analysis::MemDependencyChecker analyzer({a}, {b});

  // 创建一个简单的 Relu 示例
  /*
   * for (int x = 0; x < 10; ++x) {
   *   B[x] = Max(x, 0);
   * }
   */

  // 存储操作 bStore = B[x] = Max(x, 0)
  StorePtr bStore = Store::make(b, {x}, Max::make(x, 0, true));
  // 创建循环语句
  StmtPtr loop = For::make(x, 0, 10, bStore);

  // 将循环语句添加到代码块中
  StmtPtr stmt = Block::make({loop});

  // 分析器处理整个语句块
  stmt->accept(&analyzer);

  // 输出不依赖于输入（间接依赖）
  ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), a.node()));

  // 输出依然直接依赖于存储操作 bStore
  ASSERT_TRUE(analyzer.dependsDirectly(b.node(), bStore));

  // 使用 AccessInfo 进行检查
  auto input = analyzer.input(a.node());
  auto output = analyzer.output(b.node());

  // 输出不依赖于输入（间接依赖）
  ASSERT_FALSE(analyzer.dependsIndirectly(output, input));
}
TEST(MemDependency, MemDependencyCheckerLoopBounds) {
  BufHandle a("A", {10}, kInt);  // 创建一个名为 "A" 的缓冲区，大小为 10，元素类型为整数
  BufHandle b("B", {10}, kInt);  // 创建一个名为 "B" 的缓冲区，大小为 10，元素类型为整数
  BufHandle c("C", {10}, kInt);  // 创建一个名为 "C" 的缓冲区，大小为 10，元素类型为整数
  VarHandle x("x", kInt);  // 创建一个名为 "x" 的变量，类型为整数
  using namespace analysis;  // 引入分析命名空间

  MemDependencyChecker analyzer({a}, {c});  // 创建内存依赖分析器，分析输入为 a，输出为 c

  // 启用循环执行顺序分析，以确定某些循环是否存在自依赖关系
  analyzer.allowLoopExecutionOrderAnalysis();

  /*
   * for (int x = 1; x < 10; ++x) {
   *   B[x] = A[x];
   * }
   * for (int x = 1; x < 9; ++x) {
   *   B[x] = B[x] * 2;
   * }
   * for (int x = 3; x < 4; ++x) {
   *   C[x] = A[x];
   * }
   * for (int x = 0; x < 10; ++x) {
   *   C[x] = B[x];
   * }
   */

  // 创建包含四个语句的向量，依次为四个 for 循环语句
  std::vector<StmtPtr> stmts(
      {For::make(x, 1, 10, Store::make(b, {x}, Load::make(a, {x}))),
       For::make(
           x, 1, 9, Store::make(b, {x}, Mul::make(Load::make(b, {x}), 2))),
       For::make(x, 3, 4, Store::make(c, {x}, Load::make(a, {x}))),
       For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x})))});

  StmtPtr stmt = Block::make(stmts);  // 创建一个包含以上四个语句的代码块

  stmt->accept(&analyzer);  // 分析器分析代码块中的依赖关系

  auto input = analyzer.input(a.node());  // 获取分析器中节点 a 的输入依赖
  auto output = analyzer.output(c.node());  // 获取分析器中节点 c 的输出依赖

  // 检查间接依赖关系：输出 c 是否依赖于输入 a
  ASSERT_TRUE(analyzer.dependsIndirectly(output, input));

  // 检查各个 for 循环的依赖关系：

  // C 的最后写入依赖于 B 的两次写入，因为它们至少包含一个元素的最后写入
  ASSERT_TRUE(analyzer.dependsIndirectly(stmts[3], stmts[1]));
  ASSERT_TRUE(analyzer.dependsIndirectly(stmts[3], stmts[0]));

  // C 的最后写入不依赖于对 C 的另一次写入
  ASSERT_FALSE(analyzer.dependsIndirectly(stmts[3], stmts[2]));

  auto CB = [](int s, int e) {
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
}

// 验证即使在循环变量偏移时仍能推断边界。
TEST(MemDependency, MemDependencyCheckerLoopBoundsIndexShift) {
  // 创建一个名为 "A" 的缓冲区，大小为 10，类型为整型
  BufHandle a("A", {10}, kInt);
  // 创建一个名为 "B" 的缓冲区，大小为 10，类型为整型
  BufHandle b("B", {10}, kInt);
  // 创建一个整型变量 x
  VarHandle x("x", kInt);

  // 引入 analysis 命名空间
  using namespace analysis;

  // 创建一个内存依赖检查器，检查器关注缓冲区 a 和 b 的依赖关系
  MemDependencyChecker analyzer({a}, {b});

  // 允许分析使用循环的执行顺序来确定某些循环是否自依赖
  analyzer.allowLoopExecutionOrderAnalysis();

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * for (int x = 0; x < 9; x++) {
   *   A[x] = A[x + 1];
   * }
   * for (int x = 0; x < 9; x++) {
   *   A[9 - x] = A[8 - x];
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = A[9 - x];
   * }
   * for (int x = 0; x < 10; x++) {
   *   B[x] = A[x];
   * }
   */

  // 创建一个复合语句 stmt，包含五个循环语句，分别实现上述的赋值操作
  StmtPtr stmt = Block::make(
      {For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}))),
       For::make(x, 0, 9, Store::make(a, {x}, Load::make(a, {x + 1}))),
       For::make(
           x,
           0,
           9,
           Store::make(
               a, {ExprHandle(9) - x}, Load::make(a, {ExprHandle(8) - x}))),
       For::make(
           x, 0, 10, Store::make(a, {x}, Load::make(a, {ExprHandle(9) - x}))),
       For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x})))});

  // 使用内存依赖分析器分析 stmt
  stmt->accept(&analyzer);

  // 断言：缓冲区 b 的输出依赖于缓冲区 a 的输入
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

  // 定义一个函数 CB，用于创建界限 Bound 对象
  auto CB = [](int s, int e) {
    return Bound(alloc<IntImm>(s), alloc<IntImm>(e));
  };
  // 定义一个函数 EQ，用于比较两个 IndexBounds 对象是否相等
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
}

// Check many different cases of loop self dependency - when a load within a
// loop is dependent on a Store later in the same loop but in different
// iteration. This is affected by whether or not we can trust the execution
// order of the loop.
TEST(MemDependency, MemDependencyCheckerLoopSelfDependency) {
  // 创建一个名为 "A" 的缓冲区，大小为 5，类型为整型
  BufHandle a("A", {5}, kInt);
  // 创建一个名为 "B" 的缓冲区，大小为 5，类型为整型
  BufHandle b("B", {5}, kInt);
  // 创建整型变量 x, y, z
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  // 引入 analysis 命名空间
  using namespace analysis;

  // 定义一个函数 isSelfDependent，检查历史记录中的访问信息是否存在自依赖
  auto isSelfDependent =
      [](const std::vector<std::shared_ptr<AccessInfo>>& history) -> bool {
    return history.front()->hasDependency(history.back());
  };

  {
    /* for (int y = 0; y < 10; y++) {
     *   A[y] = (A[y]) + 1;
     * } */

    // 不是自依赖，因为所有循环迭代使用不同的 y

    // 创建内存依赖检查器
    MemDependencyChecker analyzer;
    // 创建循环语句 stmt，实现对 A[y] 的赋值操作
    StmtPtr stmt = For::make(
        y,
        0,
        10,
        Block::make({Store::make(a, {y}, Add::make(Load::make(a, {y}), 1))}));

    // 使用内存依赖分析器分析 stmt
    stmt->accept(&analyzer);

    // 断言：历史记录中不存在自依赖
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int y = 0; y < 10; y++) {
     *   A[y + 1] = (A[y + 1]) + 1;
     * }
     */

    // 不是自依赖，因为不同的 y （带有偏移）

    // 创建内存依赖检查器
    MemDependencyChecker analyzer;
    {
      /* 创建一个 For 循环语句指针 stmt，循环从 x = 1 到 x < 10：
       *   A[x] = A[x - 1];
       */
    
      // 创建一个内存依赖分析器对象
      MemDependencyChecker analyzer;
    
      // 构造 For 循环语句，从 x = 1 到 x < 10，每次迭代执行 A[x] = A[x - 1];
      StmtPtr stmt =
          For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1})));
      // 使用依赖分析器分析该循环语句
      stmt->accept(&analyzer);
    
      // 断言该循环是否自相关，因为所有循环迭代都使用 A 的前一个元素
      ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    }
    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;
    // 允许循环执行顺序分析
    analyzer.allowLoopExecutionOrderAnalysis();

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt =
        For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：即使有顺序分析，由于写入操作发生在读取操作之前，加载操作仍依赖存储操作
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[9 - x] = A[8 - x];
     * }
     */

    // 即使执行顺序反转，只要读取操作在写入操作之前，该语句仍然有效

    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;
    // 允许循环执行顺序分析
    analyzer.allowLoopExecutionOrderAnalysis();

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(9) - x}, Load::make(a, {ExprHandle(8) - x})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：在这里可以确定存储操作在加载操作之前执行
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[8 - x] = A[9 - x];
     * }
     */

    // 但如果执行顺序不同，则不再有效

    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;
    // 允许循环执行顺序分析
    analyzer.allowLoopExecutionOrderAnalysis();

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(8) - x}, Load::make(a, {ExprHandle(9) - x})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：在这里可以确定加载操作依赖存储操作
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[9 - x] = A[8 - x];
     * }
     */

    // 如果不依赖执行顺序，则同样不再有效

    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(9) - x}, Load::make(a, {ExprHandle(8) - x})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：在这里可以确定加载操作依赖存储操作
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 3; x < 10; x++) {
     *   A[x - 2] = A[x - 1];
     * }
     */

    // 正序但负索引。

    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;
    // 允许循环执行顺序分析
    analyzer.allowLoopExecutionOrderAnalysis();

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt =
        For::make(x, 3, 10, Store::make(a, {x - 2}, Load::make(a, {x - 1})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：在这里可以确定存储操作在加载操作之前执行
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2];
     * }
     */

    // 使用访问步长。

    // 创建内存依赖检查器对象
    MemDependencyChecker analyzer;
    // 由于读取和写入是完全不同的，执行顺序并不重要
    // 所以不允许循环执行顺序分析

    // 创建一个 for 循环语句，对数组 A 进行操作，每次存储操作依赖上一次加载操作
    StmtPtr stmt =
        For::make(x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2})));
    // 分析并接受该语句的依赖
    stmt->accept(&analyzer);

    // 断言：在这里可以确定读取操作不依赖存储操作
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 1];
     * }
     */

    // 使用访问步长，但读取和写入不再相同。
  {
    // 这里我们可以使用访问的公共步长来确定它们是不同的。
    // 注意，这是唯一的地方（循环自依赖），我们使用这个步长以避免不必要的依赖。

    MemDependencyChecker analyzer;
    // 执行顺序无关紧要，因为读取和写入是完全独立的。

    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 1})));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 - 1];
     * }
     */

    // 如果读取在写入之前，只要它们是不同的，结果是相同的。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 1, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 - 1})));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 2];
     * }
     */

    // 但是如果偏移量在步长中，就不能保证它们是不同的。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 2})));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 - 2];
     * }
     */

    // 对负偏移量也有效。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 1, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 - 2})));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 7];
     * }
     */

    // 当偏移量较大但不是步长的倍数时，检测到访问是不同的。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 7})));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 4];
     * }
     */

    // 对步长的倍数偏移量也有效。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 4})));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 6] = A[x * 6 + 5];
     * }
     */

    // 当偏移量在步长内时，检测到访问是不同的。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 6}, Load::make(a, {x * 6 + 5})));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 6];
     * }
     */

    // 当步长较大时，偏移量在步长内时，检测到访问是不同的。

    MemDependencyChecker analyzer;
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6})));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt =
        For::make(x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt =
        For::make(x, 0, 10, Store::make(a, {x * 4}, Load::make(a, {x * 2})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6 + 1})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6 + 4})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2 + 3}, Load::make(a, {x * 6})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 3 + 1})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt =
        For::make(x, 0, 10, Store::make(a, {x}, Load::make(a, {x + 10})));
    // 对该 For 循环语句进行内存依赖分析
    stmt->accept(&analyzer);
    // 断言这些访问是否自相依赖
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
    
    
    
    // 创建一个用于内存依赖检查的对象
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，设置循环变量 x 的范围为 0 到 9，循环体内进行存储操作和加载操作
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x}, Load::make(a, {ExprHandle(9) - x})));
    stmt->accept(&analyzer);
    
    // 分析当前语句的内存依赖关系
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[19 - x * 2];
     * }
     */

    // 或者不会，这取决于它们的起始偏移和步长
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，用于将 A[x*2] 的值设为 A[19-x*2]
    StmtPtr stmt = For::make(
        x,
        0,
        10,
        Store::make(a, {x * 2}, Load::make(a, {ExprHandle(19) - x * 2})));
    stmt->accept(&analyzer);

    // 验证当前循环是否自依赖
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x / 2] = A[x / 2];
     * }
     */

    // 如果步长不单调，则它们会重叠
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，用于将 A[x/2] 的值设为 A[x/2]
    StmtPtr stmt =
        For::make(x, 0, 10, Store::make(a, {x / 2}, Load::make(a, {x / 2})));
    stmt->accept(&analyzer);

    // 验证当前循环是否自依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x / 2] = A[x / 2] + 1;
     * }
     */

    // 如果步长不单调，则它们会重叠 - 即使有偏移量
    MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，用于将 A[x/2] 的值设为 A[x/2 + 1]
    StmtPtr stmt = For::make(
        x, 0, 10, Store::make(a, {x / 2}, Load::make(a, {x / 2 + 1})));
    stmt->accept(&analyzer);

    // 验证当前循环是否自依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x % 2] = A[x % 2];
     * }
     */

    // 同样适用于模运算
    analysis::MemDependencyChecker analyzer;
    // 创建一个 For 循环语句，用于将 A[x%2] 的值设为 A[x%2]
    StmtPtr stmt = For::make(
        x,
        0,
        10,
        Store::make(a, {Mod::make(x, 2)}, Load::make(a, {Mod::make(x, 2)})));
    stmt->accept(&analyzer);

    // 验证当前循环是否自依赖
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = y; x < z; x++) {
     *   A[x] = A[x + 1];
     * }
     */

    // 对符号循环范围也适用
    {
      MemDependencyChecker analyzer;
      // 创建一个 For 循环语句，用于将 A[x] 的值设为 A[x+1]
      StmtPtr stmt =
          For::make(x, y, z, Store::make(a, {x}, Load::make(a, {x + 1})));
      stmt->accept(&analyzer);

      // 验证当前循环是否自依赖
      ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    }

    {
      MemDependencyChecker analyzer;
      // 允许分析循环执行顺序
      analyzer.allowLoopExecutionOrderAnalysis();
      // 创建一个 For 循环语句，用于将 A[x] 的值设为 A[x+1]
      StmtPtr stmt =
          For::make(x, y, z, Store::make(a, {x}, Load::make(a, {x + 1})));
      stmt->accept(&analyzer);

      // 验证当前循环是否自依赖
      ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
    }
  }
// Verify that a strided access still works.
// TODO: actually this only works because of the size of the ranges, revisit
// this test after strided overlap is implemented.
TEST(MemDependency, MemDependencyCheckerLoopDistinctStrides) {
  // 创建名为 A 的缓冲区，包含 20 个元素，每个元素为整数类型
  BufHandle a("A", {20}, kInt);
  // 创建名为 B 的缓冲区，包含 20 个元素，每个元素为整数类型
  BufHandle b("B", {20}, kInt);
  // 创建名为 x 的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建名为 y 的变量，类型为整数
  VarHandle y("y", kInt);

  using namespace analysis;
  // 创建内存依赖检查器对象，监测缓冲区 a 的依赖情况对缓冲区 b 的影响
  MemDependencyChecker analyzer({a.node()}, {b.node()});
  // 创建一个语句块，包含两个循环：
  StmtPtr stmt = Block::make(
      {For::make(
           x, 0, 10, Store::make(b, {x * 2 + 1}, Load::make(a, {x * 2 + 1}))),
       For::make(x, 0, 10, Store::make(b, {x * 2}, Load::make(a, {x * 2})))});

  // 分析这个语句块，查看缓冲区 b 是否间接依赖于缓冲区 a
  stmt->accept(&analyzer);

  // 确保输出表明缓冲区 b 的结果依赖于缓冲区 a
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

  // 输出应有两个依赖，分别对应于每个循环中的存储操作
  auto outputAccess = analyzer.output(b.node());
  ASSERT_EQ(outputAccess->dependencies().size(), 2);
}

/* TODO(nickg) - this test will fail due to the lack of stride math in Bound
TEST(MemDependency, MemDependencyCheckerLoopDistinctStrides) {
  BufHandle a("A", {20}, kInt);
  BufHandle b("B", {20}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    analysis::MemDependencyChecker analyzer({a.node()}, {c.node()});
    StmtPtr stmt = Block::make(
        {For::make(
             x,
             0,
             10,
             Store::make(b, {x * 2 + 1}, Load::make(a, {x * 2 + 1}))),
         For::make(
             x, 0, 10, Store::make(b, {x * 2}, Load::make(a, {x * 2}))),
         For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x})))

        });
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }
}*/

// analysis on Stmts using Cond.
TEST(MemDependency, MemDependencyCheckerLoopBoundsCond) {
  // 创建名为 A 的缓冲区，包含 10 个元素，每个元素为整数类型
  BufHandle a("A", {10}, kInt);
  // 创建名为 B 的缓冲区，包含 10 个元素，每个元素为整数类型
  BufHandle b("B", {10}, kInt);
  // 创建名为 C 的缓冲区，包含 10 个元素，每个元素为整数类型
  BufHandle c("C", {10}, kInt);
  // 创建名为 x 的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建名为 y 的变量，类型为整数
  VarHandle y("y", kInt);

  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (y<5 ? 1 : 0) {
     *   C[0] = (B[0]) + 1;
     * } else {
     *   C[0] = (B[1]) + 1;
     * }
     */

    // 未来的使用可能依赖于条件语句的两个分支中的访问

    // 创建内存依赖检查器对象，监测缓冲区 a 和 b 对缓冲区 c 的影响
    MemDependencyChecker analyzer({a, b}, {c});
    // 创建一个语句块，包含一个 for 循环和一个条件语句：
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             Store::make(c, {0}, Add::make(Load::make(b, {0}), 1)),
             Store::make(c, {0}, Add::make(Load::make(b, {1}), 1)))});

    // 分析这个语句块，查看缓冲区 c 的输出是否有三个依赖，对应于三个存储操作
    stmt->accept(&analyzer);

    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 3);
}
  // C depends indirectly on A and B.
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
}

{
  /* for (int x = 0; x < 10; x++) {
   *   C[x] = A[x];
   * }
   * if (y<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     C[x] = B[x];
   *   }
   * } else {
   *   for (int x = 0; x < 10; x++) {
   *     C[x] = (B[x]) + 1;
   *   }
   * }
   */

  // Future usages may depend on accesses in both branches of a condition.

  // 创建内存依赖检查器，指定分析对象a、b以及结果c
  MemDependencyChecker analyzer({a, b}, {c});
  // 构造语句块stmt，包括一个循环将A的值存储到C中，以及一个条件语句，根据y的值选择不同的操作
  StmtPtr stmt = Block::make(
      {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
       Cond::make(
           CompareSelect::make(y, 5, CompareSelectOperation::kLT),
           For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x}))),
           For::make(
               x,
               0,
               10,
               Store::make(c, {x}, Add::make(Load::make(b, {x}), 1))))});

  stmt->accept(&analyzer);

  // 检查C的输出，应该有3个依赖，对应三个存储操作
  auto outputAccess = analyzer.output(c.node());
  ASSERT_NE(outputAccess, nullptr);
  ASSERT_EQ(outputAccess->dependencies().size(), 3);

  // TODO(nickg): actually since the true and false branch cover the total
  // range of the first store this should have 2 dependencies, but we don't
  // do that yet.

  // C depends indirectly on A and B.
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
}

{
  /* for (int x = 0; x < 10; x++) {
   *   C[x] = A[x];
   * }
   * if (y<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     C[x] = (B[x]) + 1;
   *   }
   * }
   */

  // Only has true branch.

  // 创建内存依赖检查器，指定分析对象a、b以及结果c
  MemDependencyChecker analyzer({a, b}, {c});
  // 构造语句块stmt，包括一个循环将A的值存储到C中，以及一个条件语句，根据y的值选择不同的操作
  StmtPtr stmt = Block::make(
      {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
       Cond::make(
           CompareSelect::make(y, 5, CompareSelectOperation::kLT),
           For::make(
               x,
               0,
               10,
               Store::make(c, {x}, Add::make(Load::make(b, {x}), 1))),
           nullptr)});

  stmt->accept(&analyzer);

  // 检查C的输出，应该有2个依赖，对应两个存储操作
  auto outputAccess = analyzer.output(c.node());
  ASSERT_NE(outputAccess, nullptr);
  ASSERT_EQ(outputAccess->dependencies().size(), 2);

  // C depends indirectly on A and B.
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
  ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
}

{
  /* for (int x = 0; x < 10; x++) {
   *   C[x] = A[x];
   * }
   * if (y<5 ? 1 : 0) {
   * } else {
   *   for (int x = 0; x < 10; x++) {
   *     C[x] = (B[x]) + 1;
   *   }
   * }
   */

  // Only has false branch.

  // 创建内存依赖检查器，指定分析对象a、b以及结果c
  MemDependencyChecker analyzer({a, b}, {c});
    // 创建一个指向语句对象的指针，该语句是一个由多个子语句组成的代码块
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             nullptr,
             For::make(
                 x,
                 0,
                 10,
                 Store::make(c, {x}, Add::make(Load::make(b, {x}), 1))))});

    // 将语句对象传递给分析器，执行语句对象的访问操作
    stmt->accept(&analyzer);

    // 断言：输出 C 应该有三个依赖项，即三个存储操作
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);

    // 断言：C 间接依赖于 A 和 B
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (C[0]<5 ? 1 : 0) {
     *   C[0] = 5;
     * }
     */

    // 创建一个内存依赖检查器对象，指定输入和输出的存储
    MemDependencyChecker analyzer({a}, {c});
    
    // 初始化存储操作：C[x] = A[x];
    StorePtr initStore = Store::make(c, {x}, Load::make(a, {x}));
    
    // 条件加载表达式：条件取决于 C[0] 的先前访问
    ExprHandle conditionalLoad = Load::make(c, {0});
    
    // 创建一个语句对象，包含一个循环和一个条件语句
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, initStore),
         Cond::make(
             CompareSelect::make(
                 conditionalLoad, 5, CompareSelectOperation::kLT),
             Store::make(c, {0}, 5),
             nullptr)});

    // 将语句对象传递给分析器，执行语句对象的访问操作
    stmt->accept(&analyzer);

    // 断言：C 间接依赖于 A
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));

    // 断言：条件加载表达式直接依赖于初始化存储操作
    ASSERT_TRUE(analyzer.dependsDirectly(conditionalLoad.node(), initStore));
    // 断言：条件加载表达式不直接依赖于 A
    ASSERT_FALSE(analyzer.dependsDirectly(conditionalLoad.node(), a.node()));
    // 断言：条件加载表达式间接依赖于 A
    ASSERT_TRUE(analyzer.dependsIndirectly(conditionalLoad.node(), a.node()));
  }
// Stmts using IfThenElse.
TEST(MemDependency, MemDependencyCheckerIfThenElse) {
  // 创建名为A、B、C的缓冲区对象，每个大小为10，类型为整数
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  // 创建名为x、y的变量对象，类型为整数
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  // 引入分析命名空间
  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * C[0] = (y < 5 ? (B[0]) + 1 : (B[1]) + 1;
     */

    // 未来的使用可能取决于条件分支中的访问

    // 创建内存依赖检查器对象，分析A、B到C的依赖关系
    MemDependencyChecker analyzer({a, b}, {c});
    // 创建条件分支中的存储操作，根据y的值选择不同的存储内容到C[0]
    StorePtr ifStore = Store::make(
        c,
        {0},
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Add::make(Load::make(b, {0}), 1),
            Add::make(Load::make(b, {1}), 1)));
    // 创建语句块包含一个循环和条件分支的存储操作
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
         ifStore});

    // 将分析器应用于语句块
    stmt->accept(&analyzer);

    // 检查C的输出应该有两个依赖关系，即两个存储操作
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);

    // 检查条件分支存储操作的依赖关系应该有两个
    auto ifStoreAccess = analyzer.accessFor(ifStore);
    ASSERT_EQ(ifStoreAccess->dependencies().size(), 2);

    // C间接依赖于A和B
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * C[0] = (y < 5 ? (B[0]) + 1 : 42;
     */

    // 如果加载操作只出现在条件分支的一侧，输出可能依赖于它

    // 创建内存依赖检查器对象，分析A、B到C的依赖关系
    MemDependencyChecker analyzer({a, b}, {c});
    // 创建条件分支中的存储操作，根据y的值选择不同的存储内容到C[0]
    StorePtr ifStore = Store::make(
        c,
        {0},
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Add::make(Load::make(b, {0}), 1),
            42));
    // 创建语句块包含一个循环和条件分支的存储操作
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}))),
         ifStore});

    // 将分析器应用于语句块
    stmt->accept(&analyzer);

    // C间接依赖于A和B
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = (x < 5 ? B[x] : A[x];
     * }
     */

    // 在这种情况下，C依赖于A和B

    // TODO: 在这种情况下，可以将B的范围拆分为两个边界，一个依赖于A，一个依赖于B。我们需要检查与先前遇到的循环变量相关的条件，我不确定这是否有帮助。

    // 创建内存依赖检查器对象，分析A、B到C的依赖关系
    MemDependencyChecker analyzer({a, b}, {c});
    // 创建一个名为 ifStore 的 StorePtr 对象，用于存储条件语句的结果
    StorePtr ifStore = Store::make(
        c,
        {0},
        // 创建一个 IfThenElse 语句，根据 y 是否小于 5 来选择加载 b 或者 a 的值
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Load::make(b, {x}),  // 如果 y < 5，加载变量 b 在位置 x 的值
            Load::make(a, {x})   // 否则加载变量 a 在位置 x 的值
        )
    );
    // 创建一个 stmt 对象，表示一个由 for 循环和 ifStore 语句组成的代码块
    StmtPtr stmt = Block::make({For::make(x, 0, 10, ifStore)});
    
    // 使用 analyzer 对象分析 stmt 语句
    stmt->accept(&analyzer);
    
    // 断言：C 间接依赖于 A 和 B
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
}
// Cutting a loop with single elem writes
TEST(MemDependency, MemDependencyCheckerCutLoop) {
  BufHandle a("A", {10}, kInt);  // 创建一个名为"A"的缓冲区，大小为10，类型为整数
  BufHandle b("B", {10}, kInt);  // 创建一个名为"B"的缓冲区，大小为10，类型为整数
  VarHandle x("x", kInt);        // 创建一个名为"x"的变量句柄，类型为整数

  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   B[x] = A[x];
     * }
     * B[5] = 100;
     */

    // Cutting a loop with single element writes.

    MemDependencyChecker analyzer({a}, {b});  // 创建一个内存依赖分析器，分析A到B的依赖关系
    StmtPtr stmt = Block::make(
        {For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x}))),  // 创建一个循环，将A[x]写入B[x]
         Store::make(b, {5}, 100)});  // 在循环后将B[5]设置为100

    stmt->accept(&analyzer);  // 对语句进行依赖分析

    // Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));  // 断言输出受输入影响

    // Output has 2 dependencies.
    auto outputAccess = analyzer.output(b.node());  // 获取输出B的访问对象
    ASSERT_NE(outputAccess, nullptr);  // 断言输出访问对象非空
    ASSERT_EQ(outputAccess->dependencies().size(), 2);  // 断言输出依赖数量为2
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   B[x] = A[x];
     * }
     * for (int x = 4; x < 7; x++) {
     *   B[x] = B[x] + 3;
     * }
     * B[5] = 100;
     * B[6] = 101;
     * B[7] = 102;
     */

    // Cutting a loop with a smaller loop but then totally overlap that second
    // loop with one element writes.

    MemDependencyChecker analyzer({a}, {b});  // 创建另一个内存依赖分析器，分析A到B的依赖关系
    ForPtr firstLoop =
        For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x})));  // 创建第一个循环，将A[x]写入B[x]
    StorePtr secondStore =
        Store::make(b, {x}, Add::make(Load::make(b, {x}), 1));  // 创建一个存储操作，将B[x]增加1
    ForPtr secondLoop = For::make(x, 4, 7, secondStore);  // 创建第二个循环，修改B[4]到B[6]

    StmtPtr stmt = Block::make(
        {firstLoop,
         secondLoop,
         Store::make(b, {4}, 100),  // 在循环后将B[4]设置为100
         Store::make(b, {5}, 101),  // 将B[5]设置为101
         Store::make(b, {6}, 102)});  // 将B[6]设置为102

    stmt->accept(&analyzer);  // 对语句进行依赖分析

    // Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));  // 断言输出受输入影响

    // Output has 4 dependencies.
    auto outputAccess = analyzer.output(b.node());  // 获取输出B的访问对象
    ASSERT_NE(outputAccess, nullptr);  // 断言输出访问对象非空
    ASSERT_EQ(outputAccess->dependencies().size(), 4);  // 断言输出依赖数量为4

    // Second loop depends on first loop.
    ASSERT_TRUE(analyzer.dependsDirectly(secondLoop, firstLoop));  // 断言第二个循环直接依赖于第一个循环

    // Output does not depend on second loop or store.
    ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), secondLoop));  // 断言输出不依赖第二个循环
    ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), secondStore));  // 断言输出不依赖第二个存储操作
  }
}
    {
        /*  0. Input: B[(0, 99)] - dependents: 2
         *  1. Input: A[(0, 99)] - dependents: 3
         *  2. Load: B[(0, 0)] - depends on: 0  - dependents: 3 4
         *  3. Load: A[(0, (B[0]) - 1)] - depends on: 1 2  - dependents: 4
         *  4. Store: C[(0, (B[0]) - 1)] - depends on: 2 3  - dependents: 5
         *  5. Output: C[(0, 99)] - depends on: 4
         */
    
        // Output dependent on A input.
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
        // Also dependent on B input to determine the size of the region written.
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
    
        auto history = analyzer.getHistory();
        ASSERT_EQ(history.size(), 6);
    
        // The accesses in the loop depend on the load in the stop condition.
        ASSERT_TRUE(history[4]->hasDependency(history[2]));
        ASSERT_TRUE(history[3]->hasDependency(history[2]));
    
        // Make a load from B to compare against.
        ExprHandle loadFromB = Load::make(b, {0});
    
        ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, loadFromB - 1)}));
        ASSERT_TRUE(EQ(history[4]->bounds(), {CB(0, loadFromB - 1)}));
    }
    
    {
        /* for (int x = B[0]; x < B[1]; x++) {
         *   C[x] = A[x];
         * }
         */
        MemDependencyChecker analyzer({a, b}, {c});
        StmtPtr stmt = Block::make({For::make(
            x,
            Load::make(b, {0}),
            Load::make(b, {1}),
            Store::make(c, {x}, Load::make(a, {x})))});
    
        stmt->accept(&analyzer);
    
        /*  0. Input: B[(0, 99)] - dependents: 2 3
         *  1. Input: A[(0, 99)] - dependents: 4
         *  2. Load: B[(0, 0)] - depends on: 0  - dependents: 4 5
         *  3. Load: B[(1, 1)] - depends on: 0  - dependents: 4 5
         *  4. Load: A[(B[0], (B[1]) - 1)] - depends on: 1 2 3  - dependents: 5
         *  5. Store: C[(B[0], (B[1]) - 1)] - depends on: 2 3 4  - dependents: 6
         *  6. Output: C[(0, 99)] - depends on: 5
         */
    
        // Sanity check output depends on input.
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
    
        auto history = analyzer.getHistory();
        ASSERT_EQ(history.size(), 7);
    
        // The accesses in the loop depend on the load in the start condition.
        ASSERT_TRUE(history[5]->hasDependency(history[2]));
        ASSERT_TRUE(history[4]->hasDependency(history[2]));
    
        // also the stop condition.
        ASSERT_TRUE(history[5]->hasDependency(history[3]));
        ASSERT_TRUE(history[4]->hasDependency(history[3]));
    
        // Make loads from B to compare against.
        ExprHandle loadFromB0 = Load::make(b, {0});
        ExprHandle loadFromB1 = Load::make(b, {1});
        ASSERT_TRUE(EQ(history[4]->bounds(), {CB(loadFromB0, loadFromB1 - 1)}));
        ASSERT_TRUE(EQ(history[5]->bounds(), {CB(loadFromB0, loadFromB1 - 1)}));
    }
    
    {
        /* for (int x = 0; x < 10; x++) {
         *   C[x] = A[B[x]];
         * }
         */
        MemDependencyChecker analyzer({a, b}, {c});
        StmtPtr stmt = Block::make({For::make(
            x, 0, 10, Store::make(c, {x}, Load::make(a, {Load::make(b, {x})})))});
    
        stmt->accept(&analyzer);
    
        /*  0. Input: B[(0, 99)] - dependents: 2 3
         *  1. Input: A[(0, 99)] - dependents: 4
         *  2. Load: B[(0, 0)] - depends on: 0  - dependents: 4 5
         *  3. Load: B[(1, 1)] - depends on: 0  - dependents: 4 5
         *  4. Load: A[(B[0], (B[1]) - 1)] - depends on: 1 2 3  - dependents: 5
         *  5. Store: C[(0, 9)] - depends on: 4  - dependents: 6
         *  6. Output: C[(0, 9)] - depends on: 5
         */
    
        // Sanity check output depends on input.
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
        ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
    
        auto history = analyzer.getHistory();
        ASSERT_EQ(history.size(), 7);
    
        // The accesses in the loop depend on the load from B[x].
        ASSERT_TRUE(history[5]->hasDependency(history[4]));
    
        // Make loads from B to compare against.
        ExprHandle loadFromB0 = Load::make(b, {0});
        ExprHandle loadFromB1 = Load::make(b, {1});
        ASSERT_TRUE(EQ(history[4]->bounds(), {CB(loadFromB0, loadFromB1 - 1)}));
        ASSERT_TRUE(EQ(history[5]->bounds(), {CB(0, 9)}));
    }
    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 2
     *  1. Input: A[(0, 99)] - dependents: 3
     *  2. Load: B[(0, 9)] - depends on: 0  - dependents: 3 4
     *  3. Load: A[(B[0], B[9])] - depends on: 1 2  - dependents: 4
     *  4. Store: C[(0, 9)] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // 检查输出是否间接依赖于输入。
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // 存储操作依赖于两个加载操作，加载操作 A 依赖于加载操作 B。
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    ASSERT_TRUE(history[3]->hasDependency(history[2]));

    // 索引加载操作依赖于相关的输入缓冲区。
    ASSERT_TRUE(history[3]->hasDependency(history[1]));
    ASSERT_TRUE(history[2]->hasDependency(history[0]));

    // 从 B 加载具有循环边界。
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));

    // 从 A 加载的边界为 B[0] 到 B[9]。
    ExprHandle loadFromB0 = Load::make(b, {0});
    ExprHandle loadFromB9 = Load::make(b, {9});
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(loadFromB0, loadFromB9)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[B[x]] = A[x];
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    StmtPtr stmt = Block::make({For::make(
        x, 0, 10, Store::make(c, {Load::make(b, {x})}, Load::make(a, {x})))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 3
     *  1. Input: A[(0, 99)] - dependents: 2
     *  2. Load: A[(0, 9)] - depends on: 1  - dependents: 4
     *  3. Load: B[(0, 9)] - depends on: 0  - dependents: 4
     *  4. Store: C[(B[0], B[9])] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // 检查输出是否间接依赖于输入。
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // 存储操作依赖于两个加载操作，加载操作之间无依赖关系。
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    ASSERT_FALSE(history[3]->hasDependency(history[2]));
    ASSERT_FALSE(history[2]->hasDependency(history[3]));

    // 每个加载操作依赖于它们的相关输入。（但访问顺序与上一个情况不同）。
    ASSERT_TRUE(history[3]->hasDependency(history[0]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));

    // 从 B 加载的边界为循环范围。
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, 9)}));

    // 从 A 加载的边界也是循环范围。
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));
  }
    // 创建内存依赖检查器，指定输入和输出变量
    MemDependencyChecker analyzer({a, b}, {c});

    // 创建一个语句块对象，其中包含一个循环语句
    StmtPtr stmt = Block::make({
        // 创建一个 for 循环，循环次数为 10，循环体内进行存储操作
        For::make(x, 0, 10, Store::make(
            // 存储操作：将 x 存储到 C[B[A[x]]] 中
            c, {Load::make(b, {Load::make(a, {x})})}, x))
    });

    // 使用内存依赖检查器分析该语句块
    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 3
     *  1. Input: A[(0, 99)] - dependents: 2
     *  2. Load: A[(0, 9)] - depends on: 1  - dependents: 3 4
     *  3. Load: B[(A[0], A[9])] - depends on: 0 2  - dependents: 4
     *  4. Store: C[(B[A[0]], B[A[9]])] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // 断言：存储操作的结果 c 间接依赖于输入变量 a 和 b
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    // 获取分析历史
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // 断言：存储操作依赖于两个加载操作
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    // 断言：外部加载操作依赖于内部加载操作
    ASSERT_TRUE(history[3]->hasDependency(history[2]));

    // 断言：每个加载操作依赖于其相关的输入变量
    ASSERT_TRUE(history[3]->hasDependency(history[0]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));

    // 断言：来自 A 的加载操作具有循环边界
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));
    // 断言：来自 B 的加载操作的边界是 A[0] 到 A[9]
    ExprHandle loadFromA0 = Load::make(a, {0});
    ExprHandle loadFromA9 = Load::make(a, {9});
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(loadFromA0, loadFromA9)}));

    // 断言：存储操作的边界是 B[A[0]] 到 B[A[9]]
    ExprHandle loadFromBA0 = Load::make(b, {loadFromA0});
    ExprHandle loadFromBA9 = Load::make(b, {loadFromA9});
    ASSERT_TRUE(EQ(history[4]->bounds(), {CB(loadFromBA0, loadFromBA9)}));
// Verify multi dimensional bounds work.
TEST(MemDependency, MemDependencyCheckerMultiDim) {
  // 定义三维数组的尺寸
  int M = 10, N = 9, K = 12;
  // 创建三个缓冲区对象，表示数组 A、B、C
  BufHandle a("A", {M, N, K}, kInt);
  BufHandle b("B", {M, N, K}, kInt);
  BufHandle c("C", {M, K}, kInt);
  // 创建三个变量对象 x、y、z
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  // 导入 analysis 命名空间
  using namespace analysis;

  // 定义闭包 CB，返回一个绑定对象，表示边界
  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  // 定义闭包 EQ，比较两个索引边界对象是否相等
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  {
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 9; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, y, z] = A[x, y, z];
     *     }
     *   }
     * }
     */
    // 完整范围的三重循环

    // 创建 MemDependencyChecker 分析器对象，分析数组 A 和 B 的依赖关系
    MemDependencyChecker analyzer({a}, {b});
    // 创建语句块对象 stmt，包含三重循环和存储操作，将数组 A 的值复制给数组 B
    StmtPtr stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            N,
            For::make(
                z,
                0,
                K,
                Store::make(b, {x, y, z}, Load::make(a, {x, y, z})))))});

    // 使用分析器分析语句块
    stmt->accept(&analyzer);

    // 检查：输出依赖于输入
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 个访问：输入、加载、存储、输出
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // 简单的依赖链：从输入到输出
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    // 检查边界：与完整范围相符
    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
  }

  {
    /* for (int x = 0; x < 5; x++) {
     *   for (int y = 0; y < 5; y++) {
     *     for (int z = 0; z < 5; z++) {
     *       B[x, y, z] = A[x, y, z];
     *     }
     *   }
     * }
     */
    // 部分范围的三重循环

    // 创建另一个 MemDependencyChecker 分析器对象，分析数组 A 和 B 的依赖关系
    MemDependencyChecker analyzer({a}, {b});
    // 创建语句块对象 stmt，包含三重循环和存储操作，将数组 A 的值复制给数组 B
    StmtPtr stmt = Block::make({For::make(
        x,
        0,
        5,
        For::make(
            y,
            0,
            5,
            For::make(
                z,
                0,
                5,
                Store::make(b, {x, y, z}, Load::make(a, {x, y, z})))))});

    // 使用分析器分析语句块
    stmt->accept(&analyzer);

    // 检查：输出依赖于输入
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 个访问：输入、加载、存储、输出
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // 简单的依赖链：从输入到输出
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    // 检查边界：与部分范围相符
    ASSERT_TRUE(EQ(history[1]->bounds(), {CB(0, 4), CB(0, 4), CB(0, 4)}));
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 4), CB(0, 4), CB(0, 4)}));
  }
}
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 12; y++) {
     *     B[x, 0, y] = A[x, 0, y];
     *   }
     * }
     */

    // Partial loops.

    // 创建一个内存依赖分析器，指定输入为 a，输出为 b
    MemDependencyChecker analyzer({a}, {b});
    // 创建一个语句块，包含一个嵌套的 for 循环
    // 外层循环控制变量 x，范围从 0 到 N
    // 内层循环控制变量 y，范围从 0 到 K
    // 内部语句为将数组 A 的特定元素复制到数组 B 的特定位置
    StmtPtr stmt = Block::make({For::make(
        x,
        0,
        N,
        For::make(
            y, 0, K, Store::make(b, {x, 0, y}, Load::make(a, {x, 0, y}))))});

    // 使用内存依赖分析器分析该语句块
    stmt->accept(&analyzer);

    // 测试：确认输出 b 间接依赖于输入 a
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 次内存访问：输入、加载、存储、输出
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // 简单的从输入到输出的依赖链
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    // 确认加载操作的边界
    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, N - 1), CB(0, 0), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, N - 1), CB(0, 0), CB(0, K - 1)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 100; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, 0, z] = (A[x, 0, z]) + (C[x, z]);
     *     }
     *   }
     * }
     */

    // Loops that don't correspond to an index, bufs with different
    // dimensionality.

    // 创建一个内存依赖分析器，指定输入为 a 和 c，输出为 b
    MemDependencyChecker analyzer({a, c}, {b});
    // 创建一个语句块，包含三重嵌套的 for 循环
    // 最外层循环控制变量 x，范围从 0 到 M
    // 中间循环控制变量 y，范围从 0 到 100
    // 内层循环控制变量 z，范围从 0 到 K
    // 内部语句为将数组 A 和 C 的特定元素相加后复制到数组 B 的特定位置
    StmtPtr stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            100,
            For::make(
                z,
                0,
                K,
                Store::make(
                    b,
                    {x, 0, z},
                    Add::make(
                        Load::make(a, {x, 0, z}), Load::make(c, {x, z}))))))});

    // 使用内存依赖分析器分析该语句块
    stmt->accept(&analyzer);

    // 测试：确认输出 b 间接依赖于输入 a 和 c
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), c.node()));

    // 6 次内存访问：2 个输入、2 个加载、1 个存储、1 个输出
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // 简单的从输入到输出的依赖链，涉及数组 A 的依赖链
    // history[0] 是输入 C，history[3] 是从 C 加载的部分
    ASSERT_TRUE(history[5]->hasDependency(history[4]));
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    // 存储操作还依赖于从输入 C 加载的部分
    ASSERT_TRUE(history[4]->hasDependency(history[3]));
    ASSERT_TRUE(history[3]->hasDependency(history[0]));

    // A 数组访问的边界
    ASSERT_TRUE(
        EQ(history[4]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, K - 1)}));

    // C 数组访问的边界
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, M - 1), CB(0, K - 1)}));
  }
    /* for (int x = 0; x < 9; x++) {
     *   for (int y = 0; y < 10; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, 0, 0] = (B[x, y, z]) + (A[x, y, z]);
     *     }
     *   }
     * }
     */
    // 多维度的数据处理和归约操作。

    MemDependencyChecker analyzer({a}, {b});
    // 创建一个内存依赖检查器，分析变量a和b之间的依赖关系。
    StmtPtr stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            N,
            For::make(
                z,
                0,
                K,
                Store::make(
                    b,
                    {x, 0, 0},
                    Add::make(
                        Load::make(b, {x, y, z}),
                        Load::make(a, {x, y, z}))))))});
    // 创建一个包含三层嵌套的循环语句块，用于计算和存储操作。

    stmt->accept(&analyzer);
    // 分析这个语句块，收集内存访问历史信息。

    // 检查：输出依赖于输入。
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));
    // 确保输出变量b间接依赖于输入变量a。

    // 4次内存访问：输入、两次加载、存储、输出。
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 5);
    // 确保内存访问历史记录包含5个操作。

    // 简单的从输入到输出的依赖链。
    ASSERT_TRUE(history[4]->hasDependency(history[3]));
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[3]->hasDependency(history[1]));
    ASSERT_TRUE(history[2]->hasDependency(history[0]));
    // 确保历史记录中操作之间的依赖顺序正确。

    // 从B的加载操作到B的存储操作存在依赖关系。
    ASSERT_TRUE(history[1]->hasDependency(history[3]));
    // 确保从B的加载到B的存储之间存在正确的依赖关系。

    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, 0)}));
    // 确保历史记录中各操作的边界范围符合预期。
}
// Various tests using the external Compute/Reduce API.
TEST(MemDependency, MemDependencyCheckerComputeAPI) {
  // 引入分析命名空间
  using namespace analysis;

  /* for (int m = 0; m < 4; m++) {
   *   for (int n = 0; n < 5; n++) {
   *     for (int k = 0; k < 6; k++) {
   *       broadcast_add[m, n, k] = (a[m, n]) + (b[n, k]);
   *     }
   *   }
   * }
   * for (int m_1 = 0; m_1 < 4; m_1++) {
   *   for (int n_1 = 0; n_1 < 5; n_1++) {
   *     for (int k_1 = 0; k_1 < 6; k_1++) {
   *       d[m_1, n_1, k_1] = (broadcast_add(m_1, n_1, k_1)) + float(1);
   *     }
   *   }
   * }
   */

  // 创建名为 a 和 b 的缓冲区对象
  BufHandle a_buf("a", {4, 5}, kFloat);
  BufHandle b_buf("b", {5, 6}, kFloat);

  // 使用 Compute 函数创建张量 c，代表 broadcast_add 张量的计算
  Tensor c = Compute(
      "broadcast_add",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  // 使用 Compute 函数创建张量 d 的计算
  Tensor d = Compute(
      "d",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c.load(m, n, k) + 1;
      });

  // 创建循环嵌套对象 l，并指定计算对象 c 和 d 为其输入
  LoopNest l({d}, {c, d});

  // 创建内存依赖分析器对象 analyzer，分析 a_buf 和 b_buf 对 d 的内存依赖
  MemDependencyChecker analyzer({a_buf.node(), b_buf.node()}, {d.buf()});

  // 接受分析器对象来分析循环嵌套的根语句
  l.root_stmt()->accept(&analyzer);

  // 断言：d 对象间接依赖于 a_buf 和 b_buf 对象
  ASSERT_TRUE(analyzer.dependsIndirectly(d.buf(), a_buf.node()));
  ASSERT_TRUE(analyzer.dependsIndirectly(d.buf(), b_buf.node()));

  // 断言：第二个循环依赖于第一个循环
  auto c_loop = l.getLoopStmtsFor(c)[0];
  auto d_loop = l.getLoopStmtsFor(d)[0];
  ASSERT_TRUE(analyzer.dependsDirectly(d_loop, c_loop));
}

TEST(MemDependency, MemDependencyCheckerComputeInline) {
  // 引入分析命名空间
  using namespace analysis;

  /* for (int m = 0; m < 4; m++) {
   *   for (int n = 0; n < 5; n++) {
   *     for (int k = 0; k < 6; k++) {
   *       d[m, n, k] = ((a[m, n]) + (b[n, k])) + float(1);
   *     }
   *   }
   * }
   */

  // 创建名为 a 和 b 的缓冲区对象
  BufHandle a_buf("a", {4, 5}, kFloat);
  BufHandle b_buf("b", {5, 6}, kFloat);

  // 使用 Compute 函数创建张量 c，代表 broadcast_add 张量的计算
  Tensor c = Compute(
      "broadcast_add",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  // 使用 Compute 函数创建张量 d 的计算
  Tensor d = Compute(
      "d",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c.load(m, n, k) + 1;
      });

  // 创建循环嵌套对象 l，并指定计算对象 d 和 c 为其输入
  LoopNest l({d}, {c, d});

  // 将张量 c 内联计算
  l.computeInline(c.buf());

  // 创建内存依赖分析器对象 analyzer，分析 a_buf 和 b_buf 对 d 的内存依赖
  MemDependencyChecker analyzer({a_buf.node(), b_buf.node()}, {d.buf()});
  l.root_stmt()->accept(&analyzer);

  // 断言：d 对象间接依赖于 a_buf 和 b_buf 对象
  ASSERT_TRUE(analyzer.dependsIndirectly(d.buf(), a_buf.node()));
  ASSERT_TRUE(analyzer.dependsIndirectly(d.buf(), b_buf.node()));

  // 断言：broadcast_add 张量在跟踪中不应出现
  for (auto& wi : analyzer.getHistory()) {
    ASSERT_NE(wi->var(), c.buf()->base_handle());
  }
}
TEST(MemDependency, MemDependencyCheckerComputeSplit) {
  using namespace analysis;
  // 对于内存依赖分析，测试循环分割前后的影响

  // 创建两个缓冲区对象，表示数组 a 和 b，分别为 4x5 和 5x6 的浮点数数组
  BufHandle a_buf("a", {4, 5}, kFloat);
  BufHandle b_buf("b", {5, 6}, kFloat);
  
  // 创建张量对象 c，表示大小为 4x5x6 的计算张量
  Tensor c = Compute(
      "broadcast_add",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 张量 c 的计算逻辑，读取数组 a 和 b 中的数据进行相加
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  // 创建循环嵌套对象 l，用于分析和操作张量 c 的循环
  LoopNest l({c});

  // 创建内存依赖分析器 analyzer_before，分析 a 和 b 对 c 的影响
  MemDependencyChecker analyzer_before({a_buf.node(), b_buf.node()}, {c.buf()});
  l.root_stmt()->accept(&analyzer_before);

  // 在循环嵌套中对 c 的第一个循环进行分割，保留尾部
  l.splitWithTail(l.getLoopStmtsFor(c)[0], 2);

  // 创建分割后的内存依赖分析器 analyzer_after，再次分析 a 和 b 对 c 的影响
  MemDependencyChecker analyzer_after({a_buf.node(), b_buf.node()}, {c.buf()});
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  stmt->accept(&analyzer_after);

  // 检查分割前后内存访问的历史记录是否一致
  auto history_before = analyzer_before.getHistory();
  auto history_after = analyzer_after.getHistory();

  // 断言分割前后的历史记录大小相同
  ASSERT_EQ(history_before.size(), history_after.size());

  // 逐一检查历史记录中的每一项是否一致
  for (size_t i = 0; i < history_before.size(); ++i) {
    ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
    ASSERT_EQ(history_before[i]->var(), history_after[i]->var());
    ASSERT_EQ(
        history_before[i]->bounds().size(), history_after[i]->bounds().size());
    ASSERT_TRUE(indexBoundsEquals(
        history_before[i]->bounds(), history_after[i]->bounds()));
    ASSERT_EQ(
        history_before[i]->dependencies().size(),
        history_after[i]->dependencies().size());
    ASSERT_EQ(
        history_before[i]->dependents().size(),
        history_after[i]->dependents().size());
  }
}

TEST(MemDependency, MemDependencyCheckerComputeReorder) {
  using namespace analysis;
  // 对于内存依赖分析，测试循环重新排序前后的影响

  // 创建两个缓冲区对象，表示数组 a 和 b，分别为 4x5 和 5x6 的浮点数数组
  BufHandle a_buf("a", {4, 5}, kFloat);
  BufHandle b_buf("b", {5, 6}, kFloat);
  
  // 创建张量对象 c，表示大小为 4x5x6 的计算张量
  Tensor c = Compute(
      "broadcast_add",
      {4, 5, 6},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        // 张量 c 的计算逻辑，读取数组 a 和 b 中的数据进行相加
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  // 创建循环嵌套对象 l，用于分析和操作张量 c 的循环
  LoopNest l({c});

  // 创建内存依赖分析器 analyzer_before，分析 a 和 b 对 c 的影响
  MemDependencyChecker analyzer_before({a_buf.node(), b_buf.node()}, {c.buf()});
  l.root_stmt()->accept(&analyzer_before);

  // 获取张量 c 的循环语句列表，并对其进行重新排序
  auto loops = l.getLoopStmtsFor(c);
  l.reorderAxis(loops[0], loops[1]);

  // 创建重新排序后的内存依赖分析器 analyzer_after，再次分析 a 和 b 对 c 的影响
  MemDependencyChecker analyzer_after({a_buf.node(), b_buf.node()}, {c.buf()});
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  stmt->accept(&analyzer_after);

  // 检查重新排序前后内存访问的历史记录是否一致
  auto history_before = analyzer_before.getHistory();
  auto history_after = analyzer_after.getHistory();

  // 断言重新排序前后的历史记录大小相同
  ASSERT_EQ(history_before.size(), history_after.size());

  // 逐一检查历史记录中的每一项是否一致
  for (size_t i = 0; i < history_before.size(); ++i) {
    ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
    ASSERT_EQ(history_before[i]->var(), history_after[i]->var());
    ASSERT_EQ(
        history_before[i]->bounds().size(), history_after[i]->bounds().size());
    # 断言：验证两个对象的边界是否相等
    ASSERT_TRUE(indexBoundsEquals(
        history_before[i]->bounds(), history_after[i]->bounds()));

    # 断言：验证两个对象的依赖列表大小是否相等
    ASSERT_EQ(
        history_before[i]->dependencies().size(),
        history_after[i]->dependencies().size());

    # 断言：验证两个对象的被依赖列表大小是否相等
    ASSERT_EQ(
        history_before[i]->dependents().size(),
        history_after[i]->dependents().size());
TEST(MemDependency, MemDependencyCheckerComputeGEMM) {
  // 设置矩阵维度
  int M = 1024;
  int N = 1024;
  int K = 2048;
  // 引入分析命名空间
  using namespace analysis;

  // 定义矩阵缓冲区及其形状
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  // 创建张量 CT 作为矩阵乘积的结果
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});

  // 创建循环嵌套对象 loop，并将 CT 添加进去
  LoopNest loop({CT});

  // 按照指定的掩码分割循环 m
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr m = loops[0];
    loop.splitWithMask(m, 4);
  }

  // 按照指定的掩码分割循环 n
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }

  // 重新排序轴 mi 和 no
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr mi = loops[1];
    ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }

  // 重新排序轴 ni 和 k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr ni = loops[3];
    ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }

  // 重新排序轴 mi 和 k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr mi = loops[2];
    ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }

  // 获取 CT 的循环语句列表，但没有添加到循环中
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    // 调用 loop 对象的 cacheAccesses 方法，记录访问缓存的信息，针对 C_regs 缓存，使用 loops[2] 作为索引
    loop.cacheAccesses(CT.buf(), "C_regs", loops[2]);
  }

  // 创建未降级和降级分析器对象，分别传入 loop 对象的输入和输出缓存
  MemDependencyChecker analyzer_unlowered(
      loop.getInputBufs(), loop.getOutputBufs());

  MemDependencyChecker analyzer_lowered(
      loop.getInputBufs(), loop.getOutputBufs());

  // 测试未降级和降级形式的代码
  {
    // 简化 loop 根语句并存储在 stmt 中
    StmtPtr stmt = IRSimplifier::simplify(loop.root_stmt());
    // 接受未降级分析器对象，并分析语句的依赖关系
    stmt->accept(&analyzer_unlowered);

    // 断言输出依赖于输入
    ASSERT_TRUE(analyzer_unlowered.dependsIndirectly(CT.buf(), AP.node()));
    ASSERT_TRUE(analyzer_unlowered.dependsIndirectly(CT.buf(), BP.node()));

    // 最后一次写入 gemm 应该覆盖输出的总范围
    std::shared_ptr<AccessInfo> outputAccess =
        analyzer_unlowered.output(CT.buf());
    // 一个依赖关系
    ASSERT_EQ(outputAccess->dependencies().size(), 1);

    // 获取 gemm 存储的访问信息
    std::shared_ptr<AccessInfo> gemmStore =
        outputAccess->dependencies().begin()->second;
    // 确认是存储类型
    ASSERT_EQ(gemmStore->type(), AccessType::Store);

    // 确认输出边界与 gemm 存储的边界相等
    ASSERT_TRUE(indexBoundsEquals(outputAccess->bounds(), gemmStore->bounds()));

    // 类似地，第一次从每个输入读取的范围应该覆盖整个输入的范围
    auto aInput = analyzer_unlowered.input(AP.node());
    auto bInput = analyzer_unlowered.input(BP.node());

    // 每个只有一个依赖
    ASSERT_EQ(aInput->dependents().size(), 1);
    ASSERT_EQ(bInput->dependents().size(), 1);

    // 它们都是加载类型
    std::shared_ptr<AccessInfo> aLoad = aInput->dependents().begin()->second;
    std::shared_ptr<AccessInfo> bLoad = bInput->dependents().begin()->second;
    ASSERT_EQ(aLoad->type(), AccessType::Load);
    ASSERT_EQ(bLoad->type(), AccessType::Load);

    // 确认输入边界与加载的边界相等
    ASSERT_TRUE(indexBoundsEquals(aInput->bounds(), aLoad->bounds()));
    ASSERT_TRUE(indexBoundsEquals(bInput->bounds(), bLoad->bounds()));
  }

  // 准备循环代码生成
  loop.prepareForCodegen();
  // 使用 SimpleIREvaluator 对象 cg 进行简单的 IR 评估，包括 AP、BP、CT 作为输入
  SimpleIREvaluator cg(loop.root_stmt(), {AP, BP, CT});

  // 现在检查降级后的依赖图
  {
    // 简化 cg 的语句并存储在 stmt 中
    StmtPtr stmt = IRSimplifier::simplify(cg.stmt());
    // 接受降级分析器对象，并分析语句的依赖关系
    stmt->accept(&analyzer_lowered);

    // 由于索引扁平化，边界维度改变，将插入 Allocates 和 Frees，这些操作会改变历史记录的大小
    auto history_before = analyzer_unlowered.getHistory();
    auto history_after = analyzer_lowered.getHistory();

    // 断言降级前后历史记录的大小相差 2
    ASSERT_EQ(history_before.size() + 2, history_after.size());

    // 定义一个函数，用于过滤出 Alloc 和 Free 类型的访问信息
    auto isAllocFree = [](const auto& info) {
      return info->type() == AccessType::Alloc ||
          info->type() == AccessType::Free;
    };

    // 从降级后的历史记录中移除所有的 Alloc 和 Free 类型的访问信息
    history_after.erase(
        std::remove_if(history_after.begin(), history_after.end(), isAllocFree),
        history_after.end());

    // 断言降级前后历史记录的大小相等
    ASSERT_EQ(history_before.size(), history_after.size());
    // 遍历 history_before 容器中的每个元素
    for (size_t i = 0; i < history_before.size(); ++i) {
      // 断言当前索引的历史记录对象的类型相等
      ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
      // 断言当前索引的历史记录对象的变量相等
      ASSERT_EQ(history_before[i]->var(), history_after[i]->var());

      // 如果当前历史记录对象的依赖项数量不相等
      if (history_before[i]->dependencies().size() !=
          history_after[i]->dependencies().size()) {
        // 必须依赖于一个 Alloc 类型的操作
        ASSERT_TRUE(std::any_of(
            history_after[i]->dependencies().begin(),
            history_after[i]->dependencies().end(),
            [](const auto& pair) {
              return pair.second->type() == AccessType::Alloc;
            }));

        // 断言修改后的依赖项数量是修改前的数量加一
        ASSERT_EQ(
            history_before[i]->dependencies().size() + 1,
            history_after[i]->dependencies().size());
      }

      // 如果当前历史记录对象的被依赖项数量不相等
      if (history_before[i]->dependents().size() !=
          history_after[i]->dependents().size()) {
        // 必须被一个 Free 类型的操作依赖
        ASSERT_TRUE(std::any_of(
            history_after[i]->dependents().begin(),
            history_after[i]->dependents().end(),
            [](const auto& pair) {
              return pair.second->type() == AccessType::Free;
            }));

        // 断言修改后的被依赖项数量是修改前的数量加一
        ASSERT_EQ(
            history_before[i]->dependents().size() + 1,
            history_after[i]->dependents().size());
      }

      // 对于输入和输出类型的历史记录对象，不进行扁平化处理，只处理访问操作
      if (history_before[i]->type() == AccessType::Input ||
          history_before[i]->type() == AccessType::Output) {
        // 断言修改前后的边界数量相等
        ASSERT_EQ(
            history_before[i]->bounds().size(),
            history_after[i]->bounds().size());
        // 断言修改前后的边界内容相等
        ASSERT_TRUE(indexBoundsEquals(
            history_before[i]->bounds(), history_after[i]->bounds()));
      } else {
        // 如果历史记录对象不是输入或输出类型，则断言修改后的边界数量为1
        ASSERT_EQ(history_after[i]->bounds().size(), 1);
        // 创建一个整数常量表达式指针 flat_bounds
        ExprPtr flat_bounds = alloc<IntImm>(1);

        // 遍历当前历史记录对象的边界
        for (auto& b : history_before[i]->bounds()) {
          // 更新 flat_bounds 为新的表达式，计算乘法和加法结果
          flat_bounds =
              alloc<Mul>(flat_bounds, alloc<Add>(b.end, alloc<IntImm>(1)));

          // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
          // 断言修改后的起始边界与修改前的相等
          ASSERT_TRUE(exprEquals(b.start, history_after[i]->bounds()[0].start));
        }

        // 简化 flat_bounds 表达式
        flat_bounds = IRSimplifier::simplify(flat_bounds);
        // 计算修改后的边界表达式
        ExprPtr after_bounds = IRSimplifier::simplify(
            alloc<Add>(history_after[i]->bounds()[0].end, alloc<IntImm>(1)));
        // 断言 flat_bounds 与 after_bounds 表达式相等
        ASSERT_TRUE(exprEquals(flat_bounds, after_bounds));
      }
    }
  }
}

} // namespace jit
} // namespace torch
```