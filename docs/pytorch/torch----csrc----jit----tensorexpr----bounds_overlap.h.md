# `.\pytorch\torch\csrc\jit\tensorexpr\bounds_overlap.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/tensorexpr/expr.h>
// 引入 Torch 的表达式库中的表达式头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
// 引入 Torch 的表达式库中的 IR 头文件

#include <deque>
// 引入 deque（双端队列）容器的头文件
#include <utility>
// 引入 utility（实用程序）标准库的头文件
#include <vector>
// 引入 vector（向量）容器的头文件

namespace torch {
namespace jit {
namespace tensorexpr {
namespace analysis {

// 一个简单的类，包含单个维度中范围的起始和结束
struct TORCH_API Bound {
  ExprPtr start{nullptr};
  ExprPtr end{nullptr};

  // 这个标志用来指示此 Bound 的起始和结束是否已经交换过。
  // 当 Bound 在具有负步长的循环中时会发生这种情况。
  bool swapped{false};

  Bound() = default;
  Bound(ExprPtr s, ExprPtr e) : start(std::move(s)), end(std::move(e)) {}

  // 打印 Bound 对象的内容
  void print() const;
  // 比较两个 Bound 对象是否相等
  bool equals(const Bound& other) const;

  // 比较运算符是保守的。如果比较运算符返回 true，则意味着所有元素都满足逻辑表达式。
  // 但 false 不意味着相反的比较一定成立，有时也可能成立。
  bool operator==(const Bound& other) const;
  bool operator!=(const Bound& other) const;
  bool operator<(const Bound& other) const;
  bool operator<=(const Bound& other) const;
  bool operator>(const Bound& other) const;
  bool operator>=(const Bound& other) const;

  // 交换 Bound 的起始和结束
  void swap() {
    std::swap(start, end);
    swapped = !swapped;
  }
};

// 用于对 Bound 类型对象进行哈希的函数对象
struct BoundHash {
  size_t operator()(const Bound& b) const {
    // 对 Bound 的起始和结束表达式进行哈希
    return std::hash<ExprPtr>()(b.start) ^ std::hash<ExprPtr>()(b.end);
  }
};

// 找到两个 Bound A 和 B 之间的重叠类型
// ContainedOrEqual: Bound A 中的所有元素都在 Bound B 中（包括相等的情况）
// Contains: Bound B 中的所有元素都在 Bound A 中
// PartialOverlap: Bound B 中的任意元素在 Bound A 中
// NoOverlap: Bound A 中的元素没有在 Bound B 中
enum class OverlapKind {
  ContainedOrEqual,
  Contains,
  PartialOverlap,
  NoOverlap
};

// Bound 比较的结果类型
// True: 每个 Bound 元素总是满足给定的比较运算符
// False: 每个 Bound 元素总是不满足给定的比较运算符
// NotDetermined: 一些元素满足给定的比较运算符，一些元素不满足
enum class CmpEvalResult { True, False, NotDetermined };

// 返回两个 Bound A 和 B 在单个维度上的重叠类型
OverlapKind TORCH_API boundOverlap(Bound A, Bound B);

// 进行 Bound 对象之间的比较，返回比较结果
// 这个比较是保守的，并且比较结果是确定性的。
// 意味着要比较的每个 Bound 元素都必须满足给定的比较运算符。
CmpEvalResult TORCH_API compareBound(
    const Bound& a,
    const Bound& b,
    const CompareSelectOperation& cmp_op);

// 表示一组索引的多维 Bound 类型
using IndexBounds = std::vector<Bound>;
// 使用向量容器表示一组索引的边界
// 检查两个 IndexBounds 是否等价，如果相等返回 true。
bool TORCH_API indexBoundsEquals(const IndexBounds& A, const IndexBounds& B);

// 将多维边界压缩为单一维度的边界。IndexBounds "a" 必须完全包含缓冲区的整个范围。
Bound TORCH_API flattenBounds(const IndexBounds& a);

// 确定 X 维度中的重叠类型。
OverlapKind TORCH_API overlaps(const IndexBounds& a, const IndexBounds& b);

// 返回通过从边界 B 中减去边界 A 创建的 Bound 切片。
// 如果 B 将 A 切成两个不重叠的区域，则可能返回多个 Bound。
//
// 例如：
//    subtractBound((0, 10), (2, 4)) => [(0, 1), (5, 10)]
//       边界 A: (0, 10)
//       边界 B: (2, 4)
//       如果从边界 (0, 10) 中移除切片 (2, 4)，则剩余两个切片，一个在开头 (0, 1)，一个在结尾 (5, 10)。
//       因此，这个减法的结果是 [(0, 1), (5, 10)]。
//
// 注意：这里不使用 IndexBounds，因为返回的 Bound 不表示多个不同的维度。
std::vector<Bound> TORCH_API subtractBound(Bound a, Bound b);

// 返回通过从 IndexBounds B 中减去 IndexBounds A 创建的边界切片。
std::vector<IndexBounds> TORCH_API subtractIndicesBounds(
    const IndexBounds& A,
    const IndexBounds& B,
    OverlapKind overlap);
std::vector<IndexBounds> TORCH_API
subtractIndicesBounds(const IndexBounds& A, const IndexBounds& B);
```