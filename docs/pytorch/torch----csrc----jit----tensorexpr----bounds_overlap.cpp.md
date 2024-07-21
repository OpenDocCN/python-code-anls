# `.\pytorch\torch\csrc\jit\tensorexpr\bounds_overlap.cpp`

```py
// 引入头文件以便访问相关库函数和类
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

// 引入标准输入输出流库
#include <iostream>

// 命名空间定义
namespace torch::jit::tensorexpr::analysis {

// 返回表达式是否必定为正数
static bool mustBePositive(ExprPtr e) {
  // 如果表达式是常量
  if (e->isConstant()) {
    // 获取表达式的整数值
    int e_val = immediateAs<int>(e);
    // 判断表达式是否大于0
    return e_val > 0;
  }
  return false;
}

// 返回表达式是否必定为负数
static bool mustBeNegative(ExprPtr e) {
  // 如果表达式是常量
  if (e->isConstant()) {
    // 获取表达式的整数值
    int e_val = immediateAs<int>(e);
    // 判断表达式是否小于0
    return e_val < 0;
  }
  return false;
}

// 返回表达式是否必定为零
static bool mustBeZero(ExprPtr e) {
  // 如果表达式是常量
  if (e->isConstant()) {
    // 获取表达式的整数值
    int e_val = immediateAs<int>(e);
    // 判断表达式是否等于0
    return e_val == 0;
  }
  return false;
}

// 打印边界对象的起始和结束表达式
void Bound::print() const {
  std::cout << "(" << *start << ", " << *end << ")";
}

// 比较边界对象是否相等
bool Bound::equals(const Bound& other) const {
  // 判断起始表达式和结束表达式是否相等
  return exprEquals(start, other.start) && exprEquals(end, other.end);
}

// 比较边界对象是否相等
bool Bound::operator==(const Bound& other) const {
  // 如果边界对象相等
  if (equals(other)) {
    // 简化起始和结束表达式之差，判断是否为零
    auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, end));
    return mustBeZero(ret_expr);
  }

  return false;
}

// 比较边界对象是否不相等
bool Bound::operator!=(const Bound& other) const {
  // 判断边界对象是否小于或大于另一个对象
  return (*this < other) || (*this > other);
}

// 比较边界对象是否大于等于另一个对象
bool Bound::operator>=(const Bound& other) const {
  // 如果边界对象相等
  if (*this == other) {
    return true;
  }
  // 简化起始和结束表达式之差，判断是否为正数或零
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, other.end));
  return mustBePositive(ret_expr) || mustBeZero(ret_expr);
}

// 比较边界对象是否大于另一个对象
bool Bound::operator>(const Bound& other) const {
  // 简化起始和结束表达式之差，判断是否为正数
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, other.end));
  return mustBePositive(ret_expr);
}

// 比较边界对象是否小于等于另一个对象
bool Bound::operator<=(const Bound& other) const {
  // 如果边界对象相等
  if (*this == other) {
    return true;
  }
  // 简化起始和结束表达式之差，判断是否为负数或零
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(end, other.start));
  return mustBeNegative(ret_expr) || mustBeZero(ret_expr);
}

// 比较边界对象是否小于另一个对象
bool Bound::operator<(const Bound& other) const {
  // 简化起始和结束表达式之差，判断是否为负数
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(end, other.start));
  return mustBeNegative(ret_expr);
}

// 判断两个边界对象的重叠类型
OverlapKind boundOverlap(Bound a, Bound b) {
  // 如果起始和结束表达式都相等
  bool startEqual = exprEquals(a.start, b.start);
  bool endEqual = exprEquals(a.end, b.end);
  if (startEqual && endEqual) {
  // 返回重叠类型为 ContainedOrEqual
  return OverlapKind::ContainedOrEqual;
}

// 我们需要确定边界属于以下两种情况之一：
// 1. a 在 b 之前：
//      a.start ... a.end ... b.start ... b.end
// 2. b 在 a 之前：
//      b.start ... b.end ... a.start ... a.end
//
// 因此，我们计算 "a.start - b.end" 和 "b.start - a.end"。
// 如果其中至少有一个是正数，则边界肯定不重叠。
//
// 如果差异是一个常量，则可以直接检查该常量是否为正数。
// 如果差异不是一个常量，则它将由对应于涉及缓冲区边界的变量组成。
// 这些缓冲区边界永远不可能是负数。因此，我们检查假设相关变量永远不为负时，给定表达式是否保证为正数。

ExprPtr lowDiff = IRSimplifier::simplify(alloc<Sub>(a.start, b.end));
ExprPtr highDiff = IRSimplifier::simplify(alloc<Sub>(b.start, a.end));

// 如果 lowDiff 必须为正数，则返回 NoOverlap
if (mustBePositive(lowDiff)) {
  return OverlapKind::NoOverlap;
}
// 如果 highDiff 必须为正数，则返回 NoOverlap
if (mustBePositive(highDiff)) {
  return OverlapKind::NoOverlap;
}

// 计算 b.start - a.start 和 b.end - a.end 的差异
ExprPtr diff_start = IRSimplifier::simplify(alloc<Sub>(b.start, a.start));
ExprPtr diff_end = IRSimplifier::simplify(alloc<Sub>(b.end, a.end));

// 如果一侧完全包含另一侧，则它们是相邻的。
if (diff_start->isConstant() && diff_end->isConstant()) {
  int start = immediateAs<int>(diff_start);
  int end = immediateAs<int>(diff_end);
  // 如果 diff_start 和 diff_end 有不同的符号，则表示包含。
  if (start <= 0 && end >= 0) {
    return OverlapKind::ContainedOrEqual;
  }

  if (start >= 0 && end <= 0) {
    return OverlapKind::Contains;
  }
}

// 如果无法确定没有重叠，则保守地返回 PartialOverlap
return OverlapKind::PartialOverlap;
}

CmpEvalResult TORCH_API compareBound(
    const Bound& a,
    const Bound& b,
    const CompareSelectOperation& cmp_op) {
  switch (cmp_op) {
    case CompareSelectOperation::kGT:
      // 如果 a 大于 b，则返回 True；否则根据 a 是否小于等于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a > b)
          ? CmpEvalResult::True
          : (a <= b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kGE:
      // 如果 a 大于等于 b，则返回 True；否则根据 a 是否小于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a >= b)
          ? CmpEvalResult::True
          : (a < b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kLT:
      // 如果 a 小于 b，则返回 True；否则根据 a 是否大于等于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a < b)
          ? CmpEvalResult::True
          : (a >= b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kLE:
      // 如果 a 小于等于 b，则返回 True；否则根据 a 是否大于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a <= b)
          ? CmpEvalResult::True
          : (a > b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kNE:
      // 如果 a 不等于 b，则返回 True；否则根据 a 是否等于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a != b)
          ? CmpEvalResult::True
          : (a == b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    default:
      // 断言默认情况下 cmp_op 应该是 CompareSelectOperation::kEQ
      TORCH_INTERNAL_ASSERT(cmp_op == CompareSelectOperation::kEQ)
      // 如果 a 等于 b，则返回 True；否则根据 a 是否不等于 b 返回 True 或者 False，或者返回 NotDetermined
      return (a == b)
          ? CmpEvalResult::True
          : (a != b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
  }
}

bool indexBoundsEquals(const IndexBounds& A, const IndexBounds& B) {
  // 如果 A 和 B 的大小不相等，则返回 false
  if (A.size() != B.size()) {
    return false;
  }

  // 遍历 A 和 B 的元素，如果任何对应位置的元素不相等，则返回 false
  for (size_t i = 0; i != A.size(); ++i) {
    if (!A[i].equals(B[i])) {
      return false;
    }
  }
  // 若所有元素对应位置都相等，则返回 true
  return true;
}

Bound flattenBounds(const IndexBounds& a) {
  // 如果 a 是空的，则返回一个默认构造的 Bound 对象
  if (a.empty()) {
    return Bound();
  }
  // 取出 a 的第一个元素作为 ret 的初始值
  Bound ret = a[0];

  // 遍历 a 的剩余元素，将 ret 的 start 和 end 分别与 a[i] 的 start 和 end 相乘并分配内存
  for (size_t i = 1; i < a.size(); ++i) {
    ret.start = alloc<Mul>(ret.start, a[i].start);
    ret.end = alloc<Mul>(ret.end, a[i].end);
  }

  // 简化 ret 的 start 和 end 并返回
  ret.start = IRSimplifier::simplify(ret.start);
  ret.end = IRSimplifier::simplify(ret.end);
  return ret;
}

OverlapKind overlaps(const IndexBounds& a, const IndexBounds& b) {
  // 如果 a 和 b 都为空，则返回 ContainedOrEqual
  if (a.empty() && b.empty()) {
    return OverlapKind::ContainedOrEqual;
  }

  // 如果 a 和 b 的大小不相等，则将其扁平化后再比较
  if (a.size() != b.size()) {
    return boundOverlap(flattenBounds(a), flattenBounds(b));
  }
  // 断言 a 和 b 的大小相等
  TORCH_INTERNAL_ASSERT(a.size() == b.size());

  // 初始时假设 overlap 为第一个 boundOverlap 的结果
  OverlapKind overlap = boundOverlap(a[0], b[0]);
  for (size_t i = 1; i < a.size(); ++i) {
    // 计算当前两个 boundOverlap 的结果
    OverlapKind bOverlap = boundOverlap(a[i], b[i]);
    // 如果有 NoOverlap，直接返回 NoOverlap
    if (bOverlap == OverlapKind::NoOverlap) {
      return OverlapKind::NoOverlap;
    }

    // 更新 overlap 的值根据当前的 bOverlap
    if (overlap == OverlapKind::ContainedOrEqual &&
        bOverlap == OverlapKind::Contains) {
      overlap = OverlapKind::Contains;
    }

    if (overlap == OverlapKind::Contains &&
        bOverlap == OverlapKind::ContainedOrEqual) {
      continue;
    }

    // 如果当前的 bOverlap 与之前的 overlap 不一致，则设置为 PartialOverlap 并跳出循环
    if (bOverlap != overlap) {
      overlap = OverlapKind::PartialOverlap;
      break;
    }
  }

  return overlap;
}

std::vector<Bound> subtractBound(Bound a, Bound b) {
  // 计算 a 和 b 的 boundOverlap 的结果
  OverlapKind overlap = boundOverlap(a, b);
  // 如果没有重叠，则直接返回包含 a 的 vector
  if (overlap == OverlapKind::NoOverlap) {
    return {a};
  }
  // 如果 b 包含或者等于 a，则返回一个空的 vector
  if (overlap == OverlapKind::ContainedOrEqual) {
    return {};
  }
    // 返回一个空的 vector<Bound> 对象
    return {};
  }

  // 创建一个空的 vector<Bound> 对象，用于存储结果
  std::vector<Bound> res;

  // 检查边界是否重叠
  if (a.start->isConstant() != b.start->isConstant() ||
      a.end->isConstant() != b.end->isConstant()) {
    // 如果起始点或结束点的常量性质不一致，直接返回包含 a 的 vector<Bound> 对象
    return {a};
  }

  // 计算起始点和结束点之间的差值
  ExprPtr lowDiff = IRSimplifier::simplify(alloc<Sub>(b.start, a.start));
  ExprPtr highDiff = IRSimplifier::simplify(alloc<Sub>(b.end, a.end));

  // 如果差值中只包含一个变量，尝试推测其符号
  if (!lowDiff->isConstant()) {
    // 查找差值中的变量
    auto vars = VarFinder::find(lowDiff);
    if (vars.size() == 1) {
      // 替换差值中的变量为常量 1，重新计算差值
      lowDiff = IRSimplifier::simplify(alloc<Sub>(
          SubstituteInClone(b.start, {{*vars.begin(), immLike(b.start, 1)}}),
          SubstituteInClone(a.start, {{*vars.begin(), immLike(a.start, 1)}})));
    }
  }

  if (!highDiff->isConstant()) {
    // 查找高端差值中的变量
    auto vars = VarFinder::find(highDiff);
    if (vars.size() == 1) {
      // 替换高端差值中的变量为常量 1，重新计算差值
      highDiff = IRSimplifier::simplify(alloc<Sub>(
          SubstituteInClone(b.end, {{*vars.begin(), immLike(b.end, 1)}}),
          SubstituteInClone(a.end, {{*vars.begin(), immLike(a.end, 1)}})));
    }
  }

  // 判断是否存在前导部分和尾部分
  bool hasHead = lowDiff->isConstant() && immediateAs<int>(lowDiff) > 0;
  bool hasTail = highDiff->isConstant() && immediateAs<int>(highDiff) < 0;

  // 判断起始点和结束点是否都是常量
  bool constantExtents = lowDiff->isConstant() && highDiff->isConstant();

  if (!constantExtents) {
    // 如果无法推断边界长度，无法创建安全子集，直接返回包含 a 的 vector<Bound> 对象
    return {a};
  }

  if (hasHead) {
    // 如果存在前导部分，将其添加到结果集中
    res.emplace_back(
        a.start,
        IRSimplifier::simplify(alloc<Sub>(b.start, immLike(b.start, 1))));
  }

  if (hasTail) {
    // 如果存在尾部分，计算尾部起始点并添加到结果集中
    ExprPtr tailStart =
        IRSimplifier::simplify(alloc<Add>(b.end, immLike(b.end, 1)));
    res.emplace_back(tailStart, a.end);
  }

  // 返回结果集
  return res;
// 函数定义：从索引边界 A 和 B 中减去重叠的部分，返回剩余的索引边界集合
std::vector<IndexBounds> subtractIndicesBounds(
    const IndexBounds& A,    // 输入参数 A：第一个索引边界
    const IndexBounds& B,    // 输入参数 B：第二个索引边界
    OverlapKind overlap) {   // 输入参数 overlap：重叠类型枚举

  // 如果没有重叠，直接返回 A 的边界作为结果集合
  if (overlap == OverlapKind::NoOverlap) {
    return {A};
  }

  // 如果 A 完全包含于 B 中或者相等，返回空的结果集合
  if (overlap == OverlapKind::ContainedOrEqual) {
    return {};
  }

  // 断言：所有对缓冲区的访问必须具有相同的维度
  TORCH_INTERNAL_ASSERT(A.size() == B.size(), buildErrorMessage());

  // 每个维度可以切片为多个边界片段
  std::vector<IndexBounds> boundSlices;    // 存储切片后的边界集合
  std::vector<Bound> remainingOuterBounds; // 存储剩余的外部边界

  // 对每个维度执行切片操作
  for (size_t i = 0; i < A.size(); ++i) {
    auto slices = subtractBound(A[i], B[i]);   // 切片操作得到边界片段集合

    Bound remaining = A[i];    // 初始剩余边界为 A[i]

    // 对每个切片执行处理
    for (const auto& slice : slices) {
      IndexBounds newRegion;    // 新的索引边界区域
      newRegion.reserve(A.size());  // 预留空间

      // 断言：外部剩余边界的大小必须等于当前维度 i
      TORCH_INTERNAL_ASSERT(
          remainingOuterBounds.size() == i, buildErrorMessage());

      // 构建新的区域：在当前维度 i 前面的外部边界，当前切片，以及后面的 A 维度
      for (size_t j = 0; j < i; ++j) {
        newRegion.push_back(remainingOuterBounds[j]);
      }
      newRegion.push_back(slice);
      for (size_t j = i + 1; j < A.size(); ++j) {
        newRegion.push_back(A[j]);
      }

      boundSlices.push_back(newRegion);   // 将新的区域添加到切片集合中

      // 更新剩余的边界
      if (slice.equals(A[i])) {
        remaining = A[i];
      } else {
        auto remainingSlices = subtractBound(remaining, slice);
        // 在处理单变量的差分表达式时，可能会出现空的 remainingSlices 的情况
        if (!remainingSlices.empty()) {
          TORCH_INTERNAL_ASSERT(
              remainingSlices.size() == 1, buildErrorMessage());
          remaining = remainingSlices[0];
        }
      }
    }

    remainingOuterBounds.push_back(remaining);   // 将剩余的外部边界添加到集合中
  }

  return boundSlices;   // 返回切片后的边界集合作为结果
}

// 函数定义：从索引边界 A 和 B 中减去重叠的部分，返回剩余的索引边界集合
std::vector<IndexBounds> TORCH_API
subtractIndicesBounds(const IndexBounds& A, const IndexBounds& B) {
  return subtractIndicesBounds(A, B, overlaps(A, B));   // 调用重载函数，通过 overlaps 函数确定重叠类型
}

// 命名空间结束声明：torch::jit::tensorexpr::analysis
} // namespace torch::jit::tensorexpr::analysis
```