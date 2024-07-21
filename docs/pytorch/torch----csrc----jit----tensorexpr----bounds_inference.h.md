# `.\pytorch\torch\csrc\jit\tensorexpr\bounds_inference.h`

```
#pragma once
// 防止头文件被多重包含的预处理指令

#include <map>
// 包含 C++ 标准库中的 map 头文件

#include <unordered_map>
// 包含 C++ 标准库中的 unordered_map 头文件

#include <vector>
// 包含 C++ 标准库中的 vector 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 框架中用于导出符号的头文件

#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
// 包含 Torch 框架中张量表达式 JIT 部分的内存依赖检查头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
// 声明一个表达式类

class Buf;
// 声明一个缓冲区类

class Stmt;
// 声明一个语句类

enum C10_API_ENUM TensorAccessKind { kLoad, kStore, kMutate };
// 枚举类型，表示张量访问的种类，包括加载、存储和变异

// 用于存储张量访问边界信息的结构体
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TORCH_API TensorAccessBoundsInfo {
  TensorAccessKind kind; // 记录张量访问的种类
  std::vector<ExprPtr> start; // 记录访问起始位置的表达式列表
  std::vector<ExprPtr> stop;  // 记录访问结束位置的表达式列表
};

// BoundsInfo 是一个无序映射，将缓冲区指针映射到其对应的张量访问边界信息的向量
using BoundsInfo = std::unordered_map<BufPtr, std::vector<TensorAccessBoundsInfo>>;

// 推断语句的边界信息，返回一个 BoundsInfo 结构
TORCH_API BoundsInfo inferBounds(StmtPtr s, bool distinctAccessKinds = true);

// 获取已推断的边界信息，缓存分析结果。MemDependencyChecker 必须已经运行。
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    StmtPtr s,
    bool distinctAccessKinds = true);
// 获取已推断的边界信息，缓存分析结果。MemDependencyChecker 必须已经运行。
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    ExprPtr e,
    bool distinctAccessKinds = true);

// 打印边界信息
TORCH_API void printBoundsInfo(const BoundsInfo& v);

// 获取边界维度的上限
TORCH_API std::vector<ExprPtr> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos);

// 冲突依赖的类型，按照排他性递增排序
enum class HazardKind {
  ReadAfterWrite,   // 写后读
  WriteAfterRead,   // 读后写
  WriteAfterWrite,  // 写后写
  NoDependency,     // 无依赖关系
};

// 获取可能的冲突危害类型
TORCH_API HazardKind getPotentialHazards(
    analysis::MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B);

// 判断语句 A 和 B 的访问是否存在冲突重叠
TORCH_API bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B);
// 判断存储 S1 和 S2 的访问是否存在重叠
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S1,
    StorePtr S2);
// 判断存储 S 和加载 L 的访问是否存在重叠
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S,
    LoadPtr L);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```