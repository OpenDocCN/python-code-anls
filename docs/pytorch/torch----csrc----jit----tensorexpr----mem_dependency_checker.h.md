# `.\pytorch\torch\csrc\jit\tensorexpr\mem_dependency_checker.h`

```
#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <utility>
#include <vector>

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace analysis {

// 枚举类型定义不同的访问类型，如输入、输出、加载、存储等
enum class AccessType {
  Input,
  Output,
  Load,
  Store,
  Call,
  AtomicAdd,
  Alloc,
  Free
};

// 将访问类型转换为字符串表示
const char* AccessToString(AccessType a);

// AccessInfo 类声明
class AccessInfo;
using DependencySet = std::unordered_set<std::shared_ptr<AccessInfo>>;

/* AccessInfo
 *
 * Represents a single bounded memory access to a buffer, for instance a Load or
 * a Store. Holds information relating to the specific access and links to
 * connected accesses in the dependency graph.
 */
class TORCH_API AccessInfo {
 public:
  // AccessInfo 构造函数，用于初始化访问信息对象
  AccessInfo(
      size_t id,
      AccessType type,
      StmtPtr stmt,
      VarPtr var,
      IndexBounds bounds)
      : id_(id),
        type_(type),
        stmt_(std::move(stmt)),
        expr_(nullptr),
        var_(std::move(var)),
        bounds_(std::move(bounds)) {}

  // AccessInfo 构造函数，用于初始化访问信息对象
  AccessInfo(
      size_t id,
      AccessType type,
      ExprPtr expr,
      StmtPtr stmt,
      VarPtr var,
      IndexBounds bounds)
      : id_(id),
        type_(type),
        stmt_(std::move(stmt)),
        expr_(std::move(expr)),
        var_(std::move(var)),
        bounds_(std::move(bounds)) {}

  // 返回此访问信息对象的唯一标识符
  size_t id() const {
    return id_;
  }

  // 返回此访问的类型（加载、存储等）
  AccessType type() const {
    return type_;
  }

  // 返回此访问信息对象所关联的语句
  StmtPtr stmt() const {
    return stmt_;
  }

  // 如果访问是由表达式表示（如加载或调用），则返回该表达式；否则返回nullptr
  ExprPtr expr() const {
    return expr_;
  }

  // 返回表示底层缓冲区的变量对象
  VarPtr var() const {
    return var_;
  }

  // 返回表示每个维度起始和结束表达式的边界向量
  IndexBounds& bounds() {
    return bounds_;
  }

  // 返回此访问依赖的每个访问信息对象的映射
  const std::map<size_t, std::shared_ptr<AccessInfo>>& dependencies() const {
    return dependencies_;
  }

  // 返回依赖于此访问的每个访问信息对象的映射
  std::map<size_t, std::shared_ptr<AccessInfo>> dependents() const {
    // 结果的映射，键为 size_t 类型，值为 std::shared_ptr<AccessInfo> 类型的指针
    std::map<size_t, std::shared_ptr<AccessInfo>> res;
    
    // 遍历 dependents_ 中的每一对键值对 kv，将其转换为 res 的元素，如果对应的依赖已经过期则设为 nullptr
    for (const auto& kv : dependents_) {
      res.emplace(kv.first, kv.second.lock());
    }
    
    // 返回结果映射 res
    return res;
    }
    
    // 返回此访问的索引的符号表达式
    std::vector<ExprPtr> getIndices() const;
    
    // 与另一个访问建立依赖关系或依赖关系
    void addDependency(const std::shared_ptr<AccessInfo>& write);
    void addDependent(const std::shared_ptr<AccessInfo>& read);
    
    // 检查依赖关系的辅助函数
    bool hasDependency(const std::shared_ptr<AccessInfo>& info) const;
    
    // 返回所有直接依赖（即立即依赖）此访问的节点集合
    DependencySet getDirectDependencies();
    
    // 类似地，返回所有直接依赖于此访问的节点集合
    DependencySet getDirectDependents();
    
    // 返回此访问依赖的图中所有节点的完整列表，以及它们所依赖的节点，依此类推，直到输入节点
    DependencySet getIndirectDependencies();
    
    // 类似地，返回完整列表的所有依赖于此节点的图中的节点，以及依赖于这些节点的所有节点，依此类推，直到输出节点
    DependencySet getIndirectDependents();
    
    // 此访问是否表示内存的读取（Load、ReduceOp、Call 等）
    bool isRead() const;
    
    // 此访问是否表示内存的写入（Store 等）
    bool isWrite() const;
    
    // 各种格式中访问的打印辅助函数
    void print() const;
    
    // 将访问转储为 DOT 格式的图形输出流
    void dumpDOT(std::ostream& os) const;
    
    // 访问类型的颜色
    const char* AccessTypeColour() const;
    
    private:
    size_t id_;
    AccessType type_;
    StmtPtr stmt_;
    ExprPtr expr_;
    VarPtr var_;
    IndexBounds bounds_;
    
    // 是的，这些应该是排序的
    // 此访问的依赖关系的映射，键为 size_t 类型，值为 std::shared_ptr<AccessInfo> 类型的指针
    std::map<size_t, std::shared_ptr<AccessInfo>> dependencies_;
    
    // 此访问的依赖者关系的映射，键为 size_t 类型，值为 std::weak_ptr<AccessInfo> 类型的指针
    std::map<size_t, std::weak_ptr<AccessInfo>> dependents_;
};

// 使用别名 VarBoundMap 表示一个从 VarPtr 到 Bound 的无序映射
using VarBoundMap = std::unordered_map<VarPtr, Bound>;

/* MemDependencyChecker 分析 IR 片段并构建其中访问的依赖图。

   可以获取整个图的节点对象形式，也可以作为回答依赖问题的预言器。例如：
   - analyzer.hasIndirectDependency(BufA, BufB);
   - analyzer.hasDirectDependency(LoadA, StoreB);
*/
// Scope 类的构造函数，接受一个块指针和一个父作用域的共享指针作为参数
Scope(BlockPtr b, std::shared_ptr<Scope> p)
    : block(std::move(b)), parent(std::move(p)) {}

// 当前块的指针
BlockPtr block;
// 父作用域的共享指针
std::shared_ptr<Scope> parent;

// 被屏蔽的变量边界的无序映射，从 VarPtr 到 Bound
std::unordered_map<VarPtr, Bound> shadowedVarBounds;

// 局部变量的无序集合，使用 VarPtr 表示
std::unordered_set<VarPtr> localVars;

// 访问信息对象的共享指针的向量
std::vector<std::shared_ptr<AccessInfo>> accesses_;

// 开放写入操作的映射，从 VarPtr 到 BoundRelationship 的列表的无序映射
std::unordered_map<VarPtr, std::list<BoundRelationship>> openWrites_;
};

// 当前作用域的共享指针
std::shared_ptr<Scope> currentScope_;

// 是否允许执行顺序分析的布尔值，默认为 false
bool allowExecutionOrderAnalysis_{false};

// 语句指针到访问信息对象的共享指针的多重映射
std::unordered_multimap<StmtPtr, std::shared_ptr<AccessInfo>> stmtToAccess_;

// 表达式指针到访问信息对象的共享指针的多重映射
std::unordered_multimap<ExprPtr, std::shared_ptr<AccessInfo>> exprToAccess_;

// 作用域到访问信息对象的共享指针的向量的映射
std::unordered_map<StmtPtr, std::vector<std::shared_ptr<AccessInfo>>> scopeToAccesses_;

// 已知变量边界的 VarBoundMap
VarBoundMap knownVarBounds_;

// 在作用域 v 内找到所有读取访问的依赖集合
template <typename StmtOrExprPtr>
DependencySet getAllReadsWithin(StmtOrExprPtr v) {
  DependencySet reads;
  auto insertAllReads = [&](const auto& nodes) {
    for (const auto& l : nodes) {
      auto bound = exprToAccess_.equal_range(l);
      for (auto it = bound.first; it != bound.second; ++it) {
        if (it->second->isRead()) {
          reads.insert(it->second);
        }
      }
    }
  };

  // 查找并插入所有行为像读取的节点的访问
  insertAllReads(NodeFinder<Load>::find(v));
  insertAllReads(NodeFinder<ReduceOp>::find(v));

  return reads;
}

// 在作用域 v 内找到所有写入访问的依赖集合
// 表达式中不会发生写入，因此相对简单
DependencySet getAllWritesWithin(StmtPtr v) {
  DependencySet writes;

  // 目前只有 Store 操作会写入
  auto stores = NodeFinder<Store>::find(std::move(v));
  for (const auto& s : stores) {
    auto bound = stmtToAccess_.equal_range(s);
    for (auto it = bound.first; it != bound.second; ++it) {
      if (it->second->isWrite()) {
        writes.insert(it->second);
      }
    }
  }
  return writes;
}

// 用于处理表达式或语句的模板化帮助函数
template <typename StmtOrExprPtr>
bool dependsDirectlyHelper(StmtOrExprPtr A, StmtPtr B) {
  auto aReads = getAllReadsWithin(A);
  auto bWrites = getAllWritesWithin(B);

  for (auto& read : aReads) {
    for (auto& depPair : read->dependencies()) {
      if (bWrites.count(depPair.second) != 0) {
        return true;
      }
    }
  }

  return false;
}

// 用于处理表达式或语句的间接依赖的模板化帮助函数
template <typename StmtOrExprPtr>
bool dependsIndirectlyHelper(StmtOrExprPtr A, StmtPtr B) {
    // 获取所有在 A 范围内的读操作集合
    auto aReads = getAllReadsWithin(A);
    
    // 获取所有在 B 范围内的写操作集合
    auto bWrites = getAllWritesWithin(B);
    
    // 获取所有依赖于 aReads 的写操作集合
    auto aDeps = getAllWriteDependencies(aReads);
    
    // 遍历 aDeps 中的每一个依赖项
    for (auto& dependency : aDeps) {
      // 如果 bWrites 中存在当前依赖项 dependency，则返回 true
      if (bWrites.count(dependency) != 0) {
        return true;
      }
    }
    
    // 如果未找到任何匹配的依赖项，则返回 false
    return false;
    }
    
    // 获取给定写操作集合的所有依赖项集合
    DependencySet getAllWriteDependencies(const DependencySet& products);
    
    // 由于在 IR 中没有直接表示输入和输出，因此使用 map 进行映射
    std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>> inputs_;
    std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>> outputs_;
    std::unordered_map<VarPtr, std::shared_ptr<AccessInfo>> intermediates_;
    
    // 插入 Buf 的访问信息，特别是针对输入和输出
    void insertBuffers(
        std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>>& bufs,
        AccessType type);
    
    // 更新写入历史记录，添加新的写操作，处理依赖关系并关闭可能的重叠写入
    void updateWriteHistory(
        std::list<BoundRelationship>& writeHistory,
        const std::shared_ptr<AccessInfo>& info,
        size_t latestAccessToClose,
        bool closeOverlapped = true,
        bool insert = true);
    
    // 合并子作用域到父作用域中，为父作用域中的开放写操作添加依赖关系到子作用域的访问操作
    void mergeScope(
        const std::shared_ptr<Scope>& child,
        const std::shared_ptr<Scope>& parent,
        bool closeOverlapped = true);
    
    // 获取索引中符号变量的下界和上界绑定
    std::vector<Bound> getIndicesBounds(const std::vector<ExprPtr>& indices);
    
    // 下一个访问操作的计数器
    size_t nextAccess_{0};
    
    // 最后一个语句的指针
    StmtPtr lastStmt_{nullptr};
};

// 结束了 "torch" 命名空间

} // namespace analysis
// 结束了 "analysis" 命名空间

} // namespace tensorexpr
// 结束了 "tensorexpr" 命名空间

} // namespace jit
// 结束了 "jit" 命名空间

} // namespace torch
// 结束了 "torch" 命名空间
```