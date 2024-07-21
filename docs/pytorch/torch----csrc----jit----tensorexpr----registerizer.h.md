# `.\pytorch\torch\csrc\jit\tensorexpr\registerizer.h`

```py
#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace registerizer {

/* The Registerizer performs scalar replacement by looking for common Stores and
Loads to a single item in a buffer and replacing them with a local temporary
scalar which is cheaper to write.

For example it can replace:

{
  A[0] = 0;
  for(const auto x : c10::irange(10)) {
    A[0] = (A[0]) + x;
  }
}

with:

{
  int A_ = 0;
  for(const auto x : c10::irange(10)) {
    A_ = x + A_;
  }
  A[0] = A_;
}

This is particularly useful on GPUs when parallelizing, since after replacing
loops with metavars we have a lot of accesses like this. */

class Scope;

/*  Holds analysis information about accesses to a specific range of a
 buffer, including the number of loads and stores and the lowest common parent
 Block.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class AccessInfo {
 public:
  AccessInfo() = default;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AccessInfo(
      SimplifierHashType h,             // 构造函数，初始化访问信息对象
      BufPtr b,                         // 缓冲区指针
      std::vector<ExprPtr> i,           // 索引表达式向量
      size_t accessOrder)               // 访问顺序
      : hash_(h),                       // 初始化哈希值
        buf_(std::move(b)),             // 移动缓冲区指针
        indices_(std::move(i)),         // 移动索引表达式向量
        store_cost_(alloc<IntImm>(0)),  // 存储成本初始化为0
        load_cost_(alloc<IntImm>(0)),   // 加载成本初始化为0
        accessOrder_(accessOrder) {}    // 初始化访问顺序

  // Adds a Store to this access, which is in the provided scope.
  void addStore(StorePtr store, const std::shared_ptr<Scope>& scope);

  // Adds a Load to this access, which occurs in the usage Stmt in the provided
  // scope.
  void addLoad(
      LoadPtr load,
      const std::shared_ptr<Scope>& scope,
      StmtPtr usage);

  // Merge another AccessInfo into this one.
  void merge(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the other AccessInfo's bounds may overlap this one.
  bool overlaps(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the indices of this access depend on the provided Var.
  bool dependsOnVar(VarPtr v);

  // Clone this AccessInfo, and set this as the new accesses' hiddenAccess.
  static std::shared_ptr<AccessInfo> cloneWithHiddenInfo(
      const std::shared_ptr<AccessInfo>& orig);

  // print for debugging.
  void print() const;

  SimplifierHashType hash() const {     // 获取哈希值
    return hash_;
  }

  BufPtr buf() const {                 // 获取缓冲区指针
    return buf_;
  }

  const std::vector<ExprPtr>& indices() const {  // 获取索引表达式向量
    return indices_;
  }

  BlockPtr block() const {             // 获取块指针
    return block_;
  }

  void setEnclosingBlock(BlockPtr b) { // 设置包含块
    block_ = b;
  }

  StmtPtr first_usage() const {        // 获取第一个使用语句
    return first_usage_;
  }
  StmtPtr last_usage() const {         // 获取最后一个使用语句

    return last_usage_;
  }
};

} // namespace registerizer
} // namespace tensorexpr
} // namespace jit
} // namespace torch
  // 返回最后一次使用的语句
  return last_usage_;
}

// 设置第一次和最后一次使用的语句
void setUsageMarks(StmtPtr first, StmtPtr last) {
  first_usage_ = first;
  last_usage_ = last;
}

// 返回是否第一次使用被重叠
bool firstUsageOverlapped() const {
  return firstUsageOverlapped_;
}

// 返回存储成本表达式
ExprPtr store_cost() const {
  return store_cost_;
}

// 返回加载成本表达式
ExprPtr load_cost() const {
  return load_cost_;
}

// 返回存储操作的向量引用
const std::vector<StorePtr>& stores() const {
  return stores_;
}

// 返回加载操作的向量引用
const std::vector<LoadPtr>& loads() const {
  return loads_;
}

// 提升存储和加载成本的表达式，通过乘以给定的 extent 表达式
void hoistCosts(ExprPtr extent) {
  store_cost_ = IRSimplifier::simplify(alloc<Mul>(store_cost_, extent));
  load_cost_ = IRSimplifier::simplify(alloc<Mul>(load_cost_, extent));
}

// 返回条件 ID
size_t conditionId() const {
  return conditionId_;
}

// 设置条件 ID
void setConditionId(size_t c) {
  conditionId_ = c;
}

// 返回访问顺序
size_t accessOrder() const {
  return accessOrder_;
}

// 返回隐藏的访问信息的共享指针
std::shared_ptr<AccessInfo> hiddenAccess() const {
  return hiddenAccess_;
}

// 返回替换状态相关的标量变量信息
struct ScalarReplacement {
  VarPtr var{nullptr};        // 变量指针
  BufPtr var_wrapper{nullptr}; // 缓冲区指针
  LetPtr initializer{nullptr}; // 初始化器指针
};

// 返回替换状态相关的标量替换结构
ScalarReplacement& replacement() {
  return replacement_;
}
// 使用别名 AccessHashMap 表示一个从 SimplifierHashType 到 AccessInfo 共享指针的无序映射
using AccessHashMap =
    std::unordered_map<SimplifierHashType, std::shared_ptr<AccessInfo>>;

// 表示一个作用域块，并包含其中所有访问的信息
class Scope {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，初始化作用域对象，包括块对象、父作用域、条件标识符（默认为0）
  Scope(BlockPtr b, std::shared_ptr<Scope> parent, size_t conditionId = 0)
      : block_(std::move(b)),
        parent_(std::move(parent)),
        conditionId_(conditionId) {}

  // 返回特定缓冲区（BufPtr）的访问映射
  AccessHashMap& getAccessMapByBuf(BufPtr b);

  // 返回开放访问的映射，即从缓冲区到访问映射的映射
  std::unordered_map<BufPtr, AccessHashMap>& openAccesses() {
    return openAccesses_;
  }

  // 返回已关闭的访问列表
  std::vector<std::shared_ptr<AccessInfo>>& closedAccesses() {
    return closedAccesses_;
  }

  // 返回该作用域所关联的块对象
  BlockPtr block() const {
    return block_;
  }

  // 返回该作用域的父作用域对象
  std::shared_ptr<Scope> parent() const {
    return parent_;
  }

  // 返回该作用域依赖的条件标识符
  size_t conditionId() const {
    return conditionId_;
  }

  // 返回该作用域内的局部变量集合
  const std::unordered_set<VarPtr>& localVars() const {
    return localVars_;
  }

  // 添加局部变量到局部变量集合中
  void addLocalVar(VarPtr v) {
    localVars_.insert(v);
  }

  // 关闭特定的访问信息，并将其添加到已关闭的访问列表中
  void closeAccess(const std::shared_ptr<AccessInfo>& info);

  // 过滤已关闭的访问列表，可能会从列表中删除重叠的访问信息
  void filterClosed();

 private:
  // openAccesses_：缓冲区到访问映射的映射，用于管理正在进行的访问
  std::unordered_map<BufPtr, AccessHashMap> openAccesses_;

  // closedAccesses_：已关闭的访问列表，包含已完成的访问信息
  std::vector<std::shared_ptr<AccessInfo>> closedAccesses_;

  // block_：该作用域对象关联的块对象
  BlockPtr block_;

  // parent_：父作用域对象，指向当前作用域的外部作用域
  std::shared_ptr<Scope> parent_;

  // conditionId_：表示此作用域依赖的条件块的标识符
  size_t conditionId_;

  // localVars_：该作用域内的局部变量集合，例如循环变量等
  std::unordered_set<VarPtr> localVars_;
};
/*
 * 分析图并收集对同一符号张量元素的访问，这些访问可以被替换为单个本地标量。
 *
 * 这通过以后序顺序递归地遍历树来完成，按作用域构建对同一符号元素访问的集合，
 * 然后将较低作用域合并到其封闭作用域中。
 *
 * 如果在元素的所有使用之间没有其他加载或存储可能引用它，将同一张量元素的两个访问移动到本地标量Var是安全的。
 * 在评论中，我称之为重叠访问或“切割”现有的AccessInfo。在候选用于寄存器化的情况下，
 * 可能可以通过提前将其写回到张量中来完成访问，然后在重叠访问完成后创建一个新的标量变量。我们会在节省内存访问时尝试这样做。
 *
 * 这里有几种情况使得这更具挑战性：
 *
 * - For循环：循环通过循环范围改变缓冲区的实际使用次数，但只有在我们可以将标量变量的定义和终结器从循环块中拉出来时。
 *
 * - Cond条件：条件使得将标量提升到内部作用域变得更加复杂。
 * 通常情况下，除非在更高作用域已经有对同一访问的引用，否则我们不能将访问提升到条件作用域之外，
 * 因为我们不知道条件是否保护了在更高作用域中不安全的数组访问。在评论中，我称之为条件“隐藏”访问，外部访问“取消隐藏”它。
 *
 * - IfThenElse条件：与Cond相同的情况，但由于IfThenElse是一个Expr而不是Stmt，我们不能在条件作用域内插入标量定义或终结器。
 * 在IfThenElse内部的访问可以安全地与外部访问结合，但不能完全存在于外部。
 *
 * - Let语句：通过Let语句依赖于本地变量或循环变量的访问，不能在依赖变量的作用域之外提升。
 */
class TORCH_API RegisterizerAnalysis : public IRVisitor {
 public:
  // 构造函数，初始化当前作用域为根作用域
  RegisterizerAnalysis()
      : currentScope_(std::make_shared<Scope>(nullptr, nullptr, 0)) {}
  // 析构函数，默认
  ~RegisterizerAnalysis() override = default;

  // 重写的IRVisitor方法
  // 访问For循环节点
  void visit(ForPtr v) override;
  // 访问Cond条件节点
  void visit(CondPtr v) override;
  // 访问Block块节点
  void visit(BlockPtr v) override;
  // 访问Store存储节点
  void visit(StorePtr v) override;
  // 访问Load加载节点
  void visit(LoadPtr v) override;
  // 访问IfThenElse条件节点
  void visit(IfThenElsePtr v) override;
  // 访问Let节点
  void visit(LetPtr v) override;

  // 定义宏STMT_ON_STACK，用于在堆栈上处理各种操作节点
#define STMT_ON_STACK(Op)          \
  void visit(Op##Ptr v) override { \
    stmtStack_.push_front(v);      \
    IRVisitor::visit(v);           \
    stmtStack_.pop_front();        \
  }

  // 以下宏扩展用于处理特定类型的操作节点
  STMT_ON_STACK(AtomicAdd);  // 原子加操作节点
  STMT_ON_STACK(Allocate);   // 分配操作节点
  STMT_ON_STACK(Free);       // 释放操作节点
// 定义了一个类，用于替换可寄存的访问信息为标量变量，包括定义、初始化和终结
class TORCH_API RegisterizerReplacer : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，接受一个包含访问信息共享指针的向量，并初始化内部结构
  RegisterizerReplacer(std::vector<std::shared_ptr<AccessInfo>>& vec)
      : infoSet_(vec) {
    // 调用buildReplacements方法，建立替换所需的各种ReplacerScope对象和内部映射
    buildReplacements();
  }

  // 重写了IRMutator的mutate方法，用于处理LoadPtr类型的表达式
  ExprPtr mutate(LoadPtr v) override;

  // 重写了IRMutator的mutate方法，用于处理StorePtr类型的语句
  StmtPtr mutate(StorePtr v) override;

  // 重写了IRMutator的mutate方法，用于处理BlockPtr类型的语句块
  StmtPtr mutate(BlockPtr v) override;

 private:
  // 内部结构，用于保存各种替换的作用域信息
  struct ReplacerScope {
    // 映射，将语句指针映射到AccessInfo共享指针的双端队列，用于初始化点
    std::unordered_map<StmtPtr, std::deque<std::shared_ptr<AccessInfo>>>
        initializerPoints_;
    // 映射，将语句指针映射到AccessInfo共享指针的双端队列，用于终结点
    std::unordered_map<StmtPtr, std::deque<std::shared_ptr<AccessInfo>>>
        finalizePoints_;
  };

  // 创建各种ReplacerScope对象并建立内部映射
  void buildReplacements();

  // 保存待替换访问信息的向量引用
  std::vector<std::shared_ptr<AccessInfo>>& infoSet_;

  // 映射，将StorePtr映射到AccessInfo共享指针的映射，用于访问Store信息
  std::unordered_map<StorePtr, std::shared_ptr<AccessInfo>> storeToAccess_;

  // 映射，将LoadPtr映射到AccessInfo共享指针的映射，用于访问Load信息
  std::unordered_map<LoadPtr, std::shared_ptr<AccessInfo>> loadToAccess_;

  // 映射，将BlockPtr映射到ReplacerScope结构的映射，用于父级到访问信息的映射
  std::unordered_map<BlockPtr, ReplacerScope> parentToAccesses_;

  // 集合，保存应该合并到初始化器中的StorePtr语句，以便可以被消除
  std::set<StorePtr> eliminatedIntializers_;

  // 映射，跟踪每个缓冲区被访问的次数，以便能够适当命名标量变量
  std::unordered_map<BufPtr, unsigned int> bufferAccessCounts_;

  // 获取缓冲区访问次数的函数，用于递增并返回次数
  unsigned int getBufferAccessCount(BufPtr b) {
    return ++bufferAccessCounts_[b];
  }
};
} // namespace registerizer

// 对给定的语句指针应用标量替换操作
// 要生成安全的代码，必须在处理并行化轴和原子操作之后进行此操作
TORCH_API StmtPtr registerize(StmtPtr s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```