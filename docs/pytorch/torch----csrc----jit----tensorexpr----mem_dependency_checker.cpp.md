# `.\pytorch\torch\csrc\jit\tensorexpr\mem_dependency_checker.cpp`

```
// 包含头文件，声明了一些必要的类和函数
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>

// 包含一个实用的范围迭代器
#include <c10/util/irange.h>

// 包含文件流和输入输出流的头文件
#include <fstream>
#include <iostream>

// Torch JIT TensorExpr 分析命名空间
namespace torch::jit::tensorexpr::analysis {

// 将访问类型转换为字符串形式
const char* AccessToString(AccessType a) {
  switch (a) {
    case AccessType::Input:
      return "Input";
    case AccessType::Output:
      return "Output";
    case AccessType::Load:
      return "Load";
    case AccessType::Store:
      return "Store";
    case AccessType::Call:
      return "Call";
    case AccessType::AtomicAdd:
      return "AtomicAdd";
    case AccessType::Alloc:
      return "Alloc";
    case AccessType::Free:
      return "Free";
    default:
      break;
  }
  return "Unknown";
}

// 递归获取给定访问信息节点及其依赖链的依赖集合
static void getDependencyChain(
    const std::shared_ptr<AccessInfo>& info,
    DependencySet& dependencies) {
  if (!dependencies.insert(info).second) {
    return;
  }

  // 遍历当前节点的依赖节点，并递归获取其依赖链
  for (auto& dep : info->dependencies()) {
    getDependencyChain(dep.second, dependencies);
  }
}

// 递归获取给定访问信息节点及其依赖链的依赖集合
static void getDependentsChain(
    const std::shared_ptr<AccessInfo>& info,
    DependencySet& dependents) {
  if (!dependents.insert(info).second) {
    return;
  }

  // 遍历当前节点的依赖节点，并递归获取其依赖链
  for (auto& dep : info->dependents()) {
    getDependencyChain(dep.second, dependents);
  }
}

// 实现 AccessInfo 类中的 getIndices 函数
std::vector<ExprPtr> AccessInfo::getIndices() const {
  std::vector<ExprPtr> indices;

  // 如果表达式存在，则根据表达式类型获取对应的索引
  if (expr_) {
    if (auto load = to<Load>(expr_)) {
      indices = load->indices();
    }
  } else {
    // 如果语句存在，则根据语句类型获取对应的索引
    if (auto store = to<Store>(stmt_)) {
      indices = store->indices();
    }
  }
  return indices;
}

// 添加一个依赖关系到当前访问信息节点
void AccessInfo::addDependency(const std::shared_ptr<AccessInfo>& write) {
  auto res = dependencies_.emplace(write->id(), write);
  TORCH_INTERNAL_ASSERT(
      res.second,
      buildErrorMessage("Duplicate entry in mem dep checker in the fuser."));
}

// 添加一个依赖节点到当前访问信息节点
void AccessInfo::addDependent(const std::shared_ptr<AccessInfo>& read) {
  auto res = dependents_.emplace(read->id(), read);
  TORCH_INTERNAL_ASSERT(
      res.second,
      buildErrorMessage("Duplicate entry in mem dep checker in the fuser."));
}

// 检查当前访问信息节点是否依赖于给定的访问信息节点
bool AccessInfo::hasDependency(const std::shared_ptr<AccessInfo>& info) const {
  return dependencies_.count(info->id()) != 0;
}

// 获取当前访问信息节点的直接依赖关系集合
DependencySet AccessInfo::getDirectDependencies() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    res.insert(depPair.second);
  }
  return res;
}

// 获取当前访问信息节点的间接依赖关系集合
DependencySet AccessInfo::getIndirectDependencies() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    getDependencyChain(depPair.second, res);
  }
  return res;
}

// 获取当前访问信息节点的直接依赖节点集合
DependencySet AccessInfo::getDirectDependents() {
  DependencySet res;
  for (auto& depPair : dependents_) {
    res.insert(depPair.second.lock());
  }
  return res;
}

// 获取当前访问信息节点的间接依赖节点集合
DependencySet AccessInfo::getIndirectDependents() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    getDependentsChain(depPair.second, res);
  }
  return res;
}

// 检查当前访问信息节点是否为读操作
bool AccessInfo::isRead() const {
  switch (type_) {
    // 对于输出和加载类型，认为是读操作
    case AccessType::Output:
    case AccessType::Load:
    case AccessType::Call:
        // 如果访问类型是 Call，则返回 true
        return true;
    case AccessType::AtomicAdd:
        // 如果访问类型是 AtomicAdd，则返回 true
        return true;
    default:
        // 对于其他任何访问类型，跳出 switch
        break;
  }
  // 如果没有匹配的访问类型，则返回 false
  return false;
}

// 判断访问类型是否为写操作，若是则返回true，否则返回false
bool AccessInfo::isWrite() const {
  switch (type_) {
    // 对于以下访问类型，均视为写操作
    case AccessType::Input:
    case AccessType::Store:
    case AccessType::AtomicAdd:
    case AccessType::Alloc:
    case AccessType::Free:
      return true;
    default:
      break;
  }
  // 默认返回false，表示非写操作
  return false;
}

// 打印访问信息，包括访问ID、访问类型、变量名及其维度信息，以及依赖和依赖者信息
void AccessInfo::print() const {
  std::cout << id_ << ". " << AccessToString(type_) << ": " << *var_ << "[";
  if (!bounds_.empty()) {
    for (size_t i = 0; i < bounds_.size() - 1; ++i) {
      bounds_[i].print();
      std::cout << ", ";
    }

    size_t i = bounds_.size() - 1;
    bounds_[i].print();
  }
  std::cout << "]";

  if (!dependencies_.empty()) {
    std::cout << " - depends on: ";
    for (auto& pair : dependencies_) {
      std::cout << pair.second->id() << " ";
    }
  }

  if (!dependents_.empty()) {
    std::cout << " - dependents: ";
    for (auto& pair : dependents_) {
      std::cout << pair.second.lock()->id() << " ";
    }
  }

  std::cout << "\n";
}

// 将访问信息以DOT格式输出到给定的输出流os中
void AccessInfo::dumpDOT(std::ostream& os) const {
  if (type_ == AccessType::Input || type_ == AccessType::Output ||
      type_ == AccessType::Alloc) {
    os << "n" << id_ << " [\n";
    os << "label = \"" << AccessToString(type_) << "\\n " << *var_ << "[";
    if (!bounds_.empty()) {
      for (size_t i = 0; i < bounds_.size() - 1; ++i) {
        os << *IRSimplifier::simplify(
                  alloc<Add>(bounds_[i].end, immLike(bounds_[i].end, 1)))
           << ", ";
      }

      size_t i = bounds_.size() - 1;
      os << *IRSimplifier::simplify(
          alloc<Add>(bounds_[i].end, immLike(bounds_[i].end, 1)));
      os << "]\"\n ";
    }
    if (isWrite()) {
      os << "\tshape = \"invhouse\"\n";  // 写操作的形状为倒置房子
    } else {
      os << "\tshape = \"house\"\n";  // 非写操作的形状为房子
    }
  } else {
    os << "n" << id_ << " [\n";
    os << "label = \"" << AccessToString(type_) << " (#" << id_ << ")\\n";
    os << "buf : " << *var_ << "\\n";
    os << "bounds : [";
    if (!bounds_.empty()) {
      for (size_t i = 0; i < bounds_.size() - 1; ++i) {
        os << "(" << *bounds_[i].start << ", " << *bounds_[i].end << "), ";
      }

      size_t i = bounds_.size() - 1;
      os << "(" << *bounds_[i].start << ", " << *bounds_[i].end << ")]";
    }
    os << "\"\n";
    os << "\tshape = \"box\"\n";  // 非特定类型的形状为方框
  }
  os << "\tstyle=\"filled\"\n";  // 设置节点填充样式
  os << "\tcolor=\"" << AccessTypeColour() << "\"\n";  // 设置节点颜色

  std::string edgeColour;
  if (isWrite()) {
    edgeColour = "cornflowerblue";  // 写操作的边缘颜色
  } else {
    edgeColour = "goldenrod";  // 非写操作的边缘颜色
  }
  os << "]\n";

  // 输出依赖关系边
  for (auto& pair : dependencies_) {
    os << "n" << pair.second->id() << " -> "
       << "n" << id_ << " [color=\"" << edgeColour << "\"]\n";
  }
}

// 根据访问类型返回相应的颜色
const char* AccessInfo::AccessTypeColour() const {
  switch (type_) {
    case AccessType::Input:
    case AccessType::Output:
      return "palegreen";  // 输入输出类型的颜色
    case AccessType::Load:
      return "peachpuff";  // 载入类型的颜色
    case AccessType::Store:
      return "dodgerblue";  // 存储类型的颜色
    case AccessType::Call:
      return "violet";  // 调用类型的颜色
    case AccessType::Alloc:
      return "lightblue";  // 分配类型的颜色
    default:
      return "white";  // 默认为白色
  }
}
    case AccessType::Free:
      // 当访问类型为Free时返回"sandybrown"颜色
      return "sandybrown";
    default:
      // 对于其它访问类型，执行默认操作
      break;
  }
  // 默认情况下返回"white"颜色
  return "white";
// MemDependencyChecker 类的默认构造函数
MemDependencyChecker::MemDependencyChecker() {
  // 初始化当前作用域为根作用域
  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

// MemDependencyChecker 类的构造函数，接受输入和输出的无序集合
MemDependencyChecker::MemDependencyChecker(
    const std::unordered_set<BufPtr>& inputs,
    const std::unordered_set<BufPtr>& outputs) {
  // 初始化输入集合
  for (const auto& s : inputs) {
    inputs_[s] = nullptr;
  }
  // 初始化输出集合
  for (const auto& s : outputs) {
    outputs_[s] = nullptr;
  }

  // 初始化当前作用域为根作用域
  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

// MemDependencyChecker 类的构造函数，接受输入和输出的缓冲区句柄向量
MemDependencyChecker::MemDependencyChecker(
    const std::vector<BufHandle>& inputs,
    const std::vector<BufHandle>& outputs) {
  // 初始化输入集合
  for (auto& s : inputs) {
    inputs_[s.node()] = nullptr;
  }
  // 初始化输出集合
  for (auto& s : outputs) {
    outputs_[s.node()] = nullptr;
  }

  // 初始化当前作用域为根作用域
  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

// 允许或禁止循环执行顺序分析的函数
bool MemDependencyChecker::allowLoopExecutionOrderAnalysis(bool allow) {
  // 切换循环执行顺序分析的允许状态
  std::swap(allowExecutionOrderAnalysis_, allow);
  return allow;
}

// 获取当前作用域的历史访问信息
const std::vector<std::shared_ptr<AccessInfo>>& MemDependencyChecker::
    getHistory() const {
  return currentScope_->accesses_;
}

// 将依赖关系图（DAG）输出到指定文件名的函数
void MemDependencyChecker::dumpDAG(const std::string& filename) const {
  // 打开输出文件流
  std::ofstream dotfile(filename);

  // 写入DOT格式的图头
  dotfile << "digraph {\n";
  // 对当前作用域的历史访问信息逐个生成DOT格式输出
  for (auto& wi : getHistory()) {
    wi->dumpDOT(dotfile);
  }
  // 写入DOT格式的图尾
  dotfile << "}\n";

  // 关闭输出文件流
  dotfile.close();
}

// 获取所有写依赖关系的函数
DependencySet MemDependencyChecker::getAllWriteDependencies(
    const DependencySet& products) {
  DependencySet writes;

  // 遍历产品集合中的每一个信息
  for (auto& info : products) {
    DependencySet dependencies;
    // 获取当前信息的依赖链
    getDependencyChain(info, dependencies);
    // 将依赖链中的写操作加入结果集合
    for (auto& other : dependencies) {
      if (other->isWrite()) {
        writes.insert(other);
      }
    }
  }

  return writes;
}

// 检查两个表达式间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(ExprPtr A, StmtPtr B) {
  return dependsDirectlyHelper(A, B);
}

// 检查两个语句间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(StmtPtr A, StmtPtr B) {
  return dependsDirectlyHelper(A, B);
}

// 检查输出缓冲区和语句间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(BufPtr O, StmtPtr B) {
  auto outputAccess = output(O);
  auto bWrites = getAllWritesWithin(B);

  // 遍历输出缓冲区的依赖关系，检查是否与语句的写操作有关联
  for (auto& depPair : outputAccess->dependencies()) {
    if (bWrites.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

// 检查语句和输入缓冲区间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(StmtPtr A, BufPtr I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  // 遍历输入缓冲区的依赖关系，检查是否与语句的读操作有关联
  for (auto& depPair : inputAccess->dependents()) {
    if (aReads.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

// 检查表达式和输入缓冲区间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(ExprPtr A, BufPtr I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  // 遍历输入缓冲区的依赖关系，检查是否与表达式的读操作有关联
  for (auto& depPair : inputAccess->dependents()) {
    if (aReads.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

// 检查两个访问信息对象间是否存在直接依赖关系的函数重载
bool MemDependencyChecker::dependsDirectly(
    const std::shared_ptr<AccessInfo>& A,
    const std::shared_ptr<AccessInfo>& B) {
  // 检查A是否依赖于B，并且B是写操作
  return A->hasDependency(B) && B->isWrite();
}
// 判断是否间接依赖的辅助函数，接受表达式指针 A 和语句指针 B 作为参数
bool MemDependencyChecker::dependsIndirectly(ExprPtr A, StmtPtr B) {
  // 调用辅助函数 dependsIndirectlyHelper 判断 A 和 B 是否存在间接依赖关系
  return dependsIndirectlyHelper(A, B);
}

// 判断是否间接依赖的辅助函数，接受语句指针 A 和 B 作为参数
bool MemDependencyChecker::dependsIndirectly(StmtPtr A, StmtPtr B) {
  // 调用辅助函数 dependsIndirectlyHelper 判断 A 和 B 是否存在间接依赖关系
  return dependsIndirectlyHelper(A, B);
}

// 判断是否间接依赖的辅助函数，接受缓冲区指针 O 和语句指针 B 作为参数
bool MemDependencyChecker::dependsIndirectly(BufPtr O, StmtPtr B) {
  // 获取输出缓冲区 O 的访问信息
  auto outputAccess = output(O);

  // 存储依赖集合
  DependencySet dependencies;
  // 获取输出访问的依赖链路
  getDependencyChain(outputAccess, dependencies);

  // 获取语句 B 写操作的集合
  auto bWrites = getAllWritesWithin(B);
  // 遍历依赖集合，如果存在任何依赖于 B 的写操作，则返回 true
  for (auto& dep : dependencies) {
    if (bWrites.count(dep) != 0) {
      return true;
    }
  }

  // 如果不存在依赖于 B 的写操作，则返回 false
  return false;
}

// 判断是否间接依赖的辅助函数，接受语句指针 A 和输入缓冲区指针 I 作为参数
bool MemDependencyChecker::dependsIndirectly(StmtPtr A, BufPtr I) {
  // 获取语句 A 内的所有读访问
  auto aReads = getAllReadsWithin(A);
  // 获取输入缓冲区 I 的访问信息
  auto inputAccess = input(I);

  // 获取语句 A 读访问的所有写依赖
  auto aDeps = getAllWriteDependencies(aReads);

  // 如果输入访问在依赖集合中，则返回 true，表示存在间接依赖关系
  return aDeps.count(inputAccess) != 0;
}

// 判断是否间接依赖的辅助函数，接受表达式指针 A 和输入缓冲区指针 I 作为参数
bool MemDependencyChecker::dependsIndirectly(ExprPtr A, BufPtr I) {
  // 获取表达式 A 内的所有读访问
  auto aReads = getAllReadsWithin(A);
  // 获取输入缓冲区 I 的访问信息
  auto inputAccess = input(I);

  // 获取表达式 A 读访问的所有写依赖
  auto aDeps = getAllWriteDependencies(aReads);

  // 如果输入访问在依赖集合中，则返回 true，表示存在间接依赖关系
  return aDeps.count(inputAccess) != 0;
}

// 判断是否间接依赖的辅助函数，接受输出缓冲区指针 O 和输入缓冲区指针 I 作为参数
bool MemDependencyChecker::dependsIndirectly(BufPtr O, BufPtr I) {
  // 获取输出缓冲区 O 和输入缓冲区 I 的访问信息
  auto outputAccess = output(O);
  auto inputAccess = input(I);

  // 调用另一个重载的 dependsIndirectly 函数，判断输出访问和输入访问是否间接依赖
  return dependsIndirectly(outputAccess, inputAccess);
}

// 判断是否间接依赖的辅助函数，接受共享指针 AccessInfo A 和 B 作为参数
bool MemDependencyChecker::dependsIndirectly(
    const std::shared_ptr<AccessInfo>& A,
    const std::shared_ptr<AccessInfo>& B) {
  // 如果 B 不是写操作，则直接返回 false
  if (!B->isWrite()) {
    return false;
  }

  // 存储依赖集合
  DependencySet dependencies;
  // 获取 A 的依赖链路
  getDependencyChain(A, dependencies);
  // 如果依赖集合中不包含 B，则返回 false
  if (dependencies.count(B) == 0) {
    return false;
  }

  // 否则返回 true，表示存在间接依赖关系
  return true;
}

// 返回与语句 A 相关的访问信息
std::shared_ptr<AccessInfo> MemDependencyChecker::accessFor(StmtPtr A) const {
  // 在 stmtToAccess_ 中查找与语句 A 相关的访问信息
  auto bound = stmtToAccess_.equal_range(A);
  // 遍历范围内的所有访问信息，返回第一个不含表达式的访问信息
  for (auto it = bound.first; it != bound.second; ++it) {
    if (it->second->expr() == nullptr) {
      return it->second;
    }
  }
  // 如果找不到符合条件的访问信息，则返回空指针
  return nullptr;
}

// 返回与表达式 A 相关的访问信息
std::shared_ptr<AccessInfo> MemDependencyChecker::accessFor(ExprPtr A) const {
  // TODO 表达式可能有多个访问... 我们只返回第一个，这不是最佳实践。在此无法做太多改进。
  // 在 exprToAccess_ 中查找与表达式 A 相关的访问信息
  auto bound = exprToAccess_.equal_range(A);
  // 如果范围内存在元素，则返回第一个元素对应的访问信息
  if (bound.first != exprToAccess_.end()) {
    return bound.first->second;
  }

  // 如果找不到符合条件的访问信息，则返回空指针
  return nullptr;
}

// 返回语句 A 内的所有访问信息集合
std::unordered_set<std::shared_ptr<AccessInfo>> MemDependencyChecker::
    accessesWithin(StmtPtr A) const {
  // 在 scopeToAccesses_ 中查找与语句 A 相关的所有访问信息集合
  auto it = scopeToAccesses_.find(A);
  // 如果找到相关集合，则返回集合中的所有元素
  if (it != scopeToAccesses_.end()) {
    return std::unordered_set<std::shared_ptr<AccessInfo>>(
        it->second.begin(), it->second.end());
  }

  // 如果找不到相关集合，则返回一个空的集合
  std::unordered_set<std::shared_ptr<AccessInfo>> ret;
  auto bound = stmtToAccess_.equal_range(A);
  // 遍历范围内的所有访问信息，并加入到返回集合中
  for (auto it = bound.first; it != bound.second; ++it) {
    ret.insert(it->second);
  }
  return ret;
}

// 返回表达式 A 内的所有访问信息集合
std::unordered_set<std::shared_ptr<AccessInfo>> MemDependencyChecker::
    accessesWithin(ExprPtr A) const {
  // 直接调用 accessFor 函数获取表达式 A 相关的访问信息
  return {accessFor(A)};
}

// 返回缓冲区 b 的输入访问信息
std::shared_ptr<AccessInfo> MemDependencyChecker::input(BufPtr b) const {
  // 在 inputs_ 中查找缓冲区 b 的访问信息
  auto it = inputs_.find(b);
  // 如果找到对应的访问信息，则返回该信息
  if (it == inputs_.end()) {
    return nullptr;
  }
  return it->second;
}
// 返回与给定缓冲区指针关联的访问信息的共享指针，如果未找到则返回空指针。
std::shared_ptr<AccessInfo> MemDependencyChecker::output(BufPtr b) const {
  // 在 outputs_ 中查找给定缓冲区指针 b 对应的元素
  auto it = outputs_.find(b);
  // 如果未找到，返回空指针
  if (it == outputs_.end()) {
    return nullptr;
  }
  // 返回找到的元素对应的共享指针
  return it->second;
}

// 节点访问者:

// 处理 Store 节点的访问方法
void MemDependencyChecker::visit(StorePtr v) {
  // 保存上一个语句的指针，并将当前语句设为 v
  StmtPtr last = lastStmt_;
  lastStmt_ = v;
  // 访问并处理 v 的值表达式
  v->value()->accept(this);

  // 遍历处理 v 的索引表达式
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
  // 恢复上一个语句的指针
  lastStmt_ = last;

  // 为该存储操作创建一个新的 AccessInfo 对象
  VarPtr var = v->buf()->base_handle();
  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Store, v, var, getIndicesBounds(v->indices()));

  // 添加依赖于此存储作用域内的任何访问信息（即 RHS）
  auto bound = stmtToAccess_.equal_range(v);
  for (auto it = bound.first; it != bound.second; ++it) {
    info->addDependency(it->second);
    it->second->addDependent(info);
  }

  // 将 v 与 info 映射关系存储起来
  stmtToAccess_.emplace(v, info);

  // 此写操作是开放的，会关闭完全重叠的任何开放写操作
  auto& history = currentScope_->openWrites_[var];
  updateWriteHistory(history, info, info->id());
  currentScope_->accesses_.push_back(info);
}

// 处理 Load 节点的访问方法
void MemDependencyChecker::visit(LoadPtr v) {
  // 创建一个临时作用域，用于保存发生在此 Load 索引中的任何加载操作
  auto indicesScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);
  currentScope_ = indicesScope;

  // 遍历处理 v 的索引表达式
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }

  // 为该加载操作创建一个新的 AccessInfo 对象
  VarPtr var = v->buf()->base_handle();
  auto load = std::make_shared<AccessInfo>(
      nextAccess_++,
      AccessType::Load,
      v,
      lastStmt_,
      var,
      getIndicesBounds(v->indices()));

  // 如果索引中存在加载操作，则此加载依赖于它们，并合并它们
  if (!indicesScope->accesses_.empty()) {
    for (auto& access : indicesScope->accesses_) {
      load->addDependency(access);
      access->addDependent(load);
    }
    mergeScope(indicesScope, indicesScope->parent, false);
  }

  currentScope_ = indicesScope->parent;

  // 将 lastStmt_ 与 load 的映射关系存储起来
  stmtToAccess_.emplace(lastStmt_, load);
  // 将 v 与 load 的映射关系存储起来
  exprToAccess_.emplace(v, load);

  // 这是一个读取操作，并不会关闭任何访问操作，但是我们需要建立在同一作用域内访问操作之间的依赖关系
  // 有意使用 operator[]，如果不存在，则创建
  auto& writeHistory = currentScope_->openWrites_[var];
  updateWriteHistory(writeHistory, load, load->id());
  currentScope_->accesses_.push_back(load);
}

// 此检查确定循环内两个访问是否在“安全”范围内，不考虑边界范围重叠，而是考虑边界相对于循环变量的步长
// 如果允许，则此代码段考虑迭代顺序
static bool executionSafetyCheck(
    const std::shared_ptr<AccessInfo>& info,
    const std::shared_ptr<AccessInfo>& other,
    // 引用另一个访问信息的共享指针
    const std::vector<ExprPtr>& aStrides,
    // 访问A的步长向量
    const std::vector<ExprPtr>& oStrides,
    // 访问O的步长向量
    bool parallelized) {
  // 如果任一步长向量为空，则返回false
  if (aStrides.empty() || oStrides.empty()) {
    return false;
  }
  // 断言两个访问的维度大小相同
  TORCH_INTERNAL_ASSERT(
      info->bounds().size() == other->bounds().size(),
      buildErrorMessage(
          "Dimension mismatch for two accesses in mem dep checker in the fuser."));
  // 遍历访问的维度
  for (size_t b = 0; b < info->bounds().size(); ++b) {
    // 获取当前维度的访问A和访问O的步长
    ExprPtr aIndexStride = aStrides[b];
    ExprPtr oIndexStride = oStrides[b];
    // 如果任一步长不是常数，则跳过当前维度的检查
    if (!aIndexStride->isConstant() || !oIndexStride->isConstant()) {
      continue;
    }

    // 简化并计算最小和最大步长
    ExprPtr minStride =
        IRSimplifier::simplify(alloc<Min>(aIndexStride, oIndexStride, true));
    ExprPtr maxStride =
        IRSimplifier::simplify(alloc<Max>(aIndexStride, oIndexStride, true));

    // 如果最小步长为0，则跳过当前维度的检查
    if (immediateEquals(minStride, 0)) {
      continue;
    }

    // 简化并计算最大步长对最小步长的模运算
    ExprPtr modCheck = IRSimplifier::simplify(alloc<Mod>(maxStride, minStride));

    // 如果模运算结果不为0，则跳过当前维度的检查
    if (!immediateEquals(modCheck, 0)) {
      continue;
    }

    // 简化并计算访问起始位置之间的差值
    ExprPtr startDiff = IRSimplifier::simplify(
        alloc<Sub>(info->bounds()[b].start, other->bounds()[b].start));

    // 判断差值是否为负数
    bool diffNegative = immediateIsNegative(startDiff);
    // 判断最小步长是否为负数
    bool strideNegative = immediateIsNegative(minStride);

    // 如果差值的符号与最小步长的符号不同，则取其相反数
    if (diffNegative != strideNegative) {
      startDiff =
          IRSimplifier::simplify(alloc<Sub>(immLike(startDiff, 0), startDiff));
    }

    // 如果访问A和访问O具有相同的步长，并且起始元素之间的差值小于该步长，则返回true
    if (exprEquals(minStride, maxStride)) {
      ExprPtr check1 = IRSimplifier::simplify(
          alloc<CompareSelect>(startDiff, minStride, kLT));
      if (check1->isConstant() && immediateEquals(check1, 1)) {
        return true;
      }
    }

    // 简化并计算起始差值对最小步长的模运算
    startDiff = IRSimplifier::simplify(alloc<Mod>(startDiff, minStride));

    // 根据最小步长的符号确定比较操作类型
    CompareSelectOperation op = strideNegative ? kLT : kGT;

    // 如果起始差值对最小步长的模运算结果为1，则返回true
    ExprPtr check = IRSimplifier::simplify(
        alloc<CompareSelect>(startDiff, immLike(startDiff, 0), op));

    if (check->isConstant() && immediateEquals<int>(check, 1)) {
      return true;
    }

    // 如果我们可以考虑执行顺序，并且偏移量的差异为
    // 如果条件不满足，则返回 false
    if (!parallelized && diffNegative == strideNegative &&
        immediateEquals(startDiff, 0)) {
      // 如果以上三个条件都满足，则返回 true
      return true;
    }
  }

  // 如果以上条件均不满足，则返回 false
  return false;
// 访问器函数，用于处理 For 循环节点
void MemDependencyChecker::visit(ForPtr v) {
  // 获取循环变量
  VarPtr var = v->var();

  // 保存上一个语句节点，并更新为当前的 For 循环节点
  StmtPtr last = lastStmt_;
  lastStmt_ = v;

  // 递归处理循环变量
  v->var()->accept(this);

  // 对于 For 循环起始和结束表达式内部的加载操作是特殊的。
  // 它们存在于外围作用域，但循环体内的访问可能依赖于它们，通过循环变量的使用。
  // 我们处理的方式是创建一个新的作用域，以便在范围内轻松访问访问列表。
  auto extentsScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);
  currentScope_ = extentsScope;

  // 递归处理 For 循环的起始和结束表达式
  v->start()->accept(this);
  v->stop()->accept(this);

  // 恢复当前作用域为其父级
  currentScope_ = currentScope_->parent;

  // 创建一个新的作用域，用于处理循环体
  auto newScope = std::make_shared<Scope>(v->body(), currentScope_);
  currentScope_ = newScope;

  // 递归处理 For 循环的主体
  v->body()->accept(this);

  // 恢复上一个语句节点
  lastStmt_ = last;

  // 现在我们需要确定循环中的访问是否依赖于其他循环迭代。
  //
  // 这是真正的挑战，它依赖于完全展开的边界和符号边界。

  // 索引必须单调变化，以避免交集。这很难确定，所以这是我们的保守估计希望足够保守。

  // 至少一个依赖索引的大小必须 >= 循环的大小。

  // 第一步是推断相对于每个访问的每个维度的步长，我们通过将循环变量替换为 (var+1) 进入索引表达式来实现。

  // 循环步长的向量存储
  std::vector<std::vector<ExprPtr>> loopStrides;
  loopStrides.resize(currentScope_->accesses_.size());

  // 遍历当前作用域的访问列表
  for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
    auto& info = currentScope_->accesses_[a];

    // 获取访问的索引表达式
    std::vector<ExprPtr> indices = info->getIndices();

    // 获取当前访问的循环索引步长
    std::vector<ExprPtr>& loopIndicesStride = loopStrides[a];
    loopIndicesStride.resize(indices.size());

    // 索引表达式必须以某种方式依赖于循环变量，才能有步长。
    // 遍历 indices 容器中的每一个索引 i
    for (const auto i : c10::irange(indices.size())) {
      // 创建 VarFinder 对象 vf，用于查找变量 var 的引用
      VarFinder vf;
      // 如果当前索引 i 中不含有变量 var 的引用
      if (vf.find(indices[i]).count(var) == 0) {
        // 将 loopIndicesStride[i] 设置为 immLike(indices[i], 0)
        loopIndicesStride[i] = immLike(indices[i], 0);
      } else {
        // 如果之前已经交换了此边界的起始点和结束点，需要对边界的反向应用替换
        if (info->bounds()[i].swapped) {
          // 对边界的结束点进行替换操作，用 var 替换为 v->start() 后简化
          info->bounds()[i].end = IRSimplifier::simplify(
              SubstituteInClone(info->bounds()[i].end, {{var, v->start()}}));
          // 对边界的起始点进行替换操作，用 var 替换为 Sub(v->stop(), immLike(v->stop(), 1)) 后简化
          info->bounds()[i].start = IRSimplifier::simplify(SubstituteInClone(
              info->bounds()[i].start,
              {{var, alloc<Sub>(v->stop(), immLike(v->stop(), 1))}}));

        } else {
          // 对边界的起始点进行替换操作，用 var 替换为 v->start() 后简化
          info->bounds()[i].start = IRSimplifier::simplify(
              SubstituteInClone(info->bounds()[i].start, {{var, v->start()}}));
          // 对边界的结束点进行替换操作，用 var 替换为 Sub(v->stop(), immLike(v->stop(), 1)) 后简化
          info->bounds()[i].end = IRSimplifier::simplify(SubstituteInClone(
              info->bounds()[i].end,
              {{var, alloc<Sub>(v->stop(), immLike(v->stop(), 1))}}));
        }

        // 计算循环步长
        ExprPtr zeroStep = indices[i];
        ExprPtr oneStep = SubstituteInClone(
            indices[i], {{var, alloc<Add>(var, immLike(var, 1))}});
        loopIndicesStride[i] =
            IRSimplifier::simplify(alloc<Sub>(oneStep, zeroStep));

        // 如果起始点小于结束点，则交换边界的顺序
        ExprPtr diff = IRSimplifier::simplify(
            alloc<Sub>(info->bounds()[i].end, info->bounds()[i].start));
        if (diff->isConstant() && immediateIsNegative(diff)) {
          info->bounds()[i].swap();
        }

        // 如果此访问使用循环变量，则依赖于用于计算循环变量的加载
        for (auto& extentLoad : extentsScope->accesses_) {
          info->addDependency(extentLoad);
          extentLoad->addDependent(info);
        }
      }
    }
  }

  // 现在需要更新 openWrites 中的边界，因为这是我们用来合并的内容
  for (auto& openWritePair : currentScope_->openWrites_) {
    for (auto& pair : openWritePair.second) {
      IndexBounds& bounds = pair.first;

      // 边界可能不包含循环变量，但在这种情况下 Substitute 操作不会产生影响
      for (auto& bound : bounds) {
        // 对边界的起始点进行替换操作，用 var 替换为 v->start() 后简化
        bound.start = IRSimplifier::simplify(
            SubstituteInClone(bound.start, {{var, v->start()}}));
        // 对边界的结束点进行替换操作，用 var 替换为 Sub(v->stop(), immLike(v->stop(), 1)) 后简化
        bound.end = IRSimplifier::simplify(SubstituteInClone(
            bound.end, {{var, alloc<Sub>(v->stop(), immLike(v->stop(), 1))}}));

        // 如果起始点小于结束点，则交换边界的顺序
        ExprPtr diff =
            IRSimplifier::simplify(alloc<Sub>(bound.end, bound.start));
        if (diff->isConstant() && immediateIsNegative(diff)) {
          bound.swap();
        }
      }
  }
}

// TODO this isn't a scalable way to determine parallelism.
// 指示这种方式不是一种可扩展的确定并行性的方法。

// Store buffers allocated at this scope.
// 存储在当前作用域分配的缓冲区。

// Scanning from the top of the loop, we look for accesses which may depend
// on a previous or parallel loop iteration.
// 从循环顶部开始扫描，查找可能依赖于先前或并行循环迭代的访问。

for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
  auto& info = currentScope_->accesses_[a];
  if (info->type() == AccessType::Alloc) {
    // If the access type is Allocation, insert the variable into local_intermediates.
    // 如果访问类型是分配，则将变量插入 local_intermediates 中。
    local_intermediates.insert(info->var());
    continue;
  }

  if (!info->isRead()) {
    continue;
  }

  // Vars that don't carry outside this scope can't have loop self dependence.
  // 在此作用域外部没有持续的变量不会有循环自依赖。
  if (local_intermediates.count(info->var())) {
    continue;
  }

  // Copy the bounds so we can keep track of open bounds internally without
  // affecting the merge into the enclosing scope. The open portion of the
  // bounds may be cut into multiple independent slices.
  // 复制边界，以便我们可以在不影响合并到封闭作用域的情况下内部跟踪开放边界。
  // 边界的开放部分可能被切割成多个独立的片段。
  std::vector<IndexBounds> openBounds({info->bounds()});

  // Scan from the bottom of the loop.
  // 从循环底部开始扫描。
  for (size_t j = currentScope_->accesses_.size() - 1; j > a; --j) {
    std::shared_ptr<AccessInfo> other = currentScope_->accesses_[j];
    if (!other->isWrite()) {
      continue;
    }

    if (info->var() != other->var()) {
      continue;
    }

    if (info->hasDependency(other)) {
      continue;
    }

    // Whether or not the accesses within the loop are dependent on other
    // iterations depends whether the loop could be parallelized, the
    // difference in their strides and their start offset.
    // 循环内的访问是否依赖于其他迭代取决于循环是否可以并行化，
    // 它们的步长差异以及它们的起始偏移量。
    bool iterationsDistinct = executionSafetyCheck(
        info,
        other,
        loopStrides[a],
        loopStrides[j],
        !allowExecutionOrderAnalysis_ || parallelized);

    if (iterationsDistinct) {
      continue;
    }

    std::vector<IndexBounds> newBoundSlices;
    for (auto& b : openBounds) {
      OverlapKind overlap = overlaps(b, other->bounds());
      if (overlap == OverlapKind::NoOverlap) {
        newBoundSlices.push_back(b);
        continue;
      }

      // It's dependent, link it to other.
      // 它是依赖的，将其链接到 other。
      info->addDependency(other);
      other->addDependent(info);

      if (overlap == OverlapKind::Contains) {
        continue;
      }

      // Otherwise update openBounds.
      // 否则更新 openBounds。
      auto slices = subtractIndicesBounds(b, other->bounds(), overlap);
      std::move(
          slices.begin(), slices.end(), std::back_inserter(newBoundSlices));
    }

    if (newBoundSlices.empty()) {
      break;
    }
    openBounds.swap(newBoundSlices);
  }
}

std::vector<std::shared_ptr<AccessInfo>> mergedAccesses;
// 创建一个空的智能指针共享数组，用于合并访问信息
mergedAccesses.reserve(
    extentsScope->accesses_.size() + currentScope_->accesses_.size());
// 预留足够的空间以容纳 extentsScope 和 currentScope 的所有访问信息

std::copy(
    extentsScope->accesses_.begin(),
    extentsScope->accesses_.end(),
    std::back_inserter(mergedAccesses));
// 将 extentsScope 的所有访问信息复制到 mergedAccesses 的末尾

std::copy(
    currentScope_->accesses_.begin(),
    currentScope_->accesses_.end(),
    std::back_inserter(mergedAccesses));
// 将 currentScope 的所有访问信息复制到 mergedAccesses 的末尾

scopeToAccesses_.emplace(v, mergedAccesses);
// 将变量 v 与合并后的访问信息数组 mergedAccesses 关联起来，存入 scopeToAccesses_ 中

// 在不关闭的情况下合并会稍微快一些，并且由于在开始和停止表达式之间不会发生写操作，因此我们将这样做。
mergeScope(extentsScope, extentsScope->parent, false);
// 调用 mergeScope 函数，将 extentsScope 与其父作用域 extentsScope->parent 合并，不关闭合并操作

mergeScope(currentScope_, currentScope_->parent, true);
// 调用 mergeScope 函数，将 currentScope_ 与其父作用域 currentScope_->parent 合并，并关闭合并操作

currentScope_ = currentScope_->parent;
// 将当前作用域 currentScope_ 设置为其父作用域 currentScope_->parent
void MemDependencyChecker::visit(CondPtr v) {
  // 保存上一个语句的引用
  StmtPtr last = lastStmt_;
  // 将当前语句设置为条件语句节点
  lastStmt_ = v;

  // 创建一个新的作用域，包含当前作用域和当前块
  auto enclosingScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);

  // 处理条件表达式所在的作用域
  v->condition()->accept(this);

  // 获取条件为真和条件为假的语句块
  BlockPtr true_stmt = v->true_stmt();
  BlockPtr false_stmt = v->false_stmt();

  // 创建真和假分支的作用域，以避免块访问者创建和合并新作用域
  auto trueScope = std::make_shared<Scope>(true_stmt, enclosingScope);
  auto falseScope = std::make_shared<Scope>(false_stmt, enclosingScope);

  // 如果真分支不为空，则设置当前作用域为真分支的作用域，并访问真分支
  if (true_stmt) {
    currentScope_ = trueScope;
    true_stmt->accept(this);
  }

  // 如果假分支不为空，则设置当前作用域为假分支的作用域，并访问假分支
  if (false_stmt) {
    currentScope_ = falseScope;
    false_stmt->accept(this);
  }

  // TODO(nickg): 这个逻辑不太正确，如果写入的边界范围同时出现在真分支和假分支中，
  // 那么可以在外围作用域中关闭重叠的访问。没有进行这样的分析，未来的访问可能依赖于
  // 所有三个作用域中共同范围的写入。这是一个误报，所以在短期内不太糟糕，我想。

  // 将真分支和假分支合并到父作用域中，但不关闭任何访问
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  // 将外围作用域合并到其父作用域中
  mergeScope(enclosingScope, enclosingScope->parent, false);

  // 恢复当前作用域为外围作用域，并将访问的访问列表与条件指针关联存储
  currentScope_ = enclosingScope;
  scopeToAccesses_.emplace(v, enclosingScope->accesses_);

  // 恢复当前作用域为外围作用域的父作用域，并恢复上一个语句的引用
  currentScope_ = enclosingScope->parent;
  lastStmt_ = last;
}

void MemDependencyChecker::visit(IfThenElsePtr v) {
  // 处理条件表达式所在的外围作用域
  v->condition()->accept(this);

  // 获取条件为真和条件为假的表达式
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();

  // 保存当前的外围作用域
  auto enclosingScope = currentScope_;

  // 创建用于保存下游加载的作用域。将块设置为nullptr是安全的，因为它仅由语句而不是表达式使用。
  auto trueScope = std::make_shared<Scope>(nullptr, enclosingScope);
  auto falseScope = std::make_shared<Scope>(nullptr, enclosingScope);

  // 如果真值表达式不为空，则设置当前作用域为真值作用域，并访问真值表达式
  if (true_value) {
    currentScope_ = trueScope;
    true_value->accept(this);
  }

  // 如果假值表达式不为空，则设置当前作用域为假值作用域，并访问假值表达式
  if (false_value) {
    currentScope_ = falseScope;
    false_value->accept(this);
  }

  // 这里不会出现与Cond中类似的问题，因为没有表达式是写入操作，不会从外围作用域中产生误报。

  // 将真分支和假分支合并到父作用域中，但不关闭任何访问
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  // 恢复当前作用域为外围作用域
  currentScope_ = enclosingScope;
}
// 处理 CompareSelectPtr 类型的访问方法
void MemDependencyChecker::visit(CompareSelectPtr v) {
  // condition is in enclosing scope.
  // 访问条件表达式的左侧和右侧
  v->lhs()->accept(this);
  v->rhs()->accept(this);

  // 获取条件为真和条件为假时的返回值表达式
  ExprPtr true_value = v->ret_val1();
  ExprPtr false_value = v->ret_val2();

  // 保存当前的作用域
  auto enclosingScope = currentScope_;

  // 创建用于保存下游 Loads 的作用域。由于只有 Stmt 使用 Block，可以安全地将 nullptr 传入作为 Block。
  auto trueScope = std::make_shared<Scope>(nullptr, enclosingScope);
  auto falseScope = std::make_shared<Scope>(nullptr, enclosingScope);

  // 如果存在条件为真的返回值，则切换当前作用域并访问它
  if (true_value) {
    currentScope_ = trueScope;
    true_value->accept(this);
  }

  // 如果存在条件为假的返回值，则切换当前作用域并访问它
  if (false_value) {
    currentScope_ = falseScope;
    false_value->accept(this);
  }

  // 合并真和假分支到父作用域，但不关闭任何访问
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  // 恢复当前作用域为原始的 enclosingScope
  currentScope_ = enclosingScope;
}

// 插入访问信息以处理缓冲区（例如输入和输出）
void MemDependencyChecker::insertBuffers(
    std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>>& bufs,
    AccessType type) {
  for (auto& pair : bufs) {
    // 获取缓冲区和其基本变量
    BufPtr b = pair.first;
    VarPtr var = b->base_handle();

    // 创建索引边界
    IndexBounds bounds;
    for (const auto& d : b->dims()) {
      bounds.emplace_back(
          immLike(d, 0), IRSimplifier::simplify(alloc<Sub>(d, immLike(d, 1))));
    }

    // 创建访问信息并更新历史记录
    auto info =
        std::make_shared<AccessInfo>(nextAccess_++, type, nullptr, var, bounds);

    bufs[b] = info;

    auto& history = currentScope_->openWrites_[var];
    updateWriteHistory(history, info, info->id());
    currentScope_->accesses_.push_back(info);
  }
}

// 处理 BlockPtr 类型的访问方法
void MemDependencyChecker::visit(BlockPtr v) {
  auto prev_scope = currentScope_;

  // 处理内核输入
  if (prev_scope->block == nullptr) {
    insertBuffers(inputs_, AccessType::Input);
  }

  // 如果当前作用域的块与给定的块不同，则创建新的作用域
  if (currentScope_->block != v) {
    currentScope_ = std::make_shared<Scope>((BlockPtr)v, prev_scope);
  }

  // 访问块内的每一个语句
  for (const auto& s : *v) {
    s->accept(this);
  }

  // 移除当前作用域中的本地变量
  for (const auto& v : currentScope_->localVars) {
    knownVarBounds_.erase(v);
  }

  // 更新已遮蔽变量的已知边界
  for (auto& pair : currentScope_->shadowedVarBounds) {
    knownVarBounds_[pair.first] = pair.second;
  }

  // 将当前作用域的访问信息与其对应的块关联起来
  scopeToAccesses_.emplace(v, currentScope_->accesses_);

  // 如果当前作用域与前一个作用域不同，则合并当前作用域和前一个作用域
  if (currentScope_ != prev_scope) {
    mergeScope(currentScope_, prev_scope, true);
    currentScope_ = prev_scope;
  }

  // 处理内核输出
  if (prev_scope->block == nullptr) {
    insertBuffers(outputs_, AccessType::Output);
  }
}

// 处理 LetPtr 类型的访问方法
void MemDependencyChecker::visit(LetPtr v) {
  StmtPtr last = lastStmt_;
  lastStmt_ = v;

  IRVisitor::visit(v);

  lastStmt_ = last;

  // 获取变量并检查其已知的边界信息
  VarPtr var = v->var();
  if (knownVarBounds_.count(var) != 0) {
    // ...
    # 将当前变量的已知边界复制到当前作用域的影子变量边界中
    currentScope_->shadowedVarBounds[var] = knownVarBounds_[var];
  }

  # 向当前作用域的局部变量集合中插入变量
  currentScope_->localVars.insert(var);
  # 将变量的已知边界更新为一个包含当前值的边界范围
  knownVarBounds_[var] = {v->value(), v->value()};
}

// 不支持 AtomicAdd 操作，因为它涉及读写操作，目前只在 CUDA 代码生成期间插入，所以这段代码暂时不会使用。
void MemDependencyChecker::visit(AtomicAddPtr v) {
  // 抛出运行时错误，暂未实现 AtomicAdd 操作的内存依赖检查
  throw std::runtime_error("MemDependencyChecker AtomicAdd unimplemented");
}

void MemDependencyChecker::visit(AllocatePtr v) {
  // 保存上一个语句，以便在函数结束时恢复
  StmtPtr last = lastStmt_;
  // 将当前语句设为最后访问的语句
  lastStmt_ = v;

  // 访问 Allocate 语句
  IRVisitor::visit(v);

  // 获取缓冲变量
  VarPtr var = v->buffer_var();
  IndexBounds bounds;

  // TODO: 在下面的过程中移除 "buf_flat_size" 的处理，并扩展 buf 边界检查以支持 N 维索引访问和 1 维索引访问。
  // "Allocate" 语句基于 "Buf"，支持 N 维索引访问和 1 维索引访问。目前内存分析中的写入边界检查无法识别 N 维 buf 的 1 维索引访问。
  // 因此，我们在这里将 N 维 buf 展平，以避免边界检查失败。但这不是正确的方法，应该修复这个问题。
  ExprPtr flat_size = buf_flat_size(v->buf());
  flat_size =
      IRSimplifier::simplify(alloc<Sub>(flat_size, immLike(flat_size, 1)));
  bounds.emplace_back(immLike(flat_size, 0), flat_size);

  // 创建 AccessInfo 对象
  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Alloc, nullptr, var, bounds);

  // 将 info 存储到 intermediates_ 中
  intermediates_[var] = info;

  // 更新当前作用域的写入历史记录
  auto& history = currentScope_->openWrites_[var];
  history.emplace_back(std::make_pair(info->bounds(), info));
  currentScope_->accesses_.push_back(info);

  // 恢复上一个语句
  lastStmt_ = last;
}

void MemDependencyChecker::visit(FreePtr v) {
  // 保存上一个语句，以便在函数结束时恢复
  StmtPtr last = lastStmt_;
  // 将当前语句设为最后访问的语句
  lastStmt_ = v;

  // 访问 Free 语句
  IRVisitor::visit(v);

  // 获取缓冲变量
  VarPtr var = v->buffer_var();
  // 查找变量在 intermediates_ 中的信息
  auto it = intermediates_.find(var);
  // 断言确保在内存依赖检查器中的 fuser 中找到变量的中间变量
  TORCH_INTERNAL_ASSERT(
      it != intermediates_.end(),
      buildErrorMessage(
          "Expected to find '" + var->name_hint() +
          "' in intermediate vars in mem dep checker in the fuser."));

  // 获取变量的边界
  IndexBounds bounds = it->second->bounds();
  // 创建 AccessInfo 对象
  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Free, nullptr, var, bounds);

  // 更新写入历史记录
  auto& history = currentScope_->openWrites_[var];
  updateWriteHistory(history, info, info->id());
  currentScope_->accesses_.push_back(info);

  // 恢复上一个语句
  lastStmt_ = last;
}

void MemDependencyChecker::updateWriteHistory(
    std::list<BoundRelationship>& writeHistory,
    const std::shared_ptr<AccessInfo>& info,
    size_t latestAccessToClose,
    bool closeOverlapped,
    bool insert) {
  // 判断是否为写入操作
  bool isWrite = info->isWrite();

  // 遍历写入历史记录
  for (auto it = writeHistory.begin(); it != writeHistory.end();) {
    auto& indexBounds = it->first;
    std::shared_ptr<AccessInfo> other = it->second;
    // 如果存在依赖关系，则继续下一个历史记录
    if (info->hasDependency(other)) {
      ++it;
      continue;
    }

    // 检查索引边界是否重叠
    OverlapKind overlap = overlaps(indexBounds, info->bounds());

    // 如果没有重叠，继续下一个历史记录
    if (overlap == OverlapKind::NoOverlap) {
      ++it;
      continue;
    }

    // 只有写入操作可以关闭打开的访问
    if (!isWrite) {
      info->addDependency(other);
      other->addDependent(info);
      ++it;
      continue;
    }
    // 如果我们不需要关闭重叠的访问，可以在这里停止处理。
    if (!closeOverlapped || other->id() > latestAccessToClose) {
      // 如果不需要关闭重叠的访问或者其他访问的ID大于最近要关闭的访问ID，则跳过当前迭代。
      ++it;
      continue;
    }

    // 如果重叠类型为完全包含或者相等
    if (overlap == OverlapKind::ContainedOrEqual) {
      // 完全重叠意味着新的访问完全替换了旧的访问。
      it = writeHistory.erase(it);
    } else {
      // 新的写访问部分重叠了先前的写访问。我们希望保留两者，但仅跟踪先前写访问未覆盖的部分。

      // 确定先前绑定的片段不被新信息覆盖。
      auto newBounds =
          subtractIndicesBounds(indexBounds, info->bounds(), overlap);

      // 删除旧的片段。
      it = writeHistory.erase(it);

      // 添加所有新的片段。
      for (auto& b : newBounds) {
        writeHistory.insert(it, std::make_pair(b, other));
      }
      // 由于在上面的`erase`之后迭代器已经被更新，因此不需要增加迭代器。
    }
  }

  // 如果需要插入并且是写操作
  if (insert && isWrite) {
    // 将当前写访问的边界信息和信息对象添加到写历史记录中。
    writeHistory.emplace_back(info->bounds(), info);
  }
}

// 合并作用域的方法，将子作用域的访问信息合并到父作用域中
void MemDependencyChecker::mergeScope(
    const std::shared_ptr<Scope>& child,
    const std::shared_ptr<Scope>& parent,
    bool closeOverlapped) {
  // 如果子作用域中的访问信息为空，则直接返回
  if (child->accesses_.empty()) {
    return;
  }

  // 更新依赖关系，但不添加新的开放写入操作
  for (auto& info : child->accesses_) {
    // 故意使用 operator[]，如果不存在则创建
    auto& writeHistory = parent->openWrites_[info->var()];

    // 找到最后需要关闭的最新访问的标识
    size_t latestAccessToClose = child->accesses_.front()->id();
    // 更新写入历史记录
    updateWriteHistory(
        writeHistory, info, latestAccessToClose, closeOverlapped, false);
  }

  // 复制子作用域中的开放写入操作到父作用域
  for (auto& pair : child->openWrites_) {
    VarPtr var = pair.first;

    // 故意使用 operator[]，如果不存在则创建
    auto& writeHistory = parent->openWrites_[var];

    // 将子作用域的每个关系复制到父作用域的写入历史记录中
    for (auto& rel : pair.second) {
      writeHistory.push_back(rel);
    }
  }

  // 父作用域现在负责保存所有访问信息
  parent->accesses_.insert(
      parent->accesses_.end(),
      std::make_move_iterator(child->accesses_.begin()),
      std::make_move_iterator(child->accesses_.end()));
}

// VarBoundBinder 类，应用已知边界到符号表达式的访问者
class VarBoundBinder : public IRVisitor {
 public:
  VarBoundBinder(const VarBoundMap& vars) : vars_(vars) {}

  // 获取表达式的边界信息
  Bound getBounds(ExprPtr e) {
    min_ = e;
    max_ = e;
    e->accept(this);
    min_ = IRSimplifier::simplify(min_);
    max_ = IRSimplifier::simplify(max_);
    return {min_, max_};
  }

 private:
  void visit(VarPtr v) override {
    // 查找变量在已知边界中的信息并替换表达式的最小和最大值
    auto it = vars_.find(v);
    if (it == vars_.end()) {
      return;
    }

    min_ = SubstituteInClone(min_, {{v, it->second.start}});
    max_ = SubstituteInClone(max_, {{v, it->second.end}});
  }

  ExprPtr min_{nullptr};
  ExprPtr max_{nullptr};
  const VarBoundMap& vars_;
};

// 获取索引表达式的边界信息
std::vector<Bound> MemDependencyChecker::getIndicesBounds(
    const std::vector<ExprPtr>& indices) {
  std::vector<Bound> bounds;
  bounds.reserve(indices.size());
  VarBoundBinder binder(knownVarBounds_);
  // 遍历每个索引表达式，获取其边界信息并添加到结果集合中
  for (const auto& s : indices) {
    bounds.push_back(binder.getBounds(s));
  }
  return bounds;
}

} // namespace torch::jit::tensorexpr::analysis
```