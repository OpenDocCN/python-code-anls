# `.\pytorch\torch\csrc\jit\tensorexpr\registerizer.cpp`

```
// 包含头文件 <torch/csrc/jit/tensorexpr/registerizer.h> 和 <iostream>
#include <torch/csrc/jit/tensorexpr/registerizer.h>
#include <iostream>

// 命名空间 torch::jit::tensorexpr::registerizer
namespace torch::jit::tensorexpr::registerizer {

// AccessInfo 类的实现

// 向访问信息中添加存储操作
void AccessInfo::addStore(StorePtr store, const std::shared_ptr<Scope>& scope) {
  // 如果 block_ 已经存在，则将其设置为 scope->block() 和 block_ 的共享父级；否则直接使用 scope->block()
  block_ =
      block_ ? Block::getSharedParent(block_, scope->block()) : scope->block();

  // 如果已经有使用，并且是相同的存储操作，则意味着右手边存在相同的访问。
  firstUsageOverlapped_ |= first_usage_ == store;
  // 设置 first_usage_ 为 block_->getEnclosedRoot(first_usage_) 或者 store
  first_usage_ = first_usage_ ? block_->getEnclosedRoot(first_usage_) : store;
  // 设置 last_usage_ 为 store
  last_usage_ = store;

  // 简化存储成本为 store_cost_ + 1
  store_cost_ =
      IRSimplifier::simplify(alloc<Add>(store_cost_, immLike(store_cost_, 1)));
  // 将 store 添加到 stores_ 向量中
  stores_.push_back(store);

  // 设置 conditionId_ 为 scope->conditionId()
  conditionId_ = scope->conditionId();
  // 重置 hiddenAccess_
  hiddenAccess_.reset();
}

// 向访问信息中添加加载操作
void AccessInfo::addLoad(
    LoadPtr load,
    const std::shared_ptr<Scope>& scope,
    StmtPtr usage) {
  // 如果 block_ 已经存在，则将其设置为 scope->block() 和 block_ 的共享父级；否则直接使用 scope->block()
  block_ =
      block_ ? Block::getSharedParent(block_, scope->block()) : scope->block();
  // 设置 first_usage_ 为 block_->getEnclosedRoot(first_usage_) 或者 usage
  first_usage_ = first_usage_ ? block_->getEnclosedRoot(first_usage_) : usage;
  // 设置 last_usage_ 为 usage
  last_usage_ = usage;

  // 简化加载成本为 load_cost_ + 1
  load_cost_ =
      IRSimplifier::simplify(alloc<Add>(load_cost_, immLike(load_cost_, 1)));
  // 将 load 添加到 loads_ 向量中
  loads_.push_back(load);

  // 设置 conditionId_ 为 scope->conditionId()
  conditionId_ = scope->conditionId();
  // 重置 hiddenAccess_
  hiddenAccess_.reset();
}

// 合并另一个访问信息对象的信息到当前对象中
void AccessInfo::merge(const std::shared_ptr<AccessInfo>& other) {
  // 断言哈希值 hash_ 和 other->hash() 相等
  TORCH_INTERNAL_ASSERT(
      hash_ == other->hash(),
      buildErrorMessage(
          "Expected hashes to match in registerizer in the fuser."));
  // 断言 indices_.size() 和 other->indices().size() 相等
  TORCH_INTERNAL_ASSERT(
      indices_.size() == other->indices().size(),
      buildErrorMessage(
          "Expected ranks to match in registerizer in the fuser."));

  // 设置 last_usage_ 为 other->last_usage()
  last_usage_ = other->last_usage();
  // 将 other->stores() 中的存储操作添加到 stores_ 向量中
  for (const auto& s : other->stores()) {
    stores_.push_back(s);
  }
  // 将 other->loads() 中的加载操作添加到 loads_ 向量中
  for (const auto& l : other->loads()) {
    loads_.push_back(l);
  }

  // 简化存储成本为 store_cost_ + other->store_cost()
  store_cost_ =
      IRSimplifier::simplify(alloc<Add>(store_cost_, other->store_cost()));
  // 简化加载成本为 load_cost_ + other->load_cost()
  load_cost_ =
      IRSimplifier::simplify(alloc<Add>(load_cost_, other->load_cost()));

  // 设置 block_ 为 block_ 和 other->block() 的共享父级
  block_ = Block::getSharedParent(block_, other->block());
  // 将 first_usage_ 和 last_usage_ 更新为它们在父块中的根
  first_usage_ = block_->getEnclosedRoot(first_usage_);
  last_usage_ = block_->getEnclosedRoot(last_usage_);
  // 重置 hiddenAccess_
  hiddenAccess_.reset();
}

// 检查两个访问信息对象是否重叠
bool AccessInfo::overlaps(const std::shared_ptr<AccessInfo>& other) {
  // 断言索引数量 indices_.size() 和 other->indices().size() 相等
  TORCH_INTERNAL_ASSERT(
      indices_.size() == other->indices().size(),
      buildErrorMessage(
          "Expected ranks to match in registerizer in the fuser."));

  // 获取 other 的索引
  auto& other_indices = other->indices();

  // 假设它们重叠
  bool overlap = true;
  // 遍历所有维度
  for (size_t i = 0; i < indices_.size(); ++i) {
    // 计算两个索引之间的差异
    ExprPtr diff = alloc<Sub>(indices_[i], other_indices[i]);
    // 简化差异表达式
    diff = IRSimplifier::simplify(diff);

    // 如果差异是常数且不为零，则表示维度上有保证的差异，认为不重叠
    if (diff->isConstant() && !immediateEquals(diff, 0)) {
      overlap = false;
      break;
    }
  }

  return overlap;
}

} // namespace registerizer
} // namespace torch::jit::tensorexpr
// 检查变量是否依赖于给定的变量 v
bool AccessInfo::dependsOnVar(VarPtr v) {
  // 创建一个 VarFinder 对象 vf
  VarFinder vf;
  // 遍历 indices_ 中的每一个元素 i
  for (const auto& i : indices_) {
    // 将 i 接受给 VarFinder 对象 vf 进行处理
    i->accept(&vf);
  }

  // 返回 vf 中是否包含变量 v 的结果
  return vf.vars().count(v);
}

// 克隆包含隐藏信息的 AccessInfo 对象
std::shared_ptr<AccessInfo> AccessInfo::cloneWithHiddenInfo(
    const std::shared_ptr<AccessInfo>& orig) {
  // 使用 orig 的 hash、buf、indices 和 accessOrder 创建一个新的 AccessInfo 对象 newInfo
  std::shared_ptr<AccessInfo> newInfo = std::make_shared<AccessInfo>(
      orig->hash(), orig->buf(), orig->indices(), orig->accessOrder());

  // 复制 orig 的 block、first_usage、last_usage 等信息到 newInfo
  newInfo->block_ = orig->block_;
  newInfo->first_usage_ = orig->first_usage_;
  newInfo->last_usage_ = orig->last_usage_;
  newInfo->firstUsageOverlapped_ = orig->firstUsageOverlapped_;
  newInfo->store_cost_ = orig->store_cost_;
  newInfo->load_cost_ = orig->load_cost_;

  // 将 orig 的 stores_ 和 loads_ 中的元素复制到 newInfo 的对应成员变量中
  for (const auto& s : orig->stores_) {
    newInfo->stores_.push_back(s);
  }
  for (const auto& s : orig->loads_) {
    newInfo->loads_.push_back(s);
  }

  // 复制 orig 的 conditionId_ 和 hiddenAccess_ 到 newInfo
  newInfo->conditionId_ = orig->conditionId_;
  newInfo->hiddenAccess_ = orig;

  // 返回包含隐藏信息的新的 AccessInfo 对象 newInfo
  return newInfo;
}

// 打印 AccessInfo 对象的信息
void AccessInfo::print() const {
  // 输出 buf_ 的值和 indices_ 中每个元素的值
  std::cout << "Access: " << *buf_ << "{";
  for (const auto& i : indices_) {
    std::cout << *i << " ";
  }
  // 输出 stores_ 和 loads_ 的大小以及对应的 store_cost_ 和 load_cost_ 的值
  std::cout << "} stores: " << stores_.size() << " (" << *store_cost_ << ") -";
  std::cout << " loads: " << loads_.size() << " (" << *load_cost_ << ")";
  // 如果 conditionId_ 不为 null，则输出 conditionId_ 的值
  if (conditionId_) {
    std::cout << " cond: " << conditionId_;
  }

  std::cout << "\n";
}

// 将 AccessInfo 对象添加到闭合访问列表中
void Scope::closeAccess(const std::shared_ptr<AccessInfo>& info) {
  // 将 info 添加到 closedAccesses_ 中
  closedAccesses_.push_back(info);
}

// 根据缓冲区指针获取对应的访问哈希映射
AccessHashMap& Scope::getAccessMapByBuf(BufPtr b) {
  // 查找 b 是否已存在于 openAccesses_ 中
  auto it = openAccesses_.find(b);
  // 如果不存在，则创建一个空的 AccessHashMap 并返回
  if (it == openAccesses_.end()) {
    return openAccesses_[b];
  }

  // 否则返回已存在的 AccessHashMap
  return it->second;
}

// 过滤掉存储和加载成本都小于等于 1 的闭合访问信息
void Scope::filterClosed() {
  closedAccesses_.erase(
      std::remove_if(
          closedAccesses_.begin(),
          closedAccesses_.end(),
          [](auto info) {
            return info->store_cost()->isConstant() &&
                immediateAs<int>(info->store_cost()) <= 1 &&
                info->load_cost()->isConstant() &&
                immediateAs<int>(info->load_cost()) <= 1;
          }),
      closedAccesses_.end());
}

// 将 AccessInfo 对象添加到作用域中的闭合访问列表
void RegisterizerAnalysis::closeAccessIntoScope(
    const std::shared_ptr<AccessInfo>& info,
    const std::shared_ptr<Scope>& scope) {
  // 如果 exprConditionals_ 中存在 info 的 conditionId()，则返回
  if (exprConditionals_.count(info->conditionId()) != 0) {
    return;
  }

  // 如果 info 有隐藏的访问信息，则递归调用 closeAccessIntoScope 处理隐藏访问信息
  if (info->hiddenAccess()) {
    closeAccessIntoScope(info->hiddenAccess(), scope);
    return;
  }

  // 否则将 info 添加到 scope 的闭合访问列表中
  scope->closeAccess(info);
}

// 访问 For 循环节点 v
void RegisterizerAnalysis::visit(ForPtr v) {
  // 如果循环选项中包含 GPU 相关的索引，则返回
  if (v->loop_options().is_gpu_block_index() ||
      v->loop_options().is_gpu_thread_index()) {
    # 抛出异常，指出注册化必须在并行性展开之后进行
    throw malformed_input(
        "Registerization must occur after parallelism flattening");
  }

  # 将当前作用域保存为父作用域，并创建一个新的作用域，用于处理循环体中的变量
  auto parent = currentScope_;
  currentScope_ = std::make_shared<Scope>(v->body(), parent);

  # 将循环变量添加到当前作用域的局部变量列表中
  currentScope_->addLocalVar(v->var());

  # 将循环变量推入语句堆栈的前部，并访问循环体内的语句
  stmtStack_.push_front(v);
  v->body()->accept(this);
  stmtStack_.pop_front();

  # 计算循环范围，即循环终止值减去循环起始值的简化表达式
  ExprPtr loopExtent =
      IRSimplifier::simplify(alloc<Sub>(v->stop(), v->start()));

  # 现在需要确定哪些访问可以提升出循环，它们的成本应乘以循环范围
  for (auto& pair : currentScope_->openAccesses()) {
    if (pair.second.empty()) {
      continue;
    }

    auto& childAccesses = pair.second;

    for (auto it = childAccesses.begin(); it != childAccesses.end();) {
      std::shared_ptr<AccessInfo>& candidate = it->second;

      # 如果访问是开放的但有条件，存在潜在问题。可能会有更高作用域的访问“解开”条件访问，
      # 这种情况下我们需要提升访问。如果没有更高作用域对此元素的访问，则不能安全地提升。
      # 在这个层面上，我们无法知道会发生还是不会发生这种情况。
      #
      # 这里的解决方案是分裂时空维度，并保持访问的两个版本。如果提升的访问未在上层使用，
      # 我们将回退到使用隐藏的条件性AccessInfo；如果使用了，我们将删除副本。
      if (candidate->conditionId() != 0) {
        candidate = AccessInfo::cloneWithHiddenInfo(candidate);
      }

      bool closed = false;
      # 如果此访问依赖于本地作用域变量，则无法将其提升出循环
      for (const auto& v : currentScope_->localVars()) {
        if (candidate->dependsOnVar(v)) {
          # 将访问关闭到当前作用域
          closeAccessIntoScope(candidate, currentScope_);
          closed = true;
          break;
        }
      }
      if (closed) {
        it = childAccesses.erase(it);
        continue;
      }

      # 提升访问！通过提升，我们将读取和写入操作移出循环，从而提升注册化此访问的效益乘以循环范围。
      candidate->setEnclosingBlock(parent->block());
      candidate->hoistCosts(loopExtent);

      # 在父块中，此循环语句是初始化器和终结器的插入点。
      candidate->setUsageMarks(v, v);

      ++it;
    }
  }

  # 如果访问在循环内关闭，则不能合并到现有开放访问中，但仍会关闭该现有访问。这与常规合并有所不同，因此我们需要首先处理关闭的访问。
  mergeHiddenScope(true);

  # 提升后，现在我们可以正常合并作用域。
  mergeCurrentScopeIntoParent();
};

// 访问条件语句节点的处理函数
void RegisterizerAnalysis::visit(CondPtr v) {
  // 获取条件表达式、真实分支和假实分支
  ExprPtr condition = v->condition();
  BlockPtr true_stmt = v->true_stmt();
  BlockPtr false_stmt = v->false_stmt();

  // 将当前节点压入语句栈
  stmtStack_.push_front(v);

  // 处理条件表达式（在当前作用域中）
  condition->accept(this);

  // 保存当前作用域，并创建真实分支和假实分支的作用域
  auto prev_scope = currentScope_;
  auto true_scope =
      std::make_shared<Scope>(true_stmt, prev_scope, ++conditionId_);
  auto false_scope =
      std::make_shared<Scope>(false_stmt, prev_scope, ++conditionId_);

  // 处理真实分支（如果存在）
  if (true_stmt) {
    currentScope_ = true_scope;
    true_stmt->accept(this);
    // 合并当前作用域中的隐藏作用域
    mergeHiddenScope(true);
    // 将当前作用域合并到父作用域中
    mergeCurrentScopeIntoParent();
  }
  // 处理假实分支（如果存在）
  if (false_stmt) {
    currentScope_ = false_scope;
    false_stmt->accept(this);
    // 合并当前作用域中的隐藏作用域
    mergeHiddenScope(true);
    // 将当前作用域合并到父作用域中
    mergeCurrentScopeIntoParent();
  }

  // TODO: 即使两个分支都是条件性的，如果它们在两个分支中完全重叠，我们可以合并访问，因为我们可以保证会命中其中一个定义。我们可能需要三路合并？并不像简单地先合并真和假分支那样简单。

  // 从语句栈中弹出当前节点
  stmtStack_.pop_front();
}

// IfThenElse 节点的访问处理函数
// 与 Cond 节点类似，但不是语句，因此内部不会进行寄存器化。但是，对访问的第一次引用可以在条件外部发生。
void RegisterizerAnalysis::visit(IfThenElsePtr v) {
  // 获取条件表达式、真值和假值
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();

  // 处理条件表达式（在当前作用域中）
  condition->accept(this);

  // 保存当前作用域，并创建真值和假值的作用域
  auto prev_scope = currentScope_;
  auto true_scope =
      std::make_shared<Scope>(prev_scope->block(), prev_scope, ++conditionId_);
  auto false_scope =
      std::make_shared<Scope>(prev_scope->block(), prev_scope, ++conditionId_);

  // 将 IfThenElse 节点的作用域存储在全局映射中，用于防止关闭需要在值中插入语句的任何访问，这些语句不能包含语句。
  exprConditionals_.insert(true_scope->conditionId());
  exprConditionals_.insert(false_scope->conditionId());

  // 处理真值（如果存在）
  if (true_value) {
    currentScope_ = true_scope;
    true_value->accept(this);
    // 合并当前作用域中的隐藏作用域
    mergeHiddenScope(false);
    // 将当前作用域合并到父作用域中
    mergeCurrentScopeIntoParent();
  }

  // 处理假值（如果存在）
  if (false_value) {
    currentScope_ = false_scope;
    false_value->accept(this);
    // 合并当前作用域中的隐藏作用域
    mergeHiddenScope(false);
    // 将当前作用域合并到父作用域中
    mergeCurrentScopeIntoParent();
  }
}

// 处理 Let 节点的访问处理函数
void RegisterizerAnalysis::visit(LetPtr v) {
  // 向当前作用域添加局部变量
  currentScope_->addLocalVar(v->var());

  // 将 Let 节点压入语句栈
  stmtStack_.push_front(v);
  // 处理 Let 节点的值表达式
  v->value()->accept(this);
  // 从语句栈中弹出 Let 节点
  stmtStack_.pop_front();
}

// 处理 Block 节点的访问处理函数
void RegisterizerAnalysis::visit(BlockPtr v) {
  // 保存当前作用域，并如果当前作用域的块不是当前处理的块，则创建新的作用域
  auto prev_scope = currentScope_;
  if (currentScope_->block() != v) {
    currentScope_ = std::make_shared<Scope>(v, prev_scope);
  }

  // 将 Block 节点压入语句栈
  stmtStack_.push_front(v);

  // 遍历 Block 中的每个语句，并处理
  for (const auto& s : *v) {
    s->accept(this);
    // 如果当前作用域的块不是当前处理的块，则将内部块的访问合并到这个 Block 的访问中
    if (currentScope_->block() != v) {
      mergeCurrentScopeIntoParent();
    }
  }
}
    }
  }



  stmtStack_.pop_front();



  // 如果前一个作用域没有块（block为空），则执行以下操作
  if (prev_scope->block() == nullptr) {
    // 关闭任何打开的候选项。
    for (auto& p1 : currentScope_->openAccesses()) {
      for (auto& p2 : p1.second) {
        // 将p2.second指定的访问关闭到当前作用域。
        closeAccessIntoScope(p2.second, currentScope_);
      }
    }
  }
}

void RegisterizerAnalysis::visit(StorePtr v) {
  // 将当前的 Store 指针放入语句堆栈的最前面
  stmtStack_.push_front(v);
  // 访问 Store 的值部分
  v->value()->accept(this);
  // 弹出语句堆栈中的最前面的元素（即当前的 Store 指针）
  stmtStack_.pop_front();

  // 如果 Store 操作没有索引，表明已经是标量，直接返回
  if (v->indices().empty()) {
    // 已经是标量
    return;
  }

  // 对 Store 进行哈希处理
  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (const auto& i : v->indices()) {
    // 将索引也加入哈希计算中
    accessHash = hasher_.hash_combine(accessHash, i);
  }

  // 获取当前作用域中缓冲区访问映射
  auto& bufAccesses = currentScope_->getAccessMapByBuf(v->buf());
  // 查找是否已经存在相同访问哈希的访问信息
  auto candidateIt = bufAccesses.find(accessHash);

  // 如果找到相同访问哈希的访问信息，则将当前 Store 添加到该访问信息中
  if (candidateIt != bufAccesses.end()) {
    candidateIt->second->addStore(v, currentScope_);
    return;
  }

  // 否则创建一个新的 AccessInfo，并添加当前的 Store
  auto info = std::make_shared<AccessInfo>(
      accessHash, v->buf(), v->indices(), accessOrder_++);
  info->addStore(v, currentScope_);

  // 新的访问可能与现有的打开访问重叠，此时需要关闭较旧的那个
  bool alreadyOverlapped = false;
  for (auto it = bufAccesses.begin(); it != bufAccesses.end();) {
    auto other = it->second;
    if (info->overlaps(other)) {
      if (other->last_usage() == v) {
        // 如果当前的访问已经被右侧的访问重叠覆盖
        alreadyOverlapped = true;
      }
      // 关闭较旧的访问信息，并从映射中移除
      closeAccessIntoScope(other, currentScope_);
      it = bufAccesses.erase(it);
    } else {
      ++it;
    }
  }

  // 如果当前的访问已经被右侧的访问重叠覆盖，则将当前访问信息关闭
  if (alreadyOverlapped) {
    closeAccessIntoScope(info, currentScope_);
  } else {
    // 否则将新的访问信息加入映射中
    bufAccesses.emplace(accessHash, info);
  }
}

void RegisterizerAnalysis::visit(LoadPtr v) {
  // 如果 Load 操作没有索引，表明已经是标量，直接返回
  if (v->indices().empty()) {
    // 已经是标量
    return;
  }
  // 对 Load 进行哈希处理
  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (const auto& i : v->indices()) {
    // 将索引也加入哈希计算中
    accessHash = hasher_.hash_combine(accessHash, i);
  }

  // 获取当前作用域中缓冲区访问映射
  auto& bufAccesses = currentScope_->getAccessMapByBuf(v->buf());
  // 查找是否已经存在相同访问哈希的访问信息
  auto candidateIt = bufAccesses.find(accessHash);
  // 如果找到相同访问哈希的访问信息，则将当前 Load 添加到该访问信息中
  if (candidateIt != bufAccesses.end()) {
    candidateIt->second->addLoad(v, currentScope_, stmtStack_.front());
    return;
  }

  // 否则创建一个新的 AccessInfo，并添加当前的 Load
  std::shared_ptr<AccessInfo> info = std::make_shared<AccessInfo>(
      accessHash, v->buf(), v->indices(), accessOrder_++);
  info->addLoad(v, currentScope_, stmtStack_.front());

  // 新的访问可能与现有的打开访问重叠，此时需要关闭较旧的那个
  bool alreadyOverlapped = false;
  for (auto it = bufAccesses.begin(); it != bufAccesses.end();) {
    auto other = it->second;
    if (info->overlaps(other)) {
      if (info->last_usage() == other->last_usage()) {
        // 如果这两个访问来自同一个语句，则它们已经重叠
        alreadyOverlapped = true;
      }
      // 关闭较旧的访问信息，并从映射中移除
      closeAccessIntoScope(other, currentScope_);
      it = bufAccesses.erase(it);
    } else {
      ++it;
    }
  }

  // 如果这两个访问来自同一个语句，则将当前访问信息关闭
  if (alreadyOverlapped) {
    closeAccessIntoScope(info, currentScope_);
  } else {
    bufAccesses.emplace(accessHash, info);
  }



// 如果条件成立，则调用 closeAccessIntoScope 函数，将 info 对象和当前作用域 currentScope_ 关联起来
closeAccessIntoScope(info, currentScope_);
// 如果条件不成立，则使用 accessHash 和 info 对象构建一个新的元素，并插入到 bufAccesses 容器中
} else {
    bufAccesses.emplace(accessHash, info);
}
// 合并隐藏作用域中的访问信息，根据 allowClosed 标志决定是否保留已关闭的访问信息
void RegisterizerAnalysis::mergeHiddenScope(bool allowClosed) {
  // 新建一个列表，用于存放新关闭的访问信息
  std::list<std::shared_ptr<AccessInfo>> newClosed;
  
  // 遍历当前作用域中的已关闭访问信息列表
  for (auto& info : currentScope_->closedAccesses()) {
    // 获取当前作用域中与当前访问信息关联的所有访问信息候选项
    auto& candidates = currentScope_->getAccessMapByBuf(info->buf());
    
    // 遍历候选项列表
    for (auto it = candidates.begin(); it != candidates.end();) {
      std::shared_ptr<AccessInfo> candidate = it->second;

      // 如果两个访问信息具有相同的哈希值或者有重叠部分，则将候选项加入新关闭列表，并从候选项列表中移除
      if (info->hash() == candidate->hash() || info->overlaps(candidate)) {
        newClosed.push_back(candidate);
        it = candidates.erase(it);
      } else {
        ++it;
      }
    }
  }

  // 根据 allowClosed 标志处理新关闭的访问信息列表
  if (allowClosed) {
    // 将新关闭的访问信息添加到当前作用域中
    for (auto& info : newClosed) {
      closeAccessIntoScope(info, currentScope_);
    }
  } else {
    // 清空当前作用域中的关闭访问信息列表
    currentScope_->closedAccesses().clear();
  }
}

// 将当前作用域 currentScope_ 合并到其父作用域中，并将父作用域设为新的 currentScope_
void RegisterizerAnalysis::mergeCurrentScopeIntoParent() {
  auto parent = currentScope_->parent();

  // 复制当前作用域中的已关闭访问信息到父作用域，并根据需要合并或关闭
  for (auto& candidate : currentScope_->closedAccesses()) {
    // 获取父作用域中与候选访问信息关联的访问信息列表
    auto& parentAccesses = parent->getAccessMapByBuf(candidate->buf());

    // 在父作用域中查找是否存在与候选访问信息哈希匹配的访问信息
    auto parentIt = parentAccesses.find(candidate->hash());
    if (parentIt != parentAccesses.end()) {
      std::shared_ptr<AccessInfo> pCandidate = parentIt->second;

      // 如果访问信息在条件中被关闭，只有在父作用域中的条件相同的情况下才能合并
      if (candidate->conditionId() &&
          pCandidate->conditionId() != candidate->conditionId()) {
        // 父作用域中的访问信息必须被关闭
        closeAccessIntoScope(pCandidate, parent);
        parentAccesses.erase(parentIt);

        // 将当前作用域中的访问信息插入父作用域中
        closeAccessIntoScope(candidate, parent);
        continue;
      }

      // 合并完全重叠的访问信息
      parentIt->second->merge(candidate);
      closeAccessIntoScope(parentIt->second, parent);
      parentAccesses.erase(parentIt);
      continue;
    }

    // 没有找到完全匹配的访问信息，但需要检查该缓冲区的所有开放访问信息是否部分重叠
    // 遍历父作用域中的所有访问信息
    for (auto it = parentAccesses.begin(); it != parentAccesses.end();) {
      // 获取当前父作用域中的一个访问信息候选项
      std::shared_ptr<AccessInfo> pCandidate = it->second;
      // 如果当前访问信息与候选项存在部分重叠，则关闭父作用域中的访问信息
      if (candidate->overlaps(pCandidate)) {
        closeAccessIntoScope(pCandidate, parent);
        // 从父作用域的访问信息中移除该项
        it = parentAccesses.erase(it);
        continue;
      }
      ++it;
    }

    // 将子作用域中的已关闭访问信息插入到父作用域中
    closeAccessIntoScope(candidate, parent);
  }

  // 复制当前打开访问信息，必要时进行合并
  // 对于每个具有打开访问的缓冲区：
  for (auto& pair : currentScope_->openAccesses()) {
    BufPtr buf = pair.first;
    // 如果该缓冲区的打开访问信息为空，则继续下一个缓冲区
    if (pair.second.empty()) {
      continue;
    }

    auto& parentAccesses = parent->getAccessMapByBuf(buf);

    // 对于该缓冲区在子作用域中的每个打开访问信息：
    for (auto& hpair : pair.second) {
      bool handled{false};
      std::shared_ptr<AccessInfo> candidate = hpair.second;

      // 遍历父作用域中的每个访问信息候选项
      for (auto it = parentAccesses.begin(); it != parentAccesses.end();) {
        std::shared_ptr<AccessInfo> pCandidate = it->second;

        // 如果完全重叠则进行合并
        if (candidate->hash() == pCandidate->hash()) {
          // 如果两个访问信息都在条件块中找到，则不能合并，但必须关闭较早的访问信息
          if (pCandidate->conditionId() != parent->conditionId() &&
              pCandidate->conditionId() != candidate->conditionId()) {
            closeAccessIntoScope(pCandidate, parent);
            // 从父作用域的访问信息中移除该项
            it = parentAccesses.erase(it);
            continue;
          }
          pCandidate->merge(candidate);
          handled = true;
          ++it;
          continue;
        }

        // 如果与父作用域中的访问信息存在重叠，则关闭父作用域中的访问信息
        // 子作用域中的访问信息仍可能保持打开状态
        if (candidate->overlaps(pCandidate)) {
          closeAccessIntoScope(pCandidate, parent);
          // 从父作用域的访问信息中移除该项
          it = parentAccesses.erase(it);
          continue;
        }

        ++it;
      }

      // 如果此访问信息依赖于本地作用域的变量，则不能将其提升出循环
      for (const auto& v : currentScope_->localVars()) {
        if (candidate->dependsOnVar(v)) {
          closeAccessIntoScope(candidate, parent);
          handled = true;
          break;
        }
      }

      if (!handled) {
        // 如果内部作用域不是条件性的，但外部作用域是：则所有当前访问信息现在在父作用域中都是条件性的
        if (candidate->conditionId() == 0) {
          candidate->setConditionId(parent->conditionId());
        }
        // 将访问信息候选项添加到父作用域的访问信息中
        parentAccesses[candidate->hash()] = candidate;
      }
    }
  }

  // 将当前作用域指向父作用域
  currentScope_ = parent;
}

// getCandidates() 函数：获取当前作用域内的候选项
std::vector<std::shared_ptr<AccessInfo>> RegisterizerAnalysis::getCandidates() {
  // 过滤已关闭的访问项
  currentScope_->filterClosed();
  // 对已关闭的访问项按照访问顺序排序
  std::sort(
      currentScope_->closedAccesses().begin(),
      currentScope_->closedAccesses().end(),
      [](auto i1, auto i2) { return i1->accessOrder() < i2->accessOrder(); });
  // 返回已关闭的访问项列表
  return currentScope_->closedAccesses();
}

// mutate(LoadPtr v) 函数：变异加载指针
ExprPtr RegisterizerReplacer::mutate(LoadPtr v) {
  // 查找加载指针在映射中的信息
  auto it = loadToAccess_.find(v);
  // 如果加载指针不在映射中，则无法寄存，直接返回原加载指针
  if (it == loadToAccess_.end()) {
    // 此访问无法进行寄存化
    return v;
  }

  auto& info = it->second;

  // 返回替换后的变量
  return info->replacement().var;
}

// mutate(StorePtr v) 函数：变异存储指针
StmtPtr RegisterizerReplacer::mutate(StorePtr v) {
  // 如果该存储指针是标量变量的初始化器，并且已经插入，则返回空指针
  if (eliminatedIntializers_.count(v) != 0) {
    // 此存储是已插入的标量变量的初始化器
    return nullptr;
  }

  // 查找存储指针在映射中的信息
  auto it = storeToAccess_.find(v);
  // 如果存储指针不在映射中，则无法寄存，调用父类的变异器进行处理
  if (it == storeToAccess_.end()) {
    // 此访问无法进行寄存化
    return IRMutator::mutate(v);
  }

  auto& info = it->second;

  // 对存储的值进行变异处理
  ExprPtr new_val = v->value()->accept_mutator(this);

  // 更新存储指针的值和缓冲区
  v->set_value(new_val);
  v->set_buf(info->replacement().var_wrapper);
  v->set_indices({});
  // 返回变异后的存储指针
  return v;
}

// mutate(BlockPtr v) 函数：变异块指针
StmtPtr RegisterizerReplacer::mutate(BlockPtr v) {
  // 获取父类到访问项的映射
  auto& scope = parentToAccesses_[v];

  // 新建语句列表
  std::vector<StmtPtr> stmts;
  // 遍历块中的每条语句
  for (const StmtPtr& stmt : v->stmts()) {
    {
      // 插入与该块作用域内的标量相关的初始化器
      auto it = scope.initializerPoints_.find(stmt);
      if (it != scope.initializerPoints_.end()) {
        for (auto& info : it->second) {
          // 变异处理初始化器并添加到语句列表中
          StmtPtr initializer =
              info->replacement().initializer->accept_mutator(this);
          stmts.push_back(initializer);
        }
        // 删除已处理的初始化器
        scope.initializerPoints_.erase(it);
      }
    }

    // 变异处理当前语句并添加到语句列表中
    StmtPtr stmt_new = stmt->accept_mutator(this);
    if (stmt_new) {
      if (stmt_new->get_parent()) {
        stmt_new = Stmt::clone(stmt_new);
      }
      stmts.push_back(stmt_new);
    }

    {
      // 插入与该块作用域内的标量相关的终结器
      auto it = scope.finalizePoints_.find(stmt);
      if (it != scope.finalizePoints_.end()) {
        for (auto& info : it->second) {
          // 创建并添加终结器到语句列表中
          StorePtr finalizer = alloc<Store>(
              info->buf(), info->indices(), info->replacement().var);
          stmts.push_back(finalizer);
        }
        // 删除已处理的终结器
        scope.finalizePoints_.erase(it);
      }
    }
  }

  // 返回包含变异后语句的块指针
  return alloc<Block>(stmts);
}

// buildReplacements() 函数：构建寄存器替换
void RegisterizerReplacer::buildReplacements() {
  // 遍历信息集合，创建变量并更新本地映射
  for (auto& info : infoSet_) {
    VarPtr v = alloc<Var>(
        info->buf()->name_hint() + "_" +
            std::to_string(getBufferAccessCount(info->buf())),
        info->buf()->dtype());

    info->replacement().var = v;

    // 需要将变量包装在缓冲区中，以便可以加载或存储它
    info->replacement().var_wrapper =
        alloc<Buf>(v, std::vector<ExprPtr>({}), info->buf()->dtype());

    bool first = true;
    // 遍历info对象的存储操作列表
    for (const auto& s : info->stores()) {
      // 如果是第一个存储操作，并且该存储操作是info的第一次使用且未重叠
      if (first && info->first_usage() == s && !info->firstUsageOverlapped()) {
        // 为替换操作设置初始化器，使用s的值作为参数
        info->replacement().initializer = alloc<Let>(v, s->value());
        // 将s插入到已消除初始化器的集合中
        eliminatedIntializers_.insert(s);
      } else {
        // 否则将s映射到info对象，存储到storeToAccess_映射中
        storeToAccess_[s] = info;
      }

      // 将first标记设置为false，表示第一个条件已经处理过
      first = false;
    }

    // 遍历info对象的加载操作列表
    for (const auto& s : info->loads()) {
      // 将info映射到加载操作s，存储到loadToAccess_映射中
      loadToAccess_[s] = info;
    }

    // 获取info对象所在块的作用域引用
    auto& scope = parentToAccesses_[info->block()];
    // 将info对象的第一次使用点映射到info的集合，存储到scope.initializerPoints_映射中
    scope.initializerPoints_[info->first_usage()].push_back(info);

    // 只有当标量有写操作时才进行最终化
    if (!info->stores().empty()) {
      // 将info对象的最后使用点的最终化点推入到scope.finalizePoints_映射的最前端，以反向顺序完成
      scope.finalizePoints_[info->last_usage()].push_front(info);
    }

    // 如果info对象的替换操作的初始化器为空指针
    if (info->replacement().initializer == nullptr) {
      // 创建一个默认的初始化器，通过读取访问获取的内容
      info->replacement().initializer = alloc<Let>(
          v, alloc<Load>(info->buf()->dtype(), info->buf(), info->indices()));
    }
}
} // 结束命名空间 registerizer

} // 结束命名空间

// 对给定的语句 s 应用标量替换
StmtPtr registerize(StmtPtr s) {
    // 简化语句 s
    s = IRSimplifier::simplify(s);

    // 如果 s 不是 Block 类型，则创建一个只包含 s 的 Block，用于放置外部作用域的标量
    if (!to<Block>(s)) {
        s = alloc<Block>(std::vector<StmtPtr>({s}));
    }

    // 创建 RegisterizerAnalysis 实例进行分析
    registerizer::RegisterizerAnalysis analysis;
    s->accept(&analysis);

    // 获取分析得到的候选标量
    auto candidates = analysis.getCandidates();

    // 创建 RegisterizerReplacer 实例，用分析得到的候选标量进行替换
    registerizer::RegisterizerReplacer replacer(candidates);
    s = s->accept_mutator(&replacer);

    // 返回替换后的语句 s
    return s;
}

} // 结束命名空间 torch::jit::tensorexpr
```