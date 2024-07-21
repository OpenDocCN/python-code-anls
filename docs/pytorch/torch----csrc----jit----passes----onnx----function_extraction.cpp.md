# `.\pytorch\torch\csrc\jit\passes\onnx\function_extraction.cpp`

```
// 引入 Torch JIT 日志记录的头文件
#include <torch/csrc/jit/jit_log.h>
// 引入 Torch JIT Passes 的 ONNX 相关功能提取的头文件
#include <torch/csrc/jit/passes/onnx/function_extraction.h>
// 引入 Torch JIT Passes 的 ONNX 相关命名空间功能的头文件
#include <torch/csrc/jit/passes/onnx/naming.h>

// 定义在匿名命名空间中，用于内部使用的 scope_list 别名，表示作用域的列表
namespace torch {
namespace jit {
namespace onnx {

namespace {

// 使用 scope_list 作为别名，表示作用域的列表，其中每个元素为 ScopePtr
using scope_list = std::vector<ScopePtr>;

// 从模块中检查注解获取的带注解属性，存储在此映射中，用于后续的 ONNX 本地函数子图
// 这些属性在 ONNX 本地函数子图中不被使用，因为它们不是由 PyTorch JIT 跟踪创建的，
// 但消费者可能用它们来决定是否用特定的融合内核替换函数。
static std::unordered_map<ScopePtr, Node*> scope_attr_map_;

// 创建一个共享的 Graph 对象，用于存储作用域属性的图结构
static std::shared_ptr<Graph> scope_attr_graph_ = std::make_shared<Graph>();

// 检查两个节点是否具有相同的属性
static bool HasSameAttribute(
    const Node* a,
    const Node* b,
    const c10::Symbol& attr);

// 函数提取器类，用于从模块中提取函数
struct FunctionExtractor {
 public:
  FunctionExtractor(
      std::shared_ptr<Graph>& graph,
      const std::unordered_set<std::string>& module_names,
      const std::vector<std::string>& param_names)
      : graph_(graph),
        module_names_(module_names.begin(), module_names.end()),
        param_names_(param_names.begin(), param_names.end()) {}

  // 运行函数提取器，返回节点属性名称映射
  NodeAttrNameMap run();

 private:
  // 作用域上下文结构体，用于描述作用域的状态和属性
  struct ScopeContext {
    std::unordered_set<ScopePtr> children_; // 子作用域集合
    ScopePtr scope_; // 当前作用域
    node_list nlist_; // 节点列表
    value_list inputs_; // 输入值列表
    value_list outputs_; // 输出值列表
    std::unordered_map<Value*, Value*> env_to_subgraph_; // 环境到子图的映射

    // 根据参数名称填充输入和输出列表
    void PopulateInputsOutputs(
        const std::unordered_set<std::string>& param_names);

    // 判断当前作用域是否与另一个作用域上下文相同
    bool IsIdenticalFuncion(const ScopeContext& other_ctx) const;
  };

  // ScopeContext 的指针类型别名
  using ScopeCtxPtr = ScopeContext*;
  // 作用域上下文映射，将作用域指针映射到其上下文结构体指针
  using scope_ctx_map = std::unordered_map<ScopePtr, ScopeCtxPtr>;

  // 函数上下文结构体，描述函数的状态和属性
  struct FunctionContext {
    // 构造函数，初始化函数上下文
    FunctionContext(
        ScopePtr key,
        const scope_list& scopes,
        scope_ctx_map& scope_ctxs);
    // 调试打印函数上下文信息
    void DebugPrint() const;
    // 设置节点属性名称
    void SetAttrName(Node* ref_n, Symbol attr, const std::string& name);
    // 查找节点属性名称
    std::optional<std::string> FindAttrName(Node* ref_n, Symbol attr);
    // 查找常量节点属性名称
    std::optional<std::string> FindAttrName(Node* ref_const_n);

    ScopePtr scope_key_; // 函数作用域关键字
    scope_ctx_map scope_ctxs_; // 作用域上下文映射
    std::unordered_map<
        Node*,
        std::unordered_map<Symbol, std::unordered_set<Node*>>>
        attribute_map_; // 节点属性映射，存储节点属性关系

    // 传递到序列化的信息。


This completes the annotation of the provided C++ code snippet.
    NodeAttrNameMap node_attr_to_name_;



    // 定义一个映射，将节点属性名映射到名称
    NodeAttrNameMap node_attr_to_name_;
  };

  using FunctionCtxPtr = FunctionContext*;
  using func_ctx_map = std::unordered_map<ScopePtr, FunctionCtxPtr>;



    // 使用指针定义函数上下文的映射
    using FunctionCtxPtr = FunctionContext*;
    // 使用无序映射将作用域指针映射到函数上下文指针
    using func_ctx_map = std::unordered_map<ScopePtr, FunctionCtxPtr>;

    // 静态函数：检查给定的作用域是否有效
    static bool IsValidScope(ScopePtr s);
    // 静态函数：推断给定节点的作用域，返回可选的作用域指针
    static std::optional<ScopePtr> InferScope(Node* n);
    // 静态函数：判断一个作用域是否是另一个作用域的祖先
    static bool IsAncestor(ScopePtr parent, ScopePtr child);
    // 静态函数：查找两个作用域的共同祖先，返回可选的作用域指针
    static std::optional<ScopePtr> FindCommonAncestor(ScopePtr a, ScopePtr b);
    // 静态函数：查找多个作用域的共同祖先，返回可选的作用域指针
    static std::optional<ScopePtr> FindCommonAncestor(const scope_list& scopes);
    // 构建函数图的方法，接受函数上下文的引用作为参数，返回图的共享指针
    std::shared_ptr<Graph> ConstructFuncGraph(FunctionContext& ctx);

    // 将作用域转换为函数的方法
    void ConvertScopeToFunction(
        const ScopePtr& scope_key,
        const scope_list& scope_list,
        scope_ctx_map& scope_ctxs,
        const std::shared_ptr<Graph>& graph);

    // 静态函数：处理没有作用域的节点列表
    static void HandleNoScopeNodes(scope_ctx_map&, node_list no_scope_nlist);
    // 分割块中节点按作用域分组的方法，返回作用域上下文映射和节点列表的元组
    std::tuple<scope_ctx_map, node_list> PartitionNodesByScope(Block* b);
    // 分割图中节点按作用域分组的方法，返回作用域上下文映射
    scope_ctx_map PartitionNodesByScope(const std::shared_ptr<Graph>& graph);
    // 静态函数：根据作用域上下文映射分割相同作用域的方法，返回作用域到作用域列表的映射
    static std::unordered_map<ScopePtr, scope_list> PartitionIdenticalScopes(
        scope_ctx_map& scope_ctxs);
    // 静态函数：根据最大深度对作用域列表排序的方法，返回排序后的作用域列表
    static scope_list SortScopesByMaxDepth(
        std::unordered_map<ScopePtr, scope_list>&);
    // 创建函数定义节点的方法，返回创建的节点指针
    Node* CreateFunctionDefNode(
        FunctionContext& func_ctx,
        const std::shared_ptr<Graph>& graph,
        const std::string& domain_name,
        const std::string& func_name);
    // 创建函数节点的方法，返回创建的节点指针
    Node* CreateFunctionNode(
        FunctionContext& func_ctx,
        ScopeContext& scope_ctx,
        const std::shared_ptr<Graph>& graph,
        const std::string& domain_name,
        const std::string& func_name);

    // 静态函数：调试打印作用域上下文映射的方法
    static void DebugPrintScopeContexts(const scope_ctx_map&);
    // 静态函数：调试打印带有函数的图的方法
    static void DebugPrintGraphWithFunction(const std::shared_ptr<Graph>& g);
    // 静态函数：调试打印常量差异的方法
    static void DebugPrintConstantDiff(const FunctionContext&);

    // 图的共享指针
    std::shared_ptr<Graph> graph_;
    // 模块名称的无序集合
    std::unordered_set<std::string> module_names_;
    // 参数名称的无序集合
    std::unordered_set<std::string> param_names_;
    // 跟踪具有相同模块名称但作为不同 onnx 本地函数导出的模块
    std::unordered_map<std::string, int> module_variant_count_;
    // 函数上下文映射
    func_ctx_map func_ctxs_;
};

// FunctionContext 构造函数，接收作用域键、作用域列表和作用域上下文映射
FunctionExtractor::FunctionContext::FunctionContext(
    ScopePtr key,                           // 初始化作用域键
    const scope_list& scopes,                // 初始化作用域列表
    scope_ctx_map& scope_ctxs)              // 初始化作用域上下文映射
    : scope_key_(std::move(key)) {          // 初始化作用域键成员变量

  // 输出调试信息，处理给定作用域的函数上下文
  GRAPH_UPDATE(
      "Process function context for scope ",
      scope_key_->name().toDisplayString());

  // 内部断言：作用域列表不为空
  TORCH_INTERNAL_ASSERT(!scopes.empty());

  // 获取参考上下文
  const auto& ref_ctx = scope_ctxs[scope_key_];

  // DEBUG日志：为给定作用域初始化函数上下文
  GRAPH_DEBUG(
      "Initialized function context for scope ",
      scope_key_->name().toDisplayString());

  // 遍历每个作用域
  for (const auto& scope : scopes) {
    // DEBUG日志：处理作用域的函数上下文
    GRAPH_DEBUG(
        "Process function context for scope ", scope->name().toDisplayString());

    // 内部断言：确保作用域在作用域上下文映射中
    TORCH_INTERNAL_ASSERT(scope_ctxs.find(scope) != scope_ctxs.end());

    // 将作用域上下文映射到当前函数上下文对象中
    scope_ctxs_[scope] = scope_ctxs[scope];

    // 如果当前作用域与当前函数上下文的作用域键相同，则继续下一个循环
    if (scope_key_ == scope) {
      continue;
    }

    // 获取当前作用域的上下文
    auto& scope_ctx = scope_ctxs[scope];

    // 比较两个作用域的节点列表，确保节点数和顺序一致
    const auto& ns_a = ref_ctx->nlist_;
    const auto& ns_b = scope_ctx->nlist_;
    TORCH_INTERNAL_ASSERT(ns_a.size() == ns_b.size());

    // DEBUG日志：处理作用域的节点
    GRAPH_DEBUG("Process nodes of scope ", scope->name().toDisplayString());

    // 遍历作用域中的节点
    for (const auto i : c10::irange(ns_a.size())) {
      // 内部断言：确保节点在两个作用域中具有相同的类型
      TORCH_INTERNAL_ASSERT(ns_a[i]->kind() == ns_b[i]->kind());

      // 获取节点 n_a 和 n_b
      auto n_a = ns_a[i];
      auto n_b = ns_b[i];

      // 声明存储不同和相同属性名称的向量
      std::vector<c10::Symbol> diff_attrs;
      std::vector<c10::Symbol> same_attrs;

      // 获取节点 n_a 和 n_b 的属性名称列表，并排序
      auto n_a_attr_names = n_a->attributeNames();
      auto n_b_attr_names = n_b->attributeNames();
      std::sort(n_a_attr_names.begin(), n_a_attr_names.end());
      std::sort(n_b_attr_names.begin(), n_b_attr_names.end());

      // 找到属性名称列表中的差异和相同项
      std::set_difference(
          n_a_attr_names.begin(),
          n_a_attr_names.end(),
          n_b_attr_names.begin(),
          n_b_attr_names.end(),
          std::inserter(diff_attrs, diff_attrs.begin()));
      std::set_intersection(
          n_a_attr_names.begin(),
          n_a_attr_names.end(),
          n_b_attr_names.begin(),
          n_b_attr_names.end(),
          std::inserter(same_attrs, same_attrs.begin()));

      // 遍历不同的属性名称，将 n_b 添加到属性映射中
      for (auto attr_name : diff_attrs) {
        attribute_map_[n_a][attr_name].insert(n_b);
      }

      // 遍历相同的属性名称，如果属性值不同，则将 n_b 添加到属性映射中
      for (auto attr_name : same_attrs) {
        if (!HasSameAttribute(n_a, n_b, attr_name)) {
          attribute_map_[n_a][attr_name].insert(n_b);
        }
      }
    }

    // DEBUG日志：处理作用域完成
    GRAPH_DEBUG("Process scope complete. ", scope->name().toDisplayString());
  }

  // DEBUG日志：处理函数上下文完成
  GRAPH_DEBUG(
      "Process function context complete. ",
      scope_key_->name().toDisplayString());

  // 调用调试打印函数
  DebugPrint();
}

// 函数Context的调试打印方法
void FunctionExtractor::FunctionContext::DebugPrint() const {
  // DEBUG日志：打印作用域的名称
  GRAPH_DEBUG("Scope name: ", scope_key_->name().toDisplayString());

  // 遍历属性映射，打印属性名称和不同属性值的节点
  for (const auto& it : attribute_map_) {
    for (const auto& attr_it : it.second) {
      GRAPH_DEBUG(
          "Attribute value difference for attribute ",
          attr_it.first.toDisplayString());
      GRAPH_DEBUG(*it.first);
      for (auto n : attr_it.second) {
        GRAPH_DEBUG(*n);
      }
    }
  }
}
// 设置节点属性名称，将给定节点的输出作为键查找子图环境，并将属性名称映射到指定名称
void FunctionExtractor::FunctionContext::SetAttrName(
    Node* ref_n,  // 输入的节点指针
    Symbol attr,  // 属性符号
    const std::string& name) {  // 属性名称字符串引用
  // 查找节点输出的第一个元素，以此作为键在环境到子图的映射中查找
  auto v_it = scope_ctxs_[scope_key_]->env_to_subgraph_.find(ref_n->outputs().at(0));
  TORCH_INTERNAL_ASSERT(v_it != scope_ctxs_[scope_key_]->env_to_subgraph_.end());  // 断言确保查找到结果
  auto* n_in_def = v_it->second->node();  // 获取找到的子图环境中的节点
  // 将属性名称映射到节点属性名称映射中
  auto n_attr_it = node_attr_to_name_[n_in_def][attr.toUnqualString()] = name;
}

// 查找节点属性名称，返回属性名称的可选项字符串
std::optional<std::string> FunctionExtractor::FunctionContext::FindAttrName(
    Node* ref_n,  // 输入的节点指针
    Symbol attr) {  // 属性符号
  // 查找节点输出的第一个元素，以此作为键在环境到子图的映射中查找
  auto v_it = scope_ctxs_[scope_key_]->env_to_subgraph_.find(ref_n->outputs().at(0));
  if (v_it == scope_ctxs_[scope_key_]->env_to_subgraph_.end()) {
    return c10::nullopt;  // 如果未找到，返回空的可选项
  }
  auto* n_in_def = v_it->second->node();  // 获取找到的子图环境中的节点
  auto n_attr_it = node_attr_to_name_.find(n_in_def);
  if (n_attr_it == node_attr_to_name_.end()) {
    return c10::nullopt;  // 如果未找到节点属性名称映射，返回空的可选项
  }
  auto name_it = n_attr_it->second.find(attr.toUnqualString());
  if (name_it == n_attr_it->second.end()) {
    return c10::nullopt;  // 如果未找到属性名称映射，返回空的可选项
  }
  return name_it->second;  // 返回找到的属性名称字符串
}

// 调试打印作用域上下文信息，输出作用域名称、子作用域名称和节点类型列表等
void FunctionExtractor::DebugPrintScopeContexts(
    const scope_ctx_map& scope_ctxs) {  // 作用域上下文映射引用
  for (auto& it : scope_ctxs) {
    GRAPH_UPDATE(
        "Scope name: ",
        it.first->namesFromRoot(),  // 输出作用域名称的完整路径
        " ",
        it.first->name().toDisplayString());  // 输出作用域名称的显示字符串
    GRAPH_UPDATE("Children scopes: ", [&]() {
      std::stringstream ss;
      for (const auto& child_scope : it.second->children_) {
        ss << child_scope->name().toDisplayString() << " ";  // 输出子作用域的显示字符串列表
      }
      return ss.str();
    }());
    GRAPH_UPDATE("Node types: \n", [&]() {
      std::stringstream ss;
      for (auto n : it.second->nlist_) {
        ss << "  " << *n;  // 输出作用域中节点类型的字符串表示
      }
      return ss.str();
    }());
    GRAPH_UPDATE("Node count: ", it.second->nlist_.size());  // 输出作用域中节点数量
  }
}

// 调试打印带有函数的图信息，输出本地函数定义和主图信息
void FunctionExtractor::DebugPrintGraphWithFunction(
    const std::shared_ptr<Graph>& g) {  // 共享指针指向图的引用
  GRAPH_UPDATE("Local function definitions:");  // 输出本地函数定义标题
  for (auto* n : g->nodes()) {
    if (n->kind() == Symbol::onnx("LocalFunctionDef")) {
      GRAPH_UPDATE(
          n->s(attr::name),  // 输出本地函数定义的名称
          " graph: ",
          n->g(Symbol::attr("graph"))->toString());  // 输出本地函数定义关联的图的字符串表示
    }
  }
  GRAPH_UPDATE("Main graph: ", g->toString());  // 输出主图的字符串表示
}

// 检查作用域是否有效，即不是根作用域且不是空白作用域
bool FunctionExtractor::IsValidScope(ScopePtr s) {  // 作用域指针
  return !s->isRoot() && !s->isBlank();
}

// 检查一个作用域是否是另一个作用域的祖先
bool FunctionExtractor::IsAncestor(ScopePtr parent, ScopePtr child) {  // 父作用域指针和子作用域指针
  if (!IsValidScope(parent) || !IsValidScope(child) ||
      parent->getDepth() >= child->getDepth()) {
    return false;  // 如果父或子作用域无效或者父作用域深度大于等于子作用域深度，则返回假
  }
  do {
    child = child->parent();  // 逐级向上检查子作用域的父作用域
    if (parent == child) {
      return true;  // 如果找到共同祖先，返回真
    }
  } while (IsValidScope(child));
  return false;  // 否则返回假
}

// 查找两个作用域的共同祖先
std::optional<ScopePtr> FunctionExtractor::FindCommonAncestor(
    ScopePtr a,  // 第一个作用域指针
    ScopePtr b) {  // 第二个作用域指针
  if (!IsValidScope(a) || !IsValidScope(b)) {
    return c10::nullopt;  // 如果任意一个作用域无效，返回空的可选项
  }

  auto diff =
      static_cast<int64_t>(a->getDepth()) - static_cast<int64_t>(b->getDepth());
  if (diff != 0) {
    auto deeper_scope = diff > 0 ? a : b;  // 选择深度更深的作用域
    while (diff != 0) {
      deeper_scope = deeper_scope->parent();  // 向上移动到相同深度
      diff = static_cast<int64_t>(deeper_scope->getDepth()) -
             static_cast<int64_t>(b->getDepth());
    }
    if (a == b) {
      return a;  // 如果两个作用域相同，返回其中任何一个
    }
    while (a != b) {
      a = a->parent();  // 逐级向上检查两个作用域的父作用域
      b = b->parent();
    }
  }
  return a;  // 返回找到的共同祖先
}
    // 根据 diff 的正负选择较远的作用域作为 other_scope
    auto other_scope = diff > 0 ? b : a;
    // 取 diff 的绝对值
    diff = std::abs(diff);
    // 向上遍历 deeper_scope 直到 diff 为 0
    while (diff > 0) {
      deeper_scope = deeper_scope->parent();
      diff--;
    }
    // 更新 a 和 b 的作用域
    a = deeper_scope;
    b = other_scope;
  }

  // 在有效作用域内，比较 a 和 b 的父作用域，直到找到共同的作用域或者有一个无效
  while (IsValidScope(a) && IsValidScope(b)) {
    if (a == b) {
      // 如果找到共同的作用域，返回该作用域
      return a;
    } else {
      // 否则继续向上遍历作用域链
      a = a->parent();
      b = b->parent();
    }
  }

  // 如果没有找到共同的作用域，返回空值
  return c10::nullopt;
}

std::optional<ScopePtr> FunctionExtractor::FindCommonAncestor(
    const scope_list& scopes) {
  // 如果 scopes 为空，返回空的 optional 对象
  if (scopes.empty()) {
    return c10::nullopt;
  }

  // 初始化 common_ancestor 为 scopes 中的第一个元素
  std::optional<ScopePtr> common_ancestor = scopes.at(0);
  // 遍历 scopes 中的每个 scope
  for (const auto& scope : scopes) {
    // 更新 common_ancestor 为当前 scope 和 common_ancestor 的共同祖先
    common_ancestor = FindCommonAncestor(common_ancestor.value(), scope);
    // 如果找不到共同祖先，返回空的 optional 对象
    if (!common_ancestor.has_value()) {
      return c10::nullopt;
    }
  }

  // 返回最终的 common_ancestor
  return common_ancestor;
}

std::optional<ScopePtr> FunctionExtractor::InferScope(Node* n) {
  // 节点 n 的作用域基于以下规则分配：
  // 1. 如果所有输出的使用都属于同一个作用域，则分配该作用域；否则
  // 2. 如果所有输入的节点都属于同一个作用域，则分配该作用域；否则
  // 3. 找到输出的使用的作用域的共同祖先，以及输入的节点的作用域的共同祖先。
  
  // 存储输入节点的作用域列表
  scope_list input_scopes;
  // 存储输出节点的作用域列表
  scope_list output_scopes;
  
  // 遍历节点 n 的每个输入
  for (auto input : n->inputs()) {
    // 将输入节点的作用域加入 input_scopes
    input_scopes.emplace_back(input->node()->scope());
  }
  
  // 遍历节点 n 的每个输出
  for (auto output : n->outputs()) {
    // 遍历输出节点的每个使用
    for (auto use : output->uses()) {
      // 如果使用节点的作用域无效，推断输出节点的作用域
      if (!IsValidScope(use.user->scope())) {
        auto inferred_output_scope = InferScope(use.user);
        // 如果推断得到有效的作用域，设置使用节点的作用域
        if (inferred_output_scope.has_value() &&
            IsValidScope(inferred_output_scope.value())) {
          use.user->setScope(inferred_output_scope.value());
        }
      }
      // 将使用节点的作用域加入 output_scopes
      output_scopes.emplace_back(use.user->scope());
    }
  }
  
  // 如果输出作用域列表不为空且所有作用域都相同，返回第一个作用域
  if (!output_scopes.empty() &&
      std::all_of(
          output_scopes.begin(),
          output_scopes.end(),
          [&output_scopes](ScopePtr scope) -> bool {
            return IsValidScope(scope) && scope == output_scopes.at(0);
          })) {
    return output_scopes.at(0);
  } else if (
      // 如果输入作用域列表不为空且所有作用域都相同，返回第一个作用域
      !input_scopes.empty() &&
      std::all_of(
          input_scopes.begin(),
          input_scopes.end(),
          [&input_scopes](ScopePtr scope) -> bool {
            return IsValidScope(scope) && scope == input_scopes.at(0);
          })) {
    return input_scopes.at(0);
  } else {
    // 复制有效作用域到 scopes
    scope_list scopes;
    std::copy_if(
        input_scopes.begin(),
        input_scopes.end(),
        std::back_inserter(scopes),
        IsValidScope);
    std::copy_if(
        output_scopes.begin(),
        output_scopes.end(),
        std::back_inserter(scopes),
        IsValidScope);
    // 如果 scopes 不为空，找到它们的共同祖先
    if (!scopes.empty()) {
      auto common_ancestor = FindCommonAncestor(scopes);
      // 如果找到有效的共同祖先，返回它
      if (common_ancestor.has_value() &&
          IsValidScope(common_ancestor.value())) {
        return common_ancestor.value();
      }
    }
  }

  // 如果没有找到适合的作用域，返回空的 optional 对象
  return c10::nullopt;
}
  FunctionContext& func_ctx) {
  // 获取当前函数上下文
  auto& ctx = *func_ctx.scope_ctxs_[func_ctx.scope_key_];
  // 获取当前上下文中的节点列表
  const auto& nlist = ctx.nlist_;
  // 获取当前上下文中的作用域
  const auto& scope = ctx.scope_;
  // 获取当前上下文中的环境映射
  auto& env = ctx.env_to_subgraph_;

  // 创建一个新的图对象
  auto g = std::make_shared<Graph>();
  // 输出调试信息，显示正在构建的图的作用域路径
  GRAPH_DEBUG("Constructing graph for ", scope->namesFromRoot());

  // TODO: Update input names of function to match those in Module source code
  // signature.
  // This requires mapping between function node inputs and Module inputs.
  // Due to the lack of such mapping, currently debugName is used as input
  // names.
  // 更新函数的输入名称，以匹配模块源代码中的输入名称，目前使用 debugName 作为输入名称
  ctx.PopulateInputsOutputs(param_names_);
  // 将函数的输入添加到图中，并复制元数据
  for (auto* v : ctx.inputs_) {
    env[v] = g->addInput()->copyMetadata(v);
    // 输出调试信息，显示添加的输入值及其来源
    GRAPH_DEBUG(
        "Add input value ",
        env[v]->debugName(),
        " for outer scope value ",
        v->debugName(),
        " from ",
        *v->node());
  }

  // 对于每个节点 nlist 中的节点 n，创建其克隆并添加到图中
  for (auto* n : nlist) {
    auto clone_n = g->createClone(n, [&](Value* v) {
      // 确保环境中存在该值的映射关系
      TORCH_INTERNAL_ASSERT(env.find(v) != env.end());
      return env[v];
    });
    // 将克隆节点的输出与原始节点的输出进行映射
    for (const auto i : c10::irange(clone_n->outputs().size())) {
      env[n->output(i)] = clone_n->output(i);
    }
    // 将克隆节点插入到图中
    g->insertNode(clone_n);
  }

  // 如果值在图外部被使用，则将其设置为图的输出
  for (auto* v : ctx.outputs_) {
    // 确保环境中存在该值的映射关系
    TORCH_INTERNAL_ASSERT(env.find(v) != env.end());
    g->registerOutput(env[v]);
  }

  // 输出图的字符串表示形式的调试信息
  GRAPH_DEBUG(g->toString());
  // 返回构建的图对象
  return g;
}
  // 结束 CreateFunctionDefNode 方法的实现
}

Node* FunctionExtractor::CreateFunctionDefNode(
    FunctionContext& func_ctx,                     // 函数上下文对象的引用
    const std::shared_ptr<Graph>& graph,           // 共享指针，指向图对象
    const std::string& domain_name,                // 函数域名
    const std::string& func_name) {                // 函数名称
  const auto func_def_nk = Symbol::onnx("LocalFunctionDef");  // 定义本地函数定义节点的符号
  const auto func_g_attr = Symbol::attr("graph");               // 函数定义节点的图属性符号
  const auto func_name_attr = attr::name;                       // 函数名称属性
  const auto func_domain_attr = Symbol::attr("domain");         // 函数域属性

  auto func_graph = ConstructFuncGraph(func_ctx);  // 构建函数图

  // 创建并插入本地函数定义节点
  auto func_def_n = graph->create(func_def_nk, 0);
  func_def_n->g_(func_g_attr, func_graph);         // 设置图属性
  func_def_n->s_(func_name_attr, func_name);       // 设置函数名称属性
  func_def_n->s_(func_domain_attr, domain_name);   // 设置函数域属性
  graph->prependNode(func_def_n);                  // 将节点前置到图中

  // 设置不同值的常量和属性作为函数属性。
  std::unordered_map<std::string, int> base_attr_name_count;  // 存储基本属性名称计数
  std::vector<std::string> final_attr_names;                   // 存储最终的属性名称列表

  auto adjust_attr_name = [&](std::string attr_name) {         // 调整属性名称的lambda函数
    if (base_attr_name_count.find(attr_name) != base_attr_name_count.end()) {
      attr_name = attr_name + "." + std::to_string(base_attr_name_count[attr_name]++);
    } else {
      base_attr_name_count[attr_name] = 1;
    }
    return attr_name;
  };

  // 遍历函数上下文中的属性映射
  for (const auto& n_it : func_ctx.attribute_map_) {
    auto* n = n_it.first;  // 获取节点指针
    for (const auto& attr_it : n_it.second) {
      const auto& attr = attr_it.first;  // 获取属性
      // 添加前缀 "inferred::" 到推断属性的名称，以区分于从Python模块注释中提取的已注释属性。
      auto attr_name = "inferred::" + std::string(n->kind().toUnqualString()) +
          '_' + attr.toUnqualString();
      auto final_attr_name = adjust_attr_name(attr_name);  // 调整最终的属性名称
      final_attr_names.emplace_back(final_attr_name);      // 将最终属性名称添加到列表中
      func_ctx.SetAttrName(n, attr, final_attr_name);      // 设置函数上下文中的属性名称
    }
  }

  // 设置已注释的属性
  std::unordered_set<Symbol> annotated_attr_names;  // 存储已注释的属性名称集合
  bool first_iteration = true;                      // 第一次迭代标志
  for (const auto& it : func_ctx.scope_ctxs_) {
    auto scope = it.first;                          // 获取作用域
    auto annotated_attr_node = scope_attr_map_.find(scope);  // 查找作用域属性映射
    // 检查 annotated_attr_node 是否在 scope_attr_map_ 中存在
    if (annotated_attr_node != scope_attr_map_.end()) {
      // 获取 annotated_attr_node 中的属性名列表
      auto names = annotated_attr_node->second->attributeNames();
      // 如果是第一次迭代
      if (first_iteration) {
        // 将 names 列表中的所有元素复制到 annotated_attr_names 中
        std::copy(
            names.begin(),
            names.end(),
            std::inserter(annotated_attr_names, annotated_attr_names.end()));
        // 标记已经不是第一次迭代
        first_iteration = false;
      } else {
        // 查找 names 中第一个未见过的属性名
        auto unseen_attr_name = std::find_if(
            names.begin(),
            names.end(),
            [&annotated_attr_names](const Symbol& name) {
              // 判断 annotated_attr_names 中是否包含该属性名
              return annotated_attr_names.find(name) ==
                  annotated_attr_names.end();
            });
        // 检查是否找到未见过的属性名，如果找到，则抛出错误信息
        TORCH_CHECK(
            unseen_attr_name == names.end(),
            "Found outstanding annotated attribute ",
            *unseen_attr_name,
            " from module ",
            scope->name(),
            ". Please ensure module instances of the same class have the same set of annotated attributes.");
      }
    }
  }
  // 将 annotated_attr_names 中的每个属性名转换为非限定字符串形式，并添加到 final_attr_names 中
  for (auto attr_name : annotated_attr_names) {
    final_attr_names.emplace_back(attr_name.toUnqualString());
  }

  // 在 func_def_n 的属性 "attributes" 中设置 final_attr_names
  func_def_n->ss_(Symbol::attr("attributes"), final_attr_names);

  // 返回 func_def_n，该函数定义节点现在包含了正确的属性信息
  return func_def_n;
// 创建一个新的函数节点，用于将指定作用域中的代码转换为图中的函数
Node* FunctionExtractor::CreateFunctionNode(
    FunctionContext& func_ctx,               // 函数上下文对象，包含函数相关信息
    ScopeContext& scope_ctx,                 // 作用域上下文对象，包含作用域相关信息
    const std::shared_ptr<Graph>& graph,     // 共享指针，指向图对象，用于创建节点
    const std::string& domain_name,          // 函数所在的域名
    const std::string& func_name) {          // 函数名

  // 获取函数的作用域并打印调试信息
  const auto& func_scope = func_ctx.scope_key_;
  GRAPH_DEBUG(
      "Create and insert local function for scope: ",
      func_scope->namesFromRoot());

  // 填充作用域的输入输出信息
  scope_ctx.PopulateInputsOutputs(param_names_);

  // 获取作用域中最后一个节点
  auto last_n = *scope_ctx.nlist_.rbegin();

  // 创建新的函数节点，并复制元数据
  auto func_n = graph->create(
      Symbol::fromQualString(domain_name + "::" + func_name),
      scope_ctx.outputs_.size());
  func_n->copyMetadata(last_n);

  // 将作用域的输入节点添加为新函数节点的输入
  for (auto* v : scope_ctx.inputs_) {
    func_n->addInput(v);
  }

  // 设置新函数节点的输出类型，并替换作用域中的输出节点
  for (const auto i : c10::irange(scope_ctx.outputs_.size())) {
    func_n->output(i)->setType(scope_ctx.outputs_[i]->type());
    scope_ctx.outputs_[i]->replaceAllUsesWith(func_n->output(i));
  }

  // 复制不同值的属性作为函数的属性
  auto copy_attr =
      [](Node* a, Node* b, Symbol attr, const std::string& new_name) {
#define COPY_ATTR(kind)                                \
  case AttributeKind::kind: {                          \
    b->kind##_(Symbol::attr(new_name), a->kind(attr)); \
    break;                                             \
  }
        switch (a->kindOf(attr)) {
          COPY_ATTR(f)
          COPY_ATTR(fs)
          COPY_ATTR(i)
          COPY_ATTR(is)
          COPY_ATTR(s)
          COPY_ATTR(ss)
          COPY_ATTR(t)
          COPY_ATTR(ts)
#undef COPY_ATTR
          // 对于未预期的属性类型，引发内部断言错误
          case AttributeKind::ival:
          case AttributeKind::g:
          case AttributeKind::gs:
          case AttributeKind::ty:
          case AttributeKind::tys:
          case AttributeKind::c:
          default:
            TORCH_INTERNAL_ASSERT(
                false,
                "Unexpected attribute type ",
                static_cast<int>(a->kindOf(attr)),
                " from node ",
                *a);
            break;
        }
      };

  // 遍历函数上下文中的属性映射
  for (const auto& it : func_ctx.attribute_map_) {
    auto* ref_n = it.first;
    for (const auto& attr_it : it.second) {
      const auto& attr = attr_it.first;
      auto attr_name = func_ctx.FindAttrName(ref_n, attr).value();
      // 复制属性到新函数节点
      copy_attr(ref_n, func_n, attr, attr_name);
      // 在作用域中的每个节点上查找属性并复制到新函数节点
      for (auto* n : scope_ctx.nlist_) {
        if (attr_it.second.find(n) != attr_it.second.end()) {
          copy_attr(n, func_n, attr, attr_name);
          break;
        }
      }
    }
  }

  // 获取作用域并查找其注释属性
  auto scope = scope_ctx.scope_;
  auto annotated_attr_node = scope_attr_map_.find(scope);
  if (annotated_attr_node != scope_attr_map_.end()) {
    auto node = annotated_attr_node->second;
    // 复制注释属性到新函数节点
    for (auto attr : node->attributeNames()) {
      copy_attr(node, func_n, attr, attr.toUnqualString());
    }
  }

  // 将新函数节点插入到最后一个节点之后
  func_n->insertAfter(last_n);
  return func_n;  // 返回创建的函数节点
}
  // 传入函数所需的图对象和函数上下文信息，需始终在最内层作用域调用此函数。
  // 1. 生成函数上下文，用于标识不同的常量和属性。
  // 2. 创建函数定义节点，并将其插入主图中。
  // 3. 为每次调用创建函数节点，并替换父函数中的子图节点。

  func_ctxs_.insert(std::make_pair(
      scope_key, new FunctionContext(scope_key, scope_list, scope_ctxs)));
  // 在 func_ctxs_ 中插入新的函数上下文对象，使用 scope_key 作为键，包含给定的作用域列表和上下文对象。

  auto& func_ctx = *func_ctxs_[scope_key];
  // 获取当前 scope_key 对应的函数上下文的引用。

  const std::string module_class_name(
      ONNXScopeName::className(func_ctx.scope_key_));
  // 根据 func_ctx 的 scope_key 生成模块类名。

  auto pos = module_class_name.rfind('.');
  TORCH_INTERNAL_ASSERT(pos != std::string::npos);
  // 查找模块类名中最后一个点号的位置，确保找到有效的分隔符位置。

  auto construct_unique_module_name = [&](std::string module_name) {
    // 生成唯一的模块名称函数，处理模块名称的变体。
    auto module_name_variant = module_variant_count_.find(module_name);
    if (module_name_variant != module_variant_count_.end()) {
      module_variant_count_[module_name]++;
      module_name += ("." + std::to_string(module_name_variant->second));
    } else {
      module_variant_count_[module_name] = 0;
    }
    return module_name;
  };

  const auto domain_name = module_class_name.substr(0, pos);
  // 从模块类名中提取域名部分。

  const auto func_name =
      construct_unique_module_name(module_class_name.substr(pos + 1));
  // 构造唯一的函数名称，处理模块类名中的函数名部分。

  CreateFunctionDefNode(func_ctx, graph, domain_name, func_name);
  // 调用函数创建函数定义节点，并将其插入给定的图对象中。

  // 创建并插入本地函数节点到图中。
  for (const auto& it : func_ctx.scope_ctxs_) {
    auto scope = it.first;
    auto& scope_ctx = *it.second;
    auto func_n =
        CreateFunctionNode(func_ctx, scope_ctx, graph, domain_name, func_name);
    // 对于每个函数上下文中的作用域，创建函数节点并将其插入到图中。

    std::unordered_set<Node*> old_nodes(
        scope_ctx.nlist_.begin(), scope_ctx.nlist_.end());
    // 使用作用域上下文中的节点列表创建旧节点的无序集合。

    auto last_n = *scope_ctx.nlist_.rbegin();
    // 获取作用域上下文中节点列表的最后一个节点。

    // 在父作用域中用本地函数节点替换函数体节点。
    for (auto& it : scope_ctxs) {
      const auto& parent_scope = it.first;
      auto& parent_ctx = *it.second;

      if (!IsAncestor(parent_scope, scope)) {
        continue;
      }
      // 如果当前作用域不是父作用域的祖先，则跳过。

      auto& ctx_nlist = parent_ctx.nlist_;
      GRAPH_DEBUG(
          "Replace local function node in parent scope: ",
          it.first->namesFromRoot(),
          " nodes to remove: ",
          old_nodes.size(),
          " parent total nodes: ",
          ctx_nlist.size());
      // 调试信息，替换父作用域中的本地函数节点。

      // 插入本地函数节点。
      auto last_n_it = std::find(ctx_nlist.begin(), ctx_nlist.end(), last_n);
      ctx_nlist.insert(last_n_it, func_n);

      // 移除被替换的节点。
      ctx_nlist.erase(
          std::remove_if(
              ctx_nlist.begin(),
              ctx_nlist.end(),
              [&old_nodes](Node* n) {
                return old_nodes.find(n) != old_nodes.end();
              }),
          ctx_nlist.end());

      GRAPH_DEBUG("Parent total nodes after remove: ", ctx_nlist.size());
      // 调试信息，移除节点后父作用域的节点总数。

      // 刷新输入/输出。
      parent_ctx.PopulateInputsOutputs(param_names_);
      // 更新父上下文的输入输出信息。
    }
  }

  for (const auto& it : func_ctx.scope_ctxs_) {
    // 从引用中获取当前迭代器指向的 scope_ctx 对象
    auto& scope_ctx = *it.second;
    // 在图中删除被替换的节点。
    for (auto it = scope_ctx.nlist_.rbegin(); it != scope_ctx.nlist_.rend();) {
      // 获取当前迭代器指向的节点指针
      auto* n = *it;
      // 向前移动迭代器
      it++;
      // 输出调试信息，显示将要销毁的节点
      GRAPH_DEBUG("Destroying node ", *n);
      // 销毁节点
      n->destroy();
    }
  }
}

bool FunctionExtractor::ScopeContext::IsIdenticalFuncion(
    const ScopeContext& other_ctx) const {
  // 检查两个 ScopeContext 是否表示相同的函数

  // 如果比较的是同一个对象，则认为相同
  if (&other_ctx == this) {
    return true;
  }

  // 检查作用域名称是否相同
  if (ONNXScopeName::className(this->scope_) !=
      ONNXScopeName::className(other_ctx.scope_)) {
    return false;
  }

  // 检查输入和输出的数量是否相同
  if (this->inputs_.size() != other_ctx.inputs_.size() ||
      this->outputs_.size() != other_ctx.outputs_.size()) {
    return false;
  }

  // 检查节点列表的大小及节点类型是否一致
  const auto& ns_a = this->nlist_;
  const auto& ns_b = other_ctx.nlist_;
  if (ns_a.size() != ns_b.size()) {
    return false;
  }
  for (const auto i : c10::irange(ns_a.size())) {
    if (ns_a[i]->kind() != ns_b[i]->kind()) {
      return false;
    }
  }

  // 如果以上条件都满足，则认为两个 ScopeContext 表示相同的函数
  return true;
}

void FunctionExtractor::ScopeContext::PopulateInputsOutputs(
    const std::unordered_set<std::string>& param_names) {
  // 清空输入和输出列表
  inputs_.clear();
  outputs_.clear();

  // 获取节点列表的引用
  const auto& nlist = this->nlist_;

  // 初始化值和节点的集合
  std::unordered_set<Value*> v_set;
  std::unordered_set<Node*> n_set;

  // 分别存储输入和初始化值的列表
  value_list input_list;
  value_list initializer_list;

  // 遍历节点列表，将输入值和初始化值分别加入对应的列表
  for (auto* n : nlist) {
    for (auto* v : n->inputs()) {
      if (v_set.find(v) == v_set.end()) {
        if (param_names.find(v->debugName()) != param_names.end()) {
          initializer_list.emplace_back(v);
        } else {
          input_list.emplace_back(v);
        }
        v_set.insert(v);
      }
    }
    for (auto* v : n->outputs()) {
      v_set.insert(v);
    }
    n_set.insert(n);
  }

  // 将收集到的输入值和初始化值分别加入输入列表
  for (auto* v : input_list) {
    inputs_.emplace_back(v);
  }
  for (auto* v : initializer_list) {
    inputs_.emplace_back(v);
  }

  // 根据节点的使用情况确定输出列表
  for (auto* n : nlist) {
    for (auto* v : n->outputs()) {
      bool used_outside = false;
      for (auto use : v->uses()) {
        used_outside |= (n_set.find(use.user) == n_set.end());
      }
      if (used_outside) {
        outputs_.emplace_back(v);
      }
    }
  }
}

void FunctionExtractor::HandleNoScopeNodes(
    scope_ctx_map& scope_ctxs,
    node_list no_scope_nlist) {
  // 输出无法确定作用域的节点数量
  GRAPH_UPDATE("No scope node count: ", no_scope_nlist.size());

  // 对每个无法确定作用域的节点输出警告信息
  for (auto n : no_scope_nlist) {
    TORCH_WARN(
        "ONNX function extraction cannot determine the scope for node: ", *n);
  }

  // 断言所有无法确定作用域的节点都已经处理
  TORCH_INTERNAL_ASSERT(
      no_scope_nlist.empty(),
      "ONNX function extraction cannot determine the scope for the above nodes.");
}

std::tuple<FunctionExtractor::scope_ctx_map, node_list> FunctionExtractor::
    // 根据给定的块对象 `b`，将其节点按作用域分组
    PartitionNodesByScope(Block* b) {
      // 创建空的作用域上下文映射
      scope_ctx_map scope_ctxs = {};
      // 创建空的无作用域节点列表
      node_list no_scope_nlist;
    
      // 定义函数：查找或创建指定作用域的上下文
      auto find_or_create_scope_ctx = [](scope_ctx_map& scope_ctxs,
                                         const ScopePtr& scope) {
        // 如果作用域映射中不存在该作用域，则创建新的作用域上下文并插入映射中
        if (scope_ctxs.find(scope) == scope_ctxs.end()) {
          scope_ctxs.insert(std::make_pair(scope, new ScopeContext()));
        }
        // 返回该作用域对应的上下文
        return scope_ctxs[scope];
      };
    
      // 定义函数：记录节点的作用域信息
      auto record_node_scope = [&scope_ctxs, &find_or_create_scope_ctx](Node* n) {
        // 获取节点 `n` 的作用域
        const auto& scope = n->scope();
        // 查找或创建该作用域的上下文，并设置作用域信息
        find_or_create_scope_ctx(scope_ctxs, scope)->scope_ = scope;
        auto tmp_scope = scope;
        // 将节点 `n` 添加到其所有有效的父作用域的节点列表中
        while (IsValidScope(tmp_scope)) {
          find_or_create_scope_ctx(scope_ctxs, tmp_scope)->nlist_.emplace_back(n);
          if (IsValidScope(tmp_scope->parent())) {
            // 将当前作用域 `tmp_scope` 的父作用域添加到其子作用域集合中
            find_or_create_scope_ctx(scope_ctxs, tmp_scope->parent())
                ->children_.insert(tmp_scope);
          }
          // 继续向上遍历父作用域
          tmp_scope = tmp_scope->parent();
        }
      };
    
      // 遍历块 `b` 中的所有节点
      for (auto* n : b->nodes()) {
        // 获取节点 `n` 的作用域
        auto scope = n->scope();
        // 如果节点有有效的作用域，则记录节点的作用域信息
        if (scope && IsValidScope(scope)) {
          record_node_scope(n);
        } else {
          // 否则，推断节点的作用域
          auto inferred_scope = InferScope(n);
    
          // 如果推断得到有效的作用域，则设置节点的作用域并记录作用域信息
          if (inferred_scope.has_value() && IsValidScope(inferred_scope.value())) {
            n->setScope(inferred_scope.value());
            record_node_scope(n);
          } else {
            // 否则，输出无法推断节点作用域的信息，并将节点添加到无作用域节点列表中
            GRAPH_UPDATE("Cannot infer proper scope for node: ", *n);
            no_scope_nlist.emplace_back(n);
          }
        }
    
        // 递归处理节点 `n` 的所有子块
        for (auto* sub_b : n->blocks()) {
          // 分区处理子块的节点，并获取子块的作用域上下文和无作用域节点列表
          auto [subblock_scope_ctxs, subblock_no_scope_nlist] =
              PartitionNodesByScope(sub_b);
    
          // 将子块的作用域上下文合并到当前作用域上下文映射中
          for (auto& it : subblock_scope_ctxs) {
            if (scope_ctxs.find(it.first) == scope_ctxs.end()) {
              scope_ctxs.insert(std::make_pair(it.first, it.second));
            } else {
              // 合并节点列表和子作用域集合到当前作用域上下文中
              for (auto* s_n : it.second->nlist_) {
                scope_ctxs[it.first]->nlist_.emplace_back(s_n);
              }
              for (const auto& s_child_scope : it.second->children_) {
                scope_ctxs[it.first]->children_.insert(s_child_scope);
              }
            }
          }
    
          // 将子块中的无作用域节点列表合并到当前无作用域节点列表中
          no_scope_nlist.insert(
              no_scope_nlist.end(),
              subblock_no_scope_nlist.begin(),
              subblock_no_scope_nlist.end());
        }
      }
    
      // 为每个作用域上下文设置其作用域，并填充输入输出参数名称
      for (auto& it : scope_ctxs) {
        it.second->scope_ = it.first;
        it.second->PopulateInputsOutputs(param_names_);
      }
    
      // 返回作用域上下文映射和无作用域节点列表的元组
      return std::tie(scope_ctxs, no_scope_nlist);
    }
}

FunctionExtractor::scope_ctx_map FunctionExtractor::PartitionNodesByScope(
    const std::shared_ptr<Graph>& graph) {
  // 创建空的作用域上下文映射和无作用域节点列表
  scope_ctx_map scope_ctxs;
  node_list no_scope_nlist;
  // 调用 PartitionNodesByScope 方法，将图的块分割为作用域并返回结果
  std::tie(scope_ctxs, no_scope_nlist) = PartitionNodesByScope(graph->block());

  // 处理无作用域节点
  HandleNoScopeNodes(scope_ctxs, no_scope_nlist);

  // 返回作用域上下文映射
  return scope_ctxs;
}

std::unordered_map<ScopePtr, scope_list> FunctionExtractor::
    PartitionIdenticalScopes(FunctionExtractor::scope_ctx_map& scope_ctxs) {
  // 创建无序映射表来存储相同作用域的映射
  std::unordered_map<ScopePtr, scope_list> identical_scope_map;

  // 遍历每个作用域上下文映射条目
  for (auto& it : scope_ctxs) {
    auto scope = it.first; // 获取作用域指针
    const auto& scope_ctx = it.second; // 获取作用域上下文

    bool unique = true;
    // 遍历已知相同作用域映射表
    for (auto& kv_it : identical_scope_map) {
      auto key_scope = kv_it.first; // 获取键作用域指针
      const auto& key_scope_ctx = scope_ctxs[key_scope]; // 获取键作用域上下文

      auto& key_scope_vec = kv_it.second; // 获取键作用域向量
      // 如果当前作用域与键作用域相同，则将当前作用域添加到键作用域的向量中
      if (key_scope_ctx->IsIdenticalFuncion(*scope_ctx)) {
        key_scope_vec.emplace_back(scope);
        unique = false;
        break;
      }
    }
    // 如果当前作用域是独特的，则将其添加到相同作用域映射表中
    if (unique) {
      identical_scope_map[scope].emplace_back(scope);
    }
  }

  // 返回相同作用域的映射表
  return identical_scope_map;
}

static bool HasSameAttribute(
    const Node* a,
    const Node* b,
    const c10::Symbol& attr) {
  // 如果两个节点都不具有指定属性，则它们具有相同的属性
  if (!a->hasAttribute(attr) && !b->hasAttribute(attr)) {
    return true;
  }
  // 如果其中一个节点具有指定属性而另一个节点不具有，则它们属性不同
  if (!a->hasAttribute(attr) || !b->hasAttribute(attr)) {
    return false;
  }
  auto a_kind = a->kindOf(attr); // 获取节点 a 的属性类型
  auto b_kind = b->kindOf(attr); // 获取节点 b 的属性类型
  // 如果两个节点的属性类型不同，则它们属性不同
  if (a_kind != b_kind) {
    return false;
  }

#define COMP_ATTR(kind)              \
  case AttributeKind::kind: {        \
    const auto& a_v = a->kind(attr); \
    const auto& b_v = b->kind(attr); \
    return a_v == b_v;               \
  }

  // 根据属性类型进行比较，并返回比较结果
  switch (a_kind) {
    COMP_ATTR(f) // float 类型属性比较
    COMP_ATTR(fs) // float 数组类型属性比较
    COMP_ATTR(i) // int 类型属性比较
    COMP_ATTR(is) // int 数组类型属性比较
    COMP_ATTR(s) // string 类型属性比较
    COMP_ATTR(ss) // string 数组类型属性比较
#undef COMP_ATTR
    case AttributeKind::t: { // Tensor 类型属性比较
      const auto& a_v = a->t(attr);
      const auto& b_v = b->t(attr);
      return a_v.equal(b_v); // 使用 Tensor 的 equal 方法比较
    }
    case AttributeKind::ts: { // Tensor 数组类型属性比较
      const auto& a_v = a->ts(attr);
      const auto& b_v = b->ts(attr);
      return std::equal(
          a_v.begin(),
          a_v.end(),
          b_v.begin(),
          b_v.end(),
          [](const at::Tensor& a_t, const at::Tensor& b_t) {
            return a_t.equal(b_t); // 使用 Tensor 的 equal 方法比较
          });
    }
    // 处理未预期的属性类型，抛出内部断言异常
    case AttributeKind::ival:
    case AttributeKind::g:
    case AttributeKind::gs:
    case AttributeKind::ty:
    case AttributeKind::tys:
    case AttributeKind::c:
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected attribute type ",
          static_cast<int>(a_kind),
          " from node ",
          *a);
      break;
  }

  // 默认返回属性相同
  return true;
}

scope_list FunctionExtractor::SortScopesByMaxDepth(
    std::unordered_map<ScopePtr, scope_list>& identical_scope_map) {
  // 创建作用域最大深度的无序映射表
  std::unordered_map<ScopePtr, size_t> scope_max_depth;
  // 遍历每个相同作用域映射表的条目
  for (const auto& it : identical_scope_map) {
    const auto& scopes = it.second; // 获取作用域列表
    size_t max_depth = 0;
    // 遍历scopes中的每一个作用域对象
    for (const auto& scope : scopes) {
      // 如果当前作用域的深度大于当前记录的最大深度值
      if (scope->getDepth() > max_depth) {
        // 更新最大深度值为当前作用域的深度
        max_depth = scope->getDepth();
      }
    }
    // 将当前最大深度值记录到scope_max_depth映射中，以当前迭代的键(it.first)作为索引
    scope_max_depth[it.first] = max_depth;
  }

  // 创建一个用于存储排序后作用域的列表sorted_scopes
  scope_list sorted_scopes;
  sorted_scopes.reserve(scope_max_depth.size());
  // 遍历scope_max_depth映射，将每个键(it.first)添加到sorted_scopes列表中
  for (const auto& it : scope_max_depth) {
    sorted_scopes.emplace_back(it.first);
  }
  // 对sorted_scopes列表中的作用域对象进行排序，排序依据是scope_max_depth映射中每个作用域对象的深度值
  std::sort(
      sorted_scopes.begin(),
      sorted_scopes.end(),
      [&scope_max_depth](const ScopePtr& a, const ScopePtr& b) -> bool {
        // 按照作用域对象深度值降序排序
        return scope_max_depth[a] >= scope_max_depth[b];
      });
  // 返回排序后的作用域列表sorted_scopes
  return sorted_scopes;
// FunctionExtractor 类的 run 方法，用于执行函数提取操作，更新图数据结构
NodeAttrNameMap FunctionExtractor::run() {
  // 将图中的节点按照作用域划分为不同的组
  auto scope_ctxs = PartitionNodesByScope(graph_);
  // 打印调试信息，显示各个作用域的上下文信息
  DebugPrintScopeContexts(scope_ctxs);
  // 将具有相同子图模式的作用域进行分组
  auto identical_scope_map = PartitionIdenticalScopes(scope_ctxs);
  
  // 按照最大深度排序作用域键，深度最大的作用域排在最前，保证没有其他作用域是它的子作用域
  auto sorted_scope_keys = SortScopesByMaxDepth(identical_scope_map);
  
  // 遍历排序后的作用域键
  for (const auto& scope_key : sorted_scope_keys) {
    // 如果模块名集合中存在当前作用域键对应的类名
    if (module_names_.find(ONNXScopeName::className(scope_key)) !=
        module_names_.end()) {
      // 将作用域转换为函数，并在图中进行相应的修改
      ConvertScopeToFunction(
          scope_key, identical_scope_map[scope_key], scope_ctxs, graph_);
    }
    // 打印图的调试信息，显示转换作用域为函数后的主图情况
    GRAPH_DEBUG("Main graph afterwards: ", graph_->toString());
  }
  
  // 打印调试信息，显示带有函数的图
  DebugPrintGraphWithFunction(graph_);

  // 构造节点属性到名称的映射
  NodeAttrNameMap node_attr_to_name;

  // 遍历函数上下文集合
  for (const auto& it : func_ctxs_) {
    // 获取当前函数上下文对象的节点属性到名称映射
    auto func_ref_map = it.second->node_attr_to_name_;
    // 将当前函数的映射插入到总映射中
    node_attr_to_name.insert(func_ref_map.begin(), func_ref_map.end());
  }

  // 清理作用域上下文对象
  for (auto& it : scope_ctxs) {
    delete it.second;
  }
  scope_ctxs.clear();
  
  // 清理函数上下文对象
  for (auto& it : func_ctxs_) {
    delete it.second;
  }
  func_ctxs_.clear();

  // 返回节点属性到名称的映射
  return node_attr_to_name;
}

// NodeOfMostRecentScope 函数，用于获取表示最近 ScopePtr 的节点
// 该函数仅应从模块前向钩子调用，在此时，模块前向调用已完成，并且最近的 ScopePtr 已从 TracingState 弹出
// 该函数检查节点及其子块，以查找与最近 ScopePtr 关联的节点
Node* NodeOfMostRecentScope(Node* forward_node) {
  // 断言检查，确保 forward_node 的类型是 prim::TracedModuleForward
  TORCH_INTERNAL_ASSERT(
      forward_node->kind() == prim::TracedModuleForward,
      "forward_node got kind: ",
      forward_node->kind().toDisplayString());
  // 获取 forward_node 的第一个块
  auto* block = forward_node->blocks()[0];
  // 反向遍历块中的节点
  for (auto* node : block->nodes().reverse()) {
    // 如果节点的类型是 prim::TracedModuleForward
    if (node->kind() == prim::TracedModuleForward) {
      // 递归调用 NodeOfMostRecentScope 函数，查找最近 ScopePtr 的节点
      Node* target_node = NodeOfMostRecentScope(node);
      // 如果在 scope_attr_map_ 中找不到节点的作用域，则返回目标节点
      if (scope_attr_map_.find(node->scope()) == scope_attr_map_.end()) {
        return target_node;
      }
    }
  }
  // 如果没有找到符合条件的节点，则返回 forward_node 自身
  return forward_node;
}
    // 调用宏 GRAPH_UPDATE，将模块名称集合和参数名称向量转换为字符串向量，记录到日志中
    GRAPH_UPDATE(
        "Export these module forward calls as functions: ",
        std::vector<std::string>{module_names.begin(), module_names.end()});
    // 创建 FunctionExtractor 对象 fe，传入计算图、模块名称集合和参数名称向量
    FunctionExtractor fe(graph, module_names, param_names);
    // 执行 FunctionExtractor 对象的 run 方法，返回结果
    return fe.run();
} // 结束命名空间 torch

} // 结束命名空间 jit

} // 结束命名空间 onnx
```