# `.\pytorch\torch\csrc\jit\passes\onnx\naming.cpp`

```py
// 包含头文件：torch/csrc/jit/passes/onnx/naming.h 和 torch/csrc/onnx/onnx.h
#include <torch/csrc/jit/passes/onnx/naming.h>
#include <torch/csrc/onnx/onnx.h>

// 包含标准库头文件
#include <utility>

// 声明命名空间 torch::jit::onnx
namespace torch {
namespace jit {
namespace onnx {

// 命名空间 ONNXScopeName，用于定义作用域名称相关功能
namespace ONNXScopeName {

// 定义函数指针类型 NameFunc，用于获取作用域的名称
using NameFunc = std::string (*)(torch::jit::ScopePtr scope);

// 定义作用域名称分隔符
const std::string name_separator = "::";

// 匿名命名空间，用于实现局部功能
namespace {

// 函数：从根作用域开始获取名称
std::string nameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator,
    NameFunc name_func) {
  // 获取当前作用域的名称
  std::string out = (*name_func)(scope);
  // 如果当前作用域是根作用域，则直接返回名称
  if (scope->isRoot()) {
    return out;
  }
  // 获取父作用域，并追溯直到根作用域，构建完整的作用域名称
  auto parent = scope->parent();
  while (isCompatibleScope(parent)) {
    out = std::string((*name_func)(parent)).append(layer_separator).append(out);
    parent = parent->parent();
  }
  return out;
}

// 函数：从作用域解析名称
std::pair<std::string, std::string> parseNameFromScope(
    torch::jit::ScopePtr scope) {
  // 获取作用域的完整名称
  std::string full_name = scope->name().toUnqualString();
  // 查找名称分隔符 "::"
  auto pos = full_name.find(name_separator);
  // 检查是否找到名称分隔符，否则抛出异常
  TORCH_CHECK(
      pos != std::string::npos,
      "Scope name (" + full_name + ") does not contain '" + name_separator +
          "'");
  // 返回分隔后的名称部分
  return std::make_pair(full_name.substr(0, pos), full_name.substr(pos + 2));
}

} // namespace

// 函数：创建完整的作用域名称
std::string createFullScopeName(
    const std::string& class_name,
    const std::string& variable_name) {
  // 拼接类名和变量名，并使用名称分隔符 "::" 连接
  return std::string(class_name).append(name_separator).append(variable_name);
}

// 函数：从作用域获取变量名称
std::string variableName(torch::jit::ScopePtr scope) {
  return parseNameFromScope(scope).second;
}

// 函数：从根作用域开始获取变量名称
std::string variableNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator) {
  return nameFromRoot(scope, layer_separator, &variableName);
}

// 函数：从作用域获取类名
std::string className(torch::jit::ScopePtr scope) {
  return parseNameFromScope(scope).first;
}

// 函数：从根作用域开始获取类名
std::string classNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator) {
  return nameFromRoot(scope, layer_separator, &className);
}

// 函数：判断作用域是否兼容
bool isCompatibleScope(torch::jit::ScopePtr scope) {
  // 判断作用域不是根作用域、非空白作用域，并且包含名称分隔符 "::"
  return !scope->isRoot() && !scope->isBlank() &&
      (std::string(scope->name().toUnqualString()).find(name_separator) !=
       std::string::npos);
}

} // namespace ONNXScopeName

// 匿名命名空间，用于实现类 NodeNameGenerator
namespace {

// 类：NodeNameGenerator，用于生成节点名称
class NodeNameGenerator {
 public:
  // 构造函数：初始化图形对象
  NodeNameGenerator(std::shared_ptr<Graph> g) : graph_(std::move(g)){};
  // 虚析构函数
  virtual ~NodeNameGenerator() = 0;
  // 填充节点名称的公共方法
  void PopulateNodeNames();

 protected:
  // 纯虚函数：创建节点名称
  virtual void CreateNodeName(Node* n) = 0;
  // 填充节点名称的内部方法
  void PopulateNodeNames(Block*);
  // 更新输出名称的方法
  void UpdateOutputsNames(Node* n);
  // 判断值是否为图形的输出
  bool IsGraphOutput(const Value* v, const std::shared_ptr<Graph> graph) const;

 protected:
  // 创建唯一名称的方法
  std::string CreateUniqueName(
      std::unordered_map<std::string, size_t>& base_name_count,
      std::string base_name);

  // 节点名称映射表
  std::unordered_map<const Node*, std::string> node_names_;
  // 基础节点名称计数表
  std::unordered_map<std::string, size_t> base_node_name_counts_;
  // 图形对象
  std::shared_ptr<Graph> graph_;
  // 分层分隔符
  const std::string layer_separator_ = "/";
};

// 虚析构函数的默认实现
NodeNameGenerator::~NodeNameGenerator() = default;
class ScopedNodeNameGenerator : public NodeNameGenerator {
 public:
  // ScopedNodeNameGenerator 类的构造函数，接受一个共享指针指向 Graph 对象作为参数
  ScopedNodeNameGenerator(std::shared_ptr<Graph> g) : NodeNameGenerator(g){};

 protected:
  // 重写的 CreateNodeName 函数，用于创建节点名称
  void CreateNodeName(Node* n) override;

 private:
  // 根据 Scope 返回完整的作用域名称的私有方法
  std::string GetFullScopeName(ScopePtr scope);
  // 记录每个 Scope 的完整作用域名称的哈希表
  std::unordered_map<ScopePtr, std::string> full_scope_names_;
  // 记录每个基础作用域名称计数的哈希表
  std::unordered_map<std::string, size_t> base_scope_name_counts_;
};

// NodeNameGenerator 类的成员函数，创建唯一名称的函数
std::string NodeNameGenerator::CreateUniqueName(
    std::unordered_map<std::string, size_t>& base_name_count,
    std::string base_name) {
  // 如果基础名称在 base_name_count 中不存在，则将其计数初始化为 0
  if (base_name_count.find(base_name) == base_name_count.end()) {
    base_name_count[base_name] = 0;
  } else {
    // 如果存在，则递增计数，并修改基础名称以反映其唯一性
    auto count = ++base_name_count[base_name];
    base_name += "_";
    base_name += std::to_string(count);
  }
  // 返回最终的唯一名称
  return base_name;
}

// NodeNameGenerator 类的成员函数，检查给定的值是否是图的输出
bool NodeNameGenerator::IsGraphOutput(
    const Value* v,
    const std::shared_ptr<Graph> graph) const {
  // 遍历图的所有输出，检查给定值是否是其中之一
  for (const auto* graph_output : graph->outputs()) {
    if (v == graph_output) {
      return true;
    }
  }
  return false;
}

// NodeNameGenerator 类的成员函数，更新节点输出的名称
void NodeNameGenerator::UpdateOutputsNames(Node* n) {
  // 如果节点已经有名称映射
  if (node_names_.find(n) != node_names_.end()) {
    auto node_name = node_names_[n];
    // 遍历节点的所有输出
    for (auto i : c10::irange(n->outputs().size())) {
      auto output = n->output(i);
      // 如果输出不是图的输出，则更新其调试名称
      if (!IsGraphOutput(output, graph_)) {
        auto output_name = node_name;
        output_name.append("_output_").append(std::to_string(i));
        output->setDebugName(output_name);
      }
    }
  }
}

// NodeNameGenerator 类的成员函数，填充节点名称
void NodeNameGenerator::PopulateNodeNames() {
  PopulateNodeNames(graph_->block());
}

// NodeNameGenerator 类的成员函数，填充给定块中的节点名称
void NodeNameGenerator::PopulateNodeNames(Block* b) {
  // 遍历块中的所有节点
  for (auto* n : b->nodes()) {
    // 遍历节点的所有子块
    for (auto* sub_block : n->blocks()) {
      PopulateNodeNames(sub_block);
    }
    // 创建节点名称
    CreateNodeName(n);
    // 更新节点输出的名称
    UpdateOutputsNames(n);
  }
}

// ScopedNodeNameGenerator 类的成员函数，创建节点名称
void ScopedNodeNameGenerator::CreateNodeName(Node* n) {
  // 如果节点还没有名称映射
  if (node_names_.find(n) == node_names_.end()) {
    // 如果节点的 Scope 不兼容，则返回
    if (!ONNXScopeName::isCompatibleScope(n->scope())) {
      return;
    }
    // 如果节点必须是 None，则在 JIT IR 中不允许使用属性
    if (n->mustBeNone()) {
      // JIT IR 不允许 None 节点使用属性。
      return;
    }
    // 获取节点的完整作用域名称
    auto name = GetFullScopeName(n->scope());
    // 添加分隔符和节点的类型名称，创建唯一名称
    name += layer_separator_;
    name += n->kind().toUnqualString();
    node_names_[n] = CreateUniqueName(base_node_name_counts_, name);
  }
  // 设置节点的属性名称
  n->s_(Symbol::attr(::torch::onnx::kOnnxNodeNameAttribute), node_names_[n]);
}

// ScopedNodeNameGenerator 类的成员函数，获取完整作用域名称
std::string ScopedNodeNameGenerator::GetFullScopeName(ScopePtr scope) {
  // 如果给定 Scope 的完整名称不存在，则创建唯一名称
  if (full_scope_names_.find(scope) == full_scope_names_.end()) {
    auto full_scope_name =
        ONNXScopeName::variableNameFromRoot(scope, layer_separator_);
    full_scope_names_[scope] =
        CreateUniqueName(base_scope_name_counts_, full_scope_name);
  }
  // 返回 Scope 的完整名称
  return full_scope_names_[scope];
}

// 在 onnx 命名空间中定义的函数，为图中的节点和值分配作用域名称
void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph) {
  // 创建 ScopedNodeNameGenerator 对象
  auto node_name_generator = std::make_unique<ScopedNodeNameGenerator>(graph);
  // 填充节点名称
  node_name_generator->PopulateNodeNames();
}

// 结束 onnx 命名空间
} // namespace onnx
} // namespace jit
} // namespace torch
```