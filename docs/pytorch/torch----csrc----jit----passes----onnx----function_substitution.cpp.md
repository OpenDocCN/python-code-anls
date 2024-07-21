# `.\pytorch\torch\csrc\jit\passes\onnx\function_substitution.cpp`

```
#include <torch/csrc/jit/passes/onnx/function_substitution.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/naming.h>

namespace torch {
namespace jit {

namespace {

// 定义顶层模块变量名的常量，初始化为空字符串
const std::string kTopModuleVariableName = "";

// 从TorchScript中的可选限定名中整理出整洁的类名字符串
std::string TidyClassNameFromTorchScript(
    const std::optional<c10::QualifiedName>& class_name) {
  if (!class_name) {
    return "UNKNOWN_CLASS";  // 如果类名为空，返回未知类名
  }
  std::string out = "";
  for (const auto& atom : class_name->atoms()) {
    bool is_internal_torch_atom = (atom == "__torch__");
    bool is_mangle_atom = (atom.find("__torch_mangle") != std::string::npos);
    if (!is_internal_torch_atom && !is_mangle_atom) {
      if (!out.empty()) {
        out += ".";
      }
      out += atom;
    }
  }
  return out;  // 返回整理后的类名字符串
}

// 获取调用节点的变量名
std::string GetCallNodeVariableName(const Node* call_node) {
  TORCH_INTERNAL_ASSERT(
      call_node->kind() == prim::CallFunction ||
      call_node->kind() == prim::CallMethod);
  auto module_node = call_node->input(0)->node();

  if (!module_node->hasAttribute(attr::name)) {
    return "";  // 如果模块节点没有名字属性，返回空字符串
  }
  std::string module_name = module_node->s(attr::name);
  if (module_node->inputs().empty()) {
    return module_name;  // 如果模块没有输入，直接返回模块名字
  }
  // 如果模块来自容器，则模块节点的attr::name只携带索引信息，需要检查父节点（容器）获取变量名
  auto parent_module_value = module_node->input(0);
  while (parent_module_value) {
    auto parent_module_type = parent_module_value->type()->cast<ClassType>();
    if (parent_module_type &&
        parent_module_type->name() ==
            "__torch__.torch.nn.modules.container.ModuleList") {
      auto parent_module_node = parent_module_value->node();
      module_name = parent_module_node->s(attr::name) + "." + module_name;
      parent_module_value = !parent_module_node->inputs().empty()
          ? parent_module_node->input(0)
          : nullptr;
    } else {
      break;
    }
  }

  return module_name;  // 返回完整的模块变量名
}

// 为调用方法节点创建正向调用的作用域
ScopePtr ForwardCallScope(Graph& graph, Node* call_node) {
  TORCH_INTERNAL_ASSERT(call_node->kind() == prim::CallMethod);
  const std::string& method_name = call_node->s(attr::name);
  if (method_name == "forward") {
    const auto type = call_node->input(0)->type()->expect<c10::NamedType>();
    const std::string class_name = TidyClassNameFromTorchScript(type->name());
    const std::string variable_name = GetCallNodeVariableName(call_node);
    const std::string scope_name =
        onnx::ONNXScopeName::createFullScopeName(class_name, variable_name);
    return graph.current_scope()->push(Symbol::scope(scope_name));  // 创建新的作用域并推入当前图的作用域栈
  }
  return graph.current_scope();  // 返回当前图的作用域
}

// 函数调用替换的具体实现
void functionCallSubstitution(Block* block) {
  auto graph = block->owningGraph();
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    // 进行节点处理后，记录当前图的作用域信息
    GRAPH_DEBUG(
        "Graph current scope after node process: ",
        graph->current_scope()->namesFromRoot());
  }
}

} // namespace

} // namespace jit
} // namespace torch
// 返回顶层作用域的指针，用于 ONNX 图的处理
ScopePtr ONNXGraphTopLevelScope(Graph& graph) {
  // 如果图的输入为空，则返回当前作用域
  if (graph.inputs().empty()) {
    return graph.current_scope();
  }
  // 如果图的第一个输入是 ClassType 类型
  if (auto top_module_type = graph.inputs().at(0)->type()->cast<ClassType>()) {
    // 创建完整的 ONNX 作用域名，结合类名和顶层模块变量名
    auto scope_name = ::torch::jit::onnx::ONNXScopeName::createFullScopeName(
        TidyClassNameFromTorchScript(top_module_type->name()),
        kTopModuleVariableName);
    // 将新创建的作用域推入当前图的作用域栈，并返回新的作用域
    return graph.current_scope()->push(Symbol::scope(scope_name));
  }
  // 如果条件不符合，则返回当前作用域
  return graph.current_scope();
}

} // namespace

// 该函数用于 ONNX 转换。ONNX 转换器依赖于一些已废弃的 aten 运算符。
// 这些运算符已经从 IR 中移除，并被编译后的 Python 函数代码替代。
// 然而，为了保持 ONNX 转换的行为，我们将这些函数调用替换为仍然可以被
// ONNX 转换器使用的 aten 符号。
void ONNXFunctionCallSubstitution(Graph& graph) {
  // 输出函数调用替换前的图的状态
  GRAPH_DUMP("Before function call substitution calls: ", &graph);
  // 设置当前作用域为顶层作用域，并在作用域退出时恢复原来的作用域
  WithCurrentScope top_level_scope_guard(graph, ONNXGraphTopLevelScope(graph));
  // 替换函数调用为符号化的 aten 运算符
  functionCallSubstitution(graph.block());
  // 输出函数调用替换后的图的状态
  GRAPH_DUMP("After function call substitution calls: ", &graph);
}

} // namespace jit
} // namespace torch
```