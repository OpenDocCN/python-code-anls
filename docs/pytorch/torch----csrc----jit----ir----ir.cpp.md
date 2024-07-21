# `.\pytorch\torch\csrc\jit\ir\ir.cpp`

```
// 包含了用于 JIT 编译器 IR 的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含了 ATen 库的内建函数相关头文件
#include <ATen/core/builtin_function.h>

// 包含了 ATen 库的函数相关头文件
#include <ATen/core/function.h>

// 包含了 C10 实用工具中的异常处理头文件
#include <c10/util/Exception.h>

// 包含了 C10 实用工具中的字符串处理工具头文件
#include <c10/util/StringUtil.h>

// 包含了 C10 实用工具中的迭代器范围处理头文件
#include <c10/util/irange.h>

// 包含了用于 JIT 编译器的函数实现头文件
#include <torch/csrc/jit/api/function_impl.h>

// 包含了用于 JIT 编译器前端错误报告的头文件
#include <torch/csrc/jit/frontend/error_report.h>

// 包含了用于 JIT 编译器前端模式匹配的头文件
#include <torch/csrc/jit/frontend/schema_matching.h>

// 包含了用于 JIT 编译器 IR 常量的头文件
#include <torch/csrc/jit/ir/constants.h>

// 包含了用于 JIT 运行时操作符的头文件
#include <torch/csrc/jit/runtime/operator.h>

// 包含了用于 JIT 编译器 Python 打印的头文件
#include <torch/csrc/jit/serialization/python_print.h>

// 包含了标准算法库的头文件
#include <algorithm>

// 包含了标准输入输出流的头文件
#include <iostream>

// 包含了标准本地化库的头文件
#include <locale>

// 包含了标准内存管理库的头文件
#include <memory>

// 包含了标准集合库的头文件
#include <set>

// 包含了标准字符串流库的头文件
#include <sstream>

// 包含了标准字符串处理库的头文件
#include <string>

// 包含了标准无序映射库的头文件
#include <unordered_map>

// 包含了标准无序集合库的头文件
#include <unordered_set>

// 包含了标准实用工具库的头文件
#include <utility>

// 定义了 torch::jit 命名空间
namespace torch::jit {

// 定义了 utils 命名空间
namespace utils {

// 返回给定节点的模块层次结构字符串
std::string getNodesModuleHierarchy(const Node& n) {
  // 如果节点没有调用堆栈信息，返回空字符串
  if (!n.callstack().has_value()) {
    return std::string();
  }
  // 获取节点的调用堆栈指针
  InlinedCallStackPtr callstack_ptr = n.callstack().value();
  std::string module_hierarchy;
  // 遍历调用堆栈条目
  for (auto& entry : callstack_ptr->vec()) {
    // 获取模块实例信息
    const auto& opt_module_info = std::get<kModuleInstanceInfo>(entry);
    if (opt_module_info.has_value()) {
      const auto& module_instance_info = opt_module_info.value();
      // 如果模块层次结构非空，则添加点号
      if (!module_hierarchy.empty()) {
        module_hierarchy.append(".");
      }
      // 获取模块信息并添加到模块层次结构中
      module_hierarchy.append(utils::get_module_info(module_instance_info));
    } else {
      // 如果没有模块信息，添加默认的未知实例和未知类型
      module_hierarchy += ".UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
    }
  }
  return module_hierarchy;
}

} // namespace utils

// 定义了匿名命名空间
namespace {

// 与节点的拓扑索引维护相关的常量
//
// 索引的下界和上界，包含的范围
constexpr topo_position_t kLowerBound = INT64_MIN;
constexpr topo_position_t kUpperBound = INT64_MAX;
constexpr topo_position_t kMidPoint = 0;

// 添加到图中的节点之间的间隔距离
// 应为 2^n，其中：
//   - n 是最大的重复插入次数而不重新索引
//   - 2^(64-n) 是在不重新索引的情况下可以添加到末尾的最大数量
constexpr topo_position_t kAppendInterval = 1099511627776ULL /* 2^40 */;

// 打印值引用到输出流中
void printValueRef(std::ostream& out, const Value* n) {
  out << "%" << n->debugName();
}

// 判断字符串是否为数字
bool isNumber(c10::string_view str) {
  return str.find_first_not_of("0123456789") == std::string::npos;
}

// 标准化属性名称
std::string normalizeAttrName(c10::string_view field) {
  if (isNumber(field)) {
    return "_" + std::string{field};
  }
  return std::string{field};
}

// 查找所有符合指定类型的节点
void findAllNodes(
    Block& block,
    Symbol kind,
    bool recurse,
    std::vector<Node*>& ret) {
  // 遍历块中的所有节点
  for (Node* n : block.nodes()) {
    // 如果节点的类型与指定类型相同，将其添加到结果向量中
    if (n->kind() == kind) {
      ret.push_back(n);
    }
    // 如果递归标志为真，则继续查找子块中的节点
    if (recurse) {
      for (auto b : n->blocks()) {
        findAllNodes(*b, kind, recurse, ret);
      }
    }
  }
}

} // namespace

// 注意：这个重载会在其与 Caffe2 提供的日志系统中的重载发生冲突，如果它们有交集的话。
template <typename T>
// 重载流插入运算符 << ，用于将 std::vector<T> 输出到流中
std::ostream& operator<<(std::ostream& out, const std::vector<T>& nodes) {
  // 调用 ArrayRef<T> 的输出运算符来输出 nodes
  out << at::ArrayRef<T>{nodes};
  return out;
}

// 打印 ArrayRef<T> 的值到流中
template <typename T>
static std::ostream& printValueRefs(
    std::ostream& out,
    const at::ArrayRef<T> nodes) {
  size_t i = 0;
  // 遍历 nodes 中的元素
  for (auto n : nodes) {
    if (i++ > 0) {
      out << ", "; // 每个元素之间以逗号分隔
    }
    // 调用 printValueRef 函数打印单个元素 n 到流中
    printValueRef(out, n);
  }
  return out;
}

// 重载流插入运算符 << ，用于将 ArrayRef<const Value*> 输出到流中
// 这里不能将两个重载直接设为模板，因为会与全局的 operator<< 产生歧义
static std::ostream& operator<<(
    std::ostream& out,
    const at::ArrayRef<const Value*> nodes) {
  return printValueRefs(out, nodes); // 调用 printValueRefs 函数输出 nodes 到流中
}

// 结构体 const_value_list_with_types 的输出流插入运算符重载
static std::ostream& operator<<(
    std::ostream& out,
    const const_value_list_with_types& l) {
  size_t i = 0;
  // 遍历结构体中的 values 成员
  for (auto n : l.values) {
    if (i++ > 0) {
      out << l.delim; // 如果不是第一个元素，输出分隔符 delim
    }
    // 调用 printValueRef 函数输出 values 中的每个元素到流中
    printValueRef(out, n);
    // 如果类型详细度大于等于 TypeVerbosity::Type，则输出类型信息
    if (c10::type_verbosity() >= c10::TypeVerbosity::Type) {
      out << " : ";
      out << *n->type(); // 输出元素的类型信息
    }
  }
  return out;
}

// 打印 Tensor 类型的属性到流中
static void printAttribute(std::ostream& out, const at::Tensor& tensor) {
  // 对于只有一个元素的 Tensor，以特定格式输出
  if (tensor.numel() == 1) {
    auto scalar_tensor = tensor.view(std::vector<int64_t>{}).item();
    out << "{";
    if (scalar_tensor.isFloatingPoint()) {
      out << scalar_tensor.toDouble(); // 输出浮点数值
    } else if (scalar_tensor.isComplex()) {
      out << scalar_tensor.toComplexDouble(); // 输出复数值
    } else {
      out << scalar_tensor.toLong(); // 输出整数值
    }
    out << "}";
  } else if (tensor.numel() <= max_tensor_display_size) {
    // 对于元素数量小于等于最大显示尺寸的 Tensor，以字符串形式输出（移除换行符）
    std::ostringstream tensor_ss;
    tensor_ss << tensor;
    std::string tensor_s{tensor_ss.str()};
    std::replace(tensor_s.begin(), tensor_s.end(), '\n', ' ');
    out << tensor_s;
  } else {
    out << "<Tensor>"; // 对于超过最大显示尺寸的 Tensor，以占位符形式输出
  }
}

// 打印 IValue 类型的属性到流中
static void printAttribute(std::ostream& out, const IValue& ival) {
  // 自定义格式化函数 customFormatter
  const auto customFormatter = [](std::ostream& ss, const IValue& input) {
    if (input.isTensor()) {
      printAttribute(ss, input.toTensor()); // 如果是 Tensor 类型，调用 printAttribute 打印
      return true;
    } else if (input.isTensorList()) {
      ss << "[<Tensors>]"; // 如果是 Tensor 列表，以特定格式输出
      return true;
    } else if (input.isObject() && !input.type()->is_module()) {
      ss << "object(" << &input.toObjectRef() << ")"; // 如果是非模块对象，以特定格式输出
      return true;
    }
    return false;
  };
  ival.repr(out, customFormatter); // 调用 IValue 的 repr 方法，使用自定义格式化函数输出到流中
}

// 打印 TypePtr 类型的列表到流中
static void printTypeList(
    std::ostream& out,
    const std::vector<TypePtr>& items) {
  out << "["; // 输出列表起始符号
  int i = 0;
  // 遍历 items 中的每个 TypePtr 元素
  for (auto& item : items) {
    if (i++ > 0)
      out << ", "; // 每个元素之间以逗号分隔
    out << *item; // 输出 TypePtr 所指向对象的描述信息
  }
  out << "]"; // 输出列表结束符号
}

// 打印 Node 的属性值到流中
void Node::printAttrValue(std::ostream& out, const Symbol& name) const {
  switch (kindOf(name)) {
    # 根据不同的属性类型进行处理
    case AttributeKind::c:
      # 对于 'c' 类型的属性，调用 c(name) 函数并打印其结果
      printAttribute(out, c(name));
      break;
    case AttributeKind::cs:
      # 对于 'cs' 类型的属性，打印错误消息并断言失败，需要修复
      // TODO(@anjali411): fix this
      AT_ASSERT(false);
      break;
    case AttributeKind::f:
      # 对于 'f' 类型的属性，调用 f(name) 函数并打印其结果
      printAttribute(out, f(name));
      break;
    case AttributeKind::fs:
      # 对于 'fs' 类型的属性，调用 fs(name) 函数并打印其结果
      printAttribute(out, fs(name));
      break;
    case AttributeKind::i:
      # 对于 'i' 类型的属性，调用 i(name) 函数并打印其结果
      printAttribute(out, i(name));
      break;
    case AttributeKind::is:
      # 对于 'is' 类型的属性，调用 is(name) 函数并打印其结果
      printAttribute(out, is(name));
      break;
    case AttributeKind::s:
      # 对于 's' 类型的属性，调用 s(name) 函数并打印其结果
      printAttribute(out, s(name));
      break;
    case AttributeKind::ss:
      # 对于 'ss' 类型的属性，调用 ss(name) 函数并打印其结果
      printAttribute(out, ss(name));
      break;
    case AttributeKind::t:
      # 对于 't' 类型的属性，调用 t(name) 函数并打印其结果
      printAttribute(out, t(name));
      break;
    case AttributeKind::ts:
      # 对于 'ts' 类型的属性，直接打印 "[<Tensors>]"
      out << "[<Tensors>]";
      break;
    case AttributeKind::ival:
      # 对于 'ival' 类型的属性，调用 ival(name) 函数并打印其结果
      printAttribute(out, ival(name));
      break;
    case AttributeKind::g:
      # 对于 'g' 类型的属性，直接打印 "<Graph>"
      out << "<Graph>";
      break;
    case AttributeKind::gs:
      # 对于 'gs' 类型的属性，直接打印 "[<Graphs>]"
      out << "[<Graphs>]";
      break;
    case AttributeKind::ty:
      # 对于 'ty' 类型的属性，通过指针打印其指向的类型信息
      out << *ty(name);
      break;
    case AttributeKind::tys:
      # 对于 'tys' 类型的属性，调用 printTypeList 函数打印类型列表
      printTypeList(out, tys(name));
      break;
}

// 在输出流中打印节点的属性列表，可选择忽略子图属性
void Node::printAttributes(std::ostream& out, bool ignore_subgraph = false) const {
  // 输出属性列表的起始标记
  out << "[";
  // 获取节点的所有属性名称
  auto names = attributeNames();
  int i = 0;
  // 遍历属性名称列表
  for (auto name : names) {
    // 如果忽略子图属性且当前属性为子图属性，则跳过
    if (ignore_subgraph && name == attr::Subgraph) {
      continue;
    }
    // 如果不是第一个属性，输出逗号分隔符
    if (i++ > 0) {
      out << ", ";
    }
    // 输出属性名称（不包括命名空间），并附加等号
    out << name.toUnqualString() << "=";
    // 输出属性值
    printAttrValue(out, name);
  }
  // 输出属性列表的结束标记
  out << "]";
}

// 返回节点的源代码范围
SourceRange Node::sourceRange() const {
  // 如果存在源代码范围，则返回它
  if (source_range_) {
    return *source_range_;
  }
  // 否则返回空的源代码范围对象
  return SourceRange();
}

// 在输出流中添加缩进，用于打印层级结构
static std::ostream& indent(std::ostream& out, size_t level) {
  // 根据给定的层级数，输出对应数量的缩进空格
  for (const auto i : c10::irange(level)) {
    (void)i; // 抑制未使用变量警告
    out << "  ";
  }
  return out;
}

// 在输出流中打印节点的详细信息
std::ostream& Node::print(
    std::ostream& out,
    size_t level,
    std::vector<const Node*>* groups,
    bool print_source_locations,
    bool print_attributes,
    bool print_scopes,
    bool print_body) const {
  // 获取节点的输出列表
  auto outs = outputs();
  // 打印常量值列表与类型信息
  indent(out, level) << const_value_list_with_types(outs);
  // 输出等号，表示节点的计算结果
  out << " = ";
  
  // 根据节点类型进行不同的打印处理
  if (kind() == prim::PythonOp) {
    // 如果节点类型是 PythonOp
    auto* pyOp = static_cast<const ::torch::jit::PythonOp*>(this);
    // 输出 Python 操作符的名称
    out << "^" << pyOp->name();
    // 打印节点的属性列表
    printAttributes(out, /*ignore_subgraph=*/false);
    // 输出 Python 操作符的标量值
    pyOp->writeScalars(out);
  } else if (hasAttribute(attr::Subgraph) && groups) {
    // 如果节点包含子图属性且传入了节点组向量
    out << kind().toQualString() << "_" << groups->size();
    // 如果需要打印属性且属性数量大于1，并且节点类型不是 DifferentiableGraph
    if (print_attributes && numAttributes() > 1 &&
        kind() != prim::DifferentiableGraph) {
      // 打印节点的属性列表，忽略子图属性
      printAttributes(out, /*ignore_subgraph=*/true);
    }
    // 将当前节点添加到节点组向量中
    groups->push_back(this);
  } else {
    // 对于其他节点类型
    out << kind().toQualString();
    // 如果需要打印属性且节点有属性，则打印属性列表
    if (print_attributes && hasAttributes()) {
      printAttributes(out);
    }
  }
  // 输出节点的输入列表
  out << "(" << inputs() << ")";

  // 如果需要打印作用域信息
  if (print_scopes) {
    // 获取节点的作用域名称
    std::string scName = scopeName();
    // 如果作用域名称不为空，输出作用域信息
    if (!scName.empty()) {
      out << ", ";
      out << "scope: " << scName;
    }
  }

  // 在调试打印模式下，将文件:行:列作为注释附加到每个节点后面
  if (print_source_locations) {
    // 获取节点的源代码范围
    SourceRange r = sourceRange();
    // 如果存在源代码位置
    if (sourceRange().source()) {
      // 查找生成源代码范围的原始源代码位置
      if (auto orig = sourceRange().source()->findSourceRangeThatGenerated(r)) {
        r = *orig;
      }
    }
    // 如果有文件:行:列信息，则输出在注释中附加到节点后面
    if (auto file_line_col = r.file_line_col()) {
      auto [filename, line, col] = *file_line_col;
      out << " # " << filename << ":" << line << ":" << col;
    }
  }

  // 如果不需要打印节点体，则直接返回输出流
  if (!print_body) {
    return out;
  }

  // 输出换行符，表示节点体的开始
  out << "\n";

  // 遍历节点的所有块
  for (const auto i : c10::irange(blocks().size())) {
    auto b = blocks()[i];
    // 输出块的标识符和输入列表
    indent(out, level + 1) << "block" << i << "("
                           << const_value_list_with_types(b->inputs())
                           << "):\n";
    // 递归打印块中的每个嵌套节点
    for (auto nested : b->nodes()) {
      nested->print(out, level + 2, groups);
    }
  }
    indent(out, level + 2) << "-> (" << b->outputs() << ")\n";

将输出流 `out` 进行缩进，缩进级别增加2，并输出箭头和括号，后接调用 `b->outputs()` 返回的结果，并在末尾添加换行符。


  }

结束当前的函数或代码块，此处可能是前面的循环、条件语句或函数定义的结束。
}

// 重载流操作符 << ，用于打印 Node 对象
std::ostream& operator<<(std::ostream& out, const Node& n) {
  return n.print(out, 0, nullptr);
}

// 打印 Graph 对象到输出流中，可选择打印源位置信息
std::ostream& Graph::print(std::ostream& out, bool print_source_locations)
    const {
  // 打印图的基本信息，包括输入信息
  out << "graph(" << const_value_list_with_types(inputs(), ",\n      ")
      << "):\n";
  std::vector<const Node*> groups;
  // 遍历所有节点，打印每个节点的信息
  for (auto n : nodes()) {
    n->print(out, 1, &groups, print_source_locations);
  }
  // 打印输出节点信息
  out << "  return (" << outputs() << ")\n";
  size_t i = 0;
  // 打印节点组信息
  for (auto fg : groups) {
    out << "with " << fg->kind().toQualString() << "_" << i++ << " = "
        << *fg->g(attr::Subgraph);
  }
  out.flush();

  /*
  // Uncomment this to debug all_nodes issues
  // 解除注释用于调试 all_nodes 问题
  {
    out << "\n";
    out << "all_nodes:\n";
    for (auto& n : all_nodes) {
      printNode(out, const_cast<Node*>(n), nullptr);
    }
  }
  */
  return out;
}

// 重载流操作符 << ，用于打印 Graph 对象
std::ostream& operator<<(std::ostream& out, const Graph& g) {
  return g.print(out, true);
}

// 检查节点输入输出的设备是否相同
static void checkSameDevice(const Node* node) {
  bool has_device = false;
  std::optional<at::Device> device = c10::nullopt;
  auto checkValue = [&](const Value* v) {
    if (TensorTypePtr type = v->type()->cast<TensorType>()) {
      if (type->device() && !has_device) {
        has_device = true;
        device = *type->device();
      } else {
        AT_ASSERT(device == type->device());
      }
    }
  };
  // 检查节点的输入设备
  for (auto input : node->inputs()) {
    checkValue(input);
  }
  // 检查节点的输出设备
  for (auto output : node->outputs()) {
    checkValue(output);
  }
}

// 定义节点集合类型
using node_set = std::set<const Node*>;

// 定义容器所有元素的起始和结束迭代器
#define ALL_OF(container) container.begin(), container.end()

// 这些函数有意直接操作内部成员，强迫你思考数据表示改变时不变性如何变化（即使外部 API 不变）

// 注意：此断言假设没有未连接的节点。在图的操作期间可能会出现未连接的节点。
void Node::lint() const {
  // 节点不变性检查
  // - 如果节点应该存储在列表中，nodes_iter 应保持一致
  // - 输入都被其引用的节点标记为使用
  // - 拥有的图对象非空且一致
  // - "Select" 不变性，当节点是 MultiReturn 时

  {
    size_t i = 0;
    for (auto input : inputs_) {
      // 警告：O(n^2) 的时间复杂度
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      AT_ASSERT(
          std::find(ALL_OF(input->uses_), Use(const_cast<Node*>(this), i)) !=
          input->uses_.end());
      AT_ASSERT(graph_->all_nodes.count(this) == 1);
      i++;
    }
  }

  // 检查节点的输出用法
  for (auto o : outputs()) {
    for (auto use : o->uses()) {
      // 使用不变性检查
      // - 使用与输入一致
      // - 每个使用节点都是活跃的（在图中检查）
      AT_ASSERT(use.user->inputs_[use.offset] == o);
    }
  }

  // Node subclass invariants
  switch (kind()) {
    case prim::Constant:
      // 确保常量节点没有输入
      AT_ASSERT(inputs_.empty());
      break;
    case prim::Return:
      // 返回节点没有输出
      AT_ASSERT(outputs().empty());
      break;
    case prim::Param:
      // 参数节点没有输入
      AT_ASSERT(inputs_.empty());
      break;
    case prim::PythonOp: {
      // Python 操作符的调用约定正确
      auto* value = static_cast<const PythonOp*>(this);
      value->lint_python();
      break;
    }
    case prim::Eval:
      // TODO: 添加不变性条件
      // TODO: 这些操作符不应该作为顶层操作，会使得代码情况更复杂。
      break;
    case prim::FusionGroup:
    case prim::CudaFusionGroup:
    case prim::oneDNNFusionGroup:
      // 检查同一设备上的节点
      checkSameDevice(this);
      // TODO: 对参数进行类型检查
      g(attr::Subgraph)->lint();
      break;
  }
}

// TODO: 当 lint 失败时，提供更好的指示，显示触发失败的具体指令。

void Graph::lint() const {
  // Graph invariants

  // Uncomment the following to see the graph
  // std::cout << *const_cast<Graph*>(this);

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique
  // - every use will occur later in the toposort

  // 以下是 lint 方法内部的结构体 LintScope 的定义和实现
  struct LintScope {
    LintScope() = default;
    LintScope(std::unique_ptr<LintScope> parent) : parent(std::move(parent)) {}

    // 检查值 v 是否在当前 lint 作用域内
    bool contains(const Value* v) {
      return values.count(v) > 0 || (parent && parent->contains(v));
    }

    // 检查节点 n 是否在当前 lint 作用域内
    bool contains(const Node* n) {
      return nodes.count(n) > 0 || (parent && parent->contains(n));
    }

    // 将值 v 插入当前 lint 作用域
    void insert(const Value* v) {
      AT_ASSERT(!contains(v));
      values.insert(v);
    }

    // 将节点 n 插入当前 lint 作用域
    void insert(const Node* n) {
      AT_ASSERT(!contains(n));
      nodes.insert(n);
    }

    // 父作用域
    std::unique_ptr<LintScope> parent;

   private:
    std::unordered_set<const Value*> values; // 当前 lint 作用域中的值集合
    std::unordered_set<const Node*> nodes;   // 当前 lint 作用域中的节点集合
  };

  // Struct enables mutual recursion in linting methods.
  // Putting it inside Graph::lint enables access to private Graph members
  // LintImpl 结构体实现 lint 的具体逻辑
  struct LintImpl {
    LintImpl(const Graph& g)
        : g(g),
          scope(new LintScope()),
          all_nodes_set(ALL_OF(g.all_nodes)) {} // NB: all_nodes is *unordered*

    const Graph& g;
    std::unique_ptr<LintScope> scope;           // 当前 lint 的作用域
    std::unordered_set<size_t> seen_uniques;    // 已见的 unique 值集合
    std::unordered_map<const Node*, int64_t> anticipated_uses; // 预期使用次数映射
    node_set all_nodes_set;                     // 所有节点集合
    node_set sum_set;                           // 汇总节点集合

    // 检查值 v 的 lint 规则
    void check_value(const Value* v) {
      scope->insert(v); // 插入值 v 到当前 lint 作用域
      auto b2 = seen_uniques.insert(v->unique());
      AT_ASSERT(b2.second); // 确保插入成功
      AT_ASSERT(v->unique() < g.next_unique_); // 确保 unique 值小于 next_unique_

      // 检查值 v 的每个使用情况
      for (auto use : v->uses()) {
        AT_ASSERT(!scope->contains(use.user)); // 确保使用者不在当前作用域中
        AT_ASSERT(g.all_nodes.count(use.user) == 1); // 确保使用者是有效节点
        anticipated_uses[use.user]++; // 预期使用次数加一
      }
    }

    // 检查节点 n 的 lint 规则
    void check_node(const Node* n) {
      // 检查节点 n 的每个输入值
      for (auto input : n->inputs_) {
        if (!scope->contains(input)) {
          AT_ASSERTM(0, input->unique(), " not in scope"); // 如果输入值不在作用域中，抛出异常
        }
      }
      AT_ASSERT(anticipated_uses[n] == static_cast<int64_t>(n->inputs_.size())); // 确保预期使用次数与实际输入个数相等
      anticipated_uses[n] = -1; // 表示已看到预期的使用节点
      scope->insert(n); // 将节点 n 插入当前 lint 作用域

      // 对节点 n 的每个块进行 lint 检查
      for (auto block : n->blocks()) {
        scope = std::make_unique<LintScope>(std::move(scope)); // 创建新的 lint 作用域
        check_block(block); // 对块进行 lint 检查
        scope = std::move(scope->parent); // 恢复到上一个 lint 作用域
      }

      size_t i = 0;
      // 检查节点 n 的每个输出值
      for (auto o : n->outputs()) {
        AT_ASSERT(o->node() == n); // 确保输出值的节点是 n
        AT_ASSERT(i++ == o->offset_); // 确保输出值的偏移量正确
        check_value(o); // 对输出值进行 lint 检查
      }

      n->lint(); // 对节点 n 进行 lint 检查
    }
    void check_block(const Block* b) {
      // 检查块的拓扑排序
      AT_ASSERT(b->param_node()->isBefore(*b->nodes().begin()));
      // 获取当前节点作为起始节点
      auto curNode = *b->nodes().begin();
      // 遍历直到返回节点
      while (curNode != b->return_node()) {
        AT_ASSERT(curNode->isBefore(curNode->next()));
        curNode = curNode->next();
      }
    
      // 检查块的输入参数
      for (auto input : b->inputs()) {
        check_value(input);
        AT_ASSERT(input->node()->kind_ == prim::Param);
      }
    
      // 检查块中的每个节点
      for (auto n : b->nodes()) {
        AT_ASSERT(n->kind_ != prim::Param);
        AT_ASSERT(n->kind_ != prim::Return);
        check_node(n);
      }
    
      // 检查块的输出节点
      AT_ASSERT(b->output_->kind() == prim::Return);
      check_node(b->output_);
    
      // all_nodes
      // - inputs_, output_ and nodes_ are all included in all_nodes
      // - all_nodes does not contain dead nodes??? (likely to be temporarily
      // suspended).  Weaker: all_nodes contains all inputs and returns
      // - only one return node???
    
      // 将块中的节点集合加入到节点集合中
      node_set nodes_set(ALL_OF(b->nodes()));
      node_set inputs_set{b->input_};
      node_set output_set{b->output_};
      // TODO: Make a more type safe std::includes wrapper which disallows use
      // on non-ordered containers
      // 断言所有节点集合包含当前块的节点集合、输入节点集合和输出节点集合
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(nodes_set)));
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(inputs_set)));
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(output_set)));
    
      // 将节点集合、输入节点集合和输出节点集合加入到总和集合中
      sum_set.insert(ALL_OF(nodes_set));
      sum_set.insert(ALL_OF(inputs_set));
      sum_set.insert(ALL_OF(output_set));
    }
    
    void check_graph() {
      // 将所有节点加入到节点集合中（注意：all_nodes 是无序的）
      node_set all_nodes_set(ALL_OF(g.all_nodes));
    
      // 检查整个图的块
      check_block(g.block_);
      // 检查预期使用情况的键值对，预期使用次数应为 -1
      for (auto kv : anticipated_uses) {
        AT_ASSERT(kv.second == -1);
      }
      // 断言总和集合包含所有节点集合
      AT_ASSERT(std::includes(ALL_OF(sum_set), ALL_OF(all_nodes_set)));
    }
    
    };
    LintImpl(*this).check_graph();
}

// 在标准输出流上打印当前图形对象的内容
void Graph::dump() const {
  std::cout << *this << "\n";
}

// 将给定作用域名称推入当前作用域栈中
void Graph::push_scope(const std::string& scope_name) {
  // 更新当前作用域为新的作用域
  current_scope_ = current_scope_->push(Symbol::scope(scope_name));
  // 创建一个新节点表示跟踪模块前向，并设置其作用域属性为给定的作用域名称
  Node* block_node = insertNode(create(prim::TracedModuleForward, 0));
  block_node->s_(attr::scope, scope_name);
  // 向新节点添加一个新的基本块，并将插入点设置为该块
  Block* b = block_node->addBlock();
  setInsertPoint(b);
}

// 弹出当前作用域
void Graph::pop_scope() {
  // 将当前作用域设为其父作用域
  current_scope_ = current_scope_->parent();
  // 如果当前插入点所属的块的所属节点的类型是 prim::TracedModuleForward，则将插入点移动到下一个节点
  if (insertPoint()->owningBlock()->owningNode()->kind() ==
      prim::TracedModuleForward) {
    setInsertPoint(insertPoint()->owningBlock()->owningNode()->next());
  }
}

// 对给定的图形对象进行静态检查
void LintGraph(const std::shared_ptr<Graph>& graph) {
  graph->lint();
}

// 构造函数：为块对象初始化属性和关联的图形对象
Block::Block(Graph* graph_, Node* node_)
    : graph_(graph_),
      output_(graph_->create(prim::Return, 0)),
      input_(graph_->create(prim::Param, 0)),
      owning_node_(node_) {
  // 设置输入和输出节点的连接关系
  input_->next() = output_;
  input_->prev() = output_;
  output_->next() = input_;
  output_->prev() = input_;

  // 将当前块添加到图形对象的所有块集合中
  graph_->all_blocks.emplace(this);
  output_->owning_block_ = this;
  output_->topo_position_ = kUpperBound;
  input_->owning_block_ = this;
  input_->topo_position_ = kLowerBound;
}

// 重新索引当前块内节点的拓扑顺序
void Block::reIndexTopology() {
  auto curPos = kLowerBound;
  for (auto node : nodes()) {
    AT_ASSERT(curPos <= (kUpperBound - kAppendInterval));
    curPos += kAppendInterval;
    node->topo_position_ = curPos;
  }
}

// 从源块克隆节点到当前块，并根据给定的值映射函数调整节点之间的连接关系
void Block::cloneFrom(Block* src, std::function<Value*(Value*)> value_map) {
  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value* v) {
    auto it = local_map.find(v);
    if (it != local_map.end()) {
      return it->second;
    }
    return value_map(v);
  };

  auto graph = owningGraph();
  // 克隆源块的输入节点到当前块，并复制元数据
  for (auto input : src->inputs()) {
    local_map[input] = this->addInput()->copyMetadata(input);
  }

  // 克隆源块的节点到当前块，并根据映射函数调整输出节点和元数据
  for (auto node : src->nodes()) {
    auto new_node = this->appendNode(graph->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      auto oo = node->outputs()[i];
      auto no = new_node->outputs()[i];
      local_map[oo] = no;
      no->copyMetadata(oo);
    }
  }

  // 注册源块的输出节点到当前块，并根据映射函数调整输出节点
  for (auto output : src->outputs()) {
    this->registerOutput(env(output));
  }
}

// 销毁当前块对象，包括节点和输入输出节点的连接关系
void Block::destroy() {
  // 不能销毁输出节点，因为它作为节点列表的标志节点必须保持有效
  output_->removeAllInputs();
  for (auto it = this->nodes().reverse().begin(),
            end = this->nodes().reverse().end();
       it != end;
       ++it) {
    it.destroyCurrent();
  }
  output_->destroy();
  input_->destroy();
  graph_->freeBlock(this);
}

// 从给定的源图形对象克隆当前图形对象
void Graph::cloneFrom(Graph& src) {
  auto env = [](Value* v) -> Value* {
    AT_ERROR(
        "Graph::copy() encountered a use of a value " + v->debugName() +
        " not in scope. Run lint!");
  };
  block()->cloneFrom(src.block(), env);
}

// 克隆当前图形对象，并返回其共享指针
std::shared_ptr<Graph> Graph::copy() {
  auto new_g = std::make_shared<Graph>();
  new_g->cloneFrom(*this);
  return new_g;
}
// 创建一个指向新图的唯一指针副本
std::unique_ptr<Graph> Graph::copyUnique() {
  // 使用 std::make_unique 创建一个新的 Graph 对象
  auto new_g = std::make_unique<Graph>();
  // 调用 cloneFrom 方法将当前图的内容克隆到新图中
  new_g->cloneFrom(*this);
  // 返回新图的唯一指针
  return new_g;
}

// 重新映射块中值的类型，使用提供的类型映射函数 type_map
void Block::remapTypes(const std::function<TypePtr(TypePtr)>& type_map) {
  // 遍历块中的每个输入值，更新其类型
  for (Value* input : inputs()) {
    input->setType(type_map(input->type()));
  }
  // 遍历块中的每个节点
  for (Node* node : nodes()) {
    // 更新节点的每个输出值的类型
    for (Value* output : node->outputs()) {
      output->setType(type_map(output->type()));
    }
    // 递归调用 remapTypes 方法，更新子块中的类型映射
    for (Block* sub_block : node->blocks()) {
      sub_block->remapTypes(type_map);
    }
    // 遍历节点的每个属性名
    for (Symbol name : node->attributeNames()) {
      // 如果属性是类型为 g 的图对象，则更新其类型映射
      if (node->kindOf(name) == AttributeKind::g) {
        node->g(name)->remapTypes(type_map);
      } 
      // 如果属性是类型为 gs 的图对象集合，则逐个更新其类型映射
      else if (node->kindOf(name) == AttributeKind::gs) {
        for (const auto& g : node->gs(name)) {
          g->remapTypes(type_map);
        }
      }
    }
  }
}

// 使用提供的类型映射函数 type_map 重新映射整个图的类型
void Graph::remapTypes(const std::function<TypePtr(TypePtr)>& type_map) {
  // 调用 block 方法获取图的主块，并对其进行类型重新映射
  block()->remapTypes(type_map);
}

// 从输出张量推断值的类型，并将其设置为对应的张量类型
void Value::inferTypeFrom(const at::Tensor& output) {
  // 使用 TensorType::create 方法根据输出张量创建类型，并将其设置为当前值的类型
  setType(TensorType::create(output));
}

// 从输出对象推断值的类型，并将其设置为对应的对象类型
void Value::inferTypeFrom(
    const c10::intrusive_ptr<c10::ivalue::Object>& output) {
  // 将输出对象的类型设置为当前值的类型
  setType(output->type());
}

// 判断值是否必须为 None
bool Value::mustBeNone() const {
  // 如果当前值的类型是 NoneType 或者所属节点必须是 None，则返回 true，否则返回 false
  return type()->cast<NoneType>() || node_->mustBeNone();
}

// 判断值是否不应为 None
bool Value::mustNotBeNone() const {
  // 如果节点的类型不是 prim::AutogradAdd，并且类型不是 NoneType 或者不是 OptionalType 或者不是 UnionType 且不能包含 NoneType，则返回 true，否则返回 false
  return node_->kind() != prim::AutogradAdd && type() != NoneType::get() &&
      !type()->cast<OptionalType>() &&
      !(type()->cast<UnionType>() &&
        type()->expect<UnionType>()->canHoldType(*NoneType::get()));
}

// 返回调试名称的基本部分
std::string Value::debugNameBase() const {
  // 获取调试名称
  std::string name = debugName();
  std::string name_base = name;
  // 查找最后一个点号的位置
  auto last_dot_pos = name.find_last_of('.');
  // 如果找到点号且不是最后一个字符
  if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
    // 如果点号后面的字符都是数字，则将基本名称设为去掉点号后面部分的名称
    if (name.find_first_not_of("0123456789", last_dot_pos + 1) ==
        std::string::npos) {
      name_base = name.substr(0, last_dot_pos);
    }
  }
  // 返回基本名称
  return name_base;
}

// 检查名称是否有效
bool Value::isValidName(const std::string& name) {
  // 空字符串是合法的名称
  if (name.empty()) {
    return true;
  }

  // 如果名称是数字，则不合法
  if (isNumber(name)) {
    return false;
  }

  // 其它情况都合法
  return true;
}

// 设置调试名称，如果名称无效则抛出异常
Value* Value::setDebugName(const std::string& name) {
  // 如果名称无效，则抛出运行时异常
  if (!isValidName(name)) {
    throw std::runtime_error("Invalid name: '" + name + "'");
  }

  // 获取图的唯一名称集合
  auto& names = node()->owningGraph()->unique_names_;

  // 清除映射表中的旧名称
  if (hasDebugName()) {
    names.erase(unique_name_);
    unique_name_ = "";
  }

  // 如果名称为空，则清除唯一名称
  if (name.empty()) {
    return this;
  }

  // 如果映射表中已经存在该名称，则重命名其它值
  auto old_owner_of_name = names.find(name);
  if (old_owner_of_name != names.end()) {
    size_t suffix = 1;
    std::string name_base = name;
    auto last_dot_pos = name.find_last_of('.');
    // 如果名称中存在点号
    if (last_dot_pos != std::string::npos) {
      // 获取基本名称
      name_base = name.substr(0, last_dot_pos);
    }
    // 为名称添加后缀，直到找到一个可用的唯一名称
    while (names.find(name_base + "." + std::to_string(suffix)) != names.end()) {
      suffix++;
    }
    // 使用新名称设置唯一名称
    unique_name_ = name_base + "." + std::to_string(suffix);
  } else {
    // 使用给定名称设置唯一名称
    unique_name_ = name;
  }

  // 在映射表中添加新的唯一名称
  names.insert({unique_name_, this});
  // 返回当前值的指针
  return this;
}
    // 检查是否存在文件名中的后缀部分，如果存在则提取后缀数字
    if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
      // 检查后缀部分是否只包含数字
      if (name.find_first_not_of("0123456789", last_dot_pos + 1) ==
          std::string::npos) {
        // 将后缀部分转换为长整型数值
        suffix = std::stoll(name.substr(last_dot_pos + 1));
        // 提取文件名的基础部分（去除后缀）
        name_base = name.substr(0, last_dot_pos);
      }
    }

    // 获取名字和后缀的映射表
    auto& names_suffixes = node()->owningGraph()->name_base_suffix_;
    // 查找文件名基础部分在映射表中的条目
    auto it = names_suffixes.find(name_base);
    // 如果找到，则更新后缀为当前后缀和映射表中已有后缀的最大值加一的较大值
    if (it != names_suffixes.end()) {
      suffix = std::max(suffix, it->second + 1);
    }

    // 确保新名称未被使用，并找到下一个可用的名称（如果当前后缀已被使用）
    std::string replacement_name;
    // 循环生成新名称，直到找到未被使用的名称
    do {
      // 使用流将文件名基础部分和更新后的后缀部分组合成新名称
      std::stringstream ss;
#ifndef _WIN32
      // 如果不是在 Windows 平台，需要保护 12345 这样的整数避免变成 "1,2345"，
      // 如果其他进程设置了全局语言环境。详细信息请参考：
      // https://github.com/pytorch/pytorch/issues/79583#issuecomment-1161260061
      static std::locale c_locale("C");
      ss.imbue(c_locale);
#endif
      // 使用 stringstream 拼接名称和后缀
      ss << name_base << "." << suffix++;
      // 将拼接好的字符串赋值给 replacement_name
      replacement_name = ss.str();
    } while (names.count(replacement_name) > 0);

    // 将 name_base 对应的 suffix 存入 names_suffixes 容器中
    names_suffixes[name_base] = suffix;

    // 更新 old_owner_of_name 的 debug 名称为 replacement_name
    old_owner_of_name->second->setDebugName(replacement_name);
  }

  // 将当前对象添加到 names 容器中，使用 name 作为键
  names[name] = this;
  // 设置当前对象的唯一名称为 name
  unique_name_ = name;
  // 返回当前对象的指针
  return this;
}

// 复制传入值的元数据到当前对象
Value* Value::copyMetadata(Value* from) {
  // 设置当前对象的类型与传入值相同
  setType(from->type());
  // 如果传入值有调试名称，则将其设置为当前对象的调试名称
  if (from->hasDebugName()) {
    setDebugName(from->debugName());
  }
  // 返回当前对象的指针
  return this;
}

// 替换第一个使用当前值的地方为新值
void Value::replaceFirstUseWith(Value* newValue) {
  // 断言当前值和新值都属于同一个计算图
  AT_ASSERT(owningGraph() == newValue->owningGraph());
  // 获取当前值的第一个使用方式
  auto u = uses()[0];
  // 替换使用当前值的用户的输入为新值
  u.user->inputs_[u.offset] = newValue;
  // 将新值添加到新的使用列表中
  newValue->uses_.push_back(u);
  // 移除当前值的第一个使用方式
  uses_.erase(uses_.begin());
}

// 替换所有使用当前值的地方为新值
void Value::replaceAllUsesWith(Value* newValue) {
  // 循环直到所有使用方式都被替换完毕
  while (!uses().empty()) {
    replaceFirstUseWith(newValue);
  }
}

// 替换所有在特定节点之后使用当前值的地方为新值
void Value::replaceAllUsesAfterNodeWith(const Node* node, Value* newValue) {
  // 遍历当前值的所有使用方式
  std::for_each(uses_.begin(), uses_.end(), [&node, newValue](Use& u) {
    // 如果使用方式的节点在指定节点之后，则替换为新值
    if (u.user->isAfter(node)) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
  });

  // 移除所有在指定节点之后的使用方式
  uses_.erase(
      std::remove_if(
          uses_.begin(),
          uses_.end(),
          [&node](const Use& u) { return u.user->isAfter(node); }),
      uses_.end());
}

// 替换所有在特定节点支配范围内使用当前值的地方为新值
void Value::replaceAllUsesDominatedByNodeWith(
    const Node* node,
    Value* newValue) {
  // 遍历当前值的所有使用方式
  std::for_each(uses_.begin(), uses_.end(), [&node, newValue](Use& u) {
    // 如果使用方式的节点在指定节点支配范围内，则替换为新值
    if (u.user->isDominatedBy(node)) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
  });

  // 移除所有在指定节点支配范围内的使用方式
  uses_.erase(
      std::remove_if(
          uses_.begin(),
          uses_.end(),
          [&node](const Use& u) { return u.user->isDominatedBy(node); }),
      uses_.end());
}

// 在函数模式中查找特定名称的参数位置
static size_t findArgument(
    const FunctionSchema& the_schema,
    const std::string& unqualName) {
  // 遍历函数模式中的所有参数
  for (const auto i : c10::irange(the_schema.arguments().size())) {
    const Argument* arg = &the_schema.arguments()[i];
    // 如果找到参数名称与指定名称相同，则返回其位置
    if (arg->name() == unqualName) {
      return i;
    }
  }
  // 如果未找到指定名称的参数，则抛出运行时异常
  throw std::runtime_error(
      std::string("Couldn't find an argument called ") + unqualName);
}

// 在函数模式中查找特定符号名称的参数位置
static size_t findArgument(const FunctionSchema& the_schema, Symbol name) {
  // 将符号名称转换为未限定字符串形式，再调用上述函数查找
  const auto unqualName = name.toUnqualString();
  return findArgument(the_schema, unqualName);
}

// 获取节点中指定符号名称的输入值
std::optional<IValue> Node::get(Symbol name) const {
  return toIValue(namedInput(name));
}

// 检查节点是否具有特定名称的命名输入
bool Node::hasNamedInput(const std::string& name) const {
  // 遍历节点的函数模式中的所有参数
  for (const auto& argument : schema().arguments()) {
    // 如果找到名称匹配的参数，则返回 true
    if (argument.name() == name) {
      return true;
    }
  }
  // 如果未找到匹配的参数名称，则返回 false
  return false;
}

// 获取节点中指定未限定名称的命名输入值
Value* Node::namedInput(const std::string& unqualName) const {
  // 使用先前定义的函数查找并返回指定名称的输入值
  return input(findArgument(schema(), unqualName));
}
// 返回特定名称的输入值，查找并返回与给定名称匹配的输入值
Value* Node::namedInput(Symbol name) const {
  // 调用 findArgument 函数查找对应名称的参数，并返回其输入值
  return input(findArgument(schema(), name));
}

// 检查节点是否与给定函数模式匹配
bool Node::matches(const FunctionSchema& schema) const {
  // 如果函数模式被列为不可用，则返回 false
  if (isBlockListedSchema(schema)) {
    return false;
  }
  // 如果节点的类型名称与函数模式的名称不匹配，则返回 false
  if (kind().toQualString() != schema.name()) {
    return false;
  }
  // 获取节点的实际输入值列表和函数模式的形式参数列表
  at::ArrayRef<const Value*> actuals = inputs();
  const auto& formals = schema.arguments();

  // 如果节点的实际输入值数量少于函数模式的形式参数数量，则返回 false
  if (actuals.size() < formals.size()) {
    return false;
  }

  // 创建类型环境对象
  TypeEnv type_env;
  // 遍历形式参数列表，并逐一匹配类型变量
  for (const auto i : c10::irange(formals.size())) {
    auto formal = formals[i].type();
    // 调用 matchTypeVariables 函数尝试匹配类型变量
    const MatchTypeReturn matched_type =
        matchTypeVariables(formal, actuals[i]->type(), type_env);
    // 如果匹配失败，则返回 false
    if (!matched_type.success()) {
      return false;
    }

    // 尝试解析类型变量并更新形式参数的类型
    TypePtr resolved = tryEvalTypeVariables(formal, type_env);
    if (resolved) {
      formal = resolved;
    }

    // 检查实际输入值的类型是否是形式参数类型的子类型
    if (!actuals[i]->type()->isSubtypeOf(*formal)) {
      return false;
    }
  }

  // 如果函数模式不是可变参数且实际输入值数量与形式参数数量不匹配，则返回 false
  if (!schema.is_vararg() && actuals.size() != formals.size()) {
    return false;
  }

  // 若所有条件都满足，则返回 true，表示节点与函数模式匹配
  return true;
}

// 检查节点是否与特定签名字面量及常量输入符号列表匹配
bool Node::matches(
    const char* signature_literal,
    at::ArrayRef<Symbol> const_inputs) const {
  // 获取字面量对应的操作符并检查节点是否与其模式匹配
  if (!matches(getOperatorForLiteral(signature_literal)->schema())) {
    return false;
  }
  // 检查所有的常量输入符号列表，确保它们都是常量
  for (Symbol s : const_inputs) {
    if (!is_constant(s)) {
      return false;
    }
  }
  // 若所有条件都满足，则返回 true，表示节点与指定签名匹配
  return true;
}

// 判断节点是否必须为 None
bool Node::mustBeNone() const {
  // 可以静态推断出节点返回 None 的情况包括：AutogradZero 节点、只有一个输出且输出类型为 NoneType、或者是一个常量可选类型并且没有值
  return
      kind_ == prim::AutogradZero ||
      (outputs().size() == 1 && output()->type() == NoneType::get()) ||
      (kind_ == prim::Constant && !this->hasAttributes() &&
       output()->type()->cast<OptionalType>());
}

// 打印节点信息
void Node::dump() const {
  std::cout << *this << "\n";
}

// 获取节点的函数模式
const FunctionSchema& Node::schema() const {
  // 如果操作符已存在，则返回其函数模式
  if (op_) {
    return op_->schema();
  }
  // 否则，获取操作符并返回其函数模式
  return getOperator().schema();
}

// 获取节点的函数模式，若无法获取则返回 nullptr
const FunctionSchema* Node::maybeSchema() const {
  // 如果能够获取操作符，则返回其函数模式指针
  if (auto op = maybeOperator()) {
    return &op->schema();
  }
  // 否则返回 nullptr
  return nullptr;
}

// 尝试获取节点的操作符，如果尚未确定，则根据节点类型获取匹配的操作符
const Operator* Node::maybeOperator() const {
  // 如果操作符尚未确定，则获取所有可能的操作符候选列表
  if (!op_) {
    const auto& candidates = getAllOperatorsFor(kind());
    // 遍历候选列表，找到第一个与节点匹配的操作符，并进行设定
    for (const auto& candidate : candidates) {
      if (matches(candidate->schema())) {
        op_ = candidate.get();
        break;
      }
    }
  }
  // 返回找到的操作符或者 nullptr
  return op_;
}

// 获取节点的操作符，如果操作符存在则返回其指针
const Operator& Node::getOperator() const {
  const Operator* maybe = maybeOperator();
  // 如果能够获取操作符，则返回其指针
  if (maybe)
    return *maybe;
}
    // 返回 maybe 所指向的内容
    return *maybe;

  // 创建一个错误报告，指定错误发生的源范围
  auto er = ErrorReport(sourceRange());
  // 添加关于找不到节点模式的错误信息
  er << "Schema not found for node. File a bug report.\n";
  // 添加关于节点信息的错误信息
  er << "Node: " << *this << "\n";
  // 添加输入类型信息的错误信息
  er << "Input types:";
  // 遍历输入的所有项
  for (const auto i : c10::irange(inputs().size())) {
    // 如果不是第一个输入项，添加逗号分隔符
    if (i > 0)
      er << ", ";
    // 添加当前输入项的类型信息
    er << *inputs()[i]->type();
  }
  // 获取与节点类型相符的所有操作符候选项
  const auto& candidates = getAllOperatorsFor(kind());
  // 如果存在候选操作符
  if (!candidates.empty()) {
    // 添加候选操作符的信息
    er << "\ncandidates were:\n";
    // 遍历所有候选操作符，添加其模式信息
    for (auto& candidate : candidates) {
      er << "  " << candidate->schema() << "\n";
    }
  } else {
    // 添加未找到候选操作符的信息
    er << "\nno candidates found\n";
  }
  // 添加所属图的信息
  er << "within the graph:\n";
  // 添加当前节点所属图的详细信息
  er << *owningGraph() << "\n";
  // 抛出构建好的错误报告
  throw er;
}

// 获取节点的操作信息
Operation Node::getOperation() const {
  // 注意：某些运算符要求节点生成可运行的操作，因此在这里传递 'this'。
  // getOperator() 确保 'this' 符合返回运算符的模式。
  return getOperator().getOperation(this);
}

// 判断节点是否是非确定性的
bool Node::isNondeterministic() const {
  const auto schema = maybeSchema();
  if (!kind().is_aten()) {
    return false;
  }
  // 所有 aten 操作都应该有一个 schema。但这里作为警告而非断言，以确保之前的用例不会出错。
  if (!schema) {
    TORCH_WARN("aten Schema not found.");
    return false;
  }
  torch::utils::SchemaInfo schema_info(*schema);
  if (hasNamedInput("train")) {
    auto value = constant_as<bool>(namedInput("train"));
    if (value.has_value()) {
      schema_info.addArgumentValue("train", *value);
    }
  }
  return schema_info.is_nondeterministic();
}

// 判断节点是否有副作用
bool Node::hasSideEffects() const {
  switch (kind_) {
    case prim::PythonOp:
    case prim::IgnoredPythonOp:
    case prim::Print:
    case prim::RaiseException:
    case aten::warn:
    case aten::save:
    case aten::manual_seed:
    case prim::AddStatValue:
    case prim::TimePoint:
    case prim::CallFunction:
    case prim::CallMethod:
    case prim::BailoutTemplate:
    case prim::BailOut:
    case prim::rpc_async: // 表示发送的 RPC 消息。
    case prim::rpc_sync: // 表示发送的 RPC 消息。
    case prim::rpc_remote: // 表示发送的 RPC 消息。
    case aten::wait: // 可能表示接收的 RPC 消息。
#if !defined(USE_ROCM)
    case cuda::set_stream:
    case cuda::_set_device:
    case cuda::_current_device:
    case cuda::synchronize:
#endif
    case prim::Enter:
    case prim::Exit:
      return true;
  }

  auto op = maybeOperator();
  if (!op) {
    TORCH_INTERNAL_ASSERT(
        kind_.is_prim(),
        "Only prim ops are allowed to not have a registered operator but ",
        kind_.toDisplayString(),
        " doesn't have one either. We don't know if this op has side effects.");
    return false;
  }

  if (kind_.is_prim() || kind_.is_aten() || kind_.is_cuda()) {
    // TODO: 系统中没有依赖于 aten:: 和 prim:: 操作使用 AliasAnalysisKind::FROM_SCHEMA,
    // AliasAnalysisKind::INTERNAL_SPECIAL_CASE 或 AliasAnalysisKind::CONSERVATIVE 的部分，
    // 但这是当前所有操作的预期行为和良好的错误检查。如果以后有用例需要，我们可以考虑放宽这个约束。
    # 对 op 对象进行断言，确保其别名分析类型是 INTERNAL_SPECIAL_CASE、FROM_SCHEMA 或 CONSERVATIVE 中的一种
    TORCH_INTERNAL_ASSERT(
        op->aliasAnalysisKind() == AliasAnalysisKind::INTERNAL_SPECIAL_CASE ||
            op->aliasAnalysisKind() == AliasAnalysisKind::FROM_SCHEMA ||
            op->aliasAnalysisKind() == AliasAnalysisKind::CONSERVATIVE,
        "aten:: and prim:: ops should have AliasAnalysisKind::INTERNAL_SPECIAL_CASE"
        ", AliasAnalysisKind::FROM_SCHEMA or AliasAnalysisKind::CONSERVATIVE but ",
        kind_.toDisplayString(),
        " has ",
        toString(op->aliasAnalysisKind()));
    }
    
    # 根据 op 对象的别名分析类型进行不同的处理
    switch (op->aliasAnalysisKind()) {
      case AliasAnalysisKind::PURE_FUNCTION:
      case AliasAnalysisKind::FROM_SCHEMA:
      case AliasAnalysisKind::INTERNAL_SPECIAL_CASE:
        # 如果别名分析类型是 PURE_FUNCTION、FROM_SCHEMA 或 INTERNAL_SPECIAL_CASE，则返回 false
        return false;
      case AliasAnalysisKind::CONSERVATIVE:
        # 如果别名分析类型是 CONSERVATIVE，则返回 true
        return true;
    }
    
    # 如果出现了未处理的别名分析类型，则触发断言错误
    TORCH_INTERNAL_ASSERT(false, "Unhandled AliasAnalysisKind case");
    # 默认返回 false，用于消除编译器的警告
    return false;
}

// Assign this node a topological position, to facilitate fast isBefore() and
// isAfter() queries. Must be called right after a node is inserted into the
// node list.
//
// The basic scheme is: assign every node a position (uint64_t). The common
// case (appending to the end of the graph) is made more efficient by advancing
// a fixed interval past the previous node and placing `this` there. Otherwise,
// assign `this` a position at the midpoint between its prev() and next()
// nodes.
//
// If we ever run out of space (by, e.g. inserting too much in place), we
// reindex by spreading out all the nodes again.
void Node::assignTopoPosition() {
  // Check if this node is the first in the list
  bool is_first = prev() == owningBlock()->param_node();
  // Check if this node is the last in the list
  bool is_last = next() == owningBlock()->return_node();

  // Get positions of the previous and next nodes
  const auto prevPos = prev()->topo_position_;
  const auto nextPos = next()->topo_position_;

  // Append to the end of the graph
  if (is_last) {
    if (is_first) {
      // If the node list is empty, assign the first position
      topo_position_ = kMidPoint;
      return;
    }

    // Check if appending would exceed the upper bound
    if (prevPos >= (kUpperBound - kAppendInterval)) {
      // Reindex all nodes if we're running off the edge
      owningBlock()->reIndexTopology();
      return;
    }

    // Assign position at a fixed interval past the previous node
    topo_position_ = prevPos + kAppendInterval;

  // Prepend to the graph
  } else if (is_first) {
    // If next() is the first element in the block list
    if (nextPos <= (kLowerBound + kAppendInterval)) {
      // Reindex all nodes if we're running off the edge
      owningBlock()->reIndexTopology();
      return;
    }

    // Assign position at a fixed interval before the next node
    topo_position_ = nextPos - kAppendInterval;

  // Insert between two existing nodes
  } else {
    // Calculate remaining space between prev() and next()
    int64_t remaining = nextPos - prevPos;
    AT_ASSERT(remaining > 0);

    // If no room is available for insertion
    if (remaining == 1) {
      owningBlock()->reIndexTopology();
      return;
    }

    // Predict future insertions and adjust position accordingly
    int64_t predicted_future_insertions = 0;
    if (next() == graph_->insertPoint()) {
      predicted_future_insertions = graph_->predicted_insert_count_++;
    }

    // Assign position at midpoint between prev() and next()
    topo_position_ = prevPos + std::max(int64_t(1), remaining / (2 + predicted_future_insertions));
    AT_ASSERT(prevPos < topo_position_ && topo_position_ < nextPos);
  }
}

// Constructor for Node class, initializes attributes
Node::Node(Graph* graph_, NodeKind kind_)
    : kind_(kind_),
      graph_(graph_),
      owning_block_(nullptr),
      scope_(graph_->current_scope_),
      callstack_(c10::nullopt),
      op_(nullptr),
      topo_position_(0) {
  // Add this node to the set of all nodes in the graph
  graph_->all_nodes.emplace(this);
}

// Erases an output at index i from the node
void Node::eraseOutput(size_t i) {
  AT_ASSERT(i < outputs_.size());
  AT_ASSERT(outputs_[i]->uses().empty());
  op_ = nullptr;
  Value* n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  // Update offsets of subsequent outputs
  for (const auto j : c10::irange(i, outputs_.size())) {
    outputs_[j]->offset_--;
  }
}

// Adds a new block to the node and returns it
Block* Node::addBlock() {
  op_ = nullptr;
  blocks_.push_back(new Block(owningGraph(), this));
  return blocks_.back();
}
// 检查索引是否在有效范围内，确保要删除的块索引小于当前块列表的大小
void Node::eraseBlock(size_t i) {
  AT_ASSERT(i < blocks_.size());
  // 将操作符指针设置为 nullptr
  op_ = nullptr;
  // 获取要删除的块对象，并从块列表中删除该块
  Block* n = blocks_[i];
  blocks_.erase(blocks_.begin() + i);
  // 销毁被删除的块对象
  n->destroy();
}

// 销毁节点的所有输出和输入，并从所属图中释放节点
void Node::destroy() {
  // 循环直到节点的输出列表为空，逐一删除最后一个输出
  while (!outputs().empty()) {
    eraseOutput(outputs().size() - 1);
  }
  // 循环直到节点的块列表为空，逐一删除最后一个块
  while (!blocks().empty()) {
    eraseBlock(blocks().size() - 1);
  }
  // 删除节点所有的输入
  removeAllInputs();
  // 如果节点在块列表中，从列表中移除节点
  if (inBlockList()) {
    removeFromList();
  }
  // 释放节点所属的图中的节点资源
  graph_->freeNode(this);
}

// 从另一个节点复制源码范围、作用域、属性和调用栈信息到当前节点
void Node::cloneFrom(Node* s) {
  // 复制源码范围信息
  source_range_ = s->source_range_;
  // 如果源节点具有作用域并且作用域不为空，则复制作用域信息
  if (s->scope_ && !s->scope_->isBlank()) {
    scope_ = s->scope_;
  }
  // 复制节点的所有属性
  copyAttributes(*s);
  // 复制调用栈信息
  callstack_ = s->callstack_;
}

// 用另一个节点替换当前节点的所有输出，并确保输出数目相同
void Node::replaceAllUsesWith(Node* n) {
  // 断言当前节点的输出数目与替换节点的输出数目相同
  AT_ASSERT(outputs().size() == n->outputs().size());
  size_t nOutputs = outputs().size();
  // 逐个替换当前节点的每个输出的使用
  for (const auto i : c10::irange(nOutputs)) {
    outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
  }
}

// 使用新的符号替换当前节点，并返回替换后的节点
Node* Node::replaceWithNewSymbol(Symbol new_symbol) {
  // 设置插入点为当前节点
  WithInsertPoint insert_guard{this};
  // 检查当前节点是否具有操作符
  bool had_operator = maybeOperator() != nullptr;
  auto graph = owningGraph();
  // 在所属图中插入一个新节点，并使用给定符号创建节点
  auto replace_node = graph->insertNode(graph->create(new_symbol, 0));
  // 复制当前节点的所有输入到新节点
  for (Value* v : inputs()) {
    replace_node->addInput(v);
  }
  // 复制当前节点的所有输出到新节点，并更新使用这些输出的节点
  for (Value* v : outputs()) {
    auto new_out = replace_node->addOutput()->copyMetadata(v);
    v->replaceAllUsesWith(new_out);
  }
  // 复制当前节点的元数据到新节点
  replace_node->copyMetadata(this);
  // 复制当前节点的所有属性到新节点
  replace_node->copyAttributes(*this);
  // 断言新节点是否具有操作符，与原节点操作符状态一致
  TORCH_INTERNAL_ASSERT(
      (replace_node->maybeOperator() != nullptr) == had_operator,
      "invalid symbol replacement:",
      new_symbol,
      kind());
  // 返回替换后的新节点
  return replace_node;
}

// 判断当前节点是否被给定节点所支配
bool Node::isDominatedBy(const Node* dominator) const {
  const Node* node = this;
  // 从当前节点开始向上遍历，直到到达与给定节点所属相同的块
  while (node) {
    // 如果当前节点和给定节点属于同一个块，则判断给定节点是否在当前节点之前
    if (node->owningBlock() == dominator->owningBlock()) {
      return dominator->isBefore(node);
    }
    // 否则继续向上遍历块的拥有者节点
    node = node->owningBlock()->owningNode();
  }
  // 如果未找到共同的块，则当前节点不被给定节点支配
  return false;
}

// 在指定位置插入一个输入值，并更新相关的使用信息
Value* Node::insertInput(size_t i, Value* value) {
  // 断言当前节点所属图与输入值所属图相同
  AT_ASSERT(graph_ == value->owningGraph());
  // 将操作符指针设置为 nullptr
  op_ = nullptr;
  // 更新所有位于插入位置之后的现有输入的偏移量
  // 具体来说，这些是索引为 [i, # input) 的输入。由于我们在这些输入之前插入一个输入，因此增加这些输入对于此值的使用偏移量为 1
  for (const auto use_itr : c10::irange(i, inputs_.size())) {
    // 参见注释 [用户节点不唯一标识使用]
    auto use = findUseForInput(use_itr);
    use->offset += 1;
  }
  // 在指定索引位置插入实际的输入值
  inputs_.insert(inputs_.begin() + i, value);
  // 注册新输入值作为此节点的新使用
  value->uses_.emplace_back(this, i);
  // 返回插入的输入值
  return value;
}

// 向节点添加一个输入值，并更新相关的使用信息
Value* Node::addInput(Value* value) {
  // 断言当前节点所属图与输入值所属图相同
  AT_ASSERT(graph_ == value->owningGraph());
  // 将操作符指针设置为 nullptr
  op_ = nullptr;
  // 将输入值作为最后一个输入添加到节点的输入列表中，并更新相关使用信息
  value->uses_.emplace_back(this, inputs_.size());
  inputs_.push_back(value);
  // 返回添加的输入值
  return value;
}
# 替换节点的第 i 个输入值为新值，并返回旧值
Value* Node::replaceInput(size_t i, Value* newValue) {
  # 断言新值所属的计算图与当前节点所属的计算图相同
  AT_ASSERT(newValue->owningGraph() == graph_);
  # 清空当前节点的操作符指针，表示需要重新计算
  op_ = nullptr;
  # 丢弃第 i 个输入值，并返回旧的值
  Value* old = dropInput(i);
  # 将第 i 个输入值替换为新值
  inputs_[i] = newValue;
  # 更新新值的使用列表，将当前节点和输入索引添加进去
  newValue->uses_.emplace_back(this, i);
  # 返回替换前的旧值
  return old;
}

# 将节点的输入值 from 替换为 to
void Node::replaceInputWith(Value* from, Value* to) {
  # 断言 from 和 to 所属的计算图与当前节点的计算图相同
  AT_ASSERT(from->owningGraph() == graph_);
  AT_ASSERT(to->owningGraph() == graph_);
  # 清空当前节点的操作符指针，表示需要重新计算
  op_ = nullptr;
  # 遍历当前节点的所有输入值
  size_t i = 0;
  for (auto input : inputs()) {
    # 找到与 from 相同的输入值
    if (input == from) {
      # 调用 replaceInput 方法将其替换为 to
      replaceInput(i, to);
    }
    i++;
  }
}

# 添加一个输出值到当前节点，并返回该输出值
Value* Node::addOutput() {
  # 在当前节点的输出列表末尾添加一个新的值对象
  outputs_.push_back(new Value(this, outputs_.size()));
  # 清空当前节点的操作符指针，表示需要重新计算
  op_ = nullptr;
  # 返回添加的最后一个输出值对象
  return outputs_.back();
}

# 在当前节点的指定位置 i 处插入一个新的输出值，并返回该输出值
Value* Node::insertOutput(size_t i) {
  # 清空当前节点的操作符指针，表示需要重新计算
  op_ = nullptr;
  # 在当前节点的输出列表中的位置 i 处插入一个新的值对象
  outputs_.insert(outputs_.begin() + i, new Value(this, i));
  # 更新插入位置之后的所有输出值的偏移量
  for (size_t itr = i + 1; itr < outputs_.size(); ++itr) {
    outputs_[itr]->setOffset(outputs_[itr]->offset() + 1);
  }
  # 返回插入的输出值对象
  return outputs_.at(i);
}

# 判断当前节点是否在节点 n 之前或之后，根据 moveSide 参数决定
bool Node::isBeforeOrAfter(const Node* n, MoveSide moveSide) const {
  # 如果当前节点和节点 n 属于同一个基本块
  if (this->owningBlock() == n->owningBlock()) {
    # 如果要求在节点 n 之前移动，则比较当前节点和节点 n 的拓扑顺序
    if (moveSide == MoveSide::BEFORE) {
      return this->topo_position_ < n->topo_position_;
    }
    # 如果要求在节点 n 之后移动，则比较当前节点和节点 n 的拓扑顺序
    if (moveSide == MoveSide::AFTER) {
      return this->topo_position_ > n->topo_position_;
    }
    # 断言当前节点和节点 n 相同
    AT_ASSERT(this == n);
    return false;
  }

  # 如果当前节点和节点 n 不属于同一个基本块，向上遍历基本块链，直到找到第一个公共基本块
  auto lhs = this;
  while (lhs) {
    AT_ASSERT(lhs->owningBlock());
    auto rhs = n;
    while (rhs) {
      if (!rhs->owningBlock()) {
        break;
      }
      # 如果找到了共同的基本块，则递归调用 isBeforeOrAfter 方法比较拓扑顺序
      if (lhs->owningBlock() == rhs->owningBlock()) {
        return lhs->isBeforeOrAfter(rhs, moveSide);
      }
      rhs = rhs->owningBlock()->owningNode();
    }
    lhs = lhs->owningBlock()->owningNode();
  }
  # 不应该执行到这里，因为两个节点最终都在同一个图中
  AT_ASSERT(false);
}

# 判断当前节点是否在节点 n 之前
bool Node::isBefore(const Node* n) const {
  return isBeforeOrAfter(n, MoveSide::BEFORE);
}

# 判断当前节点是否在节点 n 之后
bool Node::isAfter(const Node* n) const {
  return isBeforeOrAfter(n, MoveSide::AFTER);
}

# 在节点 n 之前插入当前节点，并返回当前节点
Node* Node::insertBefore(Node* n) {
  # 断言节点 n 已经在基本块列表中
  AT_ASSERT(n->inBlockList());
  # 调用 insertAfter 方法，在节点 n 的前面插入当前节点
  insertAfter(n->prev());
  return this;
}

# 在节点 n 之后插入当前节点，并返回当前节点
Node* Node::insertAfter(Node* n) {
  # 断言当前节点不在基本块列表中，而节点 n 在基本块列表中
  AT_ASSERT(!inBlockList() && n->inBlockList());
  # 断言节点 n 有所属的基本块
  AT_ASSERT(n->owningBlock());
  # 断言当前节点不是 prim::Return 节点，因为不能在 Return 节点之后或 Param 节点之前插入节点
  AT_ASSERTM(
      n->kind() != prim::Return,
      "Attempting to insert a Node after the Return node or before the Param node. Tried to insert",
      *this,
      " after ",
      *n,
      ".");
  # 设置当前节点的基本块为节点 n 的基本块
  this->owning_block_ = n->owningBlock();
  # 获取节点 n 的下一个节点
  Node* next = n->next();
  # 调整指针，将当前节点插入到节点 n 和其下一个节点之间
  n->next() = this;
  this->prev() = n;
  this->next() = next;
  next->prev() = this;
  # 重新分配拓扑顺序
  assignTopoPosition();
  # 返回当前节点
  return this;
}

# 将当前节点移动到节点 n 之后
void Node::moveAfter(Node* n) {
  # 从当前节点所在位置移除
  removeFromList();
  # 调用 insertAfter 方法，在节点 n 的后面插入当前节点
  insertAfter(n);
}

# 将当前节点移动到节点 n 之前
void Node::moveBefore(Node* n) {
  # 从当前节点所在位置移除
  removeFromList();
  # 调用 insertBefore 方法，在节点 n 的前面插入当前节点
  insertBefore(n);
}
// 从节点的输入列表中移除指定索引的输入
void Node::removeInput(size_t i) {
  op_ = nullptr;  // 将操作指针置为空
  dropInput(i);   // 调用私有方法删除指定索引的输入

  // 更新所有在被删除输入之后的输入使用偏移量
  for (size_t j = i + 1; j < inputs_.size(); j++) {
    auto it = findUseForInput(j);  // 查找第 j 个输入的使用列表迭代器
    it->offset--;  // 将使用偏移量减一，以保持正确的引用
  }

  inputs_.erase(inputs_.begin() + i);  // 从输入列表中删除指定索引的输入
}

// 移除节点的所有输入
void Node::removeAllInputs() {
  op_ = nullptr;  // 将操作指针置为空
  for (const auto i : c10::irange(inputs().size())) {
    dropInput(i);  // 逐个删除所有输入
  }
  inputs_.clear();  // 清空输入列表
}

// 移除节点的所有输出
void Node::removeAllOutputs() {
  op_ = nullptr;  // 将操作指针置为空
  size_t init_osize = outputs_.size();  // 记录初始输出大小
  for (auto i : c10::irange(init_osize)) {
    eraseOutput(init_osize - i - 1);  // 逆序移除所有输出
  }
}

// 根据新的顺序重新排列节点的输入
void Node::permuteInputs(const std::vector<size_t>& new_order) {
  op_ = nullptr;  // 将操作指针置为空
  AT_ASSERT(new_order.size() == inputs_.size());  // 断言新顺序的大小与输入列表相等
  std::vector<Value*> new_inputs;
  new_inputs.reserve(new_order.size());

  // 根据新顺序重排输入
  for (const auto i : c10::irange(new_order.size())) {
    AT_ASSERTM(inputs_.at(new_order[i]) != nullptr, "Repeated index");  // 断言不重复的索引
    new_inputs.push_back(inputs_.at(new_order[i]));  // 将按新顺序收集的输入添加到新列表
    auto it = findUseForInput(new_order[i]);  // 查找新顺序中每个输入的使用列表迭代器
    it->offset = i;  // 更新使用偏移量
    inputs_.at(new_order[i]) = nullptr;  // 清空原输入列表中的相应位置
  }

  inputs_ = std::move(new_inputs);  // 使用新的排列后的输入列表替换原列表
}

// 根据新的顺序重新排列节点的输出
void Node::permuteOutputs(const std::vector<size_t>& new_order) {
  op_ = nullptr;  // 将操作指针置为空
  AT_ASSERT(new_order.size() == outputs_.size());  // 断言新顺序的大小与输出列表相等
  std::vector<Value*> new_outputs;
  new_outputs.reserve(new_order.size());

  // 根据新顺序重排输出
  for (const auto i : c10::irange(new_order.size())) {
    AT_ASSERTM(outputs_.at(new_order[i]) != nullptr, "Repeated index");  // 断言不重复的索引
    new_outputs.push_back(outputs_.at(new_order[i]));  // 将按新顺序收集的输出添加到新列表
    outputs_.at(new_order[i])->setOffset(i);  // 设置输出的偏移量
    outputs_.at(new_order[i]) = nullptr;  // 清空原输出列表中的相应位置
  }

  outputs_ = std::move(new_outputs);  // 使用新的排列后的输出列表替换原列表
}

// 查找指定输入的使用列表迭代器
use_list::iterator Node::findUseForInput(size_t i) {
  auto& input_uses = inputs_[i]->uses_;  // 获取第 i 个输入的使用列表引用
  // 使用线性搜索在使用列表中查找指定的使用项
  auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
  AT_ASSERT(use_it != input_uses.end());  // 断言找到有效的使用项
  return use_it;  // 返回找到的使用列表迭代器
}

// 删除节点指定索引的输入
Value* Node::dropInput(size_t i) {
  AT_ASSERT(i < inputs_.size());  // 断言索引在有效范围内
  auto input_node = inputs_[i];  // 获取第 i 个输入的节点指针
  auto use_it = findUseForInput(i);  // 查找第 i 个输入的使用列表迭代器
  input_node->uses_.erase(use_it);  // 从使用列表中删除对该输入的引用
  inputs_[i] = nullptr;  // 将输入列表中的指定位置置为空
  return input_node;  // 返回被删除的输入节点指针
}

// 从节点列表中移除节点本身
void Node::removeFromList() {
  AT_ASSERT(inBlockList());  // 断言节点在块列表中
  this->owning_block_ = nullptr;  // 将节点所属块指针置为空
  Node* next = this->next();  // 获取下一个节点指针
  Node* prev = this->prev();  // 获取前一个节点指针
  prev->next() = next;  // 将前一个节点的下一个指针指向下一个节点
  next->prev() = prev;  // 将下一个节点的前一个指针指向前一个节点
  this->next() = nullptr;  // 将节点的下一个指针置为空
  this->prev() = nullptr;  // 将节点的前一个指针置为空
}

// 查找与另一个节点共同祖先块
Block* Node::findCommonAncestorBlockWith(Node* n) {
  if (n->owningBlock() == owningBlock()) {  // 如果节点 n 与当前节点在同一块中
    return owningBlock();  // 直接返回当前节点所属块
  }

  Node* n1 = this;  // 当前节点作为第一个节点
  Node* n2 = n;  // 给定节点作为第二个节点

  size_t d_1 = n1->blocksFromGraphBlock();  // 计算当前节点到图块的距离
  size_t d_2 = n2->blocksFromGraphBlock();  // 计算给定节点到图块的距离

  // 将节点 n1 移动到与节点 n2 同级的位置
  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();  // 将 n1 上移至其所属块的上一个节点
    // n2 包含 n1
  }

  // 将节点 n2 移动到与节点 n1 同级的位置
  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();  // 将 n2 上移至其所属块的上一个节点
    // n1 包含 n2
  }
  
  // 从节点 n1 和节点 n2 开始，向上移动，直到找到它们的共同祖先块
    n2 = n2->owningBlock()->owningNode();
  }

  // 现在它们距离图块相同数量的块，
  // 递归向上检查它们是否在同一个块中
  while (true) {
    // 检查节点 n1 和 n2 是否在同一个块中
    if (n1->owningBlock() == n2->owningBlock()) {
      // 如果在同一个块中，则返回该块
      return n1->owningBlock();
    }

    // 向上递归，将 n1 和 n2 移动到它们各自所在块的上一级节点
    n1 = n1->owningBlock()->owningNode();
    n2 = n2->owningBlock()->owningNode();

    // 断言 n1 和 n2 非空
    AT_ASSERT(n1 != nullptr);
    AT_ASSERT(n2 != nullptr);
  }
}

// 计算节点到其所属图块的距离
size_t Node::blocksFromGraphBlock() {
  Node* n = this;  // 初始化节点为当前节点
  size_t dist = 0;  // 初始化距离为0
  while (n->owningBlock()->owningNode()) {  // 循环直到节点所属块不再有所属节点
    n = n->owningBlock()->owningNode();  // 更新节点为当前节点所属块的所属节点
    ++dist;  // 增加距离计数
  }
  return dist;  // 返回节点到其所属图块的距离
}

// 返回静态的虚假源范围
inline const SourceRange& fakeRange() {
  static SourceRange range(std::make_shared<Source>(std::string("")), 0, 1);
  return range;  // 返回静态的虚假源范围对象
}

// 在图中插入操作节点
Value* Graph::insert(
    Symbol opname,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const std::optional<SourceRange>& range) {
  return emitBuiltinCall(
      range.value_or(fakeRange()), *this, opname, args, kwargs);  // 调用内建调用发射函数，并返回其结果
}

// 在图中创建操作节点
Node* Graph::create(NodeKind kind, size_t num_outputs) {
  // 注意：节点构造函数会将节点添加到所有节点列表中
  auto n = new Node(this, kind);  // 创建新节点，并将其添加到当前图的所有节点列表中
  for (const auto i : c10::irange(num_outputs)) {  // 遍历输出数量的范围
    (void)i;  // 抑制未使用变量警告
    n->addOutput();  // 添加输出到节点
  }
  return n;  // 返回新创建的节点
}

// 在图中创建操作节点
Node* Graph::create(
    NodeKind kind,
    ArrayRef<Value*> inputs,
    size_t num_outputs) {
  auto n = create(kind, num_outputs);  // 调用上述函数创建新节点
  for (auto i : inputs) {  // 遍历输入值列表
    n->addInput(i);  // 将每个输入添加到节点中
  }
  return n;  // 返回新创建的节点
}

// 创建一个自动求导零节点
Node* Graph::createAutogradZero() {
  return create(prim::AutogradZero);  // 调用创建操作节点函数创建自动求导零节点
}

// 创建一个表示None的节点
Node* Graph::createNone() {
  Node* n = create(prim::Constant);  // 调用创建操作节点函数创建常量节点
  n->output()->setType(NoneType::get());  // 设置节点输出类型为NoneType
  return n;  // 返回新创建的节点
}

// 创建一个未初始化的节点
Node* Graph::createUninitialized(TypePtr typ) {
  Node* n = create(prim::Uninitialized);  // 调用创建操作节点函数创建未初始化节点
  n->output()->setType(std::move(typ));  // 设置节点输出类型为给定的类型
  return n;  // 返回新创建的节点
}

// 创建一个带有子图的节点
Node* Graph::createWithSubgraph(Symbol kind) {
  auto n = create(kind, 0);  // 调用创建操作节点函数创建指定类型的节点
  n->g_(attr::Subgraph, std::make_shared<Graph>(current_scope()));  // 设置节点的子图属性为当前作用域的新图对象
  return n;  // 返回新创建的节点
}

// 创建一个元组节点
Node* Graph::createTuple(at::ArrayRef<Value*> values, TupleTypePtr tuple_type) {
  TORCH_INTERNAL_ASSERT(
      !tuple_type || tuple_type->schema(),
      "only pass tuple_type when creating a named tuple");  // 内部断言：只有在创建命名元组时才传递tuple_type
  if (!tuple_type) {  // 如果未提供tuple_type
    auto types = fmap(values, [](Value* v) { return v->type(); });  // 获取所有值的类型
    tuple_type = TupleType::create(std::move(types));  // 创建元组类型
  }
  auto n = create(prim::TupleConstruct, values);  // 调用创建操作节点函数创建元组构造节点

  n->output()->setType(tuple_type);  // 设置节点输出类型为元组类型
  return n;  // 返回新创建的节点
}

// 创建一个元组解包节点
Node* Graph::createTupleUnpack(Value* v) {
  TupleTypePtr tt = v->type()->expect<TupleType>();  // 获取值的类型并期望其为元组类型
  auto n = create(prim::TupleUnpack, {v}, 0);  // 调用创建操作节点函数创建元组解包节点
  for (auto& element : tt->elements()) {  // 遍历元组类型的所有元素
    n->addOutput()->setType(element);  // 为节点添加输出，并设置输出类型为当前元素类型
  }
  return n;  // 返回新创建的节点
}

// 创建一个元组索引节点
Node* Graph::createTupleIndex(
    Value* tup,
    Value* idx,
    const TypePtr& output_type) {
  auto n = create(prim::TupleIndex, {tup, idx});  // 调用创建操作节点函数创建元组索引节点
  n->output()->setType(output_type);  // 设置节点输出类型为指定的输出类型
  return n;  // 返回新创建的节点
}

// 创建一个元组切片节点
Node* Graph::createTupleSlice(
    Value* tup,
    int64_t beg,
    int64_t step_size,
    int64_t num_values) {
  std::vector<Value*> new_vals;  // 新值的向量
  TupleTypePtr tt = tup->type()->expect<TupleType>();  // 获取值的类型并期望其为元组类型
  new_vals.reserve(num_values);  // 预留新值向量的空间

  int64_t i = beg;  // 初始化索引为开始索引
  for (const auto j : c10::irange(num_values)) {  // 遍历指定数量的范围
    (void)j;  // 抑制未使用变量警告
    auto idx = insertConstant(IValue(static_cast<int64_t>(i)));  // 插入常量索引
    auto tupleIndex = insertNode(createTupleIndex(tup, idx, tt->elements()[i]));  // 插入元组索引节点

    new_vals.push_back(tupleIndex->output());  // 将新节点的输出添加到新值向量中
    i += step_size;
  }



// 增加变量 i 的值，步长为 step_size
i += step_size;



  auto n = createTuple(new_vals);
  return n;



// 使用 new_vals 创建一个元组并赋给变量 n
auto n = createTuple(new_vals);
// 返回创建的元组 n
return n;
Node* Graph::createEnumName(Value* e) {
  // 确保值 e 的类型是 EnumType
  e->type()->expect<EnumType>();
  // 使用断言确认 e 的类型确实是 EnumType
  assert(e->type()->cast<EnumType>());
  // 创建一个节点 n，表示 prim::EnumName 操作，输入为 e
  auto n = create(prim::EnumName, {e});
  // 设置节点 n 的输出类型为 StringType
  n->output()->setType(StringType::get());
  // 返回创建的节点 n
  return n;
}

Node* Graph::createEnumValue(Value* e) {
  // 期望值 e 的类型是 EnumType，并获取该类型
  auto enum_type = e->type()->expect<EnumType>();
  // 创建一个节点 n，表示 prim::EnumValue 操作，输入为 e
  auto n = create(prim::EnumValue, {e});
  // 设置节点 n 的输出类型为 enum_type 的值类型
  n->output()->setType(enum_type->getValueType());
  // 返回创建的节点 n
  return n;
}

Node* Graph::createList(
    const TypePtr& contained_type,
    at::ArrayRef<Value*> values) {
  // 创建一个节点 n，表示 prim::ListConstruct 操作，输入为 values 中的元素
  auto n = create(prim::ListConstruct, values);
  // 遍历 values 中的每个元素 v
  for (const auto& v : values) {
    // 检查 v 的类型是否是 contained_type 的子类型，否则抛出错误
    TORCH_CHECK(
        v->type()->isSubtypeOf(*contained_type),
        "Expected a list element that subtypes '",
        contained_type->repr_str(),
        "' but got an element of type '",
        v->type()->repr_str(),
        "'");
  }
  // 设置节点 n 的输出类型为包含 contained_type 的 ListType
  n->output()->setType(ListType::create(contained_type));
  // 返回创建的节点 n
  return n;
}

Node* Graph::createListUnpack(Value* v, size_t size) {
  // 期望值 v 的类型是 ListType，获取列表元素的类型
  ListTypePtr list_type = v->type()->expect<ListType>();
  TypePtr elem_type = list_type->getElementType();
  // 创建一个节点 n，表示 prim::ListUnpack 操作，输入为 v，输出为 size 个元素
  auto n = create(prim::ListUnpack, {v}, 0);
  // 对于范围在 size 内的每个 i
  for (const auto i : c10::irange(size)) {
    // 忽略未使用的变量警告
    (void)i; // Suppress unused variable warning
    // 向节点 n 添加一个输出，类型为 elem_type
    n->addOutput()->setType(elem_type);
  }
  // 返回创建的节点 n
  return n;
}

Node* Graph::createDict(
    const TypePtr& key_type,
    const TypePtr& value_type,
    at::ArrayRef<Value*> keys,
    at::ArrayRef<Value*> values) {
  // 断言 keys 和 values 的大小相等
  AT_ASSERT(keys.size() == values.size());
  // 创建一个节点 n，表示 prim::DictConstruct 操作，初始容量为 1
  auto n = create(prim::DictConstruct, 1);
  // 对于 keys 的每个索引 i
  for (const auto i : c10::irange(keys.size())) {
    // 断言 keys[i] 的类型是 key_type 的子类型
    AT_ASSERT(keys[i]->type()->isSubtypeOf(*key_type));
    // 断言 values[i] 的类型是 value_type 的子类型
    AT_ASSERT(values[i]->type()->isSubtypeOf(*value_type));

    // 向节点 n 添加键值对输入，分别为 keys[i] 和 values[i]
    n->addInput(keys[i]);
    n->addInput(values[i]);
  }
  // 设置节点 n 的输出类型为包含 key_type 和 value_type 的 DictType
  n->output()->setType(DictType::create(key_type, value_type));
  // 返回创建的节点 n
  return n;
}

Node* Graph::createNumToTensor(Value* value) {
  // 创建一个节点 result，表示 prim::NumToTensor 操作，输入为 value
  Node* result = create(prim::NumToTensor, {value});
  // 设置节点 result 的输出类型为从 value 的类型转换而来的 TensorType
  result->output()->setType(TensorType::fromNumberType(*value->type()));
  // 返回创建的节点 result
  return result;
}

Node* Graph::createObject(const ClassTypePtr& type) {
  // 创建一个节点 result，表示 prim::CreateObject 操作
  auto result = create(prim::CreateObject);
  // 设置节点 result 的输出类型为给定的类类型 type
  result->output()->setType(type);
  // 返回创建的节点 result
  return result;
}

Node* Graph::createSetAttr(
    Value* obj,
    const std::string& field,
    Value* newValue) {
  // 创建一个节点 n，表示 prim::SetAttr 操作，输入为 obj 和 newValue，无输出
  auto n = create(prim::SetAttr, {obj, newValue}, /*num_outputs=*/0);
  // 设置节点 n 的属性名称为 field
  n->s_(attr::name, field);
  // 返回创建的节点 n
  return n;
}

Node* Graph::createGetAttr(Value* obj, const std::string& field) {
  // 期望值 obj 的类型是 ClassType，并获取该类型
  const auto classType = obj->type()->expect<ClassType>();

  // 创建一个节点 n，表示 prim::GetAttr 操作，输入为 obj，输出为一个值
  auto n = create(prim::GetAttr, {obj}, /*num_outputs=*/1);
  // 设置节点 n 的属性名称为 field
  n->s_(attr::name, field);

  // 获取字段 field 的类型
  const auto outputType = classType->getAttribute(field);
  // 设置节点 n 的输出类型为字段 field 的类型
  n->output()->setType(outputType);
  // 设置节点 n 的调试名称为规范化后的字段名
  n->output()->setDebugName(normalizeAttrName(field));
  // 返回创建的节点 n
  return n;
}

Node* Graph::createStore(const std::string& name, Value* v) {
  // 创建一个节点 n，表示 prim::Store 操作，输入为 v，无输出
  auto n = create(prim::Store, {v}, /*num_outputs*/ 0);
  // 设置节点 n 的属性名称为 name
  n->s_(attr::name, name);
  // 返回创建的节点 n
  return n;
}
// 创建一个加载节点，表示从变量名 `name` 所指示的位置加载数据，并设置输出个数为1
Node* Graph::createLoad(const std::string& name, const TypePtr& type) {
  auto n = create(prim::Load, {}, /*num_outputs*/ 1);
  // 设置节点的属性 `name`，表示加载的变量名
  n->s_(attr::name, name);
  // 设置输出节点的类型为给定的 `type`
  n->output()->setType(type);
  return n;
}

// 创建一个检查实例类型的节点，判断值 `v` 是否属于 `types` 中的任一类型，并设置输出个数为1
Node* Graph::createIsInstance(Value* v, at::ArrayRef<TypePtr> types) {
  auto n = create(prim::isinstance, {v}, /*num_outputs*/ 1);
  // 设置节点的属性 `types`，表示要检查的类型列表
  n->tys_(attr::types, types.vec());
  // 设置输出节点的类型为布尔类型
  n->output()->setType(BoolType::get());
  return n;
}

// 在图中插入一个未经检查的类型转换节点，将值 `v` 转换为指定的 `type` 类型
Value* Graph::insertUncheckedCast(Value* v, TypePtr type) {
  Node* n = insertNode(create(prim::unchecked_cast, {v}));
  // 设置输出节点的类型为给定的 `type`
  n->output()->setType(std::move(type));
  return n->output();
}

// 在图中插入一个列表转换节点，将值 `v` 转换为列表形式，并设置列表的维度和基本元素类型
Value* Graph::insertToList(Value* v, TypePtr type) {
  int dim = 0;
  TypePtr ptr = type;

  // 解开类型，确定维度数量
  while (auto list_type = ptr->cast<ListType>()) {
    ptr = list_type->getElementType();
    ++dim;
  }

  // 将基本元素类型编码为整数
  int elem_ty = 0;
  if (ptr == IntType::get()) {
    elem_ty = 0;
  } else if (ptr == FloatType::get()) {
    elem_ty = 1;
  } else if (ptr == BoolType::get()) {
    elem_ty = 2;
  } else if (ptr == ComplexType::get()) {
    elem_ty = 3;
  } else {
    // 如果类型不支持，则抛出异常
    TORCH_CHECK(
        false,
        ptr->repr_str(),
        " is not one of the supported element types for tolist: int, float, complex, bool");
  }

  // 将维度数量和基本元素类型作为参数传递给操作符
  Value* dim_val = insertConstant(IValue(dim));
  Value* elem_ty_val = insertConstant(IValue(elem_ty));
  Node* n = insertNode(create(prim::tolist, {v, dim_val, elem_ty_val}));
  // 设置输出节点的类型为给定的 `type`
  n->output()->setType(std::move(type));
  return n->output();
}

// 在图中插入一个函数调用节点，调用 `callee` 函数，并传递匹配的输入参数和返回值类型
Value* Graph::insertFunctionCall(
    Function* callee,
    const MatchedSchema& matched) {
  std::string func_name = callee->name();
  // 创建一个常量节点，表示调用的函数名，并设置其类型为函数类型
  Value* fn_constant = insertNode(create(prim::Constant))
                           ->s_(attr::name, func_name)
                           ->output()
                           ->setType(FunctionType::create(callee));
  std::vector<Value*> inputs = {fn_constant};
  // 将匹配的输入参数添加到输入列表中
  inputs.insert(inputs.end(), matched.inputs.begin(), matched.inputs.end());
  // 创建一个函数调用节点，并设置返回类型为匹配的返回类型列表的第一个类型
  Value* result = insertNode(create(prim::CallFunction, inputs))
                      ->output()
                      ->setType(matched.return_types.at(0));
  return result;
}

// 在图中插入一个方法调用节点，调用具有给定方法名和匹配模式的方法
Value* Graph::insertMethodCall(
    std::string method_name,
    const MatchedSchema& matched) {
  // 创建一个方法调用节点，设置方法名和输出类型为匹配的返回类型列表的第一个类型
  Value* result = insertNode(create(prim::CallMethod, matched.inputs))
                      ->s_(attr::name, std::move(method_name))
                      ->output()
                      ->setType(matched.return_types.at(0));
  return result;
}

// 创建一个节点的克隆，复制节点 `n` 的所有输出，并根据需要进行块复制
Node* Graph::createClone(
    Node* n,
    const std::function<Value*(Value*)>& value_map,
    bool copy_blocks) {
  // 为当前图分配一个新的节点实例
  Node* r = n->allocNewInstance(this);
  for (auto o : n->outputs()) {
    // 复制输出的元数据
    r->addOutput()->copyMetadata(o);
  }
  // 从节点 `n` 复制其余部分
  r->cloneFrom(n);
  for (auto i : n->inputs()) {
    # 对于每个输入值 i，使用 value_map 函数映射后将其添加到结果 r 中作为输入
    r->addInput(value_map(i));
  }
  # 如果需要复制块（copy_blocks 为真），则遍历节点 n 的所有块
  if (copy_blocks) {
    for (auto b : n->blocks()) {
      # 将当前块 b 克隆并添加到结果 r 中，使用 value_map 对其进行值映射
      r->addBlock()->cloneFrom(b, value_map);
    }
  }
  # 返回最终结果 r
  return r;
}

// 将常量值插入图中
Value* Graph::insertConstant(
    const IValue& val,                             // 要插入的常量值
    std::optional<SourceRange> loc,                 // 可选的源代码范围
    std::optional<ScopePtr> scope) {                // 可选的作用域指针
  return jit::insertConstant(*this, val, std::move(loc), std::move(scope));
}

// 将图转换为字符串表示形式
std::string Graph::toString(bool print_source_locations) const {
  std::ostringstream oss;
  print(oss, print_source_locations);               // 将图打印到ostringstream中
  return oss.str();                                 // 返回ostringstream的字符串表示
}

// 图的析构函数
Graph::~Graph() {
  for (const Node* n : all_nodes) {                 // 遍历所有节点
    delete n;                                       // 删除节点
  }
  for (const Value* v : all_values) {               // 遍历所有值
    delete v;                                       // 删除值
  }
  for (const Block* b : all_blocks) {               // 遍历所有块
    delete b;                                       // 删除块
  }
}

// 释放节点
void Graph::freeNode(Node* n) {
  auto it = all_nodes.find(n);                      // 查找节点n
  AT_ASSERT(it != all_nodes.end());                 // 断言节点存在
  delete *it;                                       // 删除节点
  all_nodes.erase(it);                              // 从集合中移除节点
}

// 释放值
void Graph::freeValue(Value* v) {
  v->setDebugName("");                              // 清空调试名称
  auto it = all_values.find(v);                     // 查找值v
  AT_ASSERT(it != all_values.end());                // 断言值存在
  delete *it;                                       // 删除值
  all_values.erase(it);                             // 从集合中移除值
}

// 释放块
void Graph::freeBlock(Block* b) {
  auto it = all_blocks.find(b);                     // 查找块b
  AT_ASSERT(it != all_blocks.end());                // 断言块存在
  delete *it;                                       // 删除块
  all_blocks.erase(it);                             // 从集合中移除块
}

// 创建元组展开操作的数组引用
at::ArrayRef<Value*> createTupleUnpack(Value* v) {
  // 对于元组构造节点，进行小规模的优化以确保IntArrayRef属性可以转换为常量
  if (v->node()->kind() == prim::TupleConstruct) {
    return v->node()->inputs();                     // 返回节点的输入作为数组引用
  }
  auto& g = *v->owningGraph();
  return g.insertNode(g.createTupleUnpack(v))->outputs();  // 插入元组展开节点并返回其输出
}

// 内联节点的调用栈
void inlineCallStackOfNode(
    Node* n,
    std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>& new_cs_entries,
    Function* callee,
    Node* to_replace,
    std::optional<ModuleInstanceInfo> m_info) {
  auto new_node_cs = n->callstack();                // 获取节点的调用栈

  InlinedCallStack* raw_callstack_ptr =
      new_node_cs ? new_node_cs->get() : nullptr;   // 获取原始调用栈指针

  if (!new_cs_entries.count(raw_callstack_ptr)) {   // 如果未记录该调用栈指针
    if (new_node_cs) {
      new_cs_entries[raw_callstack_ptr] = c10::make_intrusive<InlinedCallStack>(
          *new_node_cs, callee, to_replace->sourceRange(), m_info);  // 创建新的内联调用栈
    } else {
      new_cs_entries[raw_callstack_ptr] = c10::make_intrusive<InlinedCallStack>(
          callee, to_replace->sourceRange(), m_info);  // 创建新的内联调用栈（无源代码范围）
    }
  }
  new_node->setCallStack(new_cs_entries.at(raw_callstack_ptr));  // 设置节点的内联调用栈

  // 更新节点块的内联调用栈
  for (auto block : new_node->blocks()) {           // 遍历节点的所有块
  # 调用函数 `inlineCallStackOfBlock`，并传递参数 `block`, `new_cs_entries`, `callee`, `to_replace`, `m_info`
  inlineCallStackOfBlock(block, new_cs_entries, callee, to_replace, m_info);
}

std::vector<Value*> inlineCallTo(
    Node* to_replace,                              // 输入参数: 要替换的节点指针
    GraphFunction* callee,                         // 输入参数: 被调用的函数图对象指针
    Graph* callee_graph) {                         // 输入参数: 被调用函数的图对象指针
  WithInsertPoint guard(to_replace);               // 设置插入点为当前要替换的节点

  std::unordered_map<Value*, Value*> value_map;    // 创建值映射表，用于存储旧值到新值的映射关系
  std::vector<torch::jit::Value*> new_outputs = insertGraph(
      *to_replace->owningGraph(),                 // 在当前节点所属图中插入被调用函数的图
      *callee_graph,                              // 被调用函数的图
      to_replace->inputs(),                       // 当前节点的输入作为被调用函数的输入
      value_map);                                 // 传入值映射表，用于更新值的映射关系

  std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>
      new_callstack_entries;                      // 创建新的内联调用栈条目映射表

  std::optional<ModuleInstanceInfo> module_instance_info = c10::nullopt;  // 可选的模块实例信息，默认为空
  if (to_replace->kind() == prim::CallMethod) {    // 如果当前节点是 prim::CallMethod 类型
    auto class_type_ptr = to_replace->input(0)->type()->cast<c10::ClassType>();  // 获取类类型指针
    if (to_replace->input(0)->node()->kind() == prim::GetAttr) {  // 如果节点的第一个输入是 prim::GetAttr 类型
      module_instance_info = c10::make_optional(ModuleInstanceInfo(
          class_type_ptr,                          // 类型指针
          to_replace->input(0)->node()->s(attr::name)));  // 获取属性名作为实例信息
    } else if (
        !to_replace->owningGraph()->inputs().empty() &&
        to_replace->input(0) == to_replace->owningGraph()->inputs()[0]) {
      // 当前 CallMethod 必须对应于同一对象的方法，该对象属于当前图
      module_instance_info =
          c10::make_optional(ModuleInstanceInfo(class_type_ptr, "SELF"));  // 设置为SELF作为实例信息
    } else {
      // 不确定是否可能到达此处
      // TODO: 移除此 else 分支或添加断言
      module_instance_info = c10::make_optional(
          ModuleInstanceInfo(class_type_ptr, "INSTANCE_NAME_UNKNOWN"));  // 设置为INSTANCE_NAME_UNKNOWN作为实例信息
    }
  }

  // TODO: 可能需要使用 nodes_map 而不是 value_map。否则，将丢失没有输出的节点（例如 prim::Print）。
  std::unordered_set<Node*> updated_nodes;         // 创建更新过的节点集合
  for (const auto& kv : value_map) {               // 遍历值映射表中的键值对
    /* 跳过旧值如果它是图输入。
     * 原因是，value_map 包含的不仅是图的节点的所有值，还包括主要输入，
     * 当第一个内联化的图成为下一个图的输入时，会创建重复项。
     * 为了避免这个问题，当旧值是 callee->optimized_graph()->inputs() 或 callee->graph()->inputs() 之一时跳过旧值。
     */
    auto is_graph_input = std::find(
        callee_graph->inputs().begin(),            // 查找是否是被调用函数的输入之一
        callee_graph->inputs().end(), kv.first);
    if (is_graph_input != callee_graph->inputs().end()) {
      continue;
    }

    Node* new_node = kv.second->node();            // 获取映射后的新节点
    if (!updated_nodes.insert(new_node).second) {  // 如果新节点已经更新过，继续下一个循环
      continue;
    }

    inlineCallStackOfNode(
        new_node,                                  // 内联节点
        new_callstack_entries,                     // 新的内联调用栈条目映射表
        callee,                                    // 被调用函数对象
        to_replace,                                // 要替换的节点
        module_instance_info);                     // 模块实例信息
  }
  const auto& old_outputs = to_replace->outputs();  // 获取当前节点的旧输出

  AT_ASSERT(new_outputs.size() == old_outputs.size());  // 断言新旧输出的数量相同
  for (const auto i : c10::irange(old_outputs.size())) {  // 遍历旧输出的索引范围
    if (old_outputs[i]->hasDebugName()) {           // 如果旧输出有调试名称
      new_outputs[i]->setDebugName(old_outputs[i]->debugName());  // 设置新输出的调试名称为旧输出的调试名称
    }
    old_outputs[i]->replaceAllUsesWith(new_outputs[i]);  // 替换所有使用旧输出的地方为新输出
  }
  to_replace->destroy();                           // 销毁要替换的节点

  return new_outputs;                              // 返回新的输出向量
}
// 在替换函数调用时，根据inline_optimized_graph参数决定是否使用优化过的GraphFunction
std::vector<Value*> inlineCallTo(
    Node* to_replace,
    GraphFunction* callee,
    bool inline_optimized_graph /*=true*/) {
  // 根据inline_optimized_graph参数选择要使用的Graph对象
  auto graph =
      inline_optimized_graph ? callee->optimized_graph() : callee->graph();
  // 调用重载的inlineCallTo函数，传入选择的Graph对象
  return inlineCallTo(to_replace, callee, graph.get());
}

// 将输出向量outputs展开为new_outputs向量，如果outputs是单个元组类型，则进行展开
std::vector<Value*> unpackOutputs(const std::vector<Value*>& outputs) {
  std::vector<Value*> new_outputs;
  // 如果outputs不是单个元素或者第一个元素不是元组类型，则直接返回outputs向量
  if (outputs.size() != 1 || outputs.at(0)->type()->kind() != TupleType::Kind) {
    return outputs;
  }

  auto tup = outputs[0];
  // 遍历创建的元组展开，并加入到new_outputs向量中
  for (Value* v : createTupleUnpack(tup)) {
    new_outputs.emplace_back(v);
  }
  // 如果tup是由prim::TupleConstruct创建且没有被使用，则销毁tup的节点
  if (tup->node()->kind() == prim::TupleConstruct && !tup->node()->hasUses()) {
    tup->node()->destroy();
  }
  return new_outputs;
}

// 在给定的Block数组中递归查找指定类型的节点，返回所有找到的节点
std::vector<Node*> findAllNodes(
    at::ArrayRef<Block*> array,
    Symbol kind,
    bool recurse) {
  std::vector<Node*> ret;
  // 遍历Block数组，调用递归版本的findAllNodes函数
  for (auto block : array) {
    findAllNodes(*block, kind, recurse, ret);
  }
  return ret;
}

// 在给定的Block中递归查找指定类型的节点，返回所有找到的节点
std::vector<Node*> findAllNodes(Block& block, Symbol kind, bool recurse) {
  return findAllNodes({&block}, kind, recurse);
}

// 在给定的Graph中递归查找指定类型的节点，返回所有找到的节点
std::vector<Node*> findAllNodes(Graph& g, Symbol kind, bool recurse) {
  return findAllNodes(*g.block(), kind, recurse);
}

// 将callee的图插入到主图g中，根据inputs映射输入值，并返回输出值向量
std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs,
    std::unordered_map<Value*, Value*>& value_map) {
  // 定义值映射的lambda函数
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  // 断言callee的输入数量与inputs的数量相同
  AT_ASSERT(callee.inputs().size() == inputs.size());
  // 将callee的输入值与inputs进行映射
  for (const auto i : c10::irange(inputs.size())) {
    value_map[callee.inputs()[i]] = inputs[i];
  }
  // 将callee的节点克隆并插入到主图g中，并更新值映射
  for (auto* node : callee.nodes()) {
    auto* new_node = g.insertNode(g.createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = new_node->outputs()[i];
    }
  }

  // 收集callee的输出值，并返回映射后的输出值向量
  std::vector<Value*> outputs;
  for (auto* output : callee.outputs()) {
    outputs.push_back(value_map_func(output));
  }

  return outputs;
}

// 将callee的图插入到主图g中，根据inputs映射输入值，并返回输出值向量
std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs) {
  std::unordered_map<Value*, Value*> value_map;
  return insertGraph(g, callee, inputs, value_map);
}

// 从另一个ProfileOp节点other_中克隆信息到当前节点，包括回调函数
void ProfileOp::cloneFrom(Node* other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<ProfileOp>();
  this->callback_ = other->getCallback();
}

// 在给定的Graph g中分配一个新的ProfileOp实例，使用空指针数组作为输入
Node* ProfileOp::allocNewInstance(Graph* g) {
  return new ProfileOp(g, {nullptr});
}

// 从另一个ProfileIValueOp节点other_中克隆信息到当前节点，包括回调函数
void ProfileIValueOp::cloneFrom(Node* other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<ProfileIValueOp>();
  this->callback_ = other->getCallback();
}

// 在给定的Graph g中分配一个新的ProfileIValueOp实例，使用空指针数组作为输入
Node* ProfileIValueOp::allocNewInstance(Graph* g) {
  return new ProfileIValueOp(g, {nullptr});
}

// 返回NamedValue对象的类型指针，如果value_存在则返回其类型，否则返回nullptr
TypePtr NamedValue::type() const {
  if (value_) {
    return value_->type();
  } else {
    // 如果value_为空指针，则返回nullptr
    return ivalue_.type();
  }
}

const Symbol ProfileOp::Kind = ::c10::prim::profile;
const Symbol ProfileIValueOp::Kind = ::c10::prim::profile_ivalue;

// OperatorSet 类的构造函数，接受一个初始化列表，将其中的签名字面值插入到操作集合中
OperatorSet::OperatorSet(std::initializer_list<const char*> sig_literals) {
  insert(sig_literals);
}

// 获取操作集合中所有操作的指针，并返回为一个共享指针的向量
std::vector<std::shared_ptr<Operator>> OperatorSet::getOps() const {
  std::vector<std::shared_ptr<Operator>> result;
  // 遍历操作集合中的每个键值对
  for (const auto& kv : ops) {
    auto ops_for_symbol = kv.second;
    // 将每个符号对应的操作指针插入到结果向量中
    result.insert(result.end(), ops_for_symbol.begin(), ops_for_symbol.end());
  }
  return result;
}

// 向操作集合中插入签名字面值的操作
void OperatorSet::insert(std::initializer_list<const char*> sig_literals) {
  // 遍历初始化列表中的每个签名字面值
  for (const char* sig : sig_literals) {
    // 获取指定签名字面值对应的操作指针
    auto op = getOperatorForLiteral(sig);
    // 将操作指针按照其名称的符号形式作为键，插入到操作集合中
    ops[Symbol::fromQualString(op->schema().name())].push_back(op);
  }
}

// 检查当前节点是否属于给定的操作集合 os
bool Node::isMemberOf(const OperatorSet& os) const {
  // 查找操作集合中是否存在当前节点的类型所对应的操作列表
  auto it = os.ops.find(kind());
  // 如果找不到对应类型的操作列表，则返回 false
  if (it == os.ops.end()) {
    return false;
  }
  // 遍历找到的操作列表，检查是否有操作的模式与当前节点的模式匹配
  for (auto& op : it->second) {
    if (matches(op->schema())) {
      // 如果找到匹配的操作模式，则返回 true
      return true;
    }
  }
  // 如果没有找到匹配的操作模式，则返回 false
  return false;
}

} // namespace torch::jit
```