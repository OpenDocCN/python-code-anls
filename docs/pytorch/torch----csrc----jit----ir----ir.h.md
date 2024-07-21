# `.\pytorch\torch\csrc\jit\ir\ir.h`

```
#pragma once

// 包含头文件，这些头文件提供了 JIT 编译器 IR 层级的各种功能和定义
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/graph_node_list.h>
#include <torch/csrc/jit/ir/named_value.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/runtime/operator.h>

// 包含导出相关的头文件和定义
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/schema_info.h>

// 包含 ATen 库的相关头文件
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

// 包含标准库和其他必要的头文件
#include <functional>
#include <iosfwd>
#include <unordered_set>
#include <vector>

// 使用 TORCH_API 命名空间下的 getNodesModuleHierarchy 函数
namespace torch {
namespace jit {
namespace utils {
TORCH_API std::string getNodesModuleHierarchy(const Node& n);
} // namespace utils

// 声明别名类型 THPObjectPtr 和 pyobj_list
class AliasDb;
using THPObjectPtr = THPPointer<PyObject>;
using pyobj_list = std::vector<THPObjectPtr>;

} // namespace jit
} // namespace torch

// 前向声明 Graph 和 Node 结构体
struct Graph;
struct Node;

// 使用 TORCH_API 命名空间下的运算符重载函数
TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);
TORCH_API std::ostream& operator<<(std::ostream& out, const Node& n);

// 前向声明 Block 结构体
struct Block;

// 包含所有 Node 使用的类型定义和命名空间
struct Value;

// 使用 TORCH_API 命名空间下的别名函数
using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::Symbol;

// 使用 ::c10::ivalue 命名空间下的类型定义
using ::c10::ivalue::Shared;

// 使用 ::c10::ivalue 命名空间下的 IValue 和 Future 类型
using ::c10::IValue;
using ::c10::ivalue::Future;

// 使用 ::c10::ivalue 命名空间下的 ConstantString 类型
using ::c10::ivalue::ConstantString;

// 使用 C10_FORALL_TYPES 宏展开的类型别名定义
#define C10_USING(T) using ::c10::T;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

// 使用 C10_FORALL_TYPES 宏展开的类型智能指针别名定义
#define C10_USING(T) using ::c10::T##Ptr;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

// 使用 ::c10 命名空间下的 Type 和 TypeEnv 类型
using ::c10::Type;
using ::c10::TypeEnv;
using ::c10::TypePtr;

// 使用 ::c10 命名空间下的 getTypePtr 和 MatchTypeReturn 函数
using ::c10::getTypePtr;
using ::c10::MatchTypeReturn;
using ::c10::TypeKind;

// 使用 ::c10 命名空间下的 fmap 函数
using ::c10::fmap;

// 使用 prim 命名空间下的类型定义
namespace prim {
using namespace ::c10::prim;
}

// 使用 attr 命名空间下的类型定义
namespace attr {
using namespace ::c10::attr;
}

// 使用 aten 命名空间下的类型定义
namespace aten {
using namespace ::c10::aten;
}

// 如果未定义 USE_ROCM，则使用 cuda 命名空间下的类型定义
namespace cuda {
#if !defined(USE_ROCM)
using namespace ::c10::cuda;
#endif
} // namespace cuda

// 声明 Function、GraphFunction 和 MatchedSchema 结构体
struct Function;
struct GraphFunction;
struct MatchedSchema;

// Graph 结构体表示一个计算函数
// 使用简单的所有权模型，图拥有其中所有的节点
// 图中所有引用都是原始指针，销毁 Graph 将使得图中所有节点的指针失效
struct Graph;

// Node 结构体是 IR 图中节点的基类，表示一个计算和依赖的列表
// 即 IR 图的基本操作
struct Node;

// Value 结构体表示节点的输入或输出，可以是 Tensor 或不透明 Handle 对象
// 其类型由 type() 决定
struct Value;
// 定义结构体 Use，表示节点使用的信息，包括用户节点和偏移量
struct Use {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;         // 用户节点指针
  size_t offset;      // 使用偏移量

  // 重载相等运算符，比较两个 Use 结构体是否相等
  bool operator==(const Use& b) {
    return user == b.user && offset == b.offset;
  }
};

// Note [User node does not uniquely identify use]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 几段时间前，我们编写了一些操作使用的代码，看起来像这样：
//
//    for (auto& use : used_val->uses_) {
//      if (use.user == this_node) {
//        use.offset += 1;
//        break;
//      }
//    }
//
// 这段代码试图找到特定使用（我们节点的使用）并更新它。
// 但这是错误的：节点 %x 可能有多个使用，例如在以下 IR 中：
//
//    %y = Add %x %x
//
// 在这种情况下，节点 'Add %x %x' 有两个使用 %x。
// 因此，“由此节点引起的使用”并不是一个明确定义的概念。
//
// 如果你正在寻找“由输入引起的使用”，最好使用 findUseForInput() 来获取它。

// 定义几个列表类型，这里故意保持简单，但是我们在这里定义它们，以便将来重构时更容易
using node_list = std::vector<Node*>;      // 节点列表类型
using value_list = std::vector<Value*>;    // 值列表类型
using use_list = std::vector<Use>;         // 使用列表类型
template <typename T>
using ArrayRef = at::ArrayRef<T>;          // 数组引用类型
using NodeKind = Symbol;                   // 节点类型
using topo_position_t = int64_t;           // 拓扑排序位置类型
using ValueSet = std::unordered_set<const Value*>;  // 值集合类型

// 这是一个包装器，允许在删除节点/值/块的 C++ 对象时安全地使 Python 对象无效
// 像图中的大部分内容一样，让不同的线程访问同一图是不安全的
template <typename T>
struct Wrap {
  explicit Wrap(T* p) : elem(p), clear_cb(nullptr) {}
  void clear() {
    if (clear_cb) {
      clear_cb(elem);
    }
    elem = nullptr;
  }
  T* elem;              // 元素指针
  void (*clear_cb)(void*);  // 清除回调函数指针
};

// Value 结构体表示计算图中的值
struct Value {
  AT_DISALLOW_COPY_AND_ASSIGN(Value);  // 禁止复制和赋值操作
  Value(Node* node_, size_t offset_);

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;              // 节点指针
  size_t offset_;           // 偏移量
  size_t unique_ = 0;       // 唯一标识符
  use_list uses_;           // 使用列表
  std::string unique_name_; // 唯一名称
  TypePtr type_;            // 类型指针
  // 一个用于管理的 Python 包装器，允许无效化
  std::shared_ptr<Wrap<Value>> wrap_;

 public:
  // 设置值的类型
  Value* setType(TypePtr type);
  TORCH_API void inferTypeFrom(const at::Tensor& output);
  TORCH_API void inferTypeFrom(
      const c10::intrusive_ptr<c10::ivalue::Object>& output);
  const TypePtr& type() const {
    AT_ASSERT(type_ != nullptr);   // 断言类型非空
    return type_;
  }
  bool requires_grad() const {     // 是否需要梯度
    return type()->requires_grad();
  }
  bool isCompleteTensor() const {   // 是否是完整张量
    if (auto pt = type()->cast<TensorType>()) {
      return pt->isComplete();
    }
    return false;
  }
  TORCH_API bool mustBeNone() const;
  TORCH_API bool mustNotBeNone() const;
  size_t unique() const {          // 返回唯一标识符
    return unique_;
  }
  bool hasDebugName() const {
    // 检查是否有调试名称
  // 返回该值是否有唯一的名称
  return !unique_name_.empty();
}

  // 静态方法：检查给定的名称是否有效
static bool isValidName(const std::string& name);

  // 设置调试名称，并返回该值的指针
TORCH_API Value* setDebugName(const std::string& name);

  // 如果有调试名称，则返回调试名称；否则返回该值的唯一标识符的字符串表示
std::string debugName() const {
  if (hasDebugName()) {
    return unique_name_;
  }
  return std::to_string(unique());
}

  // 返回该值所属的调试名称的基础部分
TORCH_API std::string debugNameBase() const;

  // 返回该值关联的节点指针
Node* node() {
  return node_;
}

  // 返回该值的偏移量
size_t offset() const {
  return offset_;
}

  // 设置该值的偏移量
void setOffset(size_t offset) {
  offset_ = offset;
}

  // 返回该值关联的节点指针（常量版本）
const Node* node() const {
  return node_;
}

  /**
   * @warning 永远不要将智能指针管理的图的原始指针传递给 Python。
   * 参考 #87343 了解详情。
   */
Graph* owningGraph();

  // 返回该值所属的图的常量指针
const Graph* owningGraph() const;

  // 返回使用该值的列表的常量引用
// TODO: 使这个方法更符合 const 语义
const use_list& uses() const {
  return uses_;
}

  // 检查该值是否有使用它的地方
bool hasUses() const {
  return !uses().empty();
}

  // 用新值替换该值的第一个使用位置
TORCH_API void replaceFirstUseWith(Value* newValue);

  // 替换所有使用该值的位置为新值
// 示例：
//   %3 = f(%1, %2)
//   %4 = g(%3)
//   %5 = h(%3, %3)
//   %3.replaceAllUsesWith(%6)
// 替换后：
//   %3 = f(%1, %2)
//   %4 = g(%6)
//   %5 = h(%6, %6)
TORCH_API void replaceAllUsesWith(Value* newValue);

  // 替换该值在指定节点之后的所有使用位置为新值
// 示例：
//   %3 = f(%1, %2)
//   %4 = g(%3)
//   %5 = inplace_(%3)
//   %6 = h(%3, %3)
//   %3.replaceAllUsesAfterNodeWith(%5.node(), %5)
// 替换后：
//   %3 = f(%1, %2)
//   %4 = g(%3)
//   %5 = inplace_(%3)
//   %6 = h(%5, %5)
// 注意：不检查作用域的合法性，请考虑使用 replaceAllUsesDominatedByNodeWith
TORCH_API void replaceAllUsesAfterNodeWith(const Node* node, Value* newValue);

  // 替换该值在指定节点主导的所有使用位置为新值
// 示例：
//   x = op(...).
//   if cond:
//      z = foo(..)
//      bar(x)
//   else:
//      print(x)
//   x.replaceAllUsesDominatedByNodeWith(foo, z) 会替换 bar(x)，但不会替换 print(x)，因为 print 不受 foo 主导。
// replaceAllUsesAfterNode 不检查主导关系，在此示例中会产生无效的 IR。
TORCH_API void replaceAllUsesDominatedByNodeWith(
    const Node* node,
    Value* newValue);

  // 复制来自另一个值的元数据到该值
TORCH_API Value* copyMetadata(Value* from);

  // 返回该值的包装（wrap）指针，如果尚未创建，则创建一个新的包装指针
std::shared_ptr<Wrap<Value>> wrap() {
  if (!wrap_) {
    wrap_ = std::make_shared<Wrap<Value>>(this);
  }
  return wrap_;
}

  // 虚析构函数：如果存在包装指针，则清除其内容
virtual ~Value() {
  if (wrap_) {
    wrap_->clear();
  }
}
};

// 结构体 Node 的定义开始
struct TORCH_API Node {
  // 禁止拷贝和赋值构造函数
  AT_DISALLOW_COPY_AND_ASSIGN(Node);
  // 声明 Graph、Block、Value 为友元类
  friend struct Graph;
  friend struct Block;
  friend struct Value;
  // 声明几个类为友元类
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

 private:
  // 节点类型
  const NodeKind kind_;
  // 输入值的列表
  std::vector<Value*> inputs_;
  // 输出值的列表
  std::vector<Value*> outputs_;
  // 子块的列表
  std::vector<Block*> blocks_;
  // 节点所属的图
  Graph* graph_;
  // 拥有该节点的块
  Block* owning_block_;
  // 源代码范围的可选项
  std::optional<SourceRange> source_range_;
  // 作用域指针
  ScopePtr scope_;
  // 内联调用堆栈的可选项
  std::optional<InlinedCallStackPtr> callstack_;
  // 缓存的操作符指针，用于属性查找时的高效访问
  // 可变因为 schema_ 实际上是一个缓存
  mutable const Operator* op_;
  // 拓扑排序位置
  topo_position_t topo_position_ = 0;
  // Python 管理包装器，允许无效化
  std::shared_ptr<Wrap<Node>> wrap_;
  // 历史模式下的操作符名称
  std::optional<std::string> historic_schema_name_ = c10::nullopt;

 protected:
  // 构造函数，声明在 graph 之后
  Node(Graph* graph_, NodeKind kind_);

 public:
  // 获取历史模式下的操作符名称
  const std::optional<std::string> getHistoricSchemaName() {
    return historic_schema_name_;
  }

  // 设置历史模式下的操作符名称
  void setHistoricSchemaName(const std::string& name) {
    historic_schema_name_ = name;
  }

  // 返回下一个节点的引用
  Node*& next() {
    return next_in_graph[kNextDirection];
  }

  // 返回前一个节点的引用
  Node*& prev() {
    return next_in_graph[kPrevDirection];
  }

  // 返回常量引用的下一个节点
  Node* const& next() const {
    return next_in_graph[kNextDirection];
  }

  // 返回常量引用的前一个节点
  Node* const& prev() const {
    return next_in_graph[kPrevDirection];
  }

  // 返回节点的类型
  NodeKind kind() const {
  // 返回成员变量 kind_
  return kind_;
}
// 设置当前节点的源代码范围，并返回当前节点指针
Node* setSourceRange(SourceRange r) {
  source_range_ = std::move(r);
  return this;
}
// 返回当前节点的源代码范围
SourceRange sourceRange() const;

/**
 * @warning 永远不要将智能指针管理的 Graph 的原始指针传递给 Python。
 * 详细信息请查看 issue #87343。
 */
Graph* owningGraph() {
  return graph_;
}
const Graph* owningGraph() const {
  return graph_;
}
// 返回拥有当前节点的 Block 指针
Block* owningBlock() {
  return owning_block_;
}
const Block* owningBlock() const {
  return owning_block_;
}
// 返回当前节点的作用域指针
ScopePtr scope() {
  return scope_;
}
// 设置当前节点的作用域
void setScope(ScopePtr scope) {
  scope_ = std::move(scope);
}
// 返回当前节点的作用域名称
std::string scopeName() const {
  if (!scope_) {
    return "";
  }
  return scope_->namesFromRoot();
}

// 从另一个节点复制源代码范围、作用域和调用堆栈信息
Node* copyMetadata(Node* from) {
  this->setSourceRange(from->sourceRange());
  this->setScope(from->scope());
  if (auto cs = from->callstack()) {
    this->setCallStack(*cs);
  }
  return this;
}

// 返回当前节点的内联调用堆栈指针，可能为空
std::optional<InlinedCallStackPtr> callstack() const {
  return callstack_;
}
// 设置当前节点的内联调用堆栈
void setCallStack(InlinedCallStackPtr cs) {
  callstack_ = std::move(cs);
}

// 返回输入值的 ArrayRef
// 注意：如果调整输入（例如使用 addInput），此返回值将失效
at::ArrayRef<Value*> inputs() {
  return inputs_;
}
// 返回输入值的 const ArrayRef
// 注意：使用原始指针以便于在 const 上进行转换
at::ArrayRef<const Value*> inputs() const {
  return {inputs_.data(), inputs_.size()};
}
// 返回输出值的 ArrayRef
// 注意：如果调整输出（例如使用 addOutput），此返回值将失效
at::ArrayRef<Value*> outputs() {
  return outputs_;
}
// 返回输出值的 const ArrayRef
// 注意：使用原始指针以便于在 const 上进行转换
at::ArrayRef<const Value*> outputs() const {
  return {outputs_.data(), outputs_.size()};
}
// 返回指定索引处的输出值指针
Value* output(size_t i) const {
  return outputs_.at(i);
}
// 检查当前节点是否有使用
bool hasUses() const {
  for (auto o : outputs()) {
    if (!o->uses().empty()) {
      return true;
    }
  }
  // 返回 false
  return false;
}

void replaceAllUsesWith(Node* n);

// 用新的节点符号替换当前节点，新节点具有相同的输入和输出，但不销毁当前节点
Node* replaceWithNewSymbol(Symbol new_symbol);

// 检查当前节点是否被 dominator 支配，即 dominator 在当前节点之前执行，并且在当前节点的作用域内
bool isDominatedBy(const Node* dominator) const;

// 许多像 chunk 这样的操作只有一个输入或一个输出，因此有一个帮助函数使访问更容易
Value* input() {
  AT_ASSERT(inputs_.size() == 1); // 断言当前节点的输入数量为1
  return inputs_.at(0); // 返回第一个输入的指针
}
Value* output() {
  AT_ASSERT(outputs_.size() == 1); // 断言当前节点的输出数量为1
  return outputs_.at(0); // 返回第一个输出的指针
}
const Value* output() const {
  AT_ASSERT(outputs_.size() == 1); // 断言当前节点的输出数量为1
  return outputs_.at(0); // 返回第一个输出的指针
}
const Value* input() const {
  AT_ASSERT(inputs_.size() == 1); // 断言当前节点的输入数量为1
  return inputs_.at(0); // 返回第一个输入的指针
}
// 访问特定的输入，这是一个受检索引
Value* input(size_t i) const {
  return inputs_.at(i); // 返回指定索引处的输入的指针
}

// 检查是否存在指定名称的命名输入
bool hasNamedInput(const std::string& unqualName) const;

// 获取指定名称的命名输入的值
Value* namedInput(const std::string& unqualName) const;

// 获取指定名称的命名输入的值，使用符号作为参数
Value* namedInput(Symbol name) const;

// 获取指定名称的值，返回一个可选的 IValue 对象
std::optional<IValue> get(Symbol name) const;

template <typename T>
// 获取指定名称的值，并将其转换为模板类型 T，返回一个可选的 T 类型对象
std::optional<T> get(Symbol name) const {
  if (auto v = get(name)) {
    return v->template to<T>();
  }
  return c10::nullopt;
}

// 如果输入名称的值在静态上是已知的，则返回 true
bool is_constant(Symbol name) const {
  return static_cast<bool>(get(name));
}
bool mustBeNone() const;

// 返回 true 如果节点是非确定性的
bool isNondeterministic() const;

// 返回 true 如果节点具有副作用
bool hasSideEffects() const;

// 返回 true 如果指令是由解释器降低并且不在优化图中运行的操作
bool notExecutedOp() const {
  // 检查当前节点类型是否为常量、profile 或 profile_ivalue
  return kind_ == prim::Constant || kind_ == prim::profile ||
      kind_ == prim::profile_ivalue;
}

// Graphs

// 注意 [拓扑不变性]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 我们始终通过 next()/prev() 链路保持所有节点的最新拓扑排序。
// 所有对图的转换必须保持这种拓扑排序不变：例如，只有在当前节点之前的拓扑节点才能使用 'addInput' 添加输入。
//
// 通常情况下，是否保持拓扑排序是显而易见的；
// 例如，如果您将节点添加到拓扑排序的末尾，那么它们不可能引用不在拓扑排序中的输入。
// 如果不明显，请相应地注释。

// 将 'node' 添加为 'this' 的输入，添加到现有参数的末尾。返回添加的节点以便链式调用。
//
// 给定：   %3 = f(%1, %2)
// 执行：   %3.addInput(%4)
// 结果：   %3 = f(%1, %2, %4)
Value* addInput(Value* value);

// 将 'value' 添加为 'this' 的输入，在参数列表的指定位置。返回添加的值以便链式调用。
Value* insertInput(size_t i, Value* value);

// 用 'newValue' 替换 'this' 在位置 'i' 的输入，并返回旧节点。
//
// 给定：   %3 = f(%1, %2)
// 执行：   %3.replaceInput(1, %4)
// 结果：   %3 = f(%1, %4)
Value* replaceInput(size_t i, Value* newValue);

// 替换此节点输入中所有出现的 'from' 为 'to'。对应于 LLVM 的 replaceUsesOfWith。
//
// 给定：   %3 = f(%1, %2, %1)
// 执行：   %3.replaceInputWith(%1, %4)
// 结果：   %3 = f(%4, %2, %4)
void replaceInputWith(Value* from, Value* to);

// 添加一个输出值，并返回该值以便链式调用。
Value* addOutput();

// 在指定位置插入一个输出值，并返回该值以便链式调用。
Value* insertOutput(size_t i);

// 删除指定位置的输出值。
void eraseOutput(size_t i);

// 添加一个新的子块。
Block* addBlock();

// 删除指定位置的子块。
void eraseBlock(size_t i);

// 每个节点可以有一个子块列表。这些用于定义结构化嵌套控制流操作，如 If 和 Loop。
// 块的含义取决于它在节点中的类型，但所有块共享以下语义：
// * 嵌套词法作用域：如果节点 'Parent' 有一个子块，其中包含节点 'Child'，则 'Child' 可以使用父节点作用域中的任何值，以及子块中在 'Child' 之前定义的任何值。
// * 块的输入列表在块的整个持续时间内有效
// * 父节点的输出对于子块不可见
// 
// 通常情况下，代表控制流的块的输入作为标准 SSA 形式中等效的 phi 节点，用于定义一个新的 Value，以表示根据控制流如何流动而具有多个定义的术语。
// 节点包含控制流的输出具有类似的作用，定义新的值，用于变量根据控制流如何流动而具有不同定义的情况。

at::ArrayRef<Block*> blocks() {
  // 返回 blocks_ 的引用
  return blocks_;
}
// 返回 blocks_ 中 Block 指针的不可变引用
at::ArrayRef<const Block*> blocks() const {
  // 向量的元素的常量性质不可转换，但原始指针可以
  return {blocks_.data(), blocks_.size()};
}

// 是否在拓扑排序中 this 在 n 之前？
bool isBefore(const Node* n) const;

// 是否在拓扑排序中 this 在 n 之后？
bool isAfter(const Node* n) const;

// 在拓扑排序中，在 n 之前插入未附加的 this 节点。
// 返回 this（用于链式调用）。
//
// 示例：
//   Given:   %3 = f(%1, %2)
//            %4 = g(%3)
//   和未附加的： %5 = h(%1)
//   执行： %5.insertBefore(%4)
//   结果：  %3 = f(%1, %2)
//           %5 = h(%1)
//           %4 = g(%3)
Node* insertBefore(Node* n);

// 在拓扑排序中，在 n 之后插入未附加的 this 节点。
// 返回 this（用于链式调用）。
//
// 示例：
//   Given: %3 = f(%1, %2)
//          %4 = g(%3)
//   和未附加的： %5 = h(%1)
//   执行： %5.insertAfter(%4)
//   结果：  %3 = f(%1, %2)
//           %4 = g(%3)
//           %5 = h(%1)
Node* insertAfter(Node* n);

// 将已经在图中的 this 移动到 n 之后的拓扑顺序中。
//
// 注意：不检查值依赖关系是否保留，参见 AliasDb::moveAfterTopologicallyValid
//
// 示例：
//   Given: %2 = f(%1)
//          %3 = g(%1)
//   执行： %2.moveAfter(%3)
//   结果： %3 = g(%1)
//          %2 = f(%1)
//
void moveAfter(Node* n);

// 将已经在图中的节点 n 移动到 this 之前的拓扑顺序中。
//
// 注意：不检查值依赖关系是否保留，参见 AliasDb::moveBeforeTopologicallyValid
//
// 示例：
//   Given: %2 = f(%1)
//          %3 = g(%1)
//   执行： %3.moveBefore(%2)
//   结果： %3 = g(%1)
//          %2 = f(%1)
void moveBefore(Node* n);

// 从节点中删除第 i 个输入。
//
// 警告：这是关于输入数量的 O(n) 操作，所以避免重复调用 removeInput。
//
// 示例：
//   Given: %3 = f(%1, %2)
//   执行： %3.removeInput(1)
//   结果： %3 = f(%1)
void removeInput(size_t i);

// 从节点中移除所有输入。
//
// 示例：
//   Given: %3 = f(%1, %2)
//   执行： %3.removeAllInputs()
//   结果： %3 = f()
void removeAllInputs();

// 从节点中移除所有输出。
//
// 示例：
//   Given: %1, %2 = f()
//   执行： removeAllInputs()
//   结果： f()
void removeAllOutputs();

// 重新排列节点的输入或输出顺序
// 给定： %3 = f(%1, %2)
// 执行： %3.permuteInputs({1, 0})
// 结果： %3 = f(%2, %1)
// 每个索引必须仅出现一次
void permuteInputs(const std::vector<size_t>& new_inputs);
void permuteOutputs(const std::vector<size_t>& new_inputs);

// 从此节点开始的节点列表迭代器
// 对于从此节点开始的搜索很有用
inline graph_node_list_iterator iterator() {
  return {this, 0};
}
inline graph_node_list_iterator reverseIterator() {
  // 返回一个迭代器并对其进行反转操作
  return iterator().reverse();
}

// 返回常量图节点列表迭代器，起始位置为索引0
inline const_graph_node_list_iterator iterator() const {
  return {this, 0};
}

// 返回常量图节点列表迭代器，并对其进行反转操作
inline const_graph_node_list_iterator reverseIterator() const {
  return iterator().reverse();
}

// 从指令列表中移除当前节点并释放其内存
//
// 不变量：当前节点的输出不能被任何其他地方使用
//
// 示例：给定 %2 = f(%1)
//       %3 = g(%1)
// 执行： %2.destroy()
// 结果： %3 = g(%1)
void destroy();

// 将当前节点动态转换为指定模板变量所示的子类，如果转换无效则返回 nullptr
//
// 示例用法：if(auto s = n.cast<Select>()) { ... }
template <typename T>
T* cast() {
  if (T::Kind == kind()) {
    return static_cast<T*>(this);
  }
  return nullptr;
}

template <typename T>
const T* cast() const {
  if (T::Kind == kind()) {
    return static_cast<const T*>(this);
  }
  return nullptr;
}

// 断言当前节点匹配给定的函数模式
bool matches(const FunctionSchema& schema) const;

// XXX: 此函数仅用于字符串字面值！
bool matches(
    const char* signature_literal,
    at::ArrayRef<Symbol> const_inputs = {}) const;

// 检查当前节点是否属于指定的运算符集合
bool isMemberOf(const OperatorSet& os) const;

// 检查当前节点是否属于指定的运算符映射集合
template <typename T>
bool isMemberOf(const OperatorMap<T>& om) const {
  auto it = om.map.find(kind());
  if (it == om.map.end()) {
    return false;
  }
  for (auto& op : it->second) {
    if (matches(op.first->schema())) {
      return true;
    }
  }
  return false;
}

// 返回当前节点的函数模式
const FunctionSchema& schema() const;

// 返回当前节点的可能函数模式，若无则返回 nullptr
const FunctionSchema* maybeSchema() const;

// 返回当前节点的操作符
const Operator& getOperator() const;

// 返回当前节点的操作
Operation getOperation() const;

// 返回当前节点的可能操作符，若无则返回 nullptr
const Operator* maybeOperator() const;

// 打印当前节点的信息
void dump() const;

// 将当前节点的信息打印到输出流中
std::ostream& print(
    std::ostream& out,
    size_t level,
    std::vector<const Node*>* groups,
    bool print_source_locations = true,
    bool print_attributes = true,
    bool print_scopes = true,
    bool print_body = true) const;

// 虚析构函数，用于清除包装
virtual ~Node() {
  if (wrap_) {
    wrap_->clear();
  }
}

// 方法用于访问属性
Node* copyAttributes(const Node& rhs) {
  values_.clear();
  for (const AVPtr& i : rhs.values_) {
    values_.push_back(i->clone());
  }
  return this;
}

// 检查当前节点是否具有指定名称的属性
bool hasAttribute(Symbol name) const {
  AT_ASSERT(name.is_attr());
  return findAttr(name, false) != values_.end();
}

// 检查当前节点是否具有指定名称的属性
bool hasAttributeS(const std::string& name) const {
  return hasAttribute(Symbol::attr(name));
}

// 返回指定属性名称的属性种类
AttributeKind kindOf(Symbol name) const {
  AT_ASSERT(name.is_attr());
  return (*findAttr(name, true))->kind();
}

// 返回指定属性名称的属性种类
AttributeKind kindOfS(const std::string& name) const {
  // 返回特定名称属性的类型
  return kindOf(Symbol::attr(name));
}
// 移除具有特定名称的属性节点
Node* removeAttribute(Symbol name) {
  AT_ASSERT(name.is_attr());
  // 通过名称查找并移除对应的属性值
  values_.erase(findAttr(name, true));
  // 返回当前节点的指针
  return this;
}
// 移除具有特定名称的属性节点
Node* removeAttributeS(const std::string& name) {
  // 调用 removeAttribute 方法，传入名称对应的符号对象
  return removeAttribute(Symbol::attr(name));
}
// 检查节点是否具有任何属性
bool hasAttributes() const {
  // 返回节点的属性集合是否为空
  return !values_.empty();
}
// 返回节点当前的属性数量
size_t numAttributes() const {
  // 返回节点属性集合的大小
  return values_.size();
}
// 返回节点属性的名称列表，按顺序排列
// 注意：这里的名称实际上是属性的索引。
std::vector<Symbol> attributeNames() const {
  // 创建一个向量用于存储属性名称
  std::vector<Symbol> names;
  // 预先分配足够的空间以避免多次重新分配
  names.reserve(values_.size());
  // 遍历属性值集合，将每个属性的名称添加到向量中
  for (const AVPtr& a : values_) {
    names.push_back(a->name);
  }
  // 返回存储属性名称的向量
  return names;
}
// 返回节点属性的名称列表，以字符串数组形式返回
std::vector<const char*> attributeNamesS() const {
  // 创建一个向量用于存储属性名称的 C 字符串形式
  std::vector<const char*> names;
  // 预先分配足够的空间以避免多次重新分配
  names.reserve(values_.size());
  // 遍历属性值集合，将每个属性的名称（转为非限定字符串形式）添加到向量中
  for (const AVPtr& a : values_) {
    names.push_back(a->name.toUnqualString());
  }
  // 返回存储属性名称的向量
  return names;
}
#define CREATE_ACCESSOR(Kind, method)                           \
  Node* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    // 调用 setAttr 方法设置指定属性名的属性值，并返回当前节点指针
    return setAttr<Kind##Attr>(                                 \
        name, std::forward<Kind##Attr::ConstructorType>(v));    \
  }                                                             \
  // 返回指定属性名的属性值引用（const 引用）
  const Kind##Attr::ValueType& method(Symbol name) const {      \
    return getAttr<Kind##Attr>(name);                           \
  }

CREATE_ACCESSOR(Float, f)
CREATE_ACCESSOR(Complex, c)
CREATE_ACCESSOR(Floats, fs)
CREATE_ACCESSOR(ComplexVals, cs)
CREATE_ACCESSOR(String, s)
CREATE_ACCESSOR(Strings, ss)
CREATE_ACCESSOR(Int, i)
CREATE_ACCESSOR(Ints, is)
CREATE_ACCESSOR(Graph, g)
CREATE_ACCESSOR(Graphs, gs)
CREATE_ACCESSOR(Type, ty)
CREATE_ACCESSOR(Types, tys)
CREATE_ACCESSOR(IValue, ival)

#undef CREATE_ACCESSOR

// Our Graphs are not very const-correct, so we need to allow returning
// non-const references too
// 返回指定属性名的 Graph 属性值引用（非 const 引用）
GraphAttr::ValueType& g(Symbol name) {
  return getAttr<GraphAttr>(name);
}

// does not use CREATE_ACCESSOR because we need additional asserts
// 调用 setAttr 方法设置 Tensor 属性值，并返回当前节点指针
Node* t_(Symbol name, TensorAttr::ConstructorType v) {
  return setAttr<TensorAttr>(
      name, std::forward<TensorAttr::ConstructorType>(v));
}
// 返回指定属性名的 Tensor 属性值引用（const 引用）
const TensorAttr::ValueType& t(Symbol name) const {
  return getAttr<TensorAttr>(name);
}

// 调用 setAttr 方法设置 Tensors 属性值，并返回当前节点指针
Node* ts_(Symbol name, TensorsAttr::ConstructorType v) {
  return setAttr<TensorsAttr>(
      name, std::forward<TensorsAttr::ConstructorType>(v));
}
// 返回指定属性名的 Tensors 属性值引用（const 引用）
const TensorsAttr::ValueType& ts(Symbol name) const {
  return getAttr<TensorsAttr>(name);
}

// 声明一个 Block 类型的指针，用于查找与当前节点共同祖先的 Block
Block* findCommonAncestorBlockWith(Node* n);

// 返回从 Graph Block 开始的 Block 数量
size_t blocksFromGraphBlock();

private:
void printAttrValue(std::ostream& out, const Symbol& name) const;
void printAttributes(std::ostream& out, bool ignore_subgraph) const;

// 模板方法：设置指定属性名的属性值
template <typename T>
Node* setAttr(Symbol name, typename T::ConstructorType v) {
  AT_ASSERT(name.is_attr());
  auto it = findAttr(name, false);
  auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 如果找不到属性，则将新属性值插入到 values_ 的末尾；否则替换原有属性值
  if (it == values_.end()) {
    values_.push_back(std::move(nv));
  } else {
    *it = std::move(nv);
  }
  return this;
}

// 模板方法：获取指定属性名的属性值引用
template <typename T>
typename T::ValueType& getAttr(Symbol name) const {
  AT_ASSERT(name.is_attr());
  auto it = findAttr(name, true);
  auto* child = dynamic_cast<T*>(it->get());
  if (child == nullptr) {
    throw IRAttributeError(name, true);
  }
  return child->value();
}

using AVPtr = AttributeValue::Ptr;
// NB: For determinism, we use a vector rather than a hash map.  This does
// mean that lookups are O(n), so you shouldn't use Attributes to store
// a big pile of messages.
// 存储节点属性值的容器，使用 vector 而非 hash map，用于保证确定性
std::vector<AVPtr> values_;

// 在 values_ 中查找指定名称的属性，如果 required 为 true，则要求属性必须存在
std::vector<AVPtr>::iterator findAttr(Symbol name, bool required) {
  AT_ASSERT(name.is_attr());
  // 在 values_ 中查找名称为 name 的属性，并返回迭代器
  auto it = std::find_if(values_.begin(), values_.end(),
                         [&name](const AVPtr& av) {
                           return av->name() == name;
                         });
  // 如果要求属性必须存在且未找到，则抛出异常
  if (required && it == values_.end()) {
    throw IRAttributeError(name, true);
  }
  return it;
}
    // 在 values_ 向量中查找满足特定条件的元素迭代器
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    // 如果 required 为 true 并且未找到对应元素，则抛出 IRAttributeError 异常
    if (required && it == values_.end()) {
      throw IRAttributeError(name, false);
    }
    // 断言检查：如果 required 为 true，则确保找到了对应元素；如果 required 为 false，则可以找到或者未找到都可以
    AT_ASSERT(!required || it != values_.end());
    // 返回找到的元素的迭代器
    return it;
  }

  // 在 values_ 向量中查找满足特定条件的元素迭代器（const 版本）
  std::vector<AVPtr>::const_iterator findAttr(Symbol name, bool required)
      const {
    // 断言检查：确保 name 是一个属性名
    AT_ASSERT(name.is_attr());
    // 在 values_ 向量中查找满足特定条件的元素迭代器
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    // 如果 required 为 true 并且未找到对应元素，则抛出 IRAttributeError 异常
    if (required && it == values_.end()) {
      throw IRAttributeError(name, false);
    }
    // 断言检查：如果 required 为 true，则确保找到了对应元素；如果 required 为 false，则可以找到或者未找到都可以
    AT_ASSERT(!required || it != values_.end());
    // 返回找到的元素的迭代器
    return it;
  }

  // 枚举类型 MoveSide 的两个值：BEFORE 和 AFTER
  enum class MoveSide { BEFORE, AFTER };

  // 检查节点 n 是在移动操作的方向 moveSide 之前还是之后
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  // 查找输入参数名为 name 的输入值，并返回值和对应的参数
  std::pair<Value*, const Argument&> findInput(Symbol name);

  // 查找输入参数 i 的使用迭代器，该迭代器对应于它在使用列表中的位置
  use_list::iterator findUseForInput(size_t i);

  // 移除输入参数 i 的使用，将其设置为 nullptr，在设置新值或从列表中擦除之前只在节点内部使用
  Value* dropInput(size_t i);

  // 检查当前节点是否在某个块（block）的列表中
  bool inBlockList() const {
    // 如果下一个节点为 nullptr，则确保前一个节点也为 nullptr（断言检查）
    if (next() == nullptr) {
      AT_ASSERT(prev() == nullptr);
    }
    // 返回当前节点是否在某个块的列表中的结果
    return next() != nullptr;
  }

  // 从节点列表中移除当前节点
  void removeFromList();

  // 对当前节点进行 lint（代码风格检查）
  void lint() const;

  // 分配拓扑位置给当前节点
  void assignTopoPosition();

 protected:
  // 子类必须重写的虚函数
  // 该函数用于在另一个图中创建克隆节点时初始化节点的新实例
  // 应该分配与 'this' 相同具体类型的新实例，但在可能不同的图 'g' 中
  virtual Node* allocNewInstance(Graph* g) {
    return new Node(g, kind());
  }

  // 将节点 s 的所有属性复制到当前节点
  // 子类如果有额外信息需要复制，应该扩展这个函数
  // 'this' 将使用 s->allocNewInstance(g) 分配，因此应该与 's' 具有相同的具体类型
  virtual void cloneFrom(Node* s);
};

struct Block {
  friend struct Node;
  friend struct Graph;

  // 禁止拷贝和赋值构造函数
  AT_DISALLOW_COPY_AND_ASSIGN(Block);
  
  // 构造函数，初始化块所属的图和起始节点
  TORCH_API Block(Graph* graph_, Node* node_);

  // 返回输入值的引用数组
  at::ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  
  // 返回常量输入值的引用数组
  at::ArrayRef<const Value*> inputs() const {
    // 获取常量输入的输出值引用数组
    const auto& inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  
  // 返回输出值的引用数组
  at::ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  
  // 返回常量输出值的引用数组
  at::ArrayRef<const Value*> outputs() const {
    // 强制类型转换为常量节点，返回其输入值引用数组
    return static_cast<const Node*>(output_)->inputs();
  }
  
  // 返回块中的节点列表
  graph_node_list nodes() {
    return {input_, kNextDirection};
  }
  
  // 返回常量块中的节点列表
  const_graph_node_list nodes() const {
    return {input_, kNextDirection};
  }
  
  // 返回输出节点
  Node* return_node() {
    return output_;
  }
  
  // 返回常量输出节点
  const Node* return_node() const {
    return output_;
  }
  
  // 返回输入节点
  Node* param_node() {
    return input_;
  }
  
  // 返回常量输入节点
  const Node* param_node() const {
    return input_;
  }
  
  /**
   * 获取所属图对象
   * @warning 绝对不要将智能指针管理的图对象的原始指针传递给Python。详情请查看#87343。
   */
  Graph* owningGraph() {
    return graph_;
  }
  
  // 获取常量所属图对象
  const Graph* owningGraph() const {
    return graph_;
  }
  
  // 获取拥有节点
  Node* owningNode() {
    return owning_node_;
  }
  
  // 获取常量拥有节点
  const Node* owningNode() const {
    return owning_node_;
  }

  // 添加输入值并设置调试名称
  Value* addInput(const std::string& name = "") {
    Value* v = input_->addOutput();
    v->setDebugName(name);
    return v;
  }
  
  // 插入输入值并设置调试名称
  Value* insertInput(size_t i, const std::string& name = "") {
    Value* v = input_->insertOutput(i);
    v->setDebugName(name);
    return v;
  }
  
  // 删除指定位置的输入值
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  
  // 删除所有输入值
  void removeAllInputs() {
    input_->removeAllOutputs();
  }
  
  // 注册输出值并返回其索引
  size_t registerOutput(Value* v) {
    output_->addInput(v);
    return outputs().size() - 1;
  }
  
  // 在指定位置插入输出值并返回其索引
  size_t insertOutput(size_t i, Value* n) {
    output_->insertInput(i, n);
    return i;
  }
  
  // 删除指定位置的输出值
  void eraseOutput(size_t i) {
    output_->removeInput(i);
  }
  
  // 删除所有输出值
  void removeAllOutputs() {
    output_->removeAllInputs();
  }

  // 替换指定位置的输出值
  void replaceOutput(size_t i, Value* n) {
    output_->replaceInput(i, n);
  }
  
  // 重新排列输出值
  void permuteOutputs(const std::vector<size_t>& new_inputs) {
    output_->permuteInputs(new_inputs);
  }
  
  // 重新排列输入值
  void permuteInputs(const std::vector<size_t>& new_inputs) {
    input_->permuteOutputs(new_inputs);
  }

  // 在块末尾添加节点
  Node* appendNode(Node* n) {
    // 断言节点属于相同的图且不在块列表中
    AT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertBefore(output_);
    return n;
  }
  
  // 在块开头添加节点
  Node* prependNode(Node* n) {
    // 断言节点属于相同的图且不在块列表中
    AT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertAfter(input_);
    return n;
  }

  // 从另一个块复制所有输入、节点和输出，并追加到当前块的输入、节点和输出
  // 当源块中的节点引用源块中的自由变量时，使用value_map查找对应的值
  TORCH_API void cloneFrom(Block* src, std::function<Value*(Value*)> value_map);
  
  // 重新映射类型
  TORCH_API void remapTypes(const std::function<TypePtr(TypePtr)>& type_map);

  // 包装为Block的智能指针
  TORCH_API std::shared_ptr<Wrap<Block>> wrap() {
    // 如果 wrap_ 指针为空，则创建一个 Wrap 对象并赋给 wrap_
    if (!wrap_) {
      wrap_ = std::make_shared<Wrap<Block>>(this);
    }
    // 返回 wrap_ 指针，可能是新创建的，也可能是之前存在的
    return wrap_;
  }

  // 虚析构函数，用于释放资源
  virtual ~Block() {
    // 如果 wrap_ 指针不为空，调用其 clear 方法
    if (wrap_) {
      wrap_->clear();
    }
  }

  // 清空 Block 对象，包括移除所有输出和所有节点
  void clear() {
    // 移除所有输出
    removeAllOutputs();
    // 以逆序销毁所有节点，以确保在销毁 Block 之前，不需要先移除内部使用的节点
    for (auto it = nodes().rbegin(); it != nodes().rend(); it++) {
      it.destroyCurrent();
    }
    // 移除所有输入
    removeAllInputs();
  }

 private:
  // 重新索引拓扑结构
  void reIndexTopology();

  // 销毁 Block 对象，用于内部调用
  void destroy();

  // 指向当前 Graph 对象的指针
  Graph* const graph_;
  // 输出节点链表的头指针
  Node* const output_;
  // 输入节点链表的头指针
  Node* const input_;
  // 拥有该 Block 的节点，若为根节点则为 nullptr
  Node* const owning_node_;
  // 用于 Python 管理的包装器，允许失效
  std::shared_ptr<Wrap<Block>> wrap_;
  };

struct Graph : std::enable_shared_from_this<Graph> {
  AT_DISALLOW_COPY_AND_ASSIGN(Graph);  // 禁止复制和赋值操作
  friend struct Node;
  friend struct Value;
  friend struct Block;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;  // 存储所有节点的集合
  std::unordered_set<const Value*> all_values;  // 存储所有值的集合
  std::unordered_set<const Block*> all_blocks;  // 存储所有块的集合
  size_t next_unique_;  // 下一个唯一标识符的值

  std::unordered_map<std::string, Value*> unique_names_;  // 存储唯一命名的值的映射表
  // name_base_suffix tracks largest suffix currently used by all names sharing
  // same name_base. Key of this map is name_base, value is largest suffix
  // numeric value.
  std::unordered_map<std::string, size_t> name_base_suffix_;  // 跟踪具有相同名称基础的名称的最大后缀数值的映射表

  ScopePtr current_scope_;  // 当前作用域指针

  Block* const block_;  // 图中的顶级块指针
  // when insertNode() is called, the node is inserted before this node
  // by default this is set to append to the top level block
  Node* insert_before_;  // 插入节点时的插入位置，默认为追加到顶级块
  int64_t predicted_insert_count_ = 0;  // 预测的插入计数，默认为0

  std::optional<size_t> op_version_;  // 操作版本号

 public:
  Graph(ScopePtr scope_root = c10::make_intrusive<Scope>())
      : next_unique_(0),
        current_scope_(std::move(scope_root)),
        block_(new Block(this, nullptr)),
        insert_before_(return_node()) {}

  at::ArrayRef<Value*> inputs() {
    return block_->inputs();  // 返回顶级块的输入值引用
  }
  at::ArrayRef<const Value*> inputs() const {
    const Block& block = *block_;
    return block.inputs();  // 返回顶级块的输入值引用（常量版本）
  }
  at::ArrayRef<Value*> outputs() {
    return block_->outputs();  // 返回顶级块的输出值引用
  }
  at::ArrayRef<const Value*> outputs() const {
    const Block& block = *block_;
    return block.outputs();  // 返回顶级块的输出值引用（常量版本）
  }
  graph_node_list nodes() {
    return block_->nodes();  // 返回顶级块的节点列表
  }
  const_graph_node_list nodes() const {
    const Block& block = *block_;
    return block.nodes();  // 返回顶级块的节点列表（常量版本）
  }
  Node* param_node() {
    return block_->param_node();  // 返回顶级块的参数节点
  }
  const Node* param_node() const {
    return block_->param_node();  // 返回顶级块的参数节点（常量版本）
  }
  Node* return_node() {
    return block_->return_node();  // 返回顶级块的返回节点
  }
  const Node* return_node() const {
    return block_->return_node();  // 返回顶级块的返回节点（常量版本）
  }
  const std::unordered_map<std::string, Value*>& debugNames() const {
    return unique_names_;  // 返回调试名称映射表
  }

  TORCH_API void push_scope(const std::string& scope_name);  // 推入作用域
  TORCH_API void pop_scope();  // 弹出作用域

  ScopePtr current_scope() {
    return current_scope_;  // 返回当前作用域指针
  }

  void set_op_version(std::optional<size_t> version) {
    op_version_ = version;  // 设置操作版本号
  }

  std::optional<size_t> get_op_version() {
    return op_version_;  // 获取操作版本号
  }

  void set_current_scope(ScopePtr scope) {
    current_scope_ = std::move(scope);  // 设置当前作用域
  }

  Value* addInput(const std::string& name = "") {
    return block_->addInput(name);  // 添加输入值到顶级块
  }
  Value* insertInput(size_t i, const std::string& name = "") {
    return block_->insertInput(i, name);  // 在指定位置插入输入值到顶级块
  }
  void eraseInput(size_t i) {
    block_->eraseInput(i);  // 删除指定位置的输入值
  }
  size_t registerOutput(Value* n) {
    return block_->registerOutput(n);  // 注册输出值到顶级块
  }
  void eraseOutput(size_t i) {
    block_->eraseOutput(i);


// 删除当前块的第 i 个输出
block_->eraseOutput(i);



  TORCH_API Node* create(NodeKind kind, size_t num_outputs = 1);


// 创建一个指定类型和输出数量的新节点
TORCH_API Node* create(NodeKind kind, size_t num_outputs = 1);



  TORCH_API Node* create(
      NodeKind kind,
      ArrayRef<Value*> inputs,
      size_t num_outputs = 1);


// 创建一个指定类型、输入和输出数量的新节点
TORCH_API Node* create(
    NodeKind kind,
    ArrayRef<Value*> inputs,
    size_t num_outputs = 1);



  TORCH_API Node* createNone();


// 创建一个表示None的节点
TORCH_API Node* createNone();



  TORCH_API Node* createAutogradZero();


// 创建一个表示Autograd Zero的节点
TORCH_API Node* createAutogradZero();



  TORCH_API Node* createUninitialized(TypePtr typ);


// 创建一个未初始化的节点，指定类型由typ确定
TORCH_API Node* createUninitialized(TypePtr typ);



  TORCH_API Node* createWithSubgraph(Symbol kind);


// 创建一个包含子图的节点，子图的类型由kind指定
TORCH_API Node* createWithSubgraph(Symbol kind);



  TORCH_API Node* createDifferentiableSubgraph();


// 创建一个可微分子图的节点
TORCH_API Node* createDifferentiableSubgraph();



  TORCH_API Node* createTuple(
      at::ArrayRef<Value*> values,
      TupleTypePtr optional_named_tuple = nullptr);


// 创建一个元组节点，包含给定的值，可选地包含命名元组类型
TORCH_API Node* createTuple(
    at::ArrayRef<Value*> values,
    TupleTypePtr optional_named_tuple = nullptr);



  TORCH_API Node* createTupleUnpack(Value* v);


// 创建一个元组解包节点，解包的对象为v
TORCH_API Node* createTupleUnpack(Value* v);



  TORCH_API Node* createTupleIndex(
      Value* tup,
      Value* idx,
      const TypePtr& output_type);


// 创建一个元组索引节点，从tup元组中取出索引为idx的元素，输出类型由output_type指定
TORCH_API Node* createTupleIndex(
    Value* tup,
    Value* idx,
    const TypePtr& output_type);



  TORCH_API Node* createTupleSlice(
      Value* tup,
      int64_t beg,
      int64_t step_size,
      int64_t num_values);


// 创建一个元组切片节点，从tup元组中切出起始位置为beg、步长为step_size、元素数量为num_values的切片
TORCH_API Node* createTupleSlice(
    Value* tup,
    int64_t beg,
    int64_t step_size,
    int64_t num_values);



  TORCH_API Node* createEnumName(Value* e);


// 创建一个枚举名称节点，表示枚举值e的名称
TORCH_API Node* createEnumName(Value* e);



  TORCH_API Node* createEnumValue(Value* e);


// 创建一个枚举值节点，表示枚举值e的值
TORCH_API Node* createEnumValue(Value* e);



  TORCH_API Node* createList(
      const TypePtr& contained_type,
      at::ArrayRef<Value*> values);


// 创建一个列表节点，列表中包含类型为contained_type的值values
TORCH_API Node* createList(
    const TypePtr& contained_type,
    at::ArrayRef<Value*> values);



  TORCH_API Node* createListUnpack(Value* v, size_t size);


// 创建一个列表解包节点，解包的对象为v，解包后的大小为size
TORCH_API Node* createListUnpack(Value* v, size_t size);



  TORCH_API Node* createDict(
      const TypePtr& key_type,
      const TypePtr& value_type,
      at::ArrayRef<Value*> keys,
      at::ArrayRef<Value*> values);


// 创建一个字典节点，字典中键的类型为key_type，值的类型为value_type，包含键值对keys和values
TORCH_API Node* createDict(
    const TypePtr& key_type,
    const TypePtr& value_type,
    at::ArrayRef<Value*> keys,
    at::ArrayRef<Value*> values);



  TORCH_API Node* createNumToTensor(Value* value);


// 创建一个将数值转换为张量的节点，数值为value
TORCH_API Node* createNumToTensor(Value* value);



  TORCH_API Node* createObject(const ClassTypePtr& type);


// 创建一个对象节点，对象的类型为type
TORCH_API Node* createObject(const ClassTypePtr& type);



  TORCH_API Node* createSetAttr(
      Value* obj,
      const std::string& field,
      Value* newValue);


// 创建一个设置对象属性节点，将obj对象的属性field设置为newValue
TORCH_API Node* createSetAttr(
    Value* obj,
    const std::string& field,
    Value* newValue);



  TORCH_API Node* createGetAttr(Value* obj, const std::string& field);


// 创建一个获取对象属性节点，获取obj对象的属性field的值
TORCH_API Node* createGetAttr(Value* obj, const std::string& field);



  Value* insertGetAttr(Value* obj, const std::string& field) {


// 在给定对象obj中插入获取属性的操作，属性为field，返回获取的值
Value* insertGetAttr(Value* obj, const std::string& field) {
    // 返回插入一个创建GetAttr对象和字段名的节点后的输出
    return insertNode(createGetAttr(obj, field))->output();
    }
    // 声明一个创建存储操作节点的函数，接受名称和值作为参数
    TORCH_API Node* createStore(const std::string& name, Value* v);
    // 声明一个创建加载操作节点的函数，接受名称和类型指针作为参数
    TORCH_API Node* createLoad(const std::string& name, const TypePtr& type);
    // 声明一个创建IsInstance操作节点的函数，接受值和类型数组作为参数
    TORCH_API Node* createIsInstance(Value* v, at::ArrayRef<TypePtr> types);
    
    // 声明一个插入未检查类型转换操作的函数，接受值和类型指针作为参数
    TORCH_API Value* insertUncheckedCast(Value* v, TypePtr type);
    
    // 声明一个插入ToList操作的函数，接受值和类型指针作为参数
    // 返回操作的输出
    TORCH_API Value* insertToList(Value* v, TypePtr type);
    
    // 声明一个插入函数调用操作的函数，接受函数指针和匹配的模式作为参数
    TORCH_API Value* insertFunctionCall(
        Function* callee,
        const MatchedSchema& matched);
    // 声明一个插入方法调用操作的函数，接受方法名和匹配的模式作为参数
    TORCH_API Value* insertMethodCall(
        std::string method_name,
        const MatchedSchema& matched);
    
    // 注意：在python_ir.cpp中定义，只能在Python扩展中使用
    // 声明一个创建Python操作节点的函数，接受Python对象指针、转换类型和标量参数数组作为参数
    Node* createPythonOp(
        THPObjectPtr&& pyobj,
        const std::string& cconv,
        pyobj_list&& scalar_args);
    // 克隆节点n，在当前图中创建一个新节点
    // 使用value_map将n的输入转换为克隆节点的输入
    // 如果copy_blocks为false，则不递归克隆此节点包含的嵌套块
    TORCH_API Node* createClone(
        Node* n,
        const std::function<Value*(Value*)>& value_map,
        bool copy_blocks = true);
    
    // 插入常量IValue到图中
    TORCH_API Value* insertConstant(
        const IValue& val,
        std::optional<SourceRange> loc = c10::nullopt,
        std::optional<ScopePtr> scope = c10::nullopt);
    
    // 基于Schema插入：
    // 根据Python参数匹配规则，将一个节点插入到图中，使用args和kwargs确定输入，并检查操作是否匹配已知的Schema
    // 如果节点成功完成，保证节点是opname的正确形式调用
    TORCH_API Value* insert(
        Symbol opname,
        at::ArrayRef<NamedValue> args,
        at::ArrayRef<NamedValue> kwargs = {},
        const std::optional<SourceRange>& range = {});
    
    // 在当前块的末尾追加节点n
    Node* appendNode(Node* n) {
      return block_->appendNode(n);
    }
    
    // 在当前块的开头插入节点n
    Node* prependNode(Node* n) {
      return block_->prependNode(n);
    }
    
    // 在insert_before_节点之前插入节点n
    // 初始值为在顶层块的末尾插入
    // 可以使用setInsertPoint()更改
    Node* insertNode(Node* n) {
      AT_ASSERT(
          insert_before_->inBlockList() &&
          "insert point node is no longer in a block list");
      return n->insertBefore(insert_before_);
    }
    
    // 将节点插入到指定块b的末尾
    void setInsertPoint(Block* b) {
      AT_ASSERT(b->owningGraph() == this);
      setInsertPoint(b->return_node());
    }
    
    // 将节点插入到指定节点n之前
    // 目前仅支持在节点之前插入
    void setInsertPoint(Node* n) {
      AT_ASSERT(n->owningGraph() == this && n->inBlockList());
      insert_before_ = n;
      predicted_insert_count_ = 0;
    }
    
    // 返回当前插入点节点
    Node* insertPoint() {
    // 返回 insert_before_ 成员变量的值
      return insert_before_;
    }
    
    // 返回 block_ 成员变量的值，用于获取顶层块
    Block* block() {
      return block_;
    }
    
    // 返回 block_ 成员变量的值的常量版本，用于获取顶层块
    const Block* block() const {
      return block_;
    }
    
    // 检查图的良好形式和不变量
    TORCH_API void lint() const;
    
    // 用于调试器中使用，打印图的内容
    TORCH_API void dump() const;
    
    // 图对象的析构函数
    TORCH_API ~Graph();
    
    // 将图对象转换成字符串表示形式
    TORCH_API std::string toString(bool print_source_locations = true) const;
    
    // 打印图对象到输出流中
    TORCH_API std::ostream& print(
        std::ostream& out,
        bool print_source_locations = true) const;
    
    // 输出流操作符重载，用于打印图对象
    friend TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);
    
    // 复制图对象，返回共享指针
    TORCH_API std::shared_ptr<Graph> copy();
    
    // 复制图对象，返回唯一指针
    TORCH_API std::unique_ptr<Graph> copyUnique();
    
    // 重新映射图中的类型，通过提供的类型映射函数
    TORCH_API void remapTypes(const std::function<TypePtr(TypePtr)>& type_map);
    
    private:
    // 友元函数，用于进行图的静态检查
    friend TORCH_API void Lint(const AliasDb* db);
    
    // 释放节点对象
    TORCH_API void freeNode(Node* n);
    
    // 释放值对象
    TORCH_API void freeValue(Value* v);
    
    // 释放块对象
    TORCH_API void freeBlock(Block* b);
    
    // 从另一个图对象 src 克隆当前图对象的内容
    void cloneFrom(Graph& src);
};

/** \brief An utility class for setting temporary insertion points.
 *
 * When an object of this class is created, it stores the current insertion
 * point, sets the new one, and restores the original insertion point when the
 * object is destroyed.
 */
struct WithInsertPoint {
  // Constructor for setting insertion point to a specific node
  WithInsertPoint(Node* n) : prev_(n->owningGraph()->insertPoint()) {
    n->owningGraph()->setInsertPoint(n);
  }
  // Constructor for setting insertion point to the return node of a block
  WithInsertPoint(Block* b) : WithInsertPoint(b->return_node()) {}

  // Destructor restores the previous insertion point
  ~WithInsertPoint() {
    prev_->owningGraph()->setInsertPoint(prev_);
  }

 private:
  Node* prev_;  ///< Pointer to the previous insertion point node
};

/** \brief An utility class for setting temporary scopes.
 *
 * When an object of this class is created, it stores the current scope, sets
 * the new one, and restores the original scope when the object is destroyed.
 */
struct WithCurrentScope {
  // Constructor sets the current scope to the provided scope
  WithCurrentScope(Graph& g, ScopePtr scope)
      : graph_(&g), prev_scope_(g.current_scope()) {
    g.set_current_scope(std::move(scope));
  }

  // Destructor restores the previous scope
  ~WithCurrentScope() {
    graph_->set_current_scope(prev_scope_);
  }

 private:
  Graph* graph_;     ///< Pointer to the graph whose scope is being managed
  ScopePtr prev_scope_;  ///< Pointer to the previous scope
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
inline Value::Value(Node* node_, size_t offset_)
    : node_(node_),
      offset_(offset_),
      unique_(node_->graph_->next_unique_++),
      type_(TensorType::get()) {
  node_->graph_->all_values.emplace(this);
}

inline Value* Value::setType(TypePtr type) {
  AT_ASSERT(type);
  if (auto dyn = type->castRaw<c10::DynamicType>()) {
    type = dyn->fallback();
  }
  type_ = std::move(type);
  // Reset all uses' operators when type is changed
  for (Use& use : uses_) {
    use.user->op_ = nullptr;
  }
  return this;
}

inline Graph* Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const {
  return node()->owningGraph();
}

/************* All nodes not required to be defined before Graph **************/
struct ProfileOp : public Node {
  static const Symbol Kind;
  
  // Constructor for ProfileOp node, setting the callback function
  ProfileOp(Graph* graph, std::function<void(std::vector<IValue>&)> callback)
      : Node(graph, ::c10::prim::profile), callback_(std::move(callback)) {}

  // Override function to clone from another node
  void cloneFrom(Node* other_) override;

  // Allocates a new instance of ProfileOp node
  Node* allocNewInstance(Graph* g) override;

  // Getter for the callback function
  const std::function<void(std::vector<IValue>&)>& getCallback() const {
    return callback_;
  }

  // Setter for the callback function
  void setCallback(std::function<void(std::vector<IValue>&)> callback) {
    callback_ = std::move(callback);
  }

  // Check if a tensor has been seen during profiling
  bool hasSeenTensor() const {
    return has_seen_tensor_;
  }

  // Set the flag indicating whether a tensor has been seen during profiling
  void setHasSeenTensor(bool has_seen_tensor) {
    has_seen_tensor_ = has_seen_tensor;
  }

 private:
  std::function<void(std::vector<IValue>&)> callback_;  ///< Callback function for profiling
  bool has_seen_tensor_ = false;  ///< Flag indicating if a tensor has been seen
};
// 定义一个继承自Node的ProfileIValueOp结构体，用于表示ProfileIValue操作节点
struct TORCH_API ProfileIValueOp : public Node {
  static const Symbol Kind;
  // 构造函数，初始化ProfileIValueOp对象
  ProfileIValueOp(
      Graph* graph,
      std::function<void(std::vector<IValue>&)> callback)
      : Node(graph, ::c10::prim::profile_ivalue),  // 调用基类Node的构造函数进行初始化
        callback_(std::move(callback)) {}  // 初始化回调函数

  void cloneFrom(Node* other_) override;  // 覆盖基类方法，实现节点的克隆操作
  Node* allocNewInstance(Graph* g) override;  // 覆盖基类方法，分配一个新的节点实例

  // 获取回调函数的引用
  const std::function<void(std::vector<IValue>&)>& getCallback() const {
    return callback_;
  }

  // 设置回调函数
  void setCallback(std::function<void(std::vector<IValue>&)> callback) {
    callback_ = std::move(callback);
  }

 private:
  std::function<void(std::vector<IValue>&)> callback_;  // 存储回调函数的成员变量
};

// 执行一个Python函数，用于我们无法优化但希望围绕其进行优化的操作
//
// 注意：实际的实现（ConcretePythonOp）定义在python_ir.cpp中，该文件不包含在libtorch.so中。
// 我们在这里包含部分PythonOp的代码以便编写通用的简单pass。一般来说，
// Python相关的部分需要移到后继的子类中。
struct TORCH_API PythonOp : public Node {
  using Node::Node;  // 继承Node类的构造函数

  virtual std::string name() const = 0;  // 纯虚函数，返回Python操作的名称
  virtual void writeScalars(std::ostream& out) const = 0;  // 纯虚函数，将标量写入流中
  void cloneFrom(Node* other_) override = 0;  // 覆盖基类方法，实现节点的克隆操作
  Node* allocNewInstance(Graph* g) override = 0;  // 覆盖基类方法，分配一个新的节点实例
  // 恢复autograd.Function实例，如果此PythonOp的函数最初是SomeFunction.apply，则在ONNX中用于发现符号
  virtual std::optional<THPObjectPtr> autogradFunction() const = 0;  // 纯虚函数

  virtual void lint_python() const = 0;  // 纯虚函数，用于Python代码的静态分析
};

// 对图进行Lint操作，检查图中的潜在问题
TORCH_API void LintGraph(const std::shared_ptr<Graph>& graph);

// 创建一个元组解包操作，将一个值解包为多个值
TORCH_API at::ArrayRef<Value*> createTupleUnpack(Value* v);

/** 插入图 \p CALLEE 到图 \p G 中，使用 \p INPUTS 作为输入值。
 * 插入操作发生在当前插入点。
 * 可选地，可以传递 \p VALUE_MAP，以获取 \p CALLEE 值和它们在 \p G 中克隆副本之间的映射。
 */
TORCH_API std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs);

/** 插入函数 \p CALLEE 到节点 \p TO_REPLACE 后，移除该节点并
 * 将其所有使用替换为插入函数的相应输出。
 * 这个函数断言原始节点和图的输出数量相同。
 */
TORCH_API std::vector<Value*> inlineCallTo(
    Node* to_replace,
    GraphFunction* callee,
    bool use_graph = true);

TORCH_API std::vector<Value*> inlineCallTo(
    Node* to_replace,
    GraphFunction* callee,
    Graph* callee_graph);

/** 如果 \p OUTPUTS 中只有一个值且其类型为Tuple，插入一个
 * 元组解包节点并返回结果值。
 */
TORCH_API std::vector<Value*> unpackOutputs(const std::vector<Value*>& outputs);
// 声明一个函数，用于在给定的图形对象中查找所有特定类型的节点，并返回节点指针的向量
TORCH_API std::vector<Node*> findAllNodes(Graph& g, Symbol kind, bool recurse);

// 声明一个函数，用于在给定的块对象中查找所有特定类型的节点，并返回节点指针的向量
TORCH_API std::vector<Node*> findAllNodes(Block& b, Symbol kind, bool recurse);

// 声明一个函数，用于在给定的块数组中查找所有特定类型的节点，并返回节点指针的向量
TORCH_API std::vector<Node*> findAllNodes(
    at::ArrayRef<Block*> a,
    Symbol kind,
    bool recurse);

// 定义一个结构体 OperatorSet
struct TORCH_API OperatorSet {
  // 构造函数，接受一个初始化列表作为参数，用于初始化操作符集合
  OperatorSet(std::initializer_list<const char*> sig_literals);
  
  // 返回内部操作符的共享指针向量
  std::vector<std::shared_ptr<Operator>> getOps() const;
  
  // 向操作符集合中插入操作符的签名字面值
  void insert(std::initializer_list<const char*> sig_literals);

 private:
  friend struct Node;
  // 符号到操作符共享指针向量的无序映射
  std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>> ops;
};

template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义一个模板结构 OperatorMap
struct OperatorMap {
  // 类型别名
  using OpMapType = typename std::pair<std::shared_ptr<Operator>, T>;
  using ValueType = std::vector<OpMapType>;
  using MapType = std::unordered_map<Symbol, ValueType>;

  // 默认构造函数
  OperatorMap() = default;
  
  // 构造函数，接受初始化列表作为参数，用于初始化操作符映射
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit OperatorMap(
      std::initializer_list<std::pair<std::shared_ptr<Operator>, T>> init) {
    insert(init);
  }
  
  // 构造函数，接受初始化列表作为参数，用于初始化操作符映射
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit OperatorMap(std::initializer_list<std::pair<const char*, T>> init) {
    insert(init);
  }

  // 插入操作符及其关联值到映射中
  void insert(const std::shared_ptr<Operator>& op, T val) {
    // 如果已存在相同操作符，则先移除
    erase(op);
    // 将操作符及其值添加到映射中
    map[Symbol::fromQualString(op->schema().name())].emplace_back(
        std::make_pair(op, val));
  }

  // 将操作符集合中的所有操作符及其关联值插入到映射中
  void insert(const OperatorSet& op_set, T val) {
    for (auto& op : op_set.getOps()) {
      insert(op, val);
    }
  }

  // 插入初始化列表中的所有操作符及其关联值到映射中
  void insert(
      std::initializer_list<std::pair<std::shared_ptr<Operator>, T>> v) {
    for (auto& el : v) {
      insert(el.first, el.second);
    }
  }

  // 插入初始化列表中的所有操作符字面值及其关联值到映射中
  void insert(std::initializer_list<std::pair<const char*, T>> v) {
    for (auto& el : v) {
      insert(getOperatorForLiteral(el.first), el.second);
    }
  }

  // 移除映射中指定的操作符
  void erase(const std::shared_ptr<Operator>& op) {
    auto it = map.find(Symbol::fromQualString(op->schema().name()));
    if (it == map.end()) {
      return;
    }
    // 遍历找到的操作符列表，删除匹配的操作符
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == op->schema()) {
        it->second.erase(vit);
        break;
      }
    }
    // 如果操作符列表为空，则从映射中删除该符号
    if (it->second.size() == 0) {
      map.erase(Symbol::fromQualString(op->schema().name()));
    }
  }

  // 检查映射中是否包含指定的操作符
  bool contains(const Operator& op) const {
    const auto it = map.find(Symbol::fromQualString(op.schema().name()));
    if (it == map.end()) {
      return false;
    }
    // 遍历找到的操作符列表，检查是否包含指定的操作符
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == op.schema()) {
        return true;
      }
    }
    return false;
  }

  // 检查映射中是否包含指定节点所属的操作符
  bool contains(const Node* n) const {
    // 如果节点存在操作符，则检查映射中是否包含该操作符
    return n->maybeOperator() && contains(n->getOperator());
  }

  // 查找映射中指定操作符的关联值，如果找到则返回其值
  std::optional<T> find(const Operator& op) {
    const auto it = map.find(Symbol::fromQualString(op.schema().name()));
    // 如果找到，则返回关联值的可选项
    if (it != map.end()) {
      return it->second;
    } else {
      return std::nullopt;
    }
  }

  // 符号到操作符及其关联值向量的映射
  MapType map;
};
    // 如果迭代器 it 指向 map 的结尾，返回空的 optional
    if (it == map.end()) {
      return c10::nullopt;
    }
    // 遍历 it 指向的映射值的 vector
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      // 如果当前 vector 元素的第一个元素（指针）指向的对象的 schema 与 op 的 schema 相同，返回第二个元素
      if (vit->first->schema() == op.schema()) {
        return vit->second;
      }
    }
    // 如果未找到匹配的 schema，返回空的 optional
    return c10::nullopt;
  }

  // TODO: return iterator
  // 获取所有键值对，返回一个 vector，其中包含所有的 OpMapType 元素
  std::vector<OpMapType> getAllKeysAndValues() const {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 创建一个空的 vector，用于存储所有的键值对
    std::vector<OpMapType> keys_values;
    // 遍历 map 中的每个 symbol_mapping
    for (auto& symbol_mapping : map) {
      // 获取当前 symbol_mapping 中的 vector 引用
      auto& vec = symbol_mapping.second;
      // 将当前 vector 中的所有元素加入到 keys_values 中
      for (auto& pair : vec) {
        keys_values.push_back(pair);
      }
    }
    // 返回包含所有键值对的 vector
    return keys_values;
  }

 private:
  friend struct Node;
  // 定义私有成员变量 map，类型为 MapType
  MapType map;
// 结构模板 `FunctionSchemaMap`，用于映射函数模式和泛型 `T` 的关系
template <typename T>
// 禁止Lint工具对成员初始化警告的检查
// `FunctionSchemaMap` 结构模板
struct FunctionSchemaMap {
  // 类型别名定义
  using FuncSchemaMapType = typename std::pair<FunctionSchema, T>;  // `FunctionSchema` 和 `T` 的键值对类型
  using ValueType = std::vector<FuncSchemaMapType>;  // 存储 `FuncSchemaMapType` 的向量类型
  using MapType = std::unordered_map<Symbol, ValueType>;  // 使用 `Symbol` 作为键的无序映射类型

  // 默认构造函数
  FunctionSchemaMap() = default;

  // 插入函数模式和对应值 `val`
  void insert(const FunctionSchema& schema, T val) {
    // 在插入之前先移除已存在的条目
    erase(schema);
    // 将函数模式名转换为符号 `Symbol`，然后将 `schema` 和 `val` 作为一对插入到映射中
    map[Symbol::fromQualString(schema.name())].emplace_back(
        std::make_pair(schema, val));
  }

  // 移除函数模式 `schema`
  void erase(const FunctionSchema& schema) {
    auto it = map.find(Symbol::fromQualString(schema.name()));  // 查找函数模式对应的符号 `Symbol`
    if (it == map.end()) {  // 如果未找到，直接返回
      return;
    }
    // 遍历该符号对应的值向量，找到并移除与 `schema` 匹配的条目
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first == schema) {
        it->second.erase(vit);
        break;
      }
    }
    // 如果移除后值向量为空，则从映射中移除该符号
    if (it->second.size() == 0) {
      map.erase(Symbol::fromQualString(schema.name()));
    }
  }

  // 检查是否包含函数模式 `schema`
  bool contains(const FunctionSchema& schema) const {
    const auto it = map.find(Symbol::fromQualString(schema.name()));  // 查找函数模式对应的符号 `Symbol`
    if (it == map.end()) {  // 如果未找到，返回 false
      return false;
    }
    // 遍历该符号对应的值向量，查找是否存在与 `schema` 匹配的条目
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == schema) {
        return true;
      }
    }
    return false;  // 未找到匹配条目，返回 false
  }

  // 查找函数模式 `schema` 对应的值 `T`
  std::optional<T> find(const FunctionSchema& schema) const {
    const auto it = map.find(Symbol::fromQualString(schema.name()));  // 查找函数模式对应的符号 `Symbol`
    if (it == map.end()) {  // 如果未找到，返回空值 `nullopt`
      return c10::nullopt;
    }
    // 遍历该符号对应的值向量，查找并返回与 `schema` 匹配的值 `T`
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first == schema) {
        return vit->second;
      }
    }
    return c10::nullopt;  // 未找到匹配条目，返回空值 `nullopt`
  }

  // 获取所有键和值的列表 `FuncSchemaMapType`
  // TODO: 返回迭代器
  std::vector<FuncSchemaMapType> getAllKeysAndValues() const {
    // 禁止Lint工具对变量初始化的检查
    std::vector<FuncSchemaMapType> keys_values;  // 声明存储键值对的向量
    for (auto& symbol_mapping : map) {  // 遍历映射中的每个符号映射
      auto& vec = symbol_mapping.second;  // 获取符号映射的值向量
      for (auto& pair : vec) {  // 遍历值向量中的每对键值对
        keys_values.push_back(pair);  // 将键值对添加到 `keys_values` 中
      }
    }
    return keys_values;  // 返回所有的键值对列表
  }

 private:
  friend struct Node;  // 允许 `Node` 结构访问私有成员
  MapType map;  // 保存函数模式和对应值的映射
};

// 结束命名空间 `jit` 和 `torch`
} // namespace jit
} // namespace torch
```