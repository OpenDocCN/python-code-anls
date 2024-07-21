# `.\pytorch\torch\csrc\jit\ir\scope.h`

```py
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 JIT 类型头文件

#include <ATen/core/symbol.h>
// 包含 ATen 库的符号头文件

#include <c10/util/Optional.h>
// 包含 C10 库的可选类型头文件

#include <c10/util/intrusive_ptr.h>
// 包含 C10 库的侵入式指针头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出头文件

#include <torch/csrc/jit/frontend/source_range.h>
// 包含 Torch JIT 前端的源范围头文件

#include <unordered_map>
// 包含标准库的无序映射头文件

namespace torch {
namespace jit {

struct ModuleInstanceInfo;
// 声明 ModuleInstanceInfo 结构体

constexpr size_t kModuleInstanceInfo = 2;
// 声明常量 kModuleInstanceInfo 并初始化为 2

namespace utils {
std::string get_module_info(const ModuleInstanceInfo& module_instance_info);
} // namespace utils
// 命名空间 utils 包含函数 get_module_info 的声明

// Scope is a node of a trie that represents the tree of nested scopes.
// Individual scopes are pushed and popped from Graph, which holds a
// pointer to the current scope. Each Node in Graph holds a pointer
// to the scope that was current when the node was created.
// The trie never needs to shrink, it only grows until it is disposed
// of when Graph is deallocated. Hence, pointers to scopes held by nodes
// will always be valid as long as Graph is alive.
// Scope 结构体表示嵌套作用域树的 Trie 节点。
// 单独的作用域从 Graph 中推入和弹出，Graph 持有当前作用域的指针。
// Graph 中的每个节点都持有在创建时当前作用域的指针。
// Trie 永远不需要收缩，只会增长，直到 Graph 被释放。
// 因此，节点持有的作用域指针在 Graph 存在期间始终有效。
struct Scope;
// 声明 Scope 结构体

using ScopePtr = c10::intrusive_ptr<Scope>;
// 使用 C10 库的侵入式指针定义 ScopePtr 类型

using c10::Symbol;
// 使用 C10 库的符号定义 Symbol 类型

struct TORCH_API Scope : public c10::intrusive_ptr_target {
  // Scope 结构体继承自 c10::intrusive_ptr_target

 private:
  ScopePtr parent_;
  // 父作用域的指针

  Symbol name_;
  // 作用域的符号名称

  ScopePtr intrusive_from_this();
  // 返回侵入式指针实例

 public:
  Scope();
  // 默认构造函数

  Scope(ScopePtr parent, Symbol name);
  // 带参构造函数，接受父作用域指针和名称符号

  ScopePtr push(Symbol name);
  // 将新作用域推入栈顶，返回指向新作用域的指针

  ScopePtr parent();
  // 返回父作用域的指针

  bool isRoot() const;
  // 判断是否为根作用域

  bool isBlank() const;
  // 判断作用域是否为空白

  ScopePtr getRoot();
  // 获取根作用域的指针

  size_t getDepth();
  // 获取作用域的深度

  Symbol name() const;
  // 返回作用域的名称符号

  std::string namesFromRoot(const std::string& separator = "/") const;
  // 返回从根作用域到当前作用域的名称路径字符串，可指定分隔符
};

struct Function;
// 声明 Function 结构体

struct InlinedCallStack;
// 声明 InlinedCallStack 结构体

/**
 * ModuleInstanceInfo is a structure to include the module type and instance
 * name. It also provide public methods to get the pointer to module type and
 * instance name.
 *
 * This structure is mainly used as a private member in InlinedCallStack, such
 * that one can follow the callstack to find the relevant module hierarchy.
 */
// ModuleInstanceInfo 结构体包含模块类型和实例名称，并提供获取模块类型指针和实例名称的公共方法。
// 此结构体主要作为 InlinedCallStack 的私有成员使用，以便跟踪调用堆栈以找到相关的模块层次结构。
struct ModuleInstanceInfo {
 private:
  c10::ClassTypePtr module_type_{nullptr};
  // 模块类型的指针，默认为空指针

  std::string instance_name_;
  // 实例名称的字符串

 public:
  ModuleInstanceInfo() = default;
  // 默认构造函数

  ModuleInstanceInfo(c10::ClassTypePtr module_type, std::string instance_name);
  // 带参构造函数，接受模块类型指针和实例名称字符串

  c10::ClassTypePtr class_type() {
    return module_type_;
  }
  // 返回模块类型的指针

  c10::ClassTypePtr class_type() const {
    return module_type_;
  }
  // 返回模块类型的指针（常量版本）

  std::string instance_name() const {
    return instance_name_;
  }
  // 返回实例名称的字符串

  bool operator==(const ModuleInstanceInfo& rhs) const {
    return (class_type() == rhs.class_type()) &&
        (instance_name() == rhs.instance_name());
  }
  // 比较运算符重载，判断两个 ModuleInstanceInfo 结构体是否相等
};

} // namespace jit
} // namespace torch
/**
 * InlinedCallStack is an element in a list representing callstack of functions
 * that have been inlined.
 *
 * Each such element holds info about the current callsite (Function and
 * SourceRange) and a pointer to the next element in the list. The last element
 * in the list represents the innermost function that was inlined.
 *
 * For instance, if a node has a callstack
 *    [foo, source_range1] -> [bar, source_range2]
 * it means that this node was originally from function 'bar' that was called
 * at 'source_range2' in function 'foo' that was called in the current function
 * at 'source_range1'.
 *
 * If a node did not come from any inlined function, its callstack will be
 * empty.
 *
 * The callstack lists only grow, we never remove elements from them, which
 * allows us to reuse same elements in different lists. For instance, if we
 * inline function 'bar' to 'foo' and then inline 'foo' to two functions 'ham'
 * and 'baz', the callstacks would look like:
 *
 *  [baz, source_range3]  --
 *                           \
 *                             --> [foo, source_range1] -> [bar, source_range2]
 *                           /
 *  [ham, source_range4]  --
 */
using InlinedCallStackPtr = c10::intrusive_ptr<InlinedCallStack>;
using InlinedCallStackEntry =
    std::tuple<Function*, SourceRange, std::optional<ModuleInstanceInfo>>;
// 定义一个结构体 InlinedCallStack，继承自 c10::intrusive_ptr_target
struct TORCH_API InlinedCallStack : public c10::intrusive_ptr_target {
 private:
  // 可选的指向 InlinedCallStackPtr 的调用者
  std::optional<InlinedCallStackPtr> callee_;
  // 指向 Function 对象的指针 fn_
  Function* fn_;
  // fn_name_ 的存在理由尽管我们已经有了 fn_
  // 序列化的调用堆栈在某些情况下使用，如移动运行时或委托后端
  // 在这些情况下，我们没有 Function*，因此存储函数名 fn_name
  // 在移动/委托后端运行时，InlinedCallStack 用于异常堆栈，fn_name_ 就足够了。
  const std::string fn_name_;
  // 源代码范围对象 source_range_
  SourceRange source_range_;
  // 通过 intrusive_from_this() 方法获取 InlinedCallStackPtr
  InlinedCallStackPtr intrusive_from_this();
  // 可选的 ModuleInstanceInfo 对象 module_instance_info_
  std::optional<ModuleInstanceInfo> module_instance_info_;

 public:
  // Leaf 调用堆栈节点的构造函数
  InlinedCallStack(Function* fn, SourceRange source_range);

  // Leaf 调用堆栈节点的构造函数，包括模块实例信息
  InlinedCallStack(
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info);

  // Leaf 调用堆栈节点的构造函数，包括模块实例信息和函数名
  InlinedCallStack(
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info,
      std::string& function_name);

  // Inner 调用堆栈节点的构造函数
  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range);

  // Inner 调用堆栈节点的构造函数，包括模块实例信息
  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info);

  // Inner 调用堆栈节点的构造函数，包括模块实例信息和函数名
  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info,
      std::string& function_name);

  // 返回调用堆栈列表中的下一个元素
  std::optional<InlinedCallStackPtr> callee() const;

  // 返回与当前元素关联的模块实例
  std::optional<ModuleInstanceInfo> module_instance() const;

  // 返回节点的源代码范围
  SourceRange source_range() const;

  // 返回节点的函数指针
  Function* function() const;

  // 返回函数名 fn_name_
  const std::string& function_name() const;

  // 将调用堆栈作为 [Function, SourceRange] 对的向量返回
  std::vector<InlinedCallStackEntry> vec();

  // 设置调用者 callee_
  void setCallee(std::optional<InlinedCallStackPtr>);

  // 相等比较运算符的重载，不需要比较 fn_，因为 source_range 足以判断相等性
  bool operator==(const InlinedCallStack& rhs) const {
    return (module_instance().has_value() ==
            rhs.module_instance().has_value()) &&
        (module_instance().has_value() &&
         module_instance().value() == rhs.module_instance().value()) &&
        callee() == rhs.callee() && source_range() == rhs.source_range();
  }

  // 不等比较运算符的重载
  bool operator!=(const InlinedCallStack& rhs) const {
    return !(*this == rhs);
  }
};

// {source range, node name, InlinedCallStack}
// 定义一个元组类型 DebugInfoTuple，用于存储调试信息
using DebugInfoTuple =
    std::tuple<SourceRange, std::string, InlinedCallStackPtr>;

// 定义常量，表示 DebugInfoTuple 中不同元素的索引
constexpr size_t kDebugInfoTupleSourceRangeIndex{0};
constexpr size_t kDebugInfoTupleNodeNameIndex{1};
constexpr size_t kDebugInfoTupleInlinedCSIndex{2};

// 命名空间 jit 中的代码，这些常量和类型属于 JIT 编译器的一部分
} // namespace jit

// 命名空间 torch 中的代码，这些常量和类型属于 PyTorch 深度学习框架的一部分
} // namespace torch
```