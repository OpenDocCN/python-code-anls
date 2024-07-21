# `.\pytorch\torch\csrc\jit\api\compilation_unit.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/name_mangler.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <torch/csrc/Export.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// 引入各种头文件，包括 ATen 和 Torch 提供的库

namespace torch::jit {

// 声明 torch::jit 命名空间

struct Def;
struct Property;
struct ClassDef;
struct SugaredValue;
struct Resolver;

using ResolverPtr = std::shared_ptr<Resolver>;

// 定义 Self 结构体
struct Self {
  virtual ~Self() = default;
  virtual std::shared_ptr<SugaredValue> makeSugared(Value* v) const = 0;
  virtual ClassTypePtr getClassType() const = 0;
  // 虚拟接口，用于生成 SugaredValue 对象和获取类类型
};

// A CompilationUnit is a list of named Functions
// with helper methods to iterate the list or invoke the function.
// Classes have a CompilationUnit holding the class methods,
// and Modules have a CompilationUnit holding the Functions that
// are used to implement their Methods

// 编译单元结构体定义
struct TORCH_API CompilationUnit {
  enum class FunctionType { Method, Hook, PreHook };
  // 函数类型枚举，用于标识方法、钩子和预钩子

  // 构造函数，接受源代码字符串进行编译
  explicit CompilationUnit(const std::string& source);
  CompilationUnit() = default;

  CompilationUnit& operator=(CompilationUnit&&) = default;
  CompilationUnit(CompilationUnit&&) = default;
  CompilationUnit& operator=(const CompilationUnit&) = delete;
  CompilationUnit(const CompilationUnit&) = delete;

  // 查找给定名称的函数
  Function* find_function(const c10::QualifiedName& name) const {
    auto it = dict_.find(name);
    if (it == dict_.end()) {
      return nullptr;
    }
    return functions_[it->second].get();
  }

  // 获取给定名称的函数，如果不存在则抛出错误
  Function& get_function(const c10::QualifiedName& name) const {
    if (auto r = find_function(name)) {
      return *r;
    }
    TORCH_CHECK(false, "attempted to get undefined function ", name.name());
  }

  // 设置优化状态（已废弃）
  void set_optimized(bool o) {
    TORCH_WARN(
        "CompilationUnit::set_optimized() is deprecated and has no effect. "
        "Please use setGraphExecutorOptimize()");
  }

  // 获取优化状态（已废弃）
  bool is_optimized() const {
    TORCH_WARN(
        "CompilationUnit::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
  // 返回布尔值 true
  return true;
}

// 由于历史原因，这些函数在 ir_emitter.cpp 中定义
// 返回刚刚定义的函数列表
std::vector<Function*> define(
    const std::optional<c10::QualifiedName>& prefix,
    const std::vector<Property>& properties,
    const std::vector<ResolverPtr>& propResolvers,
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>&
        defResolvers, /* 确定如何处理每个定义中的自由变量 */
    // 如果非空，每个定义的第一个参数绑定到此值
    const Self* self,
    // 见 [name mangling]（名称重整）
    bool shouldMangle = false,
    std::optional<size_t> operator_set_version = c10::nullopt);

// 定义钩子函数
void define_hooks(
    const std::optional<c10::QualifiedName>& prefix,
    const std::vector<Def>& hookDefs,
    const std::vector<ResolverPtr>& hookResolvers,
    const std::vector<Def>& preHookDefs,
    const std::vector<ResolverPtr>& preHookResolvers,
    const Self* self,
    bool shouldMangle = false);

// 与上述函数相同，但从源代码中解析定义
// 返回刚刚定义的函数列表
std::vector<Function*> define(
    // 将所有定义的函数放入的前缀命名空间
    const std::optional<c10::QualifiedName>& prefix,
    // 源代码字符串
    const std::string& source,
    const ResolverPtr& resolver,
    const Self* self);

// 定义接口
void define_interface(
    const c10::QualifiedName& qualifiedName,
    const ClassDef& classDef,
    ResolverPtr rcb,
    bool is_module = false);

// 创建函数
Function* create_function(
    c10::QualifiedName name,
    std::shared_ptr<Graph> graph,
    bool shouldMangle = false) {
  // 如果需要进行名称重整
  if (shouldMangle) {
    // 对名称进行重整
    name = mangle(name);
  }
  // 创建一个具有指定名称和图形的图形函数
  auto fn = std::make_unique<GraphFunction>(
      std::move(name), std::move(graph), nullptr);
  auto ret = fn.get();
  // 注册该函数
  register_function(std::move(fn));
  // 返回创建的函数
  return ret;
}

// 获取所有函数
std::vector<Function*> get_functions() const {
  // 返回所有函数的列表
  return fmap(functions_, [](const std::unique_ptr<Function>& fn) {
    return fn.get();
  });
}

/// 运行此编译单元中的一个方法。
///
/// 例如：
/// @code
///   IValue output = module->run("relu_script", a, b);
/// @endcode
///
/// 要从源字符串编译模块，请参见 torch::jit::compile
///
/// @param method_name 要运行的方法的名称
/// @param args 要传递给方法的参数
/// @return 包含返回值（如果是元组则包含多个返回值）的 IValue
template <typename... Types>
IValue run_method(const c10::QualifiedName& method_name, Types&&... args) {
  // 获取方法的函数对象并调用
  return get_function(method_name)({IValue(std::forward<Types>(args))...});
}

// 清除所有函数
void drop_all_functions() {
  // 清除字典
  dict_.clear();
  // 清除函数列表
  functions_.clear();
}

/**
 * 注册一个类作为此编译单元拥有的类。
 */
void register_type(c10::NamedTypePtr namedType) {
  // TODO: class types cannot be redefined because we have no way right now
  // of invalidating their methods. NamedTuples are fine though, since they
  // don't have methods.
  TORCH_CHECK(
      0 == classDict_.count(*namedType->name()),
      "class '",
      namedType->name()->qualifiedName(),
      "' already defined.");
  // 将新定义的类添加到 classes_ 容器中
  classes_.push_back(std::move(namedType));
  // 将新定义的类在 classDict_ 中映射到其在 classes_ 中的索引位置
  classDict_[*classes_.back()->name()] = classes_.size() - 1;
};

c10::ClassTypePtr get_class(const c10::QualifiedName& name) const {
  // 根据类名获取对应的 ClassType 指针
  auto type = get_type(name);
  if (!type) {
    return nullptr;
  }
  return type->cast<c10::ClassType>();
}

c10::InterfaceTypePtr get_interface(const c10::QualifiedName& name) const {
  // 根据接口名获取对应的 InterfaceType 指针
  auto type = get_type(name);
  if (!type) {
    return nullptr;
  }
  return type->cast<c10::InterfaceType>();
}

c10::TupleTypePtr get_named_tuple(const c10::QualifiedName& name) const {
  // 根据命名元组名获取对应的 TupleType 指针
  for (const auto& cls : classes_) {
    if (cls->name()->qualifiedName() == name.qualifiedName()) {
      return cls->expect<TupleType>();
    }
  }
  return nullptr;
}

c10::NamedTypePtr get_type(const c10::QualifiedName& name) const {
  // 根据名称在 classDict_ 中查找对应的 NamedType 指针
  auto it = classDict_.find(name);
  if (it == classDict_.end()) {
    return nullptr;
  }
  return classes_[it->second];
}

// For testing: clear all Python-defined classes to ensure that unit tests
// have isolation.
void _clear_python_cu() {
  // 删除所有与 Python 定义的类相关联的方法
  for (const auto& type : classes_) {
    if (auto cls = type->cast<ClassType>()) {
      for (auto method : cls->methods()) {
        // 在编译单元中标记方法为墓碑状态
        // 不要删除，因为 dict_ 中还会保留
        auto it = dict_.find(method->qualname());
        if (it != dict_.end()) {
          functions_[it->second] = nullptr;
          // 在大查找表中删除
          dict_.erase(it);
        }
      }
      // 类可以有多个指向相同钩子的指针，
      // 需要确保不会重复删除
      std::unordered_set<Function*> hooks_to_delete;
      for (const auto& hook : cls->getForwardHooks()) {
        hooks_to_delete.insert(hook);
      }
      for (const auto& pre_hook : cls->getForwardPreHooks()) {
        hooks_to_delete.insert(pre_hook);
      }
      for (const auto& hook : hooks_to_delete) {
        // 在编译单元中标记钩子为墓碑状态
        auto it = dict_.find(hook->qualname());
        if (it != dict_.end()) {
          functions_[it->second] = nullptr;
          // 在大查找表中删除
          dict_.erase(it);
        }
      }
    }
  }
  // 清空 classes_ 和 classDict_
  classes_.clear();
  classDict_.clear();
}

// [Internal Only] Remove method.
// Note Used for freezing.
void unsafeRemoveMethod(const c10::QualifiedName& method_name) {
    // 使用 TORCH_CHECK 确保字典中存在指定的方法，否则输出错误信息
    TORCH_CHECK(
        it != dict_.end(),
        "method '",
        method_name.qualifiedName(),
        "' does not exist.");
    // 将 functions_ 中指定索引的元素置空指针
    functions_[it->second] = nullptr;
    // 从 dict_ 中移除指定迭代器所指向的元素
    dict_.erase(it);
  }

  // [name mangling] 所有代码对象在 CompilationUnit 中必须具有唯一的限定名。
  // 在 Python 中，有时函数的限定名可能不唯一（例如，嵌套函数）。因此，我们对
  // Python 函数进行名称混淆，以确保它们具有唯一的名称。
  //
  // 我们还使用名称混淆来区分不同的 Module 实例。由于每个 Module 是单例类实例，
  // 相同 Python Module 的不同实例将具有不同的类型但相同的限定名。
  c10::QualifiedName mangle(const c10::QualifiedName& name) const {
    // 使用给定名称进行初始化
    auto mangled = name;
    // 当获取类型或查找函数时，不断进行名称混淆，直到找到唯一的名称
    while (get_type(mangled) || find_function(mangled)) {
      mangled = mangler_.mangle(mangled);
    }
    // 返回混淆后的唯一名称
    return mangled;
  }

 private:
  // 在给定前缀下定义函数
  std::unique_ptr<Function> define(
      const std::optional<c10::QualifiedName>& prefix,
      const Def& def,
      const ResolverPtr& resolver,
      const Self* self,
      const std::unordered_map<std::string, Function*>& function_table,
      bool shouldMangle = false,
      FunctionType type = FunctionType::Method,
      std::optional<size_t> version = c10::nullopt) const;

  // 定义在 self 上的属性
  struct PropertyPair;
  PropertyPair define_property(
      const std::optional<c10::QualifiedName>& prefix,
      const Property& prop,
      const ResolverPtr& resolver,
      const Self* self,
      const std::unordered_map<std::string, Function*>& function_table,
      bool shouldMangle = false) const;

  // 注册函数并返回函数的引用
  Function& register_function(std::unique_ptr<Function> fn) {
    // 确保 dict_ 中不存在同名的方法，否则输出错误信息
    TORCH_CHECK(
        0 == dict_.count(fn->qualname().qualifiedName()),
        "method '",
        fn->qualname().qualifiedName(),
        "' already defined.");
    // 将函数移动到 functions_ 后面
    functions_.emplace_back(std::move(fn));
    // 将函数名和索引添加到 dict_ 中
    dict_[functions_.back()->qualname()] = functions_.size() - 1;
    // 返回注册的函数引用
    return *functions_.back();
  }
  std::vector<std::unique_ptr<Function>> functions_;
  // 用于快速查找
  std::unordered_map<c10::QualifiedName, size_t> dict_;
  std::unordered_map<c10::QualifiedName, size_t> classDict_;

  // [class ownership] 当前存在两种类与编译单元之间的关系：
  // 1. 类内部有编译单元持有其方法。
  // 2. 加载时，任何导入类的 TypePtr 都由主模块的编译单元所拥有。
  std::vector<c10::NamedTypePtr> classes_;

  mutable NameMangler mangler_;
};

// An owning pointer to a Function. Just a pair of a raw Function ptr and it's
// owning CU. We need this because pybind requires a ref-counted way to refer to
// Functions.
// 表示一个拥有 Function 的所有权的指针。它是一个由原始 Function 指针和它所属的 CompilationUnit 组成的对。我们需要这个结构，因为 pybind 要求一种引用计数的方式来引用 Functions。
struct StrongFunctionPtr {
  StrongFunctionPtr(std::shared_ptr<CompilationUnit> cu, Function* function)
      : cu_(std::move(cu)), function_(function) {
    TORCH_INTERNAL_ASSERT(cu_);
    TORCH_INTERNAL_ASSERT(function_);
  }
  // 持有 Function 所属的 CompilationUnit 的共享指针
  std::shared_ptr<CompilationUnit> cu_;
  // 指向 Function 的指针
  Function* function_;
};

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
// 我们曾经有一个被删除的 `script::` 命名空间。这是为了向后兼容公共 API；新代码不应该使用这个类型别名。
using CompilationUnit = ::torch::jit::CompilationUnit;
} // namespace script
} // namespace torch::jit
```