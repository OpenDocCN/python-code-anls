# `.\pytorch\torch\csrc\jit\frontend\concrete_module_type.h`

```py
// 使用#pragma once确保头文件只被编译一次，防止多重包含的问题

#include <ATen/core/ivalue.h>                      // 引入ATen库中的ivalue头文件
#include <torch/csrc/jit/api/module.h>            // 引入torch库中jit模块的module头文件
#include <torch/csrc/jit/python/pybind_utils.h>   // 引入torch库中jit模块的python/pybind_utils头文件
#include <memory>                                  // 引入内存管理相关的头文件
#include <string>                                  // 引入字符串处理相关的头文件
#include <vector>                                  // 引入向量容器相关的头文件

namespace torch {                                 // 声明torch命名空间
namespace jit {                                   // 声明jit命名空间

enum class IterableModuleKind {                   // 定义枚举类IterableModuleKind，表示可迭代模块类型
    NONE,                                         // 没有特定类型
    LIST,                                         // 列表类型
    DICT,                                         // 字典类型
    PARAMLIST,                                    // 参数列表类型
    PARAMDICT                                     // 参数字典类型
};

class ConcreteModuleType;                         // 声明ConcreteModuleType类的前置声明

// nn.Module可以看作是对应于一系列JIT类型的模板。模板的“参数”是诸如常量值之类的东西。
// ConcreteModuleType对应于类型族中的单个成员，所有模板参数都已完全指定。
// 如果两个Module共享一个ConcreteModuleType，则它们可以共享一个JIT类型，反之亦然。
// ConcreteModuleType还是服务于所有ModuleValue::attr调用的事实来源。这样我们可以保证，
// 如果两个Module共享一个JIT类型（以及一个ConcreteModuleType），那么在访问它们的属性时，它们的行为是相同的。

// ConcreteModuleType有两个阶段。
// 1. 创建阶段：首先我们建立它，在ScriptModule转换过程中。这由ConcreteModuleTypeBuilder表示。
//    ...然后转换器调用ConcreteModuleTypeBuilder::build()，产生一个已准备好用于查询的ConcreteModuleType。
// 2. 查询阶段：我们使用ConcreteModuleType作为ModuleValue::attr调用的真实来源，在方法编译期间。

// 在构建过程中表示具体类型的类。我们使用它来决定是否可以在模块之间共享类型。
class VISIBILITY_HIDDEN ConcreteModuleTypeBuilder {
 public:
  explicit ConcreteModuleTypeBuilder(py::object pyClass) {    // 显式构造函数，接受一个py::object参数pyClass
    TORCH_INTERNAL_ASSERT(pyClass);                            // 断言pyClass不为空
  // 将传入的 pyClass_ 移动到当前对象的 pyClass_ 成员中
  pyClass_ = std::move(pyClass);
}

// 添加一个常量到当前模块类型构建器中，常量为 Python 对象
void addConstant(std::string name, py::object value);

// 添加一个常量到当前模块类型构建器中，常量为 Torch 的 IValue 对象
void addConstant(std::string name, IValue value);

// 添加一个属性到当前模块类型构建器中，包括属性名、类型、是否参数、是否缓冲区
void addAttribute(
    std::string name,
    const TypePtr& type,
    bool isParameter,
    bool isBuffer);

// 添加一个函数属性到当前模块类型构建器中，包括属性名、类型、Python 函数对象
void addFunctionAttribute(
    std::string name,
    const TypePtr& type,
    py::object pyFunction);

// 添加一个命名模块到当前模块类型构建器中，包括模块名和元数据
void addModule(std::string name, std::shared_ptr<ConcreteModuleType> meta);

// 添加一个前向钩子到当前模块类型构建器中，作为回调函数
void addForwardHook(py::object hook);

// 添加一个前向预处理钩子到当前模块类型构建器中，作为预处理回调函数
void addForwardPreHook(py::object pre_hook);

// 添加一个方法重载到当前模块类型构建器中，包括方法名和重载方法名列表
void addOverload(
    std::string methodName,
    std::vector<std::string> overloadedMethodNames);

// 添加一个内置函数到当前模块类型构建器中，包括函数名和符号名
void addBuiltinFunction(std::string name, const std::string& symbol_name);

// 添加一个失败的属性到当前模块类型构建器中，包括属性名和失败原因
void addFailedAttribute(std::string name, std::string failureReason);

// 添加一个被忽略的属性到当前模块类型构建器中，包括属性名
void addIgnoredAttribute(std::string name);

// 设置当前模块类型构建器的可迭代模块种类
void setIterableModuleKind(IterableModuleKind kind);

// 设置当前模块类型构建器为“有毒”，表示它永远不会等同于任何其他具体类型
void setPoisoned();

// 构建一个共享指针指向当前模块类型构建器的 ConcreteModuleType 对象
std::shared_ptr<ConcreteModuleType> build() const {
  return std::make_shared<ConcreteModuleType>(*this);
}

// 比较当前模块类型构建器与另一个是否相等
// ConcreteModuleTypeBuilder 使用 operator== 实现了有意义的比较
bool equals(const ConcreteModuleTypeBuilder& other) const;

// 函数属性结构体，包括函数类型指针和 Python 函数对象
struct FunctionAttribute {
  FunctionTypePtr function_;
  py::object pyFunction_;

  // 比较函数属性是否相等，通过比较 Python 函数对象的指针
  friend bool operator==(
      const FunctionAttribute& lhs,
      const FunctionAttribute& rhs) {
    // 函数不是一等公民，所以不能像普通属性那样进行类型比较
    // 这里通过实际的 Python 函数对象指针进行相等性检查
    return lhs.pyFunction_.is(rhs.pyFunction_);
  }
};

// 属性结构体，包括类型指针、是否参数、是否缓冲区
struct Attribute {
  Attribute(TypePtr type, bool isParam, bool isBuffer)
      : type_(std::move(type)), isParam_(isParam), isBuffer_(isBuffer) {}

  // 比较属性是否相等，比较类型和是否参数是否相等
  friend bool operator==(const Attribute& lhs, const Attribute& rhs) {
    return *(lhs.type_) == *(rhs.type_) && lhs.isParam_ == rhs.isParam_;
  }
  TypePtr type_;
  bool isParam_;
  bool isBuffer_;
};

// 模块信息结构体，包括模块名和元数据
struct ModuleInfo {
  ModuleInfo(std::string name, std::shared_ptr<ConcreteModuleType> meta)
      : name_(std::move(name)), meta_(std::move(meta)) {}

  // 比较模块信息是否相等
  friend bool operator==(const ModuleInfo& lhs, const ModuleInfo& rhs);

  std::string name_;
  // 指向具体模块类型的共享指针，用于描述模块的元信息
  std::shared_ptr<ConcreteModuleType> meta_;
};

private:
// 默认构造函数，使用默认参数
ConcreteModuleTypeBuilder() = default;

// 从当前对象创建类型的辅助函数
ClassTypePtr createTypeFromThis() const;

// 如果为真，表示此类型永远不会与其他任何东西相等。用于确保此类型不共享（例如，如果它来自跟踪模块）
bool isPoisoned_ = false;

// 模块定义的任何常量的值的映射
std::unordered_map<std::string, IValue> constants_;

// 属性的类型，使用有序字典存储
OrderedDict<std::string, Attribute> attributes_;

// 重载函数集合，格式与 Python 中的 `__overloads__` 相同
std::unordered_map<std::string, std::vector<std::string>> overloads_;

// 无法转换为 TorchScript 的任何属性及其失败原因的映射
std::unordered_map<std::string, std::string> failedAttributes_;

// 被标记为忽略的属性集合。这些属性不能在 TorchScript 中使用，但可以在 Python 中的忽略函数中使用
std::unordered_set<std::string> ignoredAttributes_;

// 函数属性集合。这些属性在类型系统中目前是特殊的，因为函数不是一级对象
std::unordered_map<std::string, FunctionAttribute> functionAttributes_;

// 内置函数调用的属性。这些直接转换为相应的 aten:: 调用。映射为属性名 -> aten 符号名
std::unordered_map<std::string, c10::Symbol> builtinFunctions_;

// 子模块的具体类型信息列表
std::vector<ModuleInfo> modules_;

// 在直接调用模块时，在前向传播前/后调用的钩子。用于确保具有不同的类型当具有不同的 Python 钩子时
// 实际的钩子在编译期间直接添加到 ClassType 中
std::vector<py::object> forwardHooks_;
std::vector<py::object> forwardPreHooks_;

// 如果某物是 ModuleDict/ModuleList，则意味着：
//   1. 子模块的顺序对比类型很重要
//   2. 编译器可以将其视为字典/元组
IterableModuleKind iterableModuleKind_ = IterableModuleKind::NONE;

// 我们从中派生此 ScriptModule 的原始 `nn.Module` 类
py::object pyClass_;

// 注意：如果将来向此结构添加任何状态，必须确保 operator== 仍然有意义！
friend ConcreteModuleType;
// 结束了类定义的尾部
};

// 表示一个已最终确定的具体类型，用于服务于方法编译期间的 ModuleValue::attr 调用
class VISIBILITY_HIDDEN ConcreteModuleType {
 public:
  // 显式构造函数，接受一个 ConcreteModuleTypeBuilder 对象作为参数
  explicit ConcreteModuleType(ConcreteModuleTypeBuilder data);

  // 从 JIT 类型 TypePtr 转换为 ConcreteModuleType 的静态方法
  static std::shared_ptr<ConcreteModuleType> fromJitType(TypePtr type);

  // 返回当前 ConcreteModuleType 对象的 JIT 类型 TypePtr
  TypePtr getJitType() const;

  // 获取与 Python 类相关联的对象，返回一个可选的 py::object
  std::optional<py::object> getPyClass() const;

  // 获取可迭代模块的种类信息
  IterableModuleKind getIterableModuleKind() const;

  // 查找指定名称的函数重载列表，返回一个可选的字符串向量
  std::optional<std::vector<std::string>> findOverloads(
      const std::string& name) const;

  // 查找指定名称的函数属性，返回一个可选的指向 Function 对象的指针
  std::optional<Function*> findFunctionAttribute(const std::string& name) const;

  // 查找内建函数的符号信息，返回一个可选的 c10::Symbol
  std::optional<c10::Symbol> findBuiltinFunction(const std::string& name) const;

  // 查找子模块的具体类型，返回一个共享指针指向 ConcreteModuleType 对象
  std::shared_ptr<ConcreteModuleType> findSubmoduleConcreteType(
      const std::string& name) const;

  // 查找指定名称的失败属性，返回一个可选的字符串
  std::optional<std::string> findFailedAttribute(const std::string& name) const;

  // 检查指定名称属性是否被忽略，返回布尔值
  bool isIgnoredAttribute(const std::string& name) const;

  // 获取常量映射到 py::object 的字典
  std::unordered_map<std::string, py::object> getConstantsPy() const;

  // 获取属性映射到 (TypePtr, bool) 对的字典
  std::unordered_map<std::string, std::pair<TypePtr, bool>> getAttributesPy()
      const;

  // 获取模块映射到 (名称, 共享指针指向 ConcreteModuleType) 对的向量
  std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>>
  getModulesPy() const;

  // 比较当前 ConcreteModuleType 对象是否等于另一个对象 other
  bool equals(const ConcreteModuleType& other) const {
    if (jitType_ == other.jitType_) {
      // 如果 JIT 类型相同，则这两个模块可以共享相同的类型
      return true;
    }

    // 否则比较数据 data_ 是否相等
    return data_.equals(other.data_);
  }

  // 比较当前 ConcreteModuleType 对象是否等于 ConcreteModuleTypeBuilder 对象 other
  bool equals(const ConcreteModuleTypeBuilder& other) const {
    return data_.equals(other);
  }

  // 打印当前 ConcreteModuleType 对象的信息
  void dump() const;

 private:
  // 默认构造函数，私有成员，不对外公开
  ConcreteModuleType() = default;

  // 从 ConcreteModuleTypeBuilder 创建的数据成员
  ConcreteModuleTypeBuilder data_;

  // 从 JIT 类型派生的具体类型 TypePtr
  TypePtr jitType_;
};

// 命名空间 jit 的结束
} // namespace jit

// 命名空间 torch 的结束
} // namespace torch
```