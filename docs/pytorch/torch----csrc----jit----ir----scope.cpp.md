# `.\pytorch\torch\csrc\jit\ir\scope.cpp`

```
// 引入Torch库中的Scope类的头文件
#include <torch/csrc/jit/ir/scope.h>

// 引入ATen库中的ClassType和Function类的头文件
#include <ATen/core/class_type.h>
#include <ATen/core/function.h>

// 定义torch::jit命名空间
namespace torch::jit {

// 实用函数的命名空间utils
namespace utils {

// 获取模块信息的函数，返回模块名称及类型信息的字符串表示
std::string get_module_info(const ModuleInstanceInfo& module_instance_info) {
  // 初始化模块信息字符串
  std::string module_info;
  
  // 获取模块实例的类类型
  const auto& class_type = module_instance_info.class_type();
  
  // 获取模块实例的名称
  std::string instance_name = module_instance_info.instance_name();
  std::string type_name;
  
  // 如果存在类类型
  if (class_type) {
    // 获取类类型的限定名称并截取最后一部分作为类型名
    type_name += class_type->name()->qualifiedName();
    type_name = type_name.substr(type_name.find_last_of('.') + 1);
  }
  
  // 如果类型名为空，则标记为UNKNOWN_TYPE
  if (type_name.empty()) {
    type_name = "UNKNOWN_TYPE";
  }
  
  // 如果实例名为空，则标记为UNKNOWN_INSTANCE
  if (instance_name.empty()) {
    instance_name = "UNKNOWN_INSTANCE";
  }
  
  // 构建模块信息字符串，格式为实例名(类型名)
  module_info.append(instance_name).append("(").append(type_name).append(")");
  
  // 返回模块信息字符串
  return module_info;
}

} // namespace utils

// Scope类的intrusive_from_this方法实现
ScopePtr Scope::intrusive_from_this() {
  // 增加原始指针的引用计数，因为要从原始this指针创建新指针
  c10::raw::intrusive_ptr::incref(this);
  
  // 通过intrusive_ptr的reclaim方法将原始指针转换为ScopePtr类型并返回
  return c10::intrusive_ptr<Scope>::reclaim(this);
}

// Scope类的默认构造函数实现，将name_初始化为一个空符号
Scope::Scope() : name_(Symbol::scope("")) {}

// Scope类的带参构造函数实现，接受parent和name参数
Scope::Scope(ScopePtr parent, Symbol name)
    : parent_(std::move(parent)), name_(name) {}

// Scope类的push方法实现，创建并返回一个新的ScopePtr对象
ScopePtr Scope::push(Symbol name) {
  return c10::make_intrusive<Scope>(intrusive_from_this(), name);
}

// Scope类的parent方法实现，返回当前Scope对象的父ScopePtr对象
ScopePtr Scope::parent() {
  if (!parent_) {
    // 如果没有父Scope，则抛出运行时错误
    throw std::runtime_error("Cannot get parent from Scope with no parent");
  }
  return parent_;
}

// Scope类的isRoot方法实现，检查当前Scope是否为根Scope
bool Scope::isRoot() const {
  return !parent_;
}

// Scope类的isBlank方法实现，检查当前Scope是否为根Scope且名称为空
bool Scope::isBlank() const {
  // 静态变量blank代表空符号
  static const Symbol blank = Symbol::scope("");
  return isRoot() && name() == blank;
}

// Scope类的getRoot方法实现，返回当前Scope的根ScopePtr对象
ScopePtr Scope::getRoot() {
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
  }
  return current;
}

// Scope类的getDepth方法实现，返回当前Scope的深度
size_t Scope::getDepth() {
  size_t d = 1;
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
    d += 1;
  }
  return d;
}

// Scope类的name方法实现，返回当前Scope的名称
Symbol Scope::name() const {
  return name_;
}

// Scope类的namesFromRoot方法实现，返回从根Scope到当前Scope的名称路径字符串
std::string Scope::namesFromRoot(const std::string& separator) const {
  // TODO: I think the answer is we shouldn't have used Symbol here
  std::string out = this->name_.toUnqualString();
  
  // 如果当前Scope为根Scope，则直接返回名称字符串
  if (this->isRoot()) {
    return out;
  }
  
  // 获取父Scope，并沿着父Scope路径构建名称字符串
  ScopePtr parent = this->parent_;
  while (!parent->isRoot()) {
    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
    out = std::string(parent->name_.toUnqualString()) + separator + out;
    parent = parent->parent_;
  }
  
  // 返回从根Scope到当前Scope的名称路径字符串
  return out;
}
InlinedCallStackPtr InlinedCallStack::intrusive_from_this() {
``` 
// 返回指向当前对象的指针 `this` 的 `InlinedCallStackPtr`
  c10::raw::intrusive_ptr::incref(this); // 我们从原始的 `this` 指针创建一个新指针
                                         // 因此需要增加引用计数
                                         // 来跟踪这个所有权
  return c10::intrusive_ptr<InlinedCallStack>::reclaim(this);
}

InlinedCallStack::InlinedCallStack(Function* fn, SourceRange source_range)
    : fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)) {}
``` 
// 构造函数，初始化成员变量 `fn_`、`fn_name_` 和 `source_range_`
// 如果 `fn` 非空，则使用 `fn` 的名称作为 `fn_name_`
// 使用 `std::move` 将 `source_range` 移动到 `source_range_`


InlinedCallStack::InlinedCallStack(
    Function* fn,
    SourceRange source_range,
    std::optional<ModuleInstanceInfo> module_instance_info)
    : fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}
``` 
// 构造函数，初始化成员变量 `fn_`、`fn_name_`、`source_range_` 和 `module_instance_info_`
// 如果 `fn` 非空，则使用 `fn` 的名称作为 `fn_name_`
// 使用 `std::move` 将 `source_range` 和 `module_instance_info` 移动到对应的成员变量


InlinedCallStack::InlinedCallStack(
    Function* fn,
    SourceRange source_range,
    std::optional<ModuleInstanceInfo> module_instance_info,
    std::string& function_name)
    : fn_(fn),
      fn_name_(std::move(function_name)),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}
``` 
// 构造函数，初始化成员变量 `fn_`、`fn_name_`、`source_range_` 和 `module_instance_info_`
// 使用 `std::move` 将 `function_name`、`source_range` 和 `module_instance_info` 移动到对应的成员变量


InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)) {}
``` 
// 构造函数，初始化成员变量 `callee_`、`fn_`、`fn_name_` 和 `source_range_`
// 使用 `std::move` 将 `callee` 和 `source_range` 移动到对应的成员变量


InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range,
    std::optional<ModuleInstanceInfo> module_instance_info,
    std::string& function_name)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(std::move(function_name)),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}
``` 
// 构造函数，初始化成员变量 `callee_`、`fn_`、`fn_name_`、`source_range_` 和 `module_instance_info_`
// 使用 `std::move` 将 `callee`、`function_name`、`source_range` 和 `module_instance_info` 移动到对应的成员变量


InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range,
    std::optional<ModuleInstanceInfo> module_instance_info)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}
``` 
// 构造函数，初始化成员变量 `callee_`、`fn_`、`fn_name_`、`source_range_` 和 `module_instance_info_`
// 使用 `std::move` 将 `callee`、`source_range` 和 `module_instance_info` 移动到对应的成员变量


std::optional<InlinedCallStackPtr> InlinedCallStack::callee() const {
``` 
// 返回成员变量 `callee_` 的可选值
  return callee_;
}

void InlinedCallStack::setCallee(std::optional<InlinedCallStackPtr> callee) {
``` 
// 设置成员变量 `callee_` 的值
  callee_ = std::move(callee);
}

std::optional<ModuleInstanceInfo> InlinedCallStack::module_instance() const {
``` 
// 返回成员变量 `module_instance_info_` 的可选值
  return module_instance_info_;
}

SourceRange InlinedCallStack::source_range() const {
``` 
// 返回成员变量 `source_range_`
  return source_range_;
}

Function* InlinedCallStack::function() const {
``` 
// 返回成员变量 `fn_`
  return fn_;
}

const std::string& InlinedCallStack::function_name() const {
``` 
// 返回成员变量 `fn_name_` 的常量引用
  return fn_name_;
}
# 返回当前内联调用堆栈的向量表示
std::vector<InlinedCallStackEntry> InlinedCallStack::vec() {
  # 创建一个空向量 r，用于存储内联调用堆栈条目
  std::vector<InlinedCallStackEntry> r;
  # 通过当前对象获取强制转换的智能指针 current
  std::optional<InlinedCallStackPtr> current = intrusive_from_this();
  # 当 current 有值时执行循环
  while (current) {
    # 向向量 r 中添加当前调用堆栈条目，包括函数名、源代码范围和模块实例信息
    r.emplace_back(
        (*current)->fn_,
        (*current)->source_range_,
        (*current)->module_instance_info_);
    # 将 current 更新为当前调用堆栈条目的调用者
    current = (*current)->callee_;
  }
  # 返回填充后的向量 r
  return r;
}

ModuleInstanceInfo::ModuleInstanceInfo(
    c10::ClassTypePtr module_type,
    std::string instance_name)
    : module_type_(std::move(module_type)),
      instance_name_(std::move(instance_name)) {}
# namespace torch::jit 命名空间的结束
} // namespace torch::jit
```