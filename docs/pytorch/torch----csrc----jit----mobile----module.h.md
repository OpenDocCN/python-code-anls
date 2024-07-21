# `.\pytorch\torch\csrc\jit\mobile\module.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <ATen/core/jit_type.h>
// 包含 ATen 库中的 jit_type.h 头文件

#include <torch/csrc/jit/mobile/debug_info.h>
// 包含 Torch 移动端调试信息的头文件

#include <torch/csrc/jit/mobile/function.h>
// 包含 Torch 移动端函数定义的头文件

#include <torch/csrc/jit/mobile/method.h>
// 包含 Torch 移动端方法定义的头文件

#include <torch/csrc/jit/mobile/quantization.h>
// 包含 Torch 移动端量化的头文件

#include <utility>
// 包含 C++ 标准库中的 utility 头文件

namespace torch {
namespace jit {
namespace mobile {
// 命名空间声明，包含了 Torch 移动端相关的内容，使得这些内容在命名空间内部可见

using Stack = std::vector<c10::IValue>;
// 使用语句，定义 Stack 类型为一个存储 c10::IValue 元素的向量

// CompilationUnit 类的定义
// 用于表示执行由轻量级解释器执行的编译单元
class CompilationUnit {
 public:
  void register_function(std::unique_ptr<Function> fn);
  // 注册一个函数对象到 CompilationUnit 中

  std::vector<std::unique_ptr<Function>>& methods() {
    return methods_;
  }
  // 返回方法对象的列表引用

  const std::vector<std::unique_ptr<Function>>& methods() const {
    return methods_;
  }
  // 返回常量方法对象的列表引用

  Function* find_function(const c10::QualifiedName& qn);
  // 查找给定限定名称的函数对象

  const Function* find_function(const c10::QualifiedName& qn) const;
  // 查找给定限定名称的常量函数对象

  void unsafeRemoveFunction(const int64_t index) {
    methods_.erase(methods_.begin() + index);
  }
  // 不安全地移除给定索引的函数对象

 private:
  std::vector<std::unique_ptr<Function>> methods_;
  // 存储函数对象的列表
};

// Module 类的定义
// 表示 Torch 移动端模块，包含数据、元数据和编译单元
class TORCH_API Module {
 public:
  Module(
      c10::intrusive_ptr<c10::ivalue::Object> object,
      std::shared_ptr<CompilationUnit> cu)
      : object_(std::move(object)), cu_(std::move(cu)) {}
  // 构造函数，初始化对象和编译单元

  Module() = default;
  // 默认构造函数

  Method get_method(const std::string& method_name) const;
  // 获取指定方法名的方法对象

  template <typename... Types>
  c10::IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }
  // 运行指定方法名的方法对象，并返回执行结果

  c10::IValue forward(std::vector<c10::IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }
  // 执行模块的前向传播方法

  std::optional<Method> find_method(const std::string& basename) const;
  // 查找指定基础名称的方法对象，返回一个 optional

  const std::string name() const {
    return object_->name();
  }
  // 返回模块的名称

  const std::vector<at::IValue>& slots() const {
    return object_->slots();
  }
  // 返回模块的槽位列表

  const c10::intrusive_ptr<c10::ivalue::Object> _ivalue() const {


以上是对给定的 C++ 代码段进行了详细的注释解释，确保每一行代码的作用和含义都得到了说明。
  // 返回私有成员变量 object_
  return object_;
}

const std::vector<at::Tensor> parameters() const;
const std::map<std::string, at::Tensor> named_parameters() const;

// 返回一个描述前向传播调试信息的字符串
std::string get_forward_method_debug_info(int64_t debug_handle) const;

// 返回模块层次结构的描述字符串
std::string getModuleHierarchy(const int64_t debug_handle) const;

// 返回调用堆栈信息的描述字符串
std::string getCallStack(const int64_t debug_handle) const;

/// 启用 "训练" 模式
void train(bool on = true);

/// 调用 train(false) 来启用 "评估" 模式
void eval() {
  train(/*on=*/false);
}

// 返回模块是否处于训练模式的布尔值
bool is_training() const;

// 返回元数据的无序映射
const std::unordered_map<std::string, std::string> getMetadata() const {
  return metadata_;
}

// 设置元数据
void setMetadata(
    const std::unordered_map<std::string, std::string>& metadata) {
  metadata_ = metadata;
}

const std::vector<Method> get_methods() const;

// 返回具有给定名称的属性值，如果找不到则返回 or_else
c10::IValue attr(const std::string& name, c10::IValue or_else) const {
  if (auto r = object_->type()->findAttributeSlot(name)) {
    return object_->getSlot(*r);
  }
  if (auto r = object_->type()->findConstantSlot(name)) {
    return object_->type()->getConstant(*r);
  }
  return or_else;
}

// 设置调试表
void setDebugTable(MobileDebugTable&& debug_table) {
  debug_table_ = std::move(debug_table);
}

// 返回调试表
const MobileDebugTable& getDebugTable() const {
  return debug_table_;
}

// 设置是否具有调试句柄
void setHasDebugHandles(bool has_debug_handles) {
  has_debug_handles_ = has_debug_handles;
}

// 返回是否具有调试句柄
bool hasDebugHandles() const {
  return has_debug_handles_;
}

// 返回编译单元的引用
const CompilationUnit& compilation_unit() const {
  return *cu_.get();
}

// 设置要删除的内存
void set_delete_memory(std::shared_ptr<char> delete_mem) {
  mem_to_delete_ = std::move(delete_mem);
}

// 设置最小运算符版本
void set_min_operator_version(int64_t version) {
  min_operator_version_ = version;
}

// 返回最小运算符版本
int64_t min_operator_version() const {
  return min_operator_version_;
}

// 设置字节码版本
void set_bytecode_version(int64_t version) {
  bytecode_version_ = version;
}

// 返回字节码版本
int64_t bytecode_version() const {
  return bytecode_version_;
}

private:
friend class quantization::PTQQuanizationHelper;

// 比较方法模式
bool compareMethodSchemas(
    const std::string& name_1,
    const std::string& name_2);

// 不安全地移除方法
void unsafeRemoveMethod(const std::string& basename);

// 不安全地复制方法
void unsafeCopyMethod(
    const std::string& new_method_name,
    const Function& to_be_copied);

// 持有一个对象的内部指针
c10::intrusive_ptr<c10::ivalue::Object> object_;

// 元数据映射
std::unordered_map<std::string, std::string> metadata_;

// 编译单元指针的共享指针
std::shared_ptr<CompilationUnit> cu_;

// 移动调试表
MobileDebugTable debug_table_;

// 是否有调试句柄的标志
bool has_debug_handles_ = false;

// 最小运算符版本
int64_t min_operator_version_ = 4;

// 字节码版本
int64_t bytecode_version_ = 4;

// 模块删除自身时要删除的内存句柄
std::shared_ptr<char> mem_to_delete_;
};

// 结构体 ModuleInfo 包含模块的字节码版本、操作符版本以及操作名到参数数量的映射、函数名集合、类型名集合
struct TORCH_API ModuleInfo {
  uint64_t bytecode_version;  // 模块的字节码版本
  uint64_t operator_version;  // 模块的操作符版本
  std::unordered_map<std::string, int> opname_to_num_args;  // 操作名到参数数量的映射
  std::unordered_set<std::string> function_names;  // 函数名集合
  std::unordered_set<std::string> type_names;  // 类型名集合
};

// 声明获取模块信息的函数，接受一个 mobile::Module 的常引用作为参数，返回 ModuleInfo 结构体
TORCH_API ModuleInfo get_module_info(const mobile::Module& module);

// namespace mobile 结束
} // namespace mobile

// namespace jit 结束
} // namespace jit

// namespace torch 结束
} // namespace torch
```