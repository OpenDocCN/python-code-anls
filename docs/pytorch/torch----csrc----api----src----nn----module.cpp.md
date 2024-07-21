# `.\pytorch\torch\csrc\api\src\nn\module.cpp`

```
// 引入 Torch 的 nn 模块定义
#include <torch/nn/module.h>

// 引入 Torch 的 ordered_dict 定义
#include <torch/ordered_dict.h>

// 引入 Torch 的 VariableType.h，包含自动求导相关内容
#include <torch/csrc/autograd/generated/VariableType.h>

// 引入 C10 库的 Exception 定义
#include <c10/util/Exception.h>

// 引入 C++ 标准库的算法库
#include <algorithm>

// 引入 C++ 标准库的函数式编程部分
#include <functional>

// 引入 C++ 标准库的映射容器定义
#include <map>

// 引入 C++ 标准库的输出流定义
#include <ostream>

// 引入 C++ 标准库的字符串处理部分
#include <string>

// 引入 C++ 标准库的类型信息
#include <typeinfo>

// Torch 命名空间
namespace torch {
// Torch 的 nn 命名空间
namespace nn {
// 匿名命名空间，用于内部函数和私有变量
namespace {

/// 将名称按层次连接起来: 如果 `name_prefix` 不为空，则为 "name_prefix.name"，否则为 "name"。
std::string join_name(const std::string& name_prefix, const std::string& name) {
  // 计算合并后的总大小
  size_t total_size = name.size();
  if (!name_prefix.empty()) {
    total_size += name_prefix.size() + 1;
  }
  // 创建用于存储合并后名称的字符串
  std::string full_name;
  full_name.reserve(total_size);
  if (!name_prefix.empty()) {
    // 如果前缀不为空，则添加前缀和一个点
    full_name += name_prefix;
    full_name.push_back('.');
  }
  // 添加名称
  full_name += name;
  return full_name;
}

} // namespace

// Module 类的默认构造函数
Module::Module()
    : parameters_("Parameter"), buffers_("Buffer"), children_("Submodule") {}

// Module 类的带名称参数的构造函数
Module::Module(std::string name) : Module() {
  // 移动赋值名称
  name_ = std::move(name);
}

// 返回模块名称的引用，若未设置名称则根据动态类型获取名称
const std::string& Module::name() const noexcept {
  // 如果名称为空，根据 RTTI 获取动态类型的名称
  if (!name_.has_value()) {
    name_ = c10::demangle(typeid(*this).name());
    // 在 Windows 系统下修正类型名称前缀
#if defined(_WIN32)
    if (name_->find("struct ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 7);
    } else if (name_->find("class ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 6);
    }
#endif // defined(_WIN32)
  }
  return *name_;
}

// 克隆模块，若未实现则抛出错误
std::shared_ptr<Module> Module::clone(const optional<Device>& device) const {
  AT_ERROR(
      "clone() has not been implemented for ",
      name(),
      ". Subclass torch::nn::Cloneable<",
      name(),
      "> instead of torch::nn::Module to inherit the ability to clone.");
}

// 对当前模块及其子模块应用给定函数
void Module::apply(const ModuleApplyFunction& function) {
  // 对当前模块应用函数
  function(*this);
  // 对子模块递归应用函数
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

// 对当前模块及其子模块应用给定的常量函数
void Module::apply(const ConstModuleApplyFunction& function) const {
  // 对当前模块应用常量函数
  function(*this);
  // 对子模块递归应用常量函数
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

// 对当前模块及其子模块应用带名称的函数
void Module::apply(
    const NamedModuleApplyFunction& function,
    // 调用函数对象 `function`，传递参数 `name_prefix` 和 `*this`
    function(/*name=*/name_prefix, *this);

    // 对每个子模块应用函数对象 `function`
    apply_to_submodules(
        // lambda 函数接受子模块的名称 `name` 和共享指针 `module`，并调用函数对象 `function`
        [&function](
            const std::string& name, const std::shared_ptr<Module>& module) {
          function(name, *module);
        },
        // 将 `name_prefix` 作为参数传递给 `apply_to_submodules` 函数
        name_prefix);
}

void Module::apply(
    const ConstNamedModuleApplyFunction& function,
    const std::string& name_prefix) const {
  // 调用给定的函数来处理当前模块和指定名称前缀
  function(/*name=*/name_prefix, *this);
  // 对子模块递归调用 apply_to_submodules 函数
  apply_to_submodules(
      [&function](
          const std::string& name, const std::shared_ptr<Module>& module) {
        // 对每个子模块调用给定的函数处理
        function(name, *module);
      },
      name_prefix);
}

void Module::apply(const ModulePointerApplyFunction& function) const {
  // 调用给定的函数来处理当前模块的共享指针
  function(shared_from_this_checked());
  // 对子模块递归调用 apply_to_submodules 函数
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        // 对每个子模块调用给定的函数处理
        function(module);
      });
}

void Module::apply(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  // 调用给定的函数来处理当前模块的名称前缀和共享指针
  function(
      /*name=*/name_prefix, shared_from_this_checked());
  // 对子模块递归调用 apply_to_submodules 函数
  apply_to_submodules(function, name_prefix);
}

std::vector<Tensor> Module::parameters(bool recurse) const {
  // 返回当前模块的所有参数张量
  return named_parameters(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_parameters(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  if (!recurse) {
    // 如果不递归，则只返回当前模块的直接参数
    for (const auto& parameter : parameters_) {
      if (parameter.value().defined()) {
        result.insert(parameter.key(), parameter.value());
      }
    }
  } else {
    // 否则，递归调用 apply 函数来获取所有子模块的参数
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& parameter : module.named_parameters(/*recurse=*/false)) {
        TORCH_INTERNAL_ASSERT(parameter.value().defined());
        result.insert(join_name(name, parameter.key()), parameter.value());
      }
    });
  }
  return result;
}

std::vector<Tensor> Module::buffers(bool recurse) const {
  // 返回当前模块的所有缓冲张量
  return named_buffers(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_buffers(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  if (!recurse) {
    // 如果不递归，则只返回当前模块的直接缓冲
    for (const auto& buffer : buffers_) {
      if (buffer.value().defined()) {
        result.insert(buffer.key(), buffer.value());
      }
    }
  } else {
    // 否则，递归调用 apply 函数来获取所有子模块的缓冲
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& buffer : module.named_buffers(/*recurse=*/false)) {
        TORCH_INTERNAL_ASSERT(buffer.value().defined());
        result.insert(join_name(name, buffer.key()), buffer.value());
      }
    });
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool include_self) const {
  std::vector<std::shared_ptr<Module>> result;
  if (include_self) {
    // 如果包括自身，则将当前模块加入结果列表
    apply([&result](const std::shared_ptr<Module>& module) {
      result.push_back(module);
    });
  } else {
    // 否则，仅将子模块加入结果列表
    apply_to_submodules(
        [&result](const std::string&, const std::shared_ptr<Module>& module) {
          result.push_back(module);
        });
  }
  return result;
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_modules(
    const std::string& name_prefix,
    bool include_self) const {
  OrderedDict<std::string, std::shared_ptr<Module>> result;
  if (include_self) {
    // 如果包括自身，则将当前模块以及其名称前缀加入结果字典
    result.insert(name_prefix, shared_from_this_checked());

  result.insert(name_prefix, shared_from_this_checked());
  // 对子模块递归调用 named_modules 函数，将结果合并到当前结果字典中
  apply_to_submodules(
      [&result](const std::string& name, const std::shared_ptr<Module>& module) {
        result.insert(join_name(name, module->name), module);
      },
      name_prefix);
}
    // 如果条件成立，执行以下代码块
    apply(
        // 对于给定的名称前缀，将每个模块插入结果字典
        [&result](
            const std::string& key, const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        name_prefix);
  } else {
    // 如果条件不成立，执行以下代码块
    apply_to_submodules(
        // 对于给定的名称前缀，将每个模块插入结果字典（子模块版本）
        [&result](
            const std::string& key, const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        name_prefix);
  }
  // 返回最终的结果字典
  return result;
}

// 返回当前模块的所有子模块的共享指针向量
std::vector<std::shared_ptr<Module>> Module::children() const {
  return children_.values();
}

// 返回一个有序字典，包含当前模块的所有命名子模块
OrderedDict<std::string, std::shared_ptr<Module>> Module::named_children() const {
  return children_;
}

// 设置模块及其所有子模块的训练状态
void Module::train(bool on) {
  for (auto& child : children_) {
    child.value()->train(on);
  }
  // 设置当前模块的训练状态
  is_training_ = on;
}

// 设置模块及其所有子模块为评估模式
void Module::eval() {
  train(/*on=*/false);
}

// 将模块及其所有子模块移动到指定的设备和数据类型
void Module::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

// 将模块及其所有子模块转换到指定的数据类型
void Module::to(torch::Dtype dtype, bool non_blocking) {
  to_impl(dtype, non_blocking);
}

// 将模块及其所有子模块移动到指定的设备
void Module::to(torch::Device device, bool non_blocking) {
  to_impl(device, non_blocking);
}

// 返回当前模块是否处于训练状态
bool Module::is_training() const noexcept {
  return is_training_;
}

// 将模块及其所有子模块的梯度置零
void Module::zero_grad(bool set_to_none) {
  for (auto& child : children_) {
    child.value()->zero_grad(set_to_none);
  }
  for (auto& parameter : named_parameters(/*recurse=*/false)) {
    auto& grad = parameter->mutable_grad();
    if (grad.defined()) {
      grad = grad.detach();

      if (set_to_none)
        grad.reset();
      else
        grad.zero_();
    }
  }
}

// 将模块及其所有参数和缓冲区保存到输出存档中
void Module::save(serialize::OutputArchive& archive) const {
  for (const auto& parameter : named_parameters(/*recurse=*/false)) {
    archive.write(parameter.key(), parameter.value());
  }
  for (const auto& buffer : named_buffers(/*recurse=*/false)) {
    archive.write(buffer.key(), buffer.value(), /*is_buffer=*/true);
  }
  for (const auto& child : children_) {
    if (child.value()->is_serializable()) {
      serialize::OutputArchive child_archive(archive.compilation_unit());
      child.value()->save(child_archive);
      archive.write(child.key(), child_archive);
    }
  }
}

// 从输入存档中加载模块及其所有参数和缓冲区
void Module::load(serialize::InputArchive& archive) {
  for (auto& parameter : named_parameters(/*recurse=*/false)) {
    archive.read(parameter.key(), parameter.value());
  }
  for (auto& buffer : named_buffers(/*recurse=*/false)) {
    archive.read(buffer.key(), buffer.value(), /*is_buffer=*/true);
  }
  for (const auto& child : children_) {
    if (child.value()->is_serializable()) {
      serialize::InputArchive child_archive;
      archive.read(child.key(), child_archive);
      child.value()->load(child_archive);
    }
  }
}

// 判断当前模块是否可序列化
bool Module::is_serializable() const {
  return true;
}

// 注册参数并返回参数的引用
Tensor& Module::register_parameter(
    std::string name,
    Tensor tensor,
    bool requires_grad) {
  TORCH_CHECK(!name.empty(), "Parameter name must not be empty");
  TORCH_CHECK(
      name.find('.') == std::string::npos,
      "Parameter name must not contain a dot (got '",
      name,
      "')");
  if (!tensor.defined()) {
    if (requires_grad) {
      TORCH_WARN(
          "An undefined tensor cannot require grad. ",
          "Ignoring the `requires_grad=true` function parameter.");
    }
  } else {
    tensor.set_requires_grad(requires_grad);
  }
  return parameters_.insert(std::move(name), std::move(tensor));
}
// 将一个缓冲区注册到模块中，返回注册后的缓冲区
Tensor& Module::register_buffer(std::string name, Tensor tensor) {
  // 检查缓冲区名字不为空
  TORCH_CHECK(!name.empty(), "Buffer name must not be empty");
  // 检查缓冲区名字中不包含点号
  TORCH_CHECK(
      name.find('.') == std::string::npos,
      "Buffer name must not contain a dot (got '",
      name,
      "')");
  // 将缓冲区插入到模块的缓冲区中
  return buffers_.insert(std::move(name), std::move(tensor));
}

// 取消注册模块
void Module::unregister_module(const std::string& name) {
  // 检查要取消注册的模块是否存在
  TORCH_CHECK(
      children_.contains(name),
      "No Module with name `",
      name,
      "` is registered");
  // 从模块中移除指定名字的子模块
  children_.erase(name);
}

// 将模块的名称输出到流中
void Module::pretty_print(std::ostream& stream) const {
  stream << name();
}

// 递归地将模块及其子模块的信息输出到流中
void Module::pretty_print_recursive(
    std::ostream& stream,
    const std::string& indentation) const {
  // 打印当前模块的名称
  pretty_print(stream);
  // 如果模块有子模块
  if (!children_.is_empty()) {
    stream << "(\n";
    // 下一级的缩进
    const std::string next_indentation = indentation + "  ";
    // 遍历每个子模块
    for (const auto& child : children_) {
      // 打印子模块的名字及其信息
      stream << next_indentation << "(" << child.key() << "): ";
      child.value()->pretty_print_recursive(stream, next_indentation);
      stream << '\n';
    }
    // 结束当前级别的模块输出
    stream << indentation << ")";
  }
}

// 克隆另一个模块的内容到当前模块
void Module::clone_(Module& other, const optional<Device>& device) {}

// 对每个子模块应用指定的函数
void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  // 遍历每个子模块
  for (const auto& child : children_) {
    // 获取完整的子模块名称
    auto qualified_name = join_name(name_prefix, child.key());
    // 对子模块应用指定函数
    function(qualified_name, child.value());
    // 递归地对子模块的子模块应用函数
    child.value()->apply_to_submodules(function, qualified_name);
  }
}

// 返回当前模块的安全共享指针
std::shared_ptr<Module> Module::shared_from_this_checked() const {
  std::shared_ptr<const Module> ptr;
  // 尝试获取当前模块的共享指针
  try {
    ptr = shared_from_this();
  } catch (const std::bad_weak_ptr&) {
    // 如果获取失败，抛出错误信息
    AT_ERROR(
        "It looks like you attempted to retrieve your top-level module "
        "as a shared_ptr, but it is not stored in a shared_ptr. "
        "Use std::make_shared<",
        name(),
        "> instead of creating your module on "
        "the stack, or alternatively do not try to access your top-level "
        "module at all by passing /*include_self=*/false "
        "to modules() or named_modules()");
  }
  // 返回当前模块的共享指针（允许修改）
  return std::const_pointer_cast<Module>(ptr);
}

// 将模块及其子模块的信息递归输出到流中
std::ostream& operator<<(std::ostream& stream, const nn::Module& module) {
  // 调用模块的递归打印函数
  module.pretty_print_recursive(stream, "");
  return stream;
}

// 将模块序列化为输出存档
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const std::shared_ptr<nn::Module>& module) {
  // 检查模块不为空
  TORCH_CHECK(module != nullptr, "Cannot serialize empty module");
  // 调用模块的保存函数
  module->save(archive);
  return archive;
}

// 从输入存档中反序列化模块
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    const std::shared_ptr<nn::Module>& module) {
  // 检查模块不为空
  TORCH_CHECK(module != nullptr, "Cannot deserialize empty module");
  // 调用模块的加载函数
  module->load(archive);
  return archive;
}
```