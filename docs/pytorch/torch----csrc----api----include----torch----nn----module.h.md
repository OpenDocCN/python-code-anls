# `.\pytorch\torch\csrc\api\include\torch\nn\module.h`

```py
#pragma once

#include <torch/nn/modules/container/any_module_holder.h>
#include <torch/nn/modules/container/any_value.h>
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <ATen/ATen.h>

#include <functional>  // 包含函数对象的头文件
#include <iosfwd>       // 包含前置声明流的头文件
#include <map>          // 包含映射容器的头文件
#include <memory>       // 包含智能指针和内存管理的头文件
#include <string>       // 包含字符串处理的头文件
#include <type_traits>  // 包含类型特性的头文件

namespace torch {
namespace nn {

/// The base class for all modules in PyTorch.
///
/// \rst
/// .. note::
///   The design and implementation of this class is largely based on the Python
///   API. You may want to consult the python documentation for
///   :py:class:`pytorch:torch.nn.Module` for further clarification on certain
///   methods or behavior.
/// \endrst
///
/// A `Module` is an abstraction over the implementation of some function or
/// algorithm, possibly associated with some persistent data. A `Module` may
/// contain further `Module`s ("submodules"), each with their own
/// implementation, persistent data and further submodules. `Module`s can thus
/// be said to form a recursive tree structure. A `Module` is registered as a
/// submodule to another `Module` by calling `register_module()`, typically from
/// within a parent module's constructor.
///
/// A distinction is made between three kinds of persistent data that may be
/// associated with a `Module`:
///
/// 1. *Parameters*: tensors that record gradients, typically weights updated
///    during the backward step (e.g. the `weight` of a `Linear` module),
/// 2. *Buffers*: tensors that do not record gradients, typically updated during
///    the forward step, such as running statistics (e.g. `mean` and `variance`
///    in the `BatchNorm` module),
/// 3. Any additional state, not necessarily tensors, required for the
///    implementation or configuration of a `Module`.
///
/// The first two kinds of state are special in that they may be registered
/// with the `Module` system to allow convenient access and batch configuration.
/// For example, registered parameters in any `Module` may be iterated over via
/// the `parameters()` accessor. Further, changing the data type of a `Module`'s
/// registered parameters can be done conveniently via `Module::to()`, e.g.
/// `module->to(torch::kCUDA)` to move all parameters to GPU memory. Lastly,
/// registered parameters and buffers are handled specially during a `clone()`
/// operation, which performs a deepcopy of a cloneable `Module` hierarchy.
///
/// Parameters are registered with a `Module` via `register_parameter`. Buffers
/// are registered separately via `register_buffer`. These methods are part of
/// the public API of `Module` and are typically invoked from within a
/// concrete `Module`s constructor.
class Module {
  public:
    // 返回假值，用于指示不需要额外的前向传播参数
    return false;
  }

  // 虚函数，返回前向传播所需的参数数量
  virtual unsigned int _forward_num_required_args() {
    // 使用 TORCH_CHECK 断言来验证条件为 false，如果条件不满足，则输出错误信息
    TORCH_CHECK(
        false,
        "torch::nn::Module subclass that has default arguments in `forward` method ",
        "must override `_forward_num_required_args` method. Please use ",
        "`FORWARD_HAS_DEFAULT_ARGS` macro to do so.");
  }

  // 虚函数，用于在 `forward` 方法中填充默认参数
  virtual std::vector<AnyValue> _forward_populate_default_args(
      std::vector<AnyValue>&& arguments) {
    // 使用 TORCH_CHECK 断言来验证条件为 false，如果条件不满足，则输出错误信息
    TORCH_CHECK(
        false,
        "torch::nn::Module subclass that has default arguments in `forward` method ",
        "must override `_forward_populate_default_args` method. Please use ",
        "`FORWARD_HAS_DEFAULT_ARGS` macro to do so.");
  }

  /// 此 `Module` 的注册参数集合。
  /// 可以通过 ParameterDict 和 ParameterList 访问 parameters_
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  OrderedDict<std::string, Tensor> parameters_;

 private:
  // 友元类声明

  // `Cloneable` 类是此类的友元类模板
  template <typename Derived>
  friend class Cloneable;

  // `AnyModuleHolder` 结构体模板是此类的友元类模板，接受 `ModuleType` 和 `ArgumentTypes...` 作为模板参数
  template <typename ModuleType, typename... ArgumentTypes>
  friend struct AnyModuleHolder;

  /// 将给定的 `Module` 漂亮地打印到 `ostream` 中。
  TORCH_API friend std::ostream& operator<<(
      std::ostream& stream,
      const nn::Module& module);

  // 数据并行使用此方法在复制步骤期间配置梯度边缘。
  template <typename ModuleType>
  friend void replicate_grad_edges(
      const std::shared_ptr<Module>& module,
      const std::vector<std::shared_ptr<ModuleType>>& replicas,
      const std::vector<Device>& devices);

  // 私有方法声明

  /// 在 `Cloneable` 实现中使用。
  virtual void clone_(Module& other, const optional<Device>& device);

  /// 各种 `to()` 方法的实现。
  template <typename... Ts>
  void to_impl(Ts&&... ts);

  /// 实现模块层次的漂亮打印。
  void pretty_print_recursive(
      std::ostream& stream,
      const std::string& indentation) const;

  /// 对每个子模块递归应用 `function`，从此 `Module` 的子代开始（因此不包括模块本身）。
  void apply_to_submodules(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  /// 返回一个安全（已检查的）方式共享指向 `this` 的 shared_ptr。
  std::shared_ptr<Module> shared_from_this_checked() const;

  /// 此 `Module` 的注册缓冲区集合。
  OrderedDict<std::string, Tensor> buffers_;

  /// 此 `Module` 的注册（直接）子模块集合。
  OrderedDict<std::string, std::shared_ptr<Module>> children_;

  /// 模块的名称（例如 "LSTM"）。
  mutable optional<std::string> name_;

  /// 模块是否处于训练模式。
  bool is_training_{true};
/// 结尾处的分号，用于结束上一段代码的语句或声明。
};

/// 将指向 `nn::Module` 的智能指针序列化到 `OutputArchive` 中。
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const std::shared_ptr<nn::Module>& module);

/// 从 `InputArchive` 中反序列化 `nn::Module`。
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    const std::shared_ptr<nn::Module>& module);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nn::Module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
typename ModuleType::ContainedType* Module::as() noexcept {
  // 使用 `ModuleHolder` 的包含类型，例如对于 `Linear` 使用 `LinearImpl`，
  // 因为 `LinearImpl` 继承自 `nn::Module`。
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType>
const typename ModuleType::ContainedType* Module::as() const noexcept {
  // 使用 `ModuleHolder` 的包含类型，例如对于 `Linear` 使用 `LinearImpl`，
  // 因为 `LinearImpl` 继承自 `nn::Module`。
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType, typename>
ModuleType* Module::as() noexcept {
  // 将当前对象尝试转换为 `ModuleType` 类型的指针。
  return dynamic_cast<ModuleType*>(this);
}

template <typename ModuleType, typename>
const ModuleType* Module::as() const noexcept {
  // 将当前对象尝试转换为 `ModuleType` 类型的常量指针。
  return dynamic_cast<const ModuleType*>(this);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    std::shared_ptr<ModuleType> module) {
  // 检查子模块名非空。
  TORCH_CHECK(!name.empty(), "Submodule name must not be empty");
  // 检查子模块名不包含点号。
  TORCH_CHECK(
      name.find('.') == std::string::npos,
      "Submodule name must not contain a dot (got '",
      name,
      "')");
  // 将模块插入到子模块集合中，并返回类型转换后的智能指针。
  auto& base_module = children_.insert(std::move(name), std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    ModuleHolder<ModuleType> module_holder) {
  // 调用前一个重载的 `register_module` 函数。
  return register_module(std::move(name), module_holder.ptr());
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    std::shared_ptr<ModuleType> module) {
  // 替换指定名字的子模块，并返回类型转换后的智能指针。
  auto& base_module = (children_[name] = std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    ModuleHolder<ModuleType> module_holder) {
  // 调用前一个重载的 `replace_module` 函数。
  return replace_module(name, module_holder.ptr());
}

template <typename... Ts>
void Module::to_impl(Ts&&... ts) {
  // 对每个子模块调用 `to()` 函数。
  for (auto& child : children_) {
    child.value()->to(ts...);
  }
  // 将每个参数移动到新的数据类型/设备。
  for (auto& parameter : named_parameters(/*recurse=*/false)) {
    parameter->set_data(autograd::Variable(*parameter).to(ts...));
  }
  // 将每个缓冲区移动到新的数据类型/设备。
  for (auto& buffer : named_buffers(/*recurse=*/false)) {
    buffer->set_data(autograd::Variable(*buffer).to(ts...));


    // 使用 buffer 指针创建一个 autograd::Variable 对象，然后将其转换为 ts 中指定的数据类型
    buffer->set_data(autograd::Variable(*buffer).to(ts...));
}

} // namespace nn
} // namespace torch
```