# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\any.h`

```py
#pragma once
// 预处理命令，确保头文件只被编译一次

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/any_module_holder.h>
#include <torch/nn/modules/container/any_value.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Device.h>

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// 存储类型擦除后的 `Module`。
///
/// PyTorch C++ API 对于 `Module` 子类的 `forward()` 没有强加统一的接口。这为设计
/// `forward()` 方法提供了完全自由。然而，这也意味着没有统一的基类型来存储任意模块
/// 以便多态调用 `forward()`。这时 `AnyModule` 就派上用场了。
///
/// `AnyModule` 可以存储任何提供 `forward()` 方法的 `nn::Module` 子类。这个 `forward()`
/// 方法可以接受任意类型的参数并返回任意类型的结果。一旦存储在 `AnyModule` 中，你可以
/// 通过调用 `AnyModule::forward()` 方法并传入与存储模块相同类型的参数来调用底层模块的
/// `forward()` 方法（尽管请注意下面的一个重要限制）。
///
/// 示例：
///
/// \rst
/// .. code-block:: cpp
///
///   struct GenericTrainer {
///     torch::nn::AnyModule module;
///
///     void train(torch::Tensor input) {
///       module.forward(input);
///     }
///   };
///
///   GenericTrainer trainer1{torch::nn::Linear(3, 4)};
///   GenericTrainer trainer2{torch::nn::Conv2d(3, 4, 2)};
/// \endrst
///
/// 由于 `AnyModule` 通过类型擦除静态类型来实现多态，参数的类型检查被移动到运行时。这意味着，
/// 向 `AnyModule` 传递类型不正确的参数会在编译时通过，但在运行时会抛出异常：
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnyModule module(torch::nn::Linear(3, 4));
///   // Linear 模块期望一个张量作为输入，但我们传递了一个整数。
///   // 这会在编译时通过，但在运行时抛出 `torch::Error` 异常。
///   module.forward(123);
/// \endrst
///
/// \rst
/// .. attention::
///   `AnyModule` 的一个值得注意的限制是，其 `forward()` 方法不支持参数类型的隐式转换。
///   例如，如果存储模块的 `forward()` 方法接受 `float`，而你调用 `any_module.forward(3.4)`
///   （其中 `3.4` 是 `double` 类型），这将会抛出异常。
/// \endrst
///
/// `AnyModule` 的 `forward()` 方法的返回类型由其模板参数决定，默认为 `torch::Tensor`。
/// 若要更改返回类型，可以编写例如 `any_module.forward<int>()`。
///
/// \rst
/// .. code-block:: cpp
///
// 创建一个 `torch::nn::AnyModule` 对象，使用 `torch::nn::Linear` 模块初始化，参数是输入维度为3，输出维度为4
torch::nn::AnyModule module(torch::nn::Linear(3, 4));

// 对已创建的 `AnyModule` 对象调用 `forward` 方法，输入参数是一个大小为 {2, 3} 的张量，并获取输出
auto output = module.forward(torch::ones({2, 3}));

// 定义一个结构体 `IntModule`，包含一个 `forward` 方法，接受整数参数并返回相同的整数
struct IntModule {
  int forward(int x) { return x; }
};

// 使用 `IntModule` 结构体初始化一个新的 `torch::nn::AnyModule` 对象
torch::nn::AnyModule module(IntModule{});

// 对新的 `AnyModule` 对象调用 `forward` 方法，传入整数参数 5，并获取输出结果
int output = module.forward<int>(5);

// `AnyModule` 类提供的唯一其他方法是 `clone()`。但是，可以通过 `.ptr()` 获取模块的句柄，
// 返回一个 `shared_ptr<nn::Module>`。此外，如果知道存储模块的具体类型，可以使用 `.get<T>()` 获取具体类型的句柄。
//
// 示例：
// 创建一个 `torch::nn::AnyModule` 对象，使用 `torch::nn::Linear` 模块初始化
torch::nn::AnyModule module(torch::nn::Linear(3, 4));

// 获取存储模块的具体类型为 `torch::nn::Linear` 的句柄
std::shared_ptr<nn::Module> ptr = module.ptr();

// 通过 `.get<T>()` 方法获取存储模块的具体类型为 `torch::nn::Linear` 的句柄
torch::nn::Linear linear(module.get<torch::nn::Linear>());
// 在 AnyModule 类中定义的成员函数，用于创建当前对象的副本
inline AnyModule AnyModule::clone(optional<Device> device) const {
  // 创建一个新的 AnyModule 对象作为副本
  AnyModule clone;
  // 如果当前对象 content_ 不为空，则调用其 clone_module 方法创建 content_ 的副本
  clone.content_ = content_ ? content_->clone_module(device) : nullptr;
  return clone;
}

// 在 AnyModule 类中定义的模板成员函数，用于对 AnyModule 进行赋值操作
template <typename ModuleType>
AnyModule& AnyModule::operator=(std::shared_ptr<ModuleType> module) {
  // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature)
  // 使用移动语义将给定的模块 module 赋值给当前的 AnyModule 对象
  return (*this = AnyModule(std::move(module)));
}

// 在 AnyModule 类中定义的模板成员函数，用于进行任意类型的 forward 操作
template <typename... ArgumentTypes>
AnyValue AnyModule::any_forward(ArgumentTypes&&... arguments) {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call forward() on an empty AnyModule");
  // 创建一个存储 AnyValue 类型值的向量，并预留空间以容纳参数的数量
  std::vector<AnyValue> values;
  values.reserve(sizeof...(ArgumentTypes));
  // 使用 apply 函数将传入的参数封装成 AnyValue 类型，并添加到 values 中
  torch::apply(
      [&values](AnyValue&& value) { values.push_back(std::move(value)); },
      AnyValue(std::forward<ArgumentTypes>(arguments))...);
  // 调用 content_ 的 forward 方法，并传入封装好的参数值，返回执行结果
  return content_->forward(std::move(values));
}

// 在 AnyModule 类中定义的模板成员函数，用于进行特定返回类型的 forward 操作
template <typename ReturnType, typename... ArgumentTypes>
ReturnType AnyModule::forward(ArgumentTypes&&... arguments) {
  // 调用 any_forward 方法执行前向传播，并将结果转换为指定的 ReturnType 返回
  return any_forward(std::forward<ArgumentTypes>(arguments)...)
      .template get<ReturnType>();
}

// 在 AnyModule 类中定义的模板成员函数，用于获取指定类型 T 的值
template <typename T, typename>
T& AnyModule::get() {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  // 调用 get_<T>() 方法获取指定类型 T 的值并返回
  return get_<T>();
}

// 在 AnyModule 类中定义的模板成员函数，用于获取指定类型 T 的常量值
template <typename T, typename>
const T& AnyModule::get() const {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  // 调用 get_<T>() 方法获取指定类型 T 的常量值并返回
  return get_<T>();
}

// 在 AnyModule 类中定义的模板成员函数，用于获取指定类型 T 的内容
template <typename T, typename ContainedType>
T AnyModule::get() const {
  // 调用 ptr<ContainedType>() 方法获取指定类型 ContainedType 的指针并返回
  return T(ptr<ContainedType>());
}

// 在 AnyModule 类中定义的成员函数，用于返回当前 content_ 指向的模块的 shared_ptr
inline std::shared_ptr<Module> AnyModule::ptr() const {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  // 返回当前 content_ 指向的模块的 shared_ptr
  return content_->ptr();
}

// 在 AnyModule 类中定义的模板成员函数，用于返回指定类型 T 的内容的 shared_ptr
template <typename T, typename>
std::shared_ptr<T> AnyModule::ptr() const {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  // 调用 get_<T>() 方法获取指定类型 T 的值，用于类型检查，然后返回转换后的 shared_ptr
  get_<T>();
  return std::dynamic_pointer_cast<T>(ptr());
}

// 在 AnyModule 类中定义的成员函数，用于返回当前 content_ 指向的模块的类型信息
inline const std::type_info& AnyModule::type_info() const {
  // 检查当前 AnyModule 对象是否为空，若为空则抛出异常
  TORCH_CHECK(!is_empty(), "Cannot call type_info() on an empty AnyModule");
  // 返回当前 content_ 指向的模块的类型信息
  return content_->type_info;
}

// 在 AnyModule 类中定义的成员函数，用于检查当前 AnyModule 对象是否为空
inline bool AnyModule::is_empty() const noexcept {
  // 返回当前 AnyModule 对象的 content_ 是否为空
  return content_ == nullptr;
}

// 在 AnyModule 类中定义的私有成员函数模板，用于创建特定模块类型的 AnyModulePlaceholder 对象
template <
    typename ModuleType,
    typename Class,
    typename ReturnType,
    typename... ArgumentTypes>
std::unique_ptr<AnyModulePlaceholder> AnyModule::make_holder(
    std::shared_ptr<ModuleType>&& module,
    ReturnType (Class::*)(ArgumentTypes...)) {
  // 静态断言，确保传入的 ArgumentTypes 不是左值引用
  static_assert(
      torch::detail::check_not_lvalue_references<ArgumentTypes...>(),
      "Modules stored inside AnyModule must not take references. "
      "Use pointers instead.");
  // 静态断言，确保 ReturnType 不是 void
  static_assert(
      !std::is_void<ReturnType>::value,
      "AnyModule cannot store modules that return void "
      "(you can return a dummy value).");
  // 创建并返回一个 AnyModuleHolder 对象，用于存储特定类型模块的信息
  return std::make_unique<
      AnyModuleHolder<std::decay_t<ModuleType>, ArgumentTypes...>>(
      std::move(module));
}
// 定义 AnyModule 类模板的成员函数 get_，返回 ModuleType 的引用
ModuleType& AnyModule::get_() const {
  // 使用类型别名 M，移除 ModuleType 的引用
  using M = typename std::remove_reference<ModuleType>::type;
  // 静态断言：检查类型 M 是否具有 forward 方法
  static_assert(
      torch::detail::has_forward<M>::value,
      "Can only call AnyModule::get<T> with a type T that has a forward method");
  // 调用具有特定参数类型签名的 get_ 函数重载
  return get_(&M::forward);
}

// AnyModule 类模板的成员函数模板 get_ 的特化版本，根据函数指针匹配类型 ModuleType
template <typename ModuleType, typename ReturnType, typename... ArgumentTypes>
ModuleType& AnyModule::get_(
    ReturnType (ModuleType::*)(ArgumentTypes...)) const {
  // 检查当前对象的类型信息与 ModuleType 是否匹配
  if (typeid(ModuleType).hash_code() == type_info().hash_code()) {
    // 将内容转换为特定类型的 AnyModuleHolder，并返回其 module 成员的引用
    return *static_cast<AnyModuleHolder<ModuleType, ArgumentTypes...>&>(
                *content_)
                .module;
  }
  // 抛出错误，尝试将类型为当前类型的模块转换为类型为 ModuleType 的模块
  AT_ERROR(
      "Attempted to cast module of type ",
      c10::demangle(type_info().name()),
      " to type ",
      c10::demangle(typeid(ModuleType).name()));
}

// 命名空间 nn 结束
} // namespace nn

// 命名空间 torch 结束
} // namespace torch
```