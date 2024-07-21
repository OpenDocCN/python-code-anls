# `.\pytorch\torch\csrc\api\include\torch\nn\pimpl.h`

```py
#pragma once

#include <torch/arg.h>  // 引入 Torch 库中的参数处理头文件
#include <torch/detail/static.h>  // 引入 Torch 库中的静态工具头文件
#include <torch/serialize/archive.h>  // 引入 Torch 库中的序列化存档头文件
#include <torch/types.h>  // 引入 Torch 库中的类型定义头文件

#include <torch/csrc/utils/variadic.h>  // 引入 Torch 库中的可变参数工具头文件

#include <memory>  // 引入 C++ 标准库中的内存管理头文件
#include <type_traits>  // 引入 C++ 标准库中的类型特性头文件
#include <utility>  // 引入 C++ 标准库中的实用工具头文件

namespace torch {
namespace detail {
// 引入模板元编程相关内容
#include <torch/csrc/api/include/torch/nn/pimpl-inl.h>
} // namespace detail

namespace nn {

/// `ModuleHolder` 是对 `std::shared_ptr<M>` 的包装，其中 `M` 是 `nn::Module`
/// 的子类，提供了方便的构造函数，用于允许我们对模块进行各种构造方式的封装。
template <typename Contained>
class ModuleHolder : torch::detail::ModuleHolderIndicator {
 protected:
  /// 该类包装的模块指针。
  /// 注意：必须放置在类顶部，以便我们可以在后面使用它作为尾返回类型。
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Contained> impl_;

 public:
  using ContainedType = Contained;

  /// 默认构造函数，如果 `Contained` 有默认构造函数，则默认构造模块，
  /// 否则产生静态错误。
  ///
  /// 注意：这利用了 C++ 模板类的行为，即只有在实际使用时才会编译构造函数（或任何方法）。
  ModuleHolder() : impl_(default_construct()) {
    static_assert(
        std::is_default_constructible<Contained>::value,
        "You are trying to default construct a module which has "
        "no default constructor. Use = nullptr to give it the empty state "
        "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
  }

  /// 使用空的 `Contained` 值构造 `ModuleHolder`。
  /// 不允许访问底层模块，并在分配值之前将抛出异常。
  /* implicit */ ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

  /// 使用包含的模块构造 `ModuleHolder`，将所有参数转发给其构造函数。
  template <
      typename Head,
      typename... Tail,
      typename = typename std::enable_if<
          !(torch::detail::is_module_holder_of<Head, ContainedType>::value &&
            (sizeof...(Tail) == 0))>::type>
  explicit ModuleHolder(Head&& head, Tail&&... tail)
      : impl_(new Contained(
            std::forward<Head>(head),
            std::forward<Tail>(tail)...)) {}

  /// 使用指向 `Contained` 类型的指针构造 `ModuleHolder`。
  /// 示例：`Linear(std::make_shared<LinearImpl>(...))`。
  /* implicit */ ModuleHolder(std::shared_ptr<Contained> module)
      : impl_(std::move(module)) {}

  /// 如果 `ModuleHolder` 包含模块则返回 true，否则返回 false（即 `nullptr`）。
  explicit operator bool() const noexcept {
    return !is_empty();
  }

  /// 转发到包含的模块。
  Contained* operator->() {
  /// Calls the `get()` method to return the contained module.
  return get();
}

/// Forwards to the contained module.
const Contained* operator->() const {
  return get();
}

/// Returns a reference to the contained module.
Contained& operator*() {
  return *get();
}

/// Returns a const reference to the contained module.
const Contained& operator*() const {
  return *get();
}

/// Returns a shared pointer to the underlying module.
const std::shared_ptr<Contained>& ptr() const {
  TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
  return impl_;
}

/// Returns a pointer to the underlying module.
Contained* get() {
  TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
  return impl_.get();
}

/// Returns a const pointer to the underlying module.
const Contained* get() const {
  TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
  return impl_.get();
}

/// Calls the `forward()` method of the contained module.
template <typename... Args>
auto operator()(Args&&... args)
    -> torch::detail::return_type_of_forward_t<Contained, Args...> {
  // This will not compile if the module does not have a `forward()` method
  // (as expected).
  // NOTE: `std::forward` is qualified to prevent VS2017 emitting
  // error C2872: 'std': ambiguous symbol
  return impl_->forward(::std::forward<Args>(args)...);
}

/// Forwards to the subscript operator of the contained module.
/// NOTE: std::forward is qualified to prevent VS2017 emitting
///       error C2872: 'std': ambiguous symbol
template <typename Arg>
decltype(auto) operator[](Arg&& arg) {
  return (*impl_)[::std::forward<Arg>(arg)];
}

/// Returns true if the `ModuleHolder` does not contain a module.
bool is_empty() const noexcept {
  return impl_ == nullptr;
}

private:
template <typename T = Contained>
std::shared_ptr<Contained> default_construct() {
  if constexpr (std::is_default_constructible_v<T>) {
    return std::make_shared<Contained>();
  } else {
    return nullptr;
  }
}
};

/// 定义一个操作符重载函数，用于将给定的 `Module` 对象打印到输出流 `ostream` 中。
template <typename ModuleType>
std::ostream& operator<<(
    std::ostream& stream,
    const nn::ModuleHolder<ModuleType>& module) {
  return stream << *module;  // 调用 Module 类的输出流操作符重载
}

/// 将 `ModuleHolder` 序列化到 `OutputArchive` 中。
template <typename ModuleType>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const nn::ModuleHolder<ModuleType>& module) {
  return archive << module.ptr();  // 序列化 ModuleHolder 持有的指针到输出存档中
}

/// 从 `InputArchive` 中反序列化 `ModuleHolder`。
template <typename ModuleType>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    nn::ModuleHolder<ModuleType>& module) {
  return archive >> module.ptr();  // 从输入存档中反序列化到 ModuleHolder 持有的指针
}

} // namespace nn
} // namespace torch

// 为了解决 CUDA 10.2 及以下版本不允许在 using 声明中使用 unused 属性的问题。
#ifdef __CUDACC__
#define TORCH_UNUSED_EXCEPT_CUDA
#else
#define TORCH_UNUSED_EXCEPT_CUDA C10_UNUSED
#endif

/// 定义一个类 `Name`，它继承自 `nn::ModuleHolder`，提供了对 `std::shared_ptr<ImplType>` 的包装。
/// `Impl` 是对 `ImplType` 的类型别名，提供一种调用 `ImplType` 的静态方法的方式。
#define TORCH_MODULE_IMPL(Name, ImplType)                              \
  class Name : public torch::nn::ModuleHolder<ImplType> { /* NOLINT */ \
   public:                                                             \
    using torch::nn::ModuleHolder<ImplType>::ModuleHolder;             \
    using Impl TORCH_UNUSED_EXCEPT_CUDA = ImplType;                    \
  }

/// 类似于 `TORCH_MODULE_IMPL`，但默认将 `ImplType` 的名称设为 `<Name>Impl`。
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)
```