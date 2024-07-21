# `.\pytorch\c10\util\OptionalArrayRef.h`

```
// This file defines OptionalArrayRef<T>, a class that has almost the same
// exact functionality as std::optional<ArrayRef<T>>, except that its
// converting constructor fixes a dangling pointer issue.
//
// The implicit converting constructor of both std::optional<ArrayRef<T>> and
// std::optional<ArrayRef<T>> can cause the underlying ArrayRef<T> to store
// a dangling pointer. OptionalArrayRef<T> prevents this by wrapping
// a std::optional<ArrayRef<T>> and fixing the constructor implementation.
//
// See https://github.com/pytorch/pytorch/issues/63645 for more on this.

#pragma once

#include <c10/util/ArrayRef.h>      // Include the definition of ArrayRef from c10/util
#include <c10/util/Optional.h>      // Include the definition of Optional from c10/util
#include <cstdint>                  // Include standard integer types
#include <initializer_list>         // Include initializer_list for initialization
#include <type_traits>              // Include type_traits for type trait support
#include <utility>                  // Include utility for std::move and other utilities

namespace c10 {

template <typename T>               // Define a template for OptionalArrayRef with type T
class OptionalArrayRef final {
 public:
  // Constructors

  // 默认构造函数，构造一个空的 OptionalArrayRef 对象
  constexpr OptionalArrayRef() noexcept = default;

  // 构造函数，接受 nullopt_t 参数，用于构造空的 OptionalArrayRef 对象
  constexpr OptionalArrayRef(nullopt_t) noexcept {}

  // 拷贝构造函数，使用默认方式实现
  OptionalArrayRef(const OptionalArrayRef& other) = default;

  // 移动构造函数，使用默认方式实现
  OptionalArrayRef(OptionalArrayRef&& other) noexcept = default;

  // 构造函数，从 optional<ArrayRef<T>> 类型对象构造 OptionalArrayRef 对象
  constexpr OptionalArrayRef(const optional<ArrayRef<T>>& other) noexcept
      : wrapped_opt_array_ref(other) {}

  // 构造函数，从 optional<ArrayRef<T>> 类型对象构造 OptionalArrayRef 对象（移动语义）
  constexpr OptionalArrayRef(optional<ArrayRef<T>>&& other) noexcept
      : wrapped_opt_array_ref(std::move(other)) {}

  // 构造函数，从 T 类型对象构造 OptionalArrayRef 对象
  constexpr OptionalArrayRef(const T& value) noexcept
      : wrapped_opt_array_ref(value) {}

  // 模板构造函数，接受类型 U，并根据条件构造 OptionalArrayRef 对象
  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
              !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
              std::is_constructible_v<ArrayRef<T>, U&&> &&
              std::is_convertible_v<U&&, ArrayRef<T>> &&
              !std::is_convertible_v<U&&, T>,
          bool> = false>
  constexpr OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  // 显式模板构造函数，接受类型 U，并根据条件显式构造 OptionalArrayRef 对象
  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
              !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
              std::is_constructible_v<ArrayRef<T>, U&&> &&
              !std::is_convertible_v<U&&, ArrayRef<T>>,
          bool> = false>
  constexpr explicit OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  // 构造函数，使用 in_place_t 构造 OptionalArrayRef 对象，并传入参数 Args
  template <typename... Args>
  constexpr explicit OptionalArrayRef(
      std::in_place_t ip,
      Args&&... args) noexcept
      : wrapped_opt_array_ref(ip, std::forward<Args>(args)...) {}

  // 构造函数，使用 in_place_t 构造 OptionalArrayRef 对象，并传入初始化列表 il 和参数 Args
  template <typename U, typename... Args>
  constexpr explicit OptionalArrayRef(
      std::in_place_t ip,
      std::initializer_list<U> il,
      Args&&... args)
      : wrapped_opt_array_ref(ip, il, std::forward<Args>(args)...) {}

  // 构造函数，从 std::initializer_list<T> 构造 OptionalArrayRef 对象
  constexpr OptionalArrayRef(const std::initializer_list<T>& Vec)
      : wrapped_opt_array_ref(ArrayRef<T>(Vec)) {}

  // Destructor

  // 析构函数，使用默认方式实现
  ~OptionalArrayRef() = default;

  // Assignment

  // 赋值运算符重载，接受 nullopt_t 参数，将 wrapped_opt_array_ref 置为 nullptr
  constexpr OptionalArrayRef& operator=(nullopt_t) noexcept {
    wrapped_opt_array_ref = c10::nullopt;
    return *this;
  }

  // 拷贝赋值运算符重载，使用默认方式实现
  OptionalArrayRef& operator=(const OptionalArrayRef& other) = default;

  // 移动赋值运算符重载，使用默认方式实现
  OptionalArrayRef& operator=(OptionalArrayRef&& other) noexcept = default;

  // 赋值运算符重载，从 optional<ArrayRef<T>> 类型对象赋值给当前对象
  constexpr OptionalArrayRef& operator=(
      const optional<ArrayRef<T>>& other) noexcept {
    wrapped_opt_array_ref = other;
    return *this;
  }

  // 赋值运算符重载，从 optional<ArrayRef<T>> 类型对象（移动语义）赋值给当前对象
  constexpr OptionalArrayRef& operator=(
      optional<ArrayRef<T>>&& other) noexcept {
    wrapped_opt_array_ref = std::move(other);
    return *this;
  }
  // 返回对象本身的引用
  }

  template <
      typename U = ArrayRef<T>,
      typename = std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
          std::is_constructible_v<ArrayRef<T>, U&&> &&
          std::is_assignable_v<ArrayRef<T>&, U&&>>>
  // 赋值运算符重载，接受右值引用并进行条件检查，确保可以构造和赋值给ArrayRef<T>
  constexpr OptionalArrayRef& operator=(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&> &&
      std::is_nothrow_assignable_v<ArrayRef<T>&, U&&>) {
    wrapped_opt_array_ref = std::forward<U>(value);
    // 返回对象本身的引用
    return *this;
  }

  // Observers

  // 返回指向wrapped_opt_array_ref.value()的指针
  constexpr ArrayRef<T>* operator->() noexcept {
    return &wrapped_opt_array_ref.value();
  }

  // 返回指向wrapped_opt_array_ref.value()的常量指针
  constexpr const ArrayRef<T>* operator->() const noexcept {
    return &wrapped_opt_array_ref.value();
  }

  // 返回wrapped_opt_array_ref.value()的引用
  constexpr ArrayRef<T>& operator*() & noexcept {
    return wrapped_opt_array_ref.value();
  }

  // 返回wrapped_opt_array_ref.value()的常量引用
  constexpr const ArrayRef<T>& operator*() const& noexcept {
    return wrapped_opt_array_ref.value();
  }

  // 返回移动后wrapped_opt_array_ref.value()的引用
  constexpr ArrayRef<T>&& operator*() && noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  // 返回移动后wrapped_opt_array_ref.value()的常量引用
  constexpr const ArrayRef<T>&& operator*() const&& noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  // 显式转换运算符，检查wrapped_opt_array_ref是否有值
  constexpr explicit operator bool() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  // 检查wrapped_opt_array_ref是否有值
  constexpr bool has_value() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  // 返回wrapped_opt_array_ref.value()的引用
  constexpr ArrayRef<T>& value() & {
    return wrapped_opt_array_ref.value();
  }

  // 返回wrapped_opt_array_ref.value()的常量引用
  constexpr const ArrayRef<T>& value() const& {
    return wrapped_opt_array_ref.value();
  }

  // 返回移动后wrapped_opt_array_ref.value()的引用
  constexpr ArrayRef<T>&& value() && {
    return std::move(wrapped_opt_array_ref.value());
  }

  // 返回移动后wrapped_opt_array_ref.value()的常量引用
  constexpr const ArrayRef<T>&& value() const&& {
    return std::move(wrapped_opt_array_ref.value());
  }

  // 如果U可以转换为ArrayRef<T>，返回value_or(std::forward<U>(default_value))
  template <typename U>
  constexpr std::
      enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>, ArrayRef<T>>
      value_or(U&& default_value) const& {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  // 如果U可以转换为ArrayRef<T>，返回value_or(std::forward<U>(default_value))
  template <typename U>
  constexpr std::
      enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>, ArrayRef<T>>
      value_or(U&& default_value) && {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  // Modifiers

  // 交换两个OptionalArrayRef对象的wrapped_opt_array_ref成员
  constexpr void swap(OptionalArrayRef& other) noexcept {
    std::swap(wrapped_opt_array_ref, other.wrapped_opt_array_ref);
  }

  // 重置wrapped_opt_array_ref对象
  constexpr void reset() noexcept {
    wrapped_opt_array_ref.reset();
  }

  // 如果参数可以构造为ArrayRef<T>，使用参数重新构造wrapped_opt_array_ref对象
  template <typename... Args>
  constexpr std::
      enable_if_t<std::is_constructible_v<ArrayRef<T>, Args&&...>, ArrayRef<T>&>
      emplace(Args&&... args) noexcept(
          std::is_nothrow_constructible_v<ArrayRef<T>, Args&&...>) {
    return wrapped_opt_array_ref.emplace(std::forward<Args>(args)...);
  }

  // 接受initializer_list<U>和参数，使用它们来重新构造wrapped_opt_array_ref对象
  template <typename U, typename... Args>
  constexpr ArrayRef<T>& emplace(
      std::initializer_list<U> il,
      Args&&... args) noexcept {
    // 将参数 il 和 std::forward<Args>(args)... 包装后插入 wrapped_opt_array_ref 中，并返回插入后的结果
    return wrapped_opt_array_ref.emplace(il, std::forward<Args>(args)...);
  }

 private:
  // 包装后的可选数组引用对象
  optional<ArrayRef<T>> wrapped_opt_array_ref;
};

// 定义一个类型别名，OptionalIntArrayRef 表示一个可选的 int64_t 数组引用
using OptionalIntArrayRef = OptionalArrayRef<int64_t>;

// 定义一个重载的相等运算符，用于比较 OptionalIntArrayRef 和 IntArrayRef 对象是否相等
inline bool operator==(
    const OptionalIntArrayRef& a1,    // 第一个参数：OptionalIntArrayRef 类型的引用 a1
    const IntArrayRef& other) {       // 第二个参数：IntArrayRef 类型的引用 other
  if (!a1.has_value()) {              // 如果 a1 没有值（即空），返回 false
    return false;
  }
  return a1.value() == other;         // 否则比较 a1 的值与 other 是否相等，并返回比较结果
}

// 定义另一个重载的相等运算符，用于比较 c10::IntArrayRef 和 c10::OptionalIntArrayRef 对象是否相等
inline bool operator==(
    const c10::IntArrayRef& a1,       // 第一个参数：c10::IntArrayRef 类型的引用 a1
    const c10::OptionalIntArrayRef& a2) {  // 第二个参数：c10::OptionalIntArrayRef 类型的引用 a2
  return a2 == a1;                    // 调用上面已定义的相等运算符，比较 a2 和 a1 是否相等，并返回比较结果
}

} // namespace c10
```