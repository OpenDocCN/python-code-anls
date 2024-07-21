# `.\pytorch\c10\util\C++17.h`

```py
#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <c10/macros/Macros.h>  // 引入 C10 库中的宏定义
#include <functional>           // 引入功能函数对象
#include <memory>               // 引入智能指针和动态内存管理相关的头文件
#include <type_traits>          // 引入类型特性相关的头文件
#include <utility>              // 引入实用工具函数

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 9
#error \
    "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
#endif

#if defined(__clang__) && __clang_major__ < 9
#error \
    "You're trying to build PyTorch with a too old version of Clang. We need Clang 9 or later."
#endif

#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)
#error You need C++17 to compile PyTorch
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

/*
 * This header adds some polyfills with C++17 functionality
 */

namespace c10 {

// std::is_pod is deprecated in C++20, std::is_standard_layout and
// std::is_trivial are introduced in C++11, std::conjunction has been introduced
// in C++17.
template <typename T>
using is_pod = std::conjunction<std::is_standard_layout<T>, std::is_trivial<T>>;

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

namespace guts {

template <typename Base, typename Child, typename... Args>
std::enable_if_t<
    !std::is_array_v<Base> && !std::is_array_v<Child> &&
        std::is_base_of_v<Base, Child>,
    std::unique_ptr<Base>>
make_unique_base(Args&&... args) {
  return std::unique_ptr<Base>(new Child(std::forward<Args>(args)...));
}

#if defined(__cpp_lib_apply) && !defined(__CUDA_ARCH__) && !defined(__HIP__)

template <class F, class Tuple>
C10_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#else

// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but
// modified)
// TODO This is an incomplete implementation of std::apply, not working for
// member functions.
namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
#if defined(_MSC_VER)
// MSVC has a problem with the decltype() return type, but it also doesn't need
// it
C10_HOST_DEVICE constexpr auto apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
#else
// GCC/Clang need the decltype() return type
C10_HOST_DEVICE constexpr decltype(auto) apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
#endif
{
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
} // namespace detail

template <class F, class Tuple>
C10_HOST_DEVICE constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return detail::apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif

template <typename Functor, typename... Args>
std::enable_if_t<
    // 使用 std::decay_t 去除 Functor 的引用修饰符，得到其基础类型
    std::is_member_pointer_v<std::decay_t<Functor>>,
    // 使用 std::invoke_result_t 获取以 Args... 为参数类型调用 Functor 后的返回类型
    typename std::invoke_result_t<Functor, Args...>>
// 调用函数对象或函数指针，使用完美转发传递参数，并返回结果
invoke(Functor&& f, Args&&... args) {
  // 如果传入的函数对象是成员函数指针，则使用 std::mem_fn 封装后调用
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

// 如果传入的函数对象不是成员函数指针，则直接调用并返回结果
template <typename Functor, typename... Args>
std::enable_if_t<
    !std::is_member_pointer_v<std::decay_t<Functor>>,
    typename std::invoke_result_t<Functor, Args...>>
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

namespace detail {
// 定义 _identity 结构体，提供一个 type_identity 成员模板和一个 operator() 方法
struct _identity final {
  // type_identity 成员模板，返回传入参数的类型
  template <class T>
  using type_identity = T;

  // operator() 方法，完美转发传入的参数，并返回其本身
  template <class T>
  decltype(auto) operator()(T&& arg) {
    return std::forward<T>(arg);
  }
};

// 检查函数对象是否接受 _identity 类型的参数，继承自 std::true_type 或 std::false_type
template <class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};

// 部分特化，当表达式 decltype(std::declval<Func>()(_identity())) 合法时，继承自 std::true_type
template <class Func>
struct function_takes_identity_argument<
    Func,
    std::void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {
};
} // namespace detail

} // namespace guts
} // namespace c10

#endif // C10_UTIL_CPP17_H_
```