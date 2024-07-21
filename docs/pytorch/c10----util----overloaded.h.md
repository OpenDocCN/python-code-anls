# `.\pytorch\c10\util\overloaded.h`

```py
#pragma once

// 预处理指令，确保头文件只被包含一次


#include <memory>

// 包含标准 C++ 库中的内存管理组件


namespace c10 {
namespace detail {

// 定义命名空间 c10 和其中的细节命名空间 detail


template <class... Ts>
struct overloaded_t {};

// 定义一个模板结构体 overloaded_t，用于存储多个可调用对象模板参数包


template <class T0>
struct overloaded_t<T0> : T0 {
  using T0::operator();
  overloaded_t(T0 t0) : T0(std::move(t0)) {}
};

// 当只有一个模板参数时的特化：继承自 T0，使其可调用；构造函数初始化基类 T0。


template <class T0, class... Ts>
struct overloaded_t<T0, Ts...> : T0, overloaded_t<Ts...> {
  using T0::operator();
  using overloaded_t<Ts...>::operator();
  overloaded_t(T0 t0, Ts... ts)
      : T0(std::move(t0)), overloaded_t<Ts...>(std::move(ts)...) {}
};

// 当有多个模板参数时的特化：继承自 T0 和后续参数包 Ts...；通过 using 声明继承的 operator()；构造函数初始化所有基类 T0 和 Ts...。


} // namespace detail

// 结束命名空间 detail


template <class... Ts>
detail::overloaded_t<Ts...> overloaded(Ts... ts) {
  return {std::move(ts)...};
}

// 定义一个 overloaded 函数模板，用于创建并返回一个包含多个可调用对象的 overloaded_t 实例。


} // namespace c10

// 结束命名空间 c10
```