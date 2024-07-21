# `.\pytorch\c10\util\Unroll.h`

```
#pragma once
// 包含头文件<c10/macros/Macros.h>，提供宏定义和其他实用工具
#include <c10/macros/Macros.h>
// 包含头文件<type_traits>，提供类型特性支持
#include <type_traits>

// Utility to guarantee complete unrolling of a loop where the bounds are known
// at compile time. Various pragmas achieve similar effects, but are not as
// portable across compilers.

// Example: c10::ForcedUnroll<4>{}(f); is equivalent to f(0); f(1); f(2); f(3);

// c10 命名空间，用于封装库中的工具和功能
namespace c10 {

// 模板类 ForcedUnroll，用于编译时已知循环边界的完全展开
template <int n>
struct ForcedUnroll {
  // 模板成员函数 operator()，接受一个函数对象和其他参数，并强制展开 n 次循环
  template <typename Func, typename... Args>
  // 使用 C10_ALWAYS_INLINE 宏，指示编译器尽可能地内联该函数
  C10_ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    // 递归调用 ForcedUnroll<n-1>{}(f, args...)，实现循环展开
    ForcedUnroll<n - 1>{}(f, args...);
    // 调用函数对象 f，并传递当前循环计数作为参数
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

// 特化模板类 ForcedUnroll，当 n 等于 1 时的情况
template <>
struct ForcedUnroll<1> {
  // 模板成员函数 operator()，接受一个函数对象和其他参数，展开单次循环
  template <typename Func, typename... Args>
  // 使用 C10_ALWAYS_INLINE 宏，指示编译器尽可能地内联该函数
  C10_ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    // 调用函数对象 f，并传递当前循环计数作为参数
    f(std::integral_constant<int, 0>{}, args...);
  }
};

} // namespace c10
```