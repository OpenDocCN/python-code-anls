# `.\pytorch\c10\util\Array.h`

```
#pragma once
// 使用 pragma once 指令，确保头文件只被编译一次

#include <array>
// 包含标准库头文件 <array>，用于定义 std::array 容器

#include <utility>
// 包含标准库头文件 <utility>，提供通用工具组件，如 std::forward

namespace c10 {
// 命名空间 c10 的开始

// This helper function creates a constexpr std::array
// From a compile time list of values, without requiring you to explicitly
// write out the length.
//
// See also https://stackoverflow.com/a/26351760/23845
// 这个辅助函数创建一个 constexpr std::array，
// 从编译时值列表中生成，无需显式指定长度。
//
// 参考链接 https://stackoverflow.com/a/26351760/23845
template <typename V, typename... T>
// 定义一个模板函数，返回值为 constexpr std::array<V, sizeof...(T)>
inline constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  // 返回一个 std::array<V, sizeof...(T)> 对象，初始化列表包含参数 t 的转发
  return {{std::forward<T>(t)...}};
}

} // namespace c10
// 命名空间 c10 的结束
```