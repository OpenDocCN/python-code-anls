# `.\pytorch\test\cpp\lite_interpreter_runtime\resources.h`

```
#pragma once

#include <string>

namespace torch {
namespace testing {

namespace detail {
// 前置声明 Path 类，定义在命名空间 detail 中
class Path;
}

/// Gets the path to the resource identified by name.
///
/// @param name identifies a resource, relative path starting from the
///             repo root
// 返回一个 detail::Path 对象，表示资源的路径
inline auto getResourcePath(std::string name) -> detail::Path;

// End interface: implementation details follow.

namespace detail {

// 定义 Path 类，用于表示资源路径
class Path {
 public:
  // 显式构造函数，接受一个字符串 rep，用于初始化路径
  explicit Path(std::string rep) : rep_(std::move(rep)) {}

  // 返回路径的字符串表示
  auto string() const -> std::string const& {
    return rep_;
  }

 private:
  std::string rep_;  // 存储路径的字符串
};

} // namespace detail

// 实现 getResourcePath 函数，返回一个 detail::Path 对象，用于表示资源的路径
inline auto getResourcePath(std::string name) -> detail::Path {
  return detail::Path(std::move(name));
}

} // namespace testing
} // namespace torch
```