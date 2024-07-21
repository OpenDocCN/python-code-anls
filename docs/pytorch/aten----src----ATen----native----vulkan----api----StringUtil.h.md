# `.\pytorch\aten\src\ATen\native\vulkan\api\StringUtil.h`

```py
#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#ifdef USE_VULKAN_API

#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace detail {

// 结构体，用于在编译时返回空字符串
struct CompileTimeEmptyString {
  // 转换运算符，返回静态空字符串
  operator const std::string&() const {
    static const std::string empty_string_literal;
    return empty_string_literal;
  }
  // 转换运算符，返回空字符指针
  operator const char*() const {
    return "";
  }
};

// 模板结构体，规范化字符串类型
template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;  // 返回常量引用
};

// 特化模板结构体，用于规范化字符数组类型
template <size_t N>
struct CanonicalizeStrTypes<char[N]> {
  using type = const char*;  // 返回字符指针
};

// 内联函数，将流返回
inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

// 模板函数，向流中插入参数并返回流
template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  ss << t;
  return ss;
}

// 特化模板函数，处理 CompileTimeEmptyString 类型的参数
template <>
inline std::ostream& _str<CompileTimeEmptyString>(
    std::ostream& ss,
    const CompileTimeEmptyString&) {
  return ss;  // 直接返回流
}

// 模板函数，递归处理多个参数插入到流中
template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

// 变长模板结构体，最终将多个参数连接成字符串返回
template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;  // 创建字符串流
    _str(ss, args...);  // 将参数插入流中
    return ss.str();  // 返回流的字符串表示
  }
};

// 特化模板结构体，处理无参数的情况
template <>
struct _str_wrapper<> final {
  static CompileTimeEmptyString call() {
    return CompileTimeEmptyString();  // 返回编译时空字符串
  }
};

} // namespace detail

// 函数模板，将多个参数连接成字符串返回
template <typename... Args>
inline std::string concat_str(const Args&... args) {
  // 调用内部的 _str_wrapper 结构体处理函数，返回连接后的字符串
  return detail::_str_wrapper<
      typename detail::CanonicalizeStrTypes<Args>::type...>::call(args...);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```