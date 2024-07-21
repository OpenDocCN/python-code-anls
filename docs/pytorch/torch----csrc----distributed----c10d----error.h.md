# `.\pytorch\torch\csrc\distributed\c10d\error.h`

```
// 版权声明及许可信息

// 预处理指令，确保本头文件只被包含一次
#pragma once

// 包含标准 C 库中的字符串操作及系统错误处理的头文件
#include <cstring>
#include <system_error>

// 包含第三方库 fmt 的格式化输出头文件
#include <fmt/format.h>

// fmt 命名空间
namespace fmt {

// 格式化器特化，用于 std::error_category 类型
template <>
struct formatter<std::error_category> {
  // 解析函数，返回格式化解析上下文的起始迭代器
  constexpr decltype(auto) parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  // 格式化函数，根据错误分类对象 cat 输出格式化后的字符串到上下文 ctx
  template <typename FormatContext>
  decltype(auto) format(const std::error_category& cat, FormatContext& ctx) const {
    // 如果错误分类名称是 "generic"，则输出 "errno"
    if (std::strcmp(cat.name(), "generic") == 0) {
      return fmt::format_to(ctx.out(), "errno");
    } else {
      // 否则输出错误分类的名称后跟 "error"
      return fmt::format_to(ctx.out(), "{} error", cat.name());
    }
  }
};

// 格式化器特化，用于 std::error_code 类型
template <>
struct formatter<std::error_code> {
  // 解析函数，返回格式化解析上下文的起始迭代器
  constexpr decltype(auto) parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  // 格式化函数，根据错误码对象 err 输出格式化后的字符串到上下文 ctx
  template <typename FormatContext>
  decltype(auto) format(const std::error_code& err, FormatContext& ctx) const {
    // 输出错误码的类别、值和消息组成的格式化字符串
    return fmt::format_to(
        ctx.out(), "({}: {} - {})", err.category(), err.value(), err.message());
  }
};

} // namespace fmt

// c10d 命名空间
namespace c10d {
namespace detail {

// 返回最后一个错误码的函数，不抛出异常
inline std::error_code lastError() noexcept {
  // 使用 errno 和通用错误类别创建并返回一个 std::error_code 对象
  return std::error_code{errno, std::generic_category()};
}

} // namespace detail
} // namespace c10d
```