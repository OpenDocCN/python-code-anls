# `.\pytorch\aten\src\ATen\native\vulkan\api\Exception.h`

```py
#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
#ifdef USE_VULKAN_API

#include <exception>
#include <ostream>
#include <string>
#include <vector>

#include <ATen/native/vulkan/api/StringUtil.h>
#include <ATen/native/vulkan/api/vk_api.h>

// 定义一个宏 VK_CHECK，用于检查 Vulkan API 调用的返回结果
#define VK_CHECK(function)                                       \
  do {                                                           \
    const VkResult result = (function);                          \
    if (VK_SUCCESS != result) {                                  \
      // 如果调用结果不是 VK_SUCCESS，抛出异常 Error
      throw ::at::native::vulkan::api::Error(                    \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          ::at::native::vulkan::api::concat_str(                 \
              #function, " returned ", result));                 \
    }                                                            \
  } while (false)

// 定义一个宏 VK_CHECK_COND，用于检查给定的条件，如果条件不满足则抛出异常 Error
#define VK_CHECK_COND(cond, ...)                                 \
  do {                                                           \
    if (!(cond)) {                                               \
      // 如果条件不满足，抛出异常 Error，记录条件和附加信息
      throw ::at::native::vulkan::api::Error(                    \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          #cond,                                                 \
          ::at::native::vulkan::api::concat_str(__VA_ARGS__));   \
    }                                                            \
  } while (false)

// 定义一个宏 VK_THROW，直接抛出异常 Error，用于一般性的错误情况
#define VK_THROW(...)                                          \
  do {                                                         \
    // 直接抛出异常 Error，记录位置和附加信息
    throw ::at::native::vulkan::api::Error(                    \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        ::at::native::vulkan::api::concat_str(__VA_ARGS__));   \
  } while (false)

namespace at {
namespace native {
namespace vulkan {
namespace api {

// 输出 VkResult 类型到输出流
std::ostream& operator<<(std::ostream& out, const VkResult loc);

// 表示代码位置的结构体，包含函数名、文件名和行号
struct SourceLocation {
  const char* function; // 函数名
  const char* file;     // 文件名
  uint32_t line;        // 行号
};

// 输出 SourceLocation 结构体到输出流
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

// Vulkan API 错误类，继承自 std::exception
class Error : public std::exception {
 public:
  Error(SourceLocation source_location, std::string msg); // 构造函数，接受位置和错误信息
  Error(SourceLocation source_location, const char* cond, std::string msg); // 构造函数，接受条件、位置和错误信息

 private:
  std::string msg_;            // 错误信息
  SourceLocation source_location_; // 错误发生位置
  std::string what_;           // 异常信息字符串

 public:
  const std::string& msg() const { // 获取错误信息的方法
    return msg_;
  }

  const char* what() const noexcept override { // 获取异常信息字符串的方法
    return what_.c_str();
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```