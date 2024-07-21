# `.\pytorch\aten\src\ATen\native\vulkan\api\Exception.cpp`

```
// 包含 Vulkan API 异常处理的头文件
#include <ATen/native/vulkan/api/Exception.h>

// 包含字符串流处理的头文件
#include <sstream>

// 定义 Vulkan 相关的命名空间
namespace at {
namespace native {
namespace vulkan {
namespace api {

// 定义宏 VK_RESULT_CASE，用于处理不同的 VkResult 枚举值
#define VK_RESULT_CASE(code) \
  case code:                 \
    out << #code;            \
    break;

// 重载 << 操作符，将 VkResult 转换为字符串输出到流 out
std::ostream& operator<<(std::ostream& out, const VkResult result) {
  switch (result) {
    VK_RESULT_CASE(VK_SUCCESS)                      // 处理 VK_SUCCESS
    VK_RESULT_CASE(VK_NOT_READY)                    // 处理 VK_NOT_READY
    VK_RESULT_CASE(VK_TIMEOUT)                      // 处理 VK_TIMEOUT
    VK_RESULT_CASE(VK_EVENT_SET)                    // 处理 VK_EVENT_SET
    VK_RESULT_CASE(VK_EVENT_RESET)                  // 处理 VK_EVENT_RESET
    VK_RESULT_CASE(VK_INCOMPLETE)                   // 处理 VK_INCOMPLETE
    VK_RESULT_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)     // 处理 VK_ERROR_OUT_OF_HOST_MEMORY
    VK_RESULT_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)   // 处理 VK_ERROR_OUT_OF_DEVICE_MEMORY
    VK_RESULT_CASE(VK_ERROR_INITIALIZATION_FAILED)  // 处理 VK_ERROR_INITIALIZATION_FAILED
    VK_RESULT_CASE(VK_ERROR_DEVICE_LOST)            // 处理 VK_ERROR_DEVICE_LOST
    VK_RESULT_CASE(VK_ERROR_MEMORY_MAP_FAILED)      // 处理 VK_ERROR_MEMORY_MAP_FAILED
    VK_RESULT_CASE(VK_ERROR_LAYER_NOT_PRESENT)      // 处理 VK_ERROR_LAYER_NOT_PRESENT
    VK_RESULT_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)  // 处理 VK_ERROR_EXTENSION_NOT_PRESENT
    VK_RESULT_CASE(VK_ERROR_FEATURE_NOT_PRESENT)    // 处理 VK_ERROR_FEATURE_NOT_PRESENT
    VK_RESULT_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)    // 处理 VK_ERROR_INCOMPATIBLE_DRIVER
    VK_RESULT_CASE(VK_ERROR_TOO_MANY_OBJECTS)       // 处理 VK_ERROR_TOO_MANY_OBJECTS
    VK_RESULT_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)   // 处理 VK_ERROR_FORMAT_NOT_SUPPORTED
    VK_RESULT_CASE(VK_ERROR_FRAGMENTED_POOL)        // 处理 VK_ERROR_FRAGMENTED_POOL
    default:
      out << "VK_ERROR_UNKNOWN (VkResult " << result << ")";  // 处理未知的 VkResult 值
      break;
  }
  return out;
}

// 取消定义 VK_RESULT_CASE 宏
#undef VK_RESULT_CASE

//
// SourceLocation
//

// 重载 << 操作符，将 SourceLocation 转换为字符串输出到流 out
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;  // 输出函数名、文件名和行号
  return out;
}

//
// Exception
//

// 定义 Error 类的构造函数，接受源位置和异常信息
Error::Error(SourceLocation source_location, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  // 使用字符串流创建异常信息
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";  // 输出异常来源位置
  oss << msg_;  // 输出异常信息
  what_ = oss.str();  // 获取完整异常信息字符串
}

// 定义 Error 类的构造函数，接受源位置、条件和异常信息
Error::Error(SourceLocation source_location, const char* cond, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  // 使用字符串流创建异常信息
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";  // 输出异常来源位置
  oss << "(" << cond << ") is false! ";  // 输出条件不满足的信息
  oss << msg_;  // 输出异常信息
  what_ = oss.str();  // 获取完整异常信息字符串
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```