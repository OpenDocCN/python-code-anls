# `.\pytorch\torch\csrc\distributed\c10d\logging.h`

```py
// 定义命名空间 c10d 下的 detail 命名空间，用于存放内部实现细节
namespace c10d {
namespace detail {

// 定义日志级别枚举，包括 Trace、Debug、Info、Warning、Error
enum class LogLevel { Trace, Debug, Info, Warning, Error };

// 声明一个函数 isLogLevelEnabled，用于检查给定的日志级别是否启用
TORCH_API bool isLogLevelEnabled(LogLevel level) noexcept;

// 声明一个模板函数 formatLogMessage，用于格式化日志消息
template <typename... T>
std::string formatLogMessage(fmt::string_view fmt, T&&... args) {
  return fmt::vformat(fmt, fmt::make_format_args(args...));
}

} // namespace detail
} // namespace c10d

// 定义宏 C10D_ERROR，用于记录 Error 级别的日志
#define C10D_ERROR(...)                                                      \
  LOG_IF(                                                                    \
      ERROR, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Error)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// 定义宏 C10D_WARNING，用于记录 Warning 级别的日志
#define C10D_WARNING(...)                                               \
  LOG_IF(                                                               \
      WARNING,                                                          \
      c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// 定义宏 C10D_INFO，用于记录 Info 级别的日志
#define C10D_INFO(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// 定义宏 C10D_DEBUG，用于记录 Debug 级别的日志
#define C10D_DEBUG(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
      << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// 定义宏 C10D_TRACE，用于记录 Trace 级别的日志
#define C10D_TRACE(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
      << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)
```