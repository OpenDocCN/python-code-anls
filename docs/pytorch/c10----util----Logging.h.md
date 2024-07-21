# `.\pytorch\c10\util\Logging.h`

```py
#ifndef C10_UTIL_LOGGING_H_
#define C10_UTIL_LOGGING_H_

#include <climits>
#include <exception>
#include <functional>
#include <limits>
#include <sstream>

#include <c10/macros/Macros.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <c10/util/StringUtil.h>

// CAFFE2_LOG_THRESHOLD is a compile time flag that would allow us to turn off
// logging at compile time so no logging message below that level is produced
// at all. The value should be between INT_MIN and CAFFE_FATAL.
#ifndef CAFFE2_LOG_THRESHOLD
// If we have not defined the compile time log threshold, we keep all the
// log cases.
#define CAFFE2_LOG_THRESHOLD INT_MIN
#endif // CAFFE2_LOG_THRESHOLD

// Below are different implementations for glog and non-glog cases.
#ifdef C10_USE_GLOG
#include <c10/util/logging_is_google_glog.h>
#else // !C10_USE_GLOG
#include <c10/util/logging_is_not_google_glog.h>
#endif // C10_USE_GLOG

C10_DECLARE_int(caffe2_log_level);
C10_DECLARE_bool(caffe2_use_fatal_for_enforce);

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS. If it's
// not available - just short-circuit to the always working one one.
// We define the C10_ name to avoid confusing other files
#ifdef LOG_EVERY_MS
// Macro to log a message every ms milliseconds if supported, otherwise log at severity level
#define C10_LOG_EVERY_MS(severity, ms) LOG_EVERY_MS(severity, ms)
#else
// Fallback to regular LOG macro if LOG_EVERY_MS is not defined
#define C10_LOG_EVERY_MS(severity, ms) LOG(severity)
#endif

// Same for LOG_FIRST_N
#ifdef LOG_FIRST_N
// Macro to log the message for the first n times if supported, otherwise log at severity level
#define C10_LOG_FIRST_N(severity, n) LOG_FIRST_N(severity, n)
#else
// Fallback to regular LOG macro if LOG_FIRST_N is not defined
#define C10_LOG_FIRST_N(severity, n) LOG(severity)
#endif

// Same for LOG_EVERY_N
#ifdef LOG_EVERY_N
// Macro to log the message every n times if supported, otherwise log at severity level
#define C10_LOG_EVERY_N(severity, n) LOG_EVERY_N(severity, n)
#else
// Fallback to regular LOG macro if LOG_EVERY_N is not defined
#define C10_LOG_EVERY_N(severity, n) LOG(severity)
#endif

namespace c10 {

using std::string;

// Functions that we use for initialization.

// 初始化 Caffe 日志系统，返回是否成功
C10_API bool InitCaffeLogging(int* argc, char** argv);

// 从标志更新日志级别
C10_API void UpdateLoggingLevelsFromFlags();

// ThrowEnforceNotMet 函数的声明

// 抛出执行不满足的异常，包含文件、行数、条件、消息、调用者信息
[[noreturn]] C10_API void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller = nullptr);

// 抛出执行不满足的异常，包含文件、行数、条件、消息字符串、调用者信息
[[noreturn]] C10_API void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller = nullptr);

// 抛出执行不满足的异常，包含文件、行数、条件、空消息字符串、调用者信息
[[noreturn]] C10_API inline void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    detail::CompileTimeEmptyString /*msg*/,
    const void* caller = nullptr) {
  ThrowEnforceNotMet(file, line, condition, "", caller);
}

// 抛出执行不满足有限条件的异常，包含文件、行数、条件、消息、调用者信息
[[noreturn]] C10_API void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller = nullptr);

// 抛出执行不满足有限条件的异常，包含文件、行数、条件、消息字符串、调用者信息
[[noreturn]] C10_API void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller = nullptr);

// 抛出执行不满足有限条件的异常，包含文件、行数、条件、空消息字符串、调用者信息
[[noreturn]] C10_API inline void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    // 定义一个接收常量字符指针的参数 condition
    const char* condition,
    // 定义一个空的编译时字符串参数 msg，使用命名空间 detail
    detail::CompileTimeEmptyString /*msg*/,
    // 定义一个可选的空指针参数 caller，默认为 nullptr
    const void* caller = nullptr) {
  // 调用 ThrowEnforceFiniteNotMet 函数，抛出一个条件未满足的异常，传入文件名、行号、条件、空消息和可选的调用者指针
  ThrowEnforceFiniteNotMet(file, line, condition, "", caller);
/**
 * Check if Google Logging (GLOG) is enabled based on C10_USE_GLOG macro.
 * Returns true if enabled, false otherwise.
 */
constexpr bool IsUsingGoogleLogging() {
#ifdef C10_USE_GLOG
  return true;
#else
  return false;
#endif
}

/**
 * A utility function to enable logging to stderr after program starts.
 *
 * This function is similar to enabling GLOG's --logtostderr flag or setting
 * caffe2_log_level to a level below INFO. It's recommended to use this only
 * sparingly, such as for writing tutorials. Normally, use command-line flags
 * to set the log level.
 */
C10_API void ShowLogInfoToStderr();

/**
 * Set a custom function to fetch stack trace information.
 *
 * @param fetcher A function that returns a ::c10::Backtrace object.
 */
C10_API void SetStackTraceFetcher(std::function<::c10::Backtrace()> fetcher);

/**
 * Set a custom function to fetch stack trace information.
 * This overload is preferred when stringifying the backtrace is expensive.
 *
 * @param fetcher A function that returns a string representation of the backtrace.
 */
C10_API void SetStackTraceFetcher(std::function<std::string()> fetcher);

/**
 * Macro to enforce a condition. Throws ::c10::Error if the condition is false.
 *
 * @param condition The condition to check.
 * @param ... Optional message or arguments to include in the error message.
 */
#define CAFFE_ENFORCE(condition, ...)                               \
  do {                                                              \
    if (C10_UNLIKELY(!(condition))) {                               \
      ::c10::ThrowEnforceNotMet(                                    \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__)); \
    }                                                               \
  } while (false)

/**
 * Macro to enforce a condition that checks for finiteness.
 * Throws ::c10::Error if the condition is false.
 *
 * @param condition The condition to check for finiteness.
 * @param ... Optional message or arguments to include in the error message.
 */
#define CAFFE_ENFORCE_FINITE(condition, ...)                        \
  do {                                                              \
    if (C10_UNLIKELY(!(condition))) {                               \
      ::c10::ThrowEnforceFiniteNotMet(                              \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__)); \
    }                                                               \
  } while (false)

/**
 * Macro to enforce a condition with a caller context.
 * Throws ::c10::Error if the condition is false.
 *
 * @param condition The condition to check.
 * @param ... Optional message or arguments to include in the error message.
 */
#define CAFFE_ENFORCE_WITH_CALLER(condition, ...)                         \
  do {                                                                    \
    if (C10_UNLIKELY(!(condition))) {                                     \
      ::c10::ThrowEnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__), this); \
    }                                                                     \
  } while (false)

/**
 * Macro to throw an error message.
 *
 * @param ... Message or arguments to include in the error message.
 */
#define CAFFE_THROW(...) \
  ::c10::ThrowEnforceNotMet(__FILE__, __LINE__, "", ::c10::str(__VA_ARGS__))
/**
 * Rich logging messages
 *
 * CAFFE_ENFORCE_THAT can be used with one of the "checker functions" that
 * capture input argument values and add it to the exception message. E.g.
 * `CAFFE_ENFORCE_THAT(Equals(foo(x), bar(y)), "Optional additional message")`
 * would evaluate both foo and bar only once and if the results are not equal -
 * include them in the exception message.
 *
 * Some of the basic checker functions like Equals or Greater are already
 * defined below. Other header might define customized checkers by adding
 * functions to caffe2::enforce_detail namespace. For example:
 *
 *   namespace caffe2 { namespace enforce_detail {
 *   inline EnforceFailMessage IsVector(const vector<int64_t>& shape) {
 *     if (shape.size() == 1) { return EnforceOK(); }
 *     return c10::str("Shape ", shape, " is not a vector");
 *   }
 *   }}
 *
 * With further usages like `CAFFE_ENFORCE_THAT(IsVector(Input(0).dims()))`
 *
 * Convenient wrappers for binary operations like CAFFE_ENFORCE_EQ are provided
 * too. Please use them instead of TORCH_CHECK_EQ and friends for failures in
 * user-provided input.
 */

// 实现丰富的日志消息功能

namespace enforce_detail {

// 模板函数，用于生成失败消息字符串
template <typename T1, typename T2>
std::string enforceFailMsgImpl(const T1& x, const T2& y) {
  return c10::str(x, " vs ", y);
}

// 可变参数版本的模板函数，用于生成失败消息字符串
template <typename T1, typename T2, typename... Args>
std::string enforceFailMsgImpl(const T1& x, const T2& y, const Args&... args) {
  return c10::str(x, " vs ", y, ". ", args...);
}

// 实际的执行函数模板，用于执行断言检查
template <typename Pred, typename T1, typename T2, typename GetFailMsgFunc>
void enforceThatImpl(
    Pred p,
    const T1& lhs,
    const T2& rhs,
    const char* file,
    int line,
    const char* expr,
    const void* caller,
    GetFailMsgFunc getFailMsg) {
  if (C10_UNLIKELY(!(p(lhs, rhs)))) {
    // 如果断言失败，则抛出异常
    ::c10::ThrowEnforceNotMet(file, line, expr, getFailMsg(lhs, rhs), caller);
  }
}

// 宏定义，用于生成CAFFE_ENFORCE_THAT的具体实现
#define CAFFE_ENFORCE_THAT_IMPL(op, lhs, rhs, expr, ...)  \
  ::c10::enforce_detail::enforceThatImpl(                 \
      op,                                                 \
      (lhs),                                              \
      (rhs),                                              \
      __FILE__,                                           \
      __LINE__,                                           \
      expr,                                               \
      nullptr,                                            \
      [&](const auto& arg1, const auto& arg2) {           \
        return ::c10::enforce_detail::enforceFailMsgImpl( \
            arg1, arg2, ##__VA_ARGS__);                   \
      })
// 定义宏 CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER，调用 enforceThatImpl 函数来实现断言检查
#define CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(op, lhs, rhs, expr, ...) \
  ::c10::enforce_detail::enforceThatImpl(                            \
      op,                                                            \
      (lhs),                                                         \
      (rhs),                                                         \
      __FILE__,                                                      \
      __LINE__,                                                      \
      expr,                                                          \
      this,                                                          \
      [&](const auto& arg1, const auto& arg2) {                      \
        return ::c10::enforce_detail::enforceFailMsgImpl(            \
            arg1, arg2, ##__VA_ARGS__);                              \
      })

// 定义宏 CAFFE_ENFORCE_THAT，调用 CAFFE_ENFORCE_THAT_IMPL 宏来进行比较和断言检查
#define CAFFE_ENFORCE_THAT(cmp, op, lhs, rhs, ...) \
  CAFFE_ENFORCE_THAT_IMPL(cmp, lhs, rhs, #lhs " " #op " " #rhs, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_BINARY_OP，调用 CAFFE_ENFORCE_THAT_IMPL 宏来进行二元操作的比较和断言检查
#define CAFFE_ENFORCE_BINARY_OP(cmp, op, x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_EQ，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现相等（==）比较的断言检查
#define CAFFE_ENFORCE_EQ(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::equal_to<void>(), ==, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_NE，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现不相等（!=）比较的断言检查
#define CAFFE_ENFORCE_NE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_LE，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现小于等于（<=）比较的断言检查
#define CAFFE_ENFORCE_LE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::less_equal<void>(), <=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_LT，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现小于（<）比较的断言检查
#define CAFFE_ENFORCE_LT(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::less<void>(), <, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_GE，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现大于等于（>=）比较的断言检查
#define CAFFE_ENFORCE_GE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::greater_equal<void>(), >=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_GT，调用 CAFFE_ENFORCE_BINARY_OP 宏，实现大于（>）比较的断言检查
#define CAFFE_ENFORCE_GT(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::greater<void>(), >, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER，调用 CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER 宏，实现带有调用者信息的二元操作的比较和断言检查
#define CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(cmp, op, x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(                          \
      cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_EQ_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的相等（==）比较的断言检查
#define CAFFE_ENFORCE_EQ_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::equal_to<void>(), ==, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_NE_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的不相等（!=）比较的断言检查
#define CAFFE_ENFORCE_NE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_LE_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的小于等于（<=）比较的断言检查
#define CAFFE_ENFORCE_LE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::less_equal<void>(), <=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_LT_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的小于（<）比较的断言检查
#define CAFFE_ENFORCE_LT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(std::less<void>(), <, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_GE_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的大于等于（>=）比较的断言检查
#define CAFFE_ENFORCE_GE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::greater_equal<void>(), >=, x, y, ##__VA_ARGS__)

// 定义宏 CAFFE_ENFORCE_GT_WITH_CALLER，调用 CAFFE_ENFORCE_BINARY_OP_WITH_CALLER 宏，实现带有调用者信息的大于（>）比较的断言检查
#define CAFFE_ENFORCE_GT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::greater<void>(), >, x, y, ##__VA_ARGS__)
/**
 * Lightweight logging mechanism for tracking first-time API usage. This is useful 
 * in large applications to monitor individual function usage.
 *
 * To maintain lightweight logging, a static variable trick is used: LogAPIUsageOnce
 * will only trigger once, subsequent invocations will perform an atomic check.
 *
 * Example:
 *   // Logs caller information with an arbitrary text event, if API is used.
 *   C10_LOG_API_USAGE_ONCE("my_api");
 */
#define C10_LOG_API_USAGE_ONCE(...)                        \
  C10_UNUSED static bool C10_ANONYMOUS_VARIABLE(logFlag) = \
      ::c10::detail::LogAPIUsageFakeReturn(__VA_ARGS__);

// Declaration of API usage logging functions
C10_API void SetAPIUsageLogger(std::function<void(const std::string&)> logger);
C10_API void LogAPIUsage(const std::string& context);

C10_API void SetAPIUsageMetadataLogger(
    std::function<void(
        const std::string&,
        const std::map<std::string, std::string>& metadata_map)> logger);
C10_API void LogAPIUsageMetadata(
    const std::string& context,
    const std::map<std::string, std::string>& metadata_map);

// Declaration of PyTorch DDP (Distributed Data Parallel) usage logging functions
// DDPLoggingData structure for logging application data
struct DDPLoggingData {
  std::map<std::string, std::string> strs_map; // Map for string-type logging fields
  std::map<std::string, int64_t> ints_map;     // Map for int64_t-type logging fields
};

C10_API void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger);
C10_API void LogPyTorchDDPUsage(const DDPLoggingData& ddpData);

namespace detail {
// Function to return a value needed for static variable initialization trick
C10_API bool LogAPIUsageFakeReturn(const std::string& context);
} // namespace detail

// Initializes the c10 logging system
C10_API void initLogging();

// Sets the global rank to be included in log messages
C10_API void SetGlobalRank(int64_t rank);

} // namespace c10

#endif // C10_UTIL_LOGGING_H_
```