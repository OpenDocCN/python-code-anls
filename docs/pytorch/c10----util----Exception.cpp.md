# `.\pytorch\c10\util\Exception.cpp`

```
// 包含异常处理、日志和类型相关的头文件
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Type.h>

// 包含字符串流、字符串、实用工具相关的头文件
#include <sstream>
#include <string>
#include <utility>

// c10 命名空间
namespace c10 {

// 构造函数：使用给定的消息、回溯信息和调用者指针来初始化 Error 对象
Error::Error(std::string msg, Backtrace backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
  // 刷新 what_ 字符串
  refresh_what();
}

// 这个构造函数定义在 Logging.cpp 中，用于处理 PyTorch 风格的错误消息

// 另一种构造函数：使用文件名、行号、条件、消息、回溯信息和调用者指针来初始化 Error 对象
Error::Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    Backtrace backtrace,
    const void* caller)
    : Error(
          str("[enforce fail at ",
              detail::StripBasename(file),
              ":",
              line,
              "] ",
              condition,
              ". ",
              msg),
          std::move(backtrace),
          caller) {}

// 计算并返回包含上下文信息和回溯信息的错误消息字符串
std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;

  // 添加错误消息
  oss << msg_;

  // 如果只有一个上下文信息，将错误和上下文信息合并到一行
  if (context_.size() == 1) {
    oss << " (" << context_[0] << ")";
  } else {
    // 多个上下文信息，每个信息占据一行
    for (const auto& c : context_) {
      oss << "\n  " << c;
    }
  }

  // 如果需要包含回溯信息并且回溯信息存在，则添加回溯信息
  if (include_backtrace && backtrace_) {
    oss << "\n" << backtrace_->get();
  }

  // 返回构建的错误消息字符串
  return oss.str();
}

// 返回错误的回溯信息对象
const Backtrace& Error::backtrace() const {
  return backtrace_;
}

// 返回错误消息 C 风格的字符串表示
const char* Error::what() const noexcept {
  return what_
      .ensure([this] {
        try {
          // 返回计算的错误消息，包括回溯信息
          return compute_what(/*include_backtrace*/ true);
        } catch (...) {
          // 如果计算失败，则返回一个占位符字符串
          return std::string{"<Error computing Error::what()>"};
        }
      })
      .c_str();
}

// 刷新错误消息字符串
void Error::refresh_what() {
  // 重置 what_ 字符串，以便在首次访问时重新计算
  what_.reset();
  // 计算不包含回溯信息的错误消息字符串
  what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

// 添加上下文信息到错误对象中
void Error::add_context(std::string new_msg) {
  // 将新的上下文信息添加到 context_ 容器中
  context_.push_back(std::move(new_msg));
  // 刷新错误消息字符串，重新计算包括回溯信息的完整错误消息
  refresh_what();
}

// detail 命名空间

// 抛出一个使用函数名、文件名、行号和消息构建的 Error 对象
void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg) {
  throw ::c10::Error({func, file, line}, msg);
}

// 另一个 torchCheckFail 函数的重载版本，抛出一个 Error 对象
void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condition,
    const std::string& msg) {
  throw ::c10::Error(
      file, line, condition, msg, Backtrace::current(), static_cast<const void*>(func));
}

} // namespace detail

} // namespace c10
    const char* msg) {
  throw ::c10::Error({func, file, line}, msg);


// 抛出一个 C10 错误异常，包含函数名、文件名和行号信息，以及指定的错误消息
throw ::c10::Error({func, file, line}, msg);
}

// 实现 torchInternalAssertFail 函数，用于处理内部断言失败情况
void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const char* userMsg) {
  // 调用 torchCheckFail 函数处理断言失败的详细信息
  torchCheckFail(func, file, line, c10::str(condMsg, userMsg));
}

// 实现 torchInternalAssertFail 函数的重载版本，处理内部断言失败情况
// 这个版本接受 std::string 类型的用户消息
void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg) {
  // 调用 torchCheckFail 函数处理断言失败的详细信息
  torchCheckFail(func, file, line, c10::str(condMsg, userMsg));
}

} // namespace detail

namespace WarningUtils {

namespace {
// 获取基础警告处理器的静态实例
WarningHandler* getBaseHandler() {
  static WarningHandler base_warning_handler_ = WarningHandler();
  return &base_warning_handler_;
}

// 线程本地的警告处理器类
class ThreadWarningHandler {
 public:
  ThreadWarningHandler() = delete;

  // 获取线程本地的警告处理器
  static WarningHandler* get_handler() {
    if (!warning_handler_) {
      warning_handler_ = getBaseHandler();
    }
    return warning_handler_;
  }

  // 设置线程本地的警告处理器
  static void set_handler(WarningHandler* handler) {
    warning_handler_ = handler;
  }

 private:
  static thread_local WarningHandler* warning_handler_;
};

// 线程本地存储的警告处理器指针初始化为 nullptr
thread_local WarningHandler* ThreadWarningHandler::warning_handler_ = nullptr;

} // namespace

// 设置全局的警告处理器
void set_warning_handler(WarningHandler* handler) noexcept(true) {
  ThreadWarningHandler::set_handler(handler);
}

// 获取当前线程的警告处理器
WarningHandler* get_warning_handler() noexcept(true) {
  return ThreadWarningHandler::get_handler();
}

// 控制是否始终发出警告的全局变量，默认为 false
bool warn_always = false;

// 设置是否始终发出警告的全局函数
void set_warnAlways(bool setting) noexcept(true) {
  warn_always = setting;
}

// 获取是否始终发出警告的全局函数
bool get_warnAlways() noexcept(true) {
  return warn_always;
}

// 构造函数，用于设置是否始终发出警告
WarnAlways::WarnAlways(bool setting /*=true*/)
    : prev_setting(get_warnAlways()) {
  set_warnAlways(setting);
}

// 析构函数，用于恢复之前的警告设置
WarnAlways::~WarnAlways() {
  set_warnAlways(prev_setting);
}

} // namespace WarningUtils

// 发出警告的全局函数，委托给线程本地的警告处理器处理
void warn(const Warning& warning) {
  WarningUtils::ThreadWarningHandler::get_handler()->process(warning);
}

// 构造函数，用于创建警告对象
Warning::Warning(
    warning_variant_t type,
    const SourceLocation& source_location,
    std::string msg,
    const bool verbatim)
    : type_(type),
      source_location_(source_location),
      msg_(std::move(msg)),
      verbatim_(verbatim) {}

// 构造函数，用于创建警告对象，处理编译时空字符串的情况
Warning::Warning(
    warning_variant_t type,
    SourceLocation source_location,
    detail::CompileTimeEmptyString msg,
    const bool verbatim)
    : Warning(type, source_location, "", verbatim) {}

// 构造函数，用于创建警告对象，接受 C 字符串作为消息
Warning::Warning(
    warning_variant_t type,
    SourceLocation source_location,
    const char* msg,
    const bool verbatim)
    : type_(type),
      source_location_(source_location),
      msg_(std::string(msg)),
      verbatim_(verbatim) {}

// 获取警告类型的方法
Warning::warning_variant_t Warning::type() const {
  return type_;
}

// 获取警告来源位置的方法
const SourceLocation& Warning::source_location() const {
  return source_location_;
}

// 获取警告消息的方法
const std::string& Warning::msg() const {
  return msg_;
}

// 获取警告是否直接使用的方法
bool Warning::verbatim() const {
  return verbatim_;
}
// 处理警告的函数，将警告信息记录到日志中
void WarningHandler::process(const Warning& warning) {
  // 调用 LOG_AT_FILE_LINE 宏记录警告信息的文件位置和行号
  LOG_AT_FILE_LINE(
      WARNING, warning.source_location().file, warning.source_location().line)
      // 输出警告信息，包括警告消息和发生警告的函数名
      << "Warning: " << warning.msg() << " (function "
      << warning.source_location().function << ")";
}

// 获取异常信息的字符串表示
std::string GetExceptionString(const std::exception& e) {
  // 如果支持 RTTI (__GXX_RTTI 宏定义存在)，使用 demangle 函数解析异常类型名，并附加异常的 what() 方法返回的消息
#ifdef __GXX_RTTI
  return demangle(typeid(e).name()) + ": " + e.what();
// 如果不支持 RTTI，则简单返回异常类型为未知，并附加异常的 what() 方法返回的消息
#else
  return std::string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

// 结束 c10 命名空间的定义
} // namespace c10
```