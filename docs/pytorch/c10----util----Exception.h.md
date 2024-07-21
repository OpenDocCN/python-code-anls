# `.\pytorch\c10\util\Exception.h`

```
#ifndef C10_UTIL_EXCEPTION_H_  // 如果未定义 C10_UTIL_EXCEPTION_H_ 宏，则开始头文件保护
#define C10_UTIL_EXCEPTION_H_

#include <c10/macros/Export.h>  // 引入 Export.h 文件，用于导出符号
#include <c10/macros/Macros.h>  // 引入 Macros.h 文件，包含一些宏定义
#include <c10/util/Backtrace.h>  // 引入 Backtrace.h 文件，用于获取堆栈信息
#include <c10/util/Lazy.h>  // 引入 Lazy.h 文件，提供延迟初始化的功能
#include <c10/util/StringUtil.h>  // 引入 StringUtil.h 文件，包含字符串处理工具

#include <cstdint>  // 引入标准整数类型
#include <exception>  // 引入异常处理相关的标准库头文件
#include <memory>  // 引入智能指针和动态内存管理的标准库头文件
#include <string>  // 引入字符串处理相关的标准库头文件
#include <variant>  // 引入 variant，C++17 中引入的用于保存任意类型值的容器
#include <vector>  // 引入向量容器的标准库头文件

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__  // 如果是早期版本的 MSVC，使用 __FUNCTION__ 宏定义
#endif

namespace c10 {

/// 主要的 ATen 错误类。
/// 通过 `what()` 方法提供带有源代码位置信息的完整错误消息，
/// 通过 `what_without_backtrace()` 方法提供更简洁的错误消息。
/// 不要直接抛出此类异常，应使用 TORCH_CHECK/TORCH_INTERNAL_ASSERT 宏来处理。
///
/// 注意：c10::Error 在默认的 torch 中有特殊处理，用于抑制堆栈跟踪信息，详见 torch/csrc/Exceptions.h
// C10_API Error 类的定义，继承自 std::exception
class C10_API Error : public std::exception {
 private:
  // 实际的错误消息字符串
  std::string msg_;

  // 消息的上下文（按特定顺序排列）。上下文会自动适当地格式化，因此不需要在此向量中的字符串添加额外的前导/尾随换行符
  std::vector<std::string> context_;

  // 异常抛出时的 C++ 回溯信息。如果没有有效的回溯信息，则可能为空。（我们这里不使用 optional 来减少该文件的依赖性。）
  Backtrace backtrace_;

  // 这两个字段是从 msg_stack_ 和 backtrace_ 派生出来的字段，但我们需要字符串字段以便能够返回 const char*（符合 std::exception 的签名要求）。
  // 目前的不变条件是，这些字段始终与 msg_stack_ 和 backtrace_ 一致地填充。
  mutable OptimisticLazy<std::string> what_;
  std::string what_without_backtrace_;

  // 这是一个小的调试技巧：您可以在 caller 中存储一个相关的指针，然后在捕获异常时，可以与手头上的指针进行比较，以获取有关异常来源的更多信息。
  // 在 Caffe2 中，这用于确定哪个运算符引发了异常。
  const void* caller_;

 public:
  // PyTorch 风格的 Error 构造函数。注意：此函数的实现实际上在 Logging.cpp 中。
  Error(SourceLocation source_location, std::string msg);

  // Caffe2 风格的错误消息构造函数
  Error(
      const char* file,
      const uint32_t line,
      const char* condition,
      const std::string& msg,
      Backtrace backtrace,
      const void* caller = nullptr);

  // 基础构造函数
  Error(
      std::string msg,
      Backtrace backtrace = nullptr,
      const void* caller = nullptr);

  // 向消息栈添加新的上下文。在打印时，最后添加的上下文将被格式化在上下文列表的末尾。
  // 注意：此方法的时间复杂度为 O(n)，其中 n 是栈的大小，因此不要在错误消息中添加过多的上下文。
  void add_context(std::string msg);

  // 返回错误消息字符串的引用
  const std::string& msg() const {
    return msg_;
  }

  // 返回上下文字符串向量的引用
  const std::vector<std::string>& context() const {
    return context_;
  }

  // 返回异常回溯信息的引用
  const Backtrace& backtrace() const;

  /// 返回完整的错误消息，包括源位置。
  /// 如果在此对象上调用 add_context()，则返回的指针将失效。
  const char* what() const noexcept override;

  // 返回仅包含错误消息字符串的指针，不包括回溯信息。
  // 如果在此对象上调用 add_context()，则返回的指针将失效。
  virtual const char* what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }



    # 返回成员变量 what_without_backtrace_ 的 C 风格字符串指针
    return what_without_backtrace_.c_str();
  }



 private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;



    # 声明私有成员函数 refresh_what()，用于刷新异常消息内容
    void refresh_what();
    # 声明私有成员函数 compute_what()，返回一个 std::string 类型的异常消息，
    # 可选择是否包含回溯信息（include_backtrace 参数）
    std::string compute_what(bool include_backtrace) const;
};

// 警告类，用于表示不同类型的警告
class C10_API Warning {
 public:
  // 用户警告类
  class C10_API UserWarning {};
  // 弃用警告类
  class C10_API DeprecationWarning {};

  // 警告类型的变体
  using warning_variant_t = std::variant<UserWarning, DeprecationWarning>;

  // 构造函数，根据警告类型、源位置、消息和是否直接显示警告初始化对象
  Warning(
      warning_variant_t type,
      const SourceLocation& source_location,
      std::string msg,
      bool verbatim);

  // 构造函数，根据警告类型、源位置、消息 C 字符串和是否直接显示警告初始化对象
  Warning(
      warning_variant_t type,
      SourceLocation source_location,
      const char* msg,
      bool verbatim);

  // 构造函数，根据警告类型、源位置、空消息字符串和是否直接显示警告初始化对象
  Warning(
      warning_variant_t type,
      SourceLocation source_location,
      ::c10::detail::CompileTimeEmptyString msg,
      bool verbatim);

  // 获取成员函数的访问器
  warning_variant_t type() const; // 获取警告类型
  const SourceLocation& source_location() const; // 获取警告发生位置
  const std::string& msg() const; // 获取警告消息
  bool verbatim() const; // 是否直接显示警告

 private:
  // 警告的类型
  warning_variant_t type_;
  // 警告发生的位置
  SourceLocation source_location_;
  // 实际的警告消息
  std::string msg_;
  // 参见注释: [Verbatim Warnings]
  bool verbatim_;
};

// 简化警告类型的别名
using UserWarning = Warning::UserWarning;
using DeprecationWarning = Warning::DeprecationWarning;

// 发出警告，输出给定的警告消息，由当前警告处理程序处理
void C10_API warn(const Warning& warning);

// 警告处理程序基类，处理警告信息的抽象接口
class C10_API WarningHandler {
 public:
  virtual ~WarningHandler() = default;
  // 默认警告处理函数，将警告消息输出到 stderr
  virtual void process(const Warning& warning);
};

// 警告工具命名空间
namespace WarningUtils {

// 注释: [Verbatim Warnings]
// 来自 C++ 代码的警告可能对 Python 用户而言位置不符合预期：
// 用户在 Python 中运行一行代码，但警告可能引用 C++ 中的一行代码。
// PyTorch 的某些部分（如 JIT）意识到这种不匹配并将警告映射回用户的程序，
// 但大多数情况下 PyTorch 只会抛出一个无上下文的警告。为了允许警告处理程序在适当时添加上下文，
// warn 函数接受 "verbatim" 标志。当这个标志为 false 时，警告处理程序可能会将 C++ 警告附加到
// Python 警告消息中，以将警告与用户的程序相关联。已经在警告消息中考虑了上下文的调用者应将 verbatim 设置为 true，
// 以便他们的警告消息不被修改地显示出来。

// 设置全局警告处理程序，这不是线程安全的，因此通常在初始化期间或者在使用 python 的程序中持有 GIL 时调用一次。
// 用户负责保持 WarningHandler 在其不再需要时保持存活状态。
C10_API void set_warning_handler(WarningHandler* handler) noexcept(true);
// 获取全局警告处理程序
C10_API WarningHandler* get_warning_handler() noexcept(true);

// 警告处理程序的守卫类，用于管理警告处理程序的生命周期
class C10_API WarningHandlerGuard {
  WarningHandler* prev_handler_;

 public:
  // 构造函数，将当前警告处理程序设为新的警告处理程序，并保存之前的处理程序
  WarningHandlerGuard(WarningHandler* new_handler)
      : prev_handler_(c10::WarningUtils::get_warning_handler()) {
    c10::WarningUtils::set_warning_handler(new_handler);
  }
  // 析构函数，恢复之前保存的警告处理程序
  ~WarningHandlerGuard() {
    // 使用 WarningUtils 类中的静态方法 set_warning_handler() 设置警告处理程序为 prev_handler_
    c10::WarningUtils::set_warning_handler(prev_handler_);
  }
};

/// The TORCH_WARN_ONCE macro is difficult to test for. Use
/// setWarnAlways(true) to turn it into TORCH_WARN, which can be
/// tested for more easily.
C10_API void set_warnAlways(bool) noexcept(true);
C10_API bool get_warnAlways() noexcept(true);

// A RAII guard that sets warn_always (not thread-local) on
// construction, and sets it back to the original value upon destruction.
struct C10_API WarnAlways {
 public:
  // 构造函数，设置 warn_always 为 true 或者指定的 setting 值
  explicit WarnAlways(bool setting = true);
  // 析构函数，恢复 warn_always 的前一个设置
  ~WarnAlways();

 private:
  // 保存前一个 warn_always 设置的值
  bool prev_setting;
};

} // namespace WarningUtils

// Like Error, but we always report the C++ backtrace, instead of only
// reporting when TORCH_SHOW_CPP_STACKTRACES
class C10_API ErrorAlwaysShowCppStacktrace : public Error {
  using Error::Error;
  // 返回没有包含回溯信息的 what() 函数结果
  const char* what_without_backtrace() const noexcept override {
    return what();
  }
};

// Used in ATen for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).  These turn into
// IndexError when they cross to Python.
class C10_API IndexError : public Error {
  using Error::Error;
};

// Used in ATen for invalid values.  These turn into
// ValueError when they cross to Python.
class C10_API ValueError : public Error {
  using Error::Error;
};

// Used in ATen for invalid types.  These turn into
// TypeError when they cross to Python.
class C10_API TypeError : public Error {
  using Error::Error;
};

// Used in ATen for functionality that is not implemented.  These turn into
// NotImplementedError when they cross to Python.
class C10_API NotImplementedError : public Error {
  using Error::Error;
};

// Used in ATen for non finite indices.  These turn into
// ExitException when they cross to Python.
class C10_API EnforceFiniteError : public Error {
  using Error::Error;
};

// Used in Onnxifi backend lowering.  These turn into
// ExitException when they cross to Python.
class C10_API OnnxfiBackendSystemError : public Error {
  using Error::Error;
};

// Used for numerical errors from the linalg module. These
// turn into LinAlgError when they cross into Python.
class C10_API LinAlgError : public Error {
  using Error::Error;
};

class C10_API OutOfMemoryError : public Error {
  using Error::Error;
};

// Base error type for all distributed errors.
// These turn into DistError when they cross into Python.
class C10_API DistError : public Error {
  using Error::Error;
};

// Used for collective communication library errors from the distributed module.
// These turn into DistBackendError when they cross into Python.
class C10_API DistBackendError : public DistError {
  using DistError::DistError;
};

// Used for errors originating from the store.
// These turn into DistStoreError when they cross into Python.
class C10_API DistStoreError : public DistError {
  using DistError::DistError;
};

// Used for errors originating from the TCP/IP stack and not from collective
// libraries. These turn into DistNetworkError when they cross into Python.
// 定义一个派生自 DistError 的 DistNetworkError 异常类
class C10_API DistNetworkError : public DistError {
  using DistError::DistError;
};

// 定义一个实用函数，用于返回异常 std::string，其内容为异常类型和 what() 方法返回的内容的组合
C10_API std::string GetExceptionString(const std::exception& e);

} // namespace c10

// 用于实现 TORCH_INTERNAL_ASSERT 和 TORCH_CHECK 的私有辅助宏
//
// 注意：在调试构建中，对于 MSVC，__LINE__ 可能是 long 类型（即 int32_t），而不是 `SourceLocation` 的定义所需的 unsigned int 类型（即 uint32_t），
// 这可能会导致编译错误，错误消息为：error C2397: conversion from 'long' to 'uint32_t' requires a narrowing conversion。
// 这里使用 static_cast 来解决这个问题。如果这个宏在 lambda 内部使用，__func__ 宏会扩展为 operator()，这并不是很有用，但很难在宏中修复，所以抑制了警告。
#define C10_THROW_ERROR(err_type, msg) \
  throw ::c10::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

#define C10_BUILD_ERROR(err_type, msg) \
  ::c10::err_type({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

// 用于解决 MSVC 对涉及 __VA_ARGS__ 的嵌套宏展开的误解问题的私有辅助宏
// 参考：https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define C10_EXPAND_MSVC_WORKAROUND(x) x

// 在 nvcc 中，C10_UNLIKELY 会破坏缺失返回语句分析。在不确定表达式可能是常量的情况下，使用这个宏来确保返回语句分析继续工作
// （以不获取 nvcc 上的 likely/unlikely 注释为代价）。
// 参考：https://github.com/pytorch/pytorch/issues/21418
//
// 目前，此宏仅在下面的错误报告宏中使用。如果要更广泛地使用它，请将其移到 Macros.h 文件中
//
// TODO：Brian Vaughan 提出，我们可能通过编写某种类型的 C++ 重载来区分 constexpr 输入和非 constexpr 输入，以使其在 nvcc 上工作。
// 由于没有证据表明在 nvcc 上失去 C10_UNLIKELY 会导致性能问题，因此尚未实现这一点，但这可能是一个有趣的 C++ 代码片段，可以由有冒险精神的初级开发者编写。
#if defined(__CUDACC__)
#define C10_UNLIKELY_OR_CONST(e) e
#else
#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)
#endif

// ----------------------------------------------------------------------------
// 错误报告宏
// ----------------------------------------------------------------------------

#ifdef STRIP_ERROR_MESSAGES
// 如果定义了 STRIP_ERROR_MESSAGES，简单地重新抛出异常
#define TORCH_RETHROW(e, ...) throw
#else
// 否则，在异常中添加上下文信息并重新抛出
#define TORCH_RETHROW(e, ...)               \
  do {                                      \
    e.add_context(::c10::str(__VA_ARGS__)); \
    throw;                                  \
  } while (false)
#endif

// 提供类似 assert() 的功能的实用宏；即强制条件检查，如果条件为 false，则抛出异常
// 定义一个内部断言宏 TORCH_INTERNAL_ASSERT，用于在条件不满足时触发异常。
// 如果宏 STRIP_ERROR_MESSAGES 被定义，只会输出基本的失败信息，否则输出详细的错误信息。
#ifdef STRIP_ERROR_MESSAGES
#define TORCH_INTERNAL_ASSERT(cond, ...)                              \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                               \
    ::c10::detail::torchCheckFail(                                    \
        __func__,                                                     \
        __FILE__,                                                     \
        static_cast<uint32_t>(__LINE__),                              \
        #cond " INTERNAL ASSERT FAILED at " C10_STRINGIZE(__FILE__)); \
  }
#else
// 如果条件不满足，则调用 torchInternalAssertFail 函数打印失败信息，包括函数名、文件名、行号和条件表达式。
// 这个宏支持额外的参数，用于在失败时打印更多的调试信息。
#define TORCH_INTERNAL_ASSERT(cond, ...)                                         \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                                          \
    ::c10::detail::torchInternalAssertFail(                                      \
        __func__,                                                                \
        __FILE__,                                                                \
        static_cast<uint32_t>(__LINE__),                                         \
        #cond                                                                    \
        " INTERNAL ASSERT FAILED at " C10_STRINGIZE(__FILE__) ":" C10_STRINGIZE( \
            __LINE__) ", please report a bug to PyTorch. ",                      \
        c10::str(__VA_ARGS__));                                                  \
  }
#endif
//    TORCH_CHECK(should_be_true); // A default error message will be provided
//                                 // in this case; but we recommend writing an
//                                 // explicit error message, as it is more
//                                 // user friendly.
//    TORCH_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception.  If this exception propagates
// to Python, it will convert into a Python RuntimeError.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike CHECK() from glog.)
//
#define TORCH_CHECK_WITH(error_t, cond, ...) \
  TORCH_CHECK_WITH_MSG(error_t, cond, "", __VA_ARGS__)

#ifdef STRIP_ERROR_MESSAGES
// Define a macro TORCH_CHECK_MSG(cond, type, ...) that creates an error message string.
#define TORCH_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
// Define a macro TORCH_CHECK_WITH_MSG(error_t, cond, type, ...) to perform a checked assertion.
#define TORCH_CHECK_WITH_MSG(error_t, cond, type, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                               \
    C10_THROW_ERROR(Error, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); \
  }
#else
// The following code is inside the c10::detail namespace.

// Implementation details for creating error messages with varying arguments.

namespace c10::detail {
// Function template torchCheckMsgImpl that handles creation of error messages.
template <typename... Args>
decltype(auto) torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  return ::c10::str(args...);
}
// Specialization for handling a single const char* argument.
inline C10_API const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline C10_API const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}
} // namespace c10::detail

// Macro TORCH_CHECK_MSG(cond, type, ...) creates an error message string based on conditions and additional arguments.
#define TORCH_CHECK_MSG(cond, type, ...)                   \
  (::c10::detail::torchCheckMsgImpl(                       \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
// Macro TORCH_CHECK_WITH_MSG(error_t, cond, type, ...) performs a checked assertion with an error message.
#define TORCH_CHECK_WITH_MSG(error_t, cond, type, ...)                  \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                                 \
    C10_THROW_ERROR(error_t, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); \
  }
#endif

// The c10::detail namespace contains functions for error handling.

namespace c10::detail {
// Function prototypes for error handling functions that do not return.
[[noreturn]] C10_API void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg);
[[noreturn]] C10_API void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

// Function prototype for internal assertion failure with detailed error message.
[[noreturn]] C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const char* userMsg);
}
// 当前函数用于在断言失败时调用，触发torchCheckFail函数
[[noreturn]] inline C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    ::c10::detail::CompileTimeEmptyString /*userMsg*/) {
  // 调用torchCheckFail函数，传递函数名、文件名、行号以及条件消息
  torchCheckFail(func, file, line, condMsg);
}

// 当前函数用于在断言失败时调用，触发torchCheckFail函数
[[noreturn]] C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg);

} // namespace c10::detail

// 如果定义了STRIP_ERROR_MESSAGES宏，则使用优化后的TORCH_CHECK宏
#ifdef STRIP_ERROR_MESSAGES
// 定义TORCH_CHECK宏，检查条件cond，如果不满足则触发torchCheckFail函数
#define TORCH_CHECK(cond, ...)                   \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {          \
    ::c10::detail::torchCheckFail(               \
        __func__,                                \
        __FILE__,                                \
        static_cast<uint32_t>(__LINE__),         \
        TORCH_CHECK_MSG(cond, "", __VA_ARGS__)); \
  }
// 否则使用默认的TORCH_CHECK宏
#else
// 定义TORCH_CHECK宏，检查条件cond，如果不满足则触发torchCheckFail函数
#define TORCH_CHECK(cond, ...)                     \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {            \
    ::c10::detail::torchCheckFail(                 \
        __func__,                                  \
        __FILE__,                                  \
        static_cast<uint32_t>(__LINE__),           \
        TORCH_CHECK_MSG(cond, "", ##__VA_ARGS__)); \
  }
#endif

// 如果当前是CUDA或HIP环境，则定义TORCH_CHECK_IF_NOT_ON_CUDA为空宏
// 否则定义TORCH_CHECK_IF_NOT_ON_CUDA为TORCH_CHECK宏
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TORCH_CHECK_IF_NOT_ON_CUDA(cond, ...)
#else
#define TORCH_CHECK_IF_NOT_ON_CUDA(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)
#endif

// 在调试模式下，定义TORCH_INTERNAL_ASSERT_DEBUG_ONLY宏为TORCH_INTERNAL_ASSERT宏
// 否则优化为不生成任何代码的形式
#ifdef NDEBUG
// 优化版本，不生成任何代码
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  while (false)                               \
  C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
#else
// 调试版本，调用TORCH_INTERNAL_ASSERT宏
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
#endif

// TODO: We're going to get a lot of similar looking string literals
// this way; check if this actually affects binary size.

// 类似于TORCH_CHECK，但引发LinAlgError而不是Error。
#define TORCH_CHECK_LINALG(cond, ...) \
  TORCH_CHECK_WITH_MSG(LinAlgError, cond, "LINALG", __VA_ARGS__)

// 类似于TORCH_CHECK，但引发IndexError而不是Error。
#define TORCH_CHECK_INDEX(cond, ...) \
  TORCH_CHECK_WITH_MSG(IndexError, cond, "INDEX", __VA_ARGS__)

// 类似于TORCH_CHECK，但引发ValueError而不是Error。
#define TORCH_CHECK_VALUE(cond, ...) \
  TORCH_CHECK_WITH_MSG(ValueError, cond, "VALUE", __VA_ARGS__)

// 类似于TORCH_CHECK，但引发TypeError而不是Error。
#define TORCH_CHECK_TYPE(cond, ...) \  // 定义宏TORCH_CHECK_TYPE，用于检查条件是否满足，若不满足则抛出TypeError，可附加参数进行说明
  TORCH_CHECK_WITH_MSG(TypeError, cond, "TYPE", __VA_ARGS__)  // 调用TORCH_CHECK_WITH_MSG宏，传递TypeError和额外参数，检查条件是否满足

// Like TORCH_CHECK, but raises NotImplementedErrors instead of Errors.
#define TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \  // 定义宏TORCH_CHECK_NOT_IMPLEMENTED，功能类似TORCH_CHECK，但条件不满足时抛出NotImplementedError
  TORCH_CHECK_WITH_MSG(NotImplementedError, cond, "TYPE", __VA_ARGS__)  // 调用TORCH_CHECK_WITH_MSG宏，传递NotImplementedError和额外参数，检查条件是否满足

#define TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(cond, ...) \  // 定义宏TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE，总是显示C++堆栈跟踪信息
  TORCH_CHECK_WITH_MSG(                                   \  // 调用TORCH_CHECK_WITH_MSG宏，传递ErrorAlwaysShowCppStacktrace和额外参数，检查条件是否满足
      ErrorAlwaysShowCppStacktrace, cond, "TYPE", ##__VA_ARGS__)

#ifdef STRIP_ERROR_MESSAGES
#define WARNING_MESSAGE_STRING(...) \  // 如果定义了STRIP_ERROR_MESSAGES宏，则定义空字符串WARNING_MESSAGE_STRING
  ::c10::detail::CompileTimeEmptyString {}  // 使用C10库的CompileTimeEmptyString表示空字符串
#else
#define WARNING_MESSAGE_STRING(...) ::c10::str(__VA_ARGS__)  // 否则，使用C10库的str函数将额外参数转换为字符串形式作为警告信息
#endif

// Report a warning to the user.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#ifdef DISABLE_WARN
#define _TORCH_WARN_WITH(...) ((void)0);  // 如果定义了DISABLE_WARN宏，则定义_TORCH_WARN_WITH为空操作
#else
#define _TORCH_WARN_WITH(warning_t, ...)                     \  // 定义_TORCH_WARN_WITH宏，用于生成警告信息
  ::c10::warn(::c10::Warning(                                \  // 调用C10库的warn函数，传递警告类型和其他参数生成警告
      warning_t(),                                           \  // 警告类型
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \  // 函数名、文件名和行号
      WARNING_MESSAGE_STRING(__VA_ARGS__),                   \  // 警告信息
      false));                                                // 是否在控制台打印警告

#define TORCH_WARN(...) _TORCH_WARN_WITH(::c10::UserWarning, __VA_ARGS__);  // 定义TORCH_WARN宏，生成用户警告，调用_TORCH_WARN_WITH传递UserWarning类型和额外参数

#define TORCH_WARN_DEPRECATION(...) \  // 定义TORCH_WARN_DEPRECATION宏，生成过时警告，调用_TORCH_WARN_WITH传递DeprecationWarning类型和额外参数
  _TORCH_WARN_WITH(::c10::DeprecationWarning, __VA_ARGS__);

// Report a warning to the user only once.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#define _TORCH_WARN_ONCE(...)                                             \  // 定义_TORCH_WARN_ONCE宏，只生成一次警告
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(torch_warn_once_) = \  // 定义静态变量用于标记是否已经发出过警告
      [&] {                                                               \  // 使用lambda表达式定义初始化函数
        TORCH_WARN(__VA_ARGS__);                                          \  // 调用TORCH_WARN生成警告
        return true;                                                      \  // 返回true表示已经生成过警告
      }()

#ifdef DISABLE_WARN
#define TORCH_WARN_ONCE(...) ((void)0);  // 如果定义了DISABLE_WARN宏，则TORCH_WARN_ONCE为空操作
#else
#define TORCH_WARN_ONCE(...)                   \  // 否则定义TORCH_WARN_ONCE宏，根据全局设置判断是否生成警告
  if (::c10::WarningUtils::get_warnAlways()) { \  // 如果全局设置总是生成警告
    TORCH_WARN(__VA_ARGS__);                   \  // 直接调用TORCH_WARN生成警告
  } else {                                     \  // 否则，仅生成一次警告
    _TORCH_WARN_ONCE(__VA_ARGS__);             \
  }
#endif

// Report an error with a specific argument
// NOTE: using the argument name in TORCH_CHECK's message is preferred
#define TORCH_CHECK_ARG(cond, argN, ...) \  // 定义TORCH_CHECK_ARG宏，检查条件是否满足，并报告特定参数的错误
  TORCH_CHECK(cond, "invalid argument ", argN, ": ", __VA_ARGS__)  // 调用TORCH_CHECK宏，检查条件是否满足，同时附加参数作为错误信息

// ----------------------------------------------------------------------------
// Deprecated macros
// ----------------------------------------------------------------------------

namespace c10::detail {

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ERROR(msg) is deprecated, use TORCH_CHECK(false, msg)
instead.")
*/
inline void deprecated_AT_ERROR() {}  // 定义deprecated_AT_ERROR函数，该函数已被废弃

/*
// Deprecation disabled until we fix sites in our codebase
/*
// 定义一个名为 deprecated_AT_ASSERT 的内联函数，不建议使用此函数，建议使用 TORCH_INTERNAL_ASSERT 或 TORCH_CHECK
inline void deprecated_AT_ASSERT() {}

/*
// 在我们修复代码库中的问题之前，禁用此处的过时消息
// 建议使用 TORCH_INTERNAL_ASSERT 或 TORCH_CHECK
*/
inline void deprecated_AT_ASSERTM() {}

} // namespace c10::detail

// AT_ASSERT 宏的定义，用于向后兼容，但已被弃用
// 建议使用 TORCH_INTERNAL_ASSERT 或 TORCH_CHECK 替代
#define AT_ASSERT(...)                                              \
  do {                                                              \
    ::c10::detail::deprecated_AT_ASSERT();                          \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__)); \
  } while (false)

// AT_ASSERTM 宏的定义，用于向后兼容，但已被弃用
// 建议使用 TORCH_INTERNAL_ASSERT 或 TORCH_CHECK 替代
#define AT_ASSERTM(cond, ...)                                             \
  do {                                                                    \
    ::c10::detail::deprecated_AT_ASSERTM();                               \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__)); \
  } while (false)

// AT_ERROR 宏的定义，用于向后兼容，但已被弃用
// 建议使用 TORCH_CHECK 或 TORCH_INTERNAL_ASSERT 来在代码行上无条件失败
#define AT_ERROR(...)                                                        \
  do {                                                                       \
    ::c10::detail::deprecated_AT_ERROR();                                    \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(false, ::c10::str(__VA_ARGS__))); \
  } while (false)

#endif // C10_UTIL_EXCEPTION_H_
```