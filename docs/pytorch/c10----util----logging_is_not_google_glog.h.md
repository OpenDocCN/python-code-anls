# `.\pytorch\c10\util\logging_is_not_google_glog.h`

```py
#ifndef C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_

#include <chrono>
#include <climits>
#include <ctime>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <c10/util/Flags.h>

// 定义一个常量字符串，表示日志严重程度的前缀
const char CAFFE2_SEVERITY_PREFIX[] = "FEWIV";

namespace c10 {

// 定义日志严重程度的常量
const int GLOG_FATAL = 3;
const int GLOG_ERROR = 2;
const int GLOG_WARNING = 1;
const int GLOG_INFO = 0;

// 日志消息记录器类，用于记录日志消息
class C10_API MessageLogger {
 public:
  // 构造函数，初始化日志消息记录器
  MessageLogger(const char* file, int line, int severity);
  // 析构函数，释放资源
  ~MessageLogger();
  // 返回与日志记录器对象关联的流对象
  std::stringstream& stream() {
    return stream_;
  }

 private:
  // 处理致命日志的私有方法，简单地中止程序
  void DealWithFatal() {
    abort();
  }

  const char* tag_;  // 标签，暂未使用
  std::stringstream stream_;  // 字符串流，用于构建日志消息
  int severity_;  // 日志消息的严重程度
};

// 该类用于在条件日志宏中显式忽略值，避免编译器警告
class C10_API LoggerVoidify {
 public:
  LoggerVoidify() = default;
  // 该操作符必须具有比 << 更低但比 ?: 更高的优先级
  void operator&(const std::ostream& s) {}
};

// 记录消息并终止程序的模板函数
template <class T>
void LogMessageFatal(const char* file, int line, const T& message) {
  MessageLogger(file, line, GLOG_FATAL).stream() << message;
}

// TORCH_CHECK_NOTNULL() 的辅助函数，支持原始指针和智能指针
template <typename T>
T& CheckNotNullCommon(const char* file, int line, const char* names, T& t) {
  // 如果指针为空，则记录致命日志
  if (t == nullptr) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}

// 检查指针非空的模板函数，支持原始指针和智能指针
template <typename T>
T* CheckNotNull(const char* file, int line, const char* names, T* t) {
  return CheckNotNullCommon(file, line, names, t);
}

// 检查引用非空的模板函数，支持原始指针和智能指针
template <typename T>
T& CheckNotNull(const char* file, int line, const char* names, T& t) {
  return CheckNotNullCommon(file, line, names, t);
}
} // namespace c10

// ---------------------- Logging Macro definitions --------------------------

// 断言：CAFFE2_LOG_THRESHOLD 至多等于 GLOG_FATAL
static_assert(
    CAFFE2_LOG_THRESHOLD <= ::c10::GLOG_FATAL,
    "CAFFE2_LOG_THRESHOLD should at most be GLOG_FATAL.");

// 如果日志级别 n 不小于编译时 CAFFE2_LOG_THRESHOLD，则记录日志消息
#define LOG(n)                                 \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()

// 如果 VLOG(n) 的负数不小于编译时 CAFFE2_LOG_THRESHOLD，则记录日志消息
#define VLOG(n)                   \
  if (-n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(__FILE__, __LINE__, -n).stream()

// 如果条件为真且日志级别 n 不小于 CAFFE2_LOG_THRESHOLD，则记录日志消息
#define LOG_IF(n, condition)                                  \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD && (condition)) \
  ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()

#endif // C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_
// 定义一个宏，用于根据给定的日志级别和条件输出日志消息
#define VLOG_IF(n, condition)                    \
  if (-n >= CAFFE2_LOG_THRESHOLD && (condition)) \
  ::c10::MessageLogger(__FILE__, __LINE__, -n).stream()

// 定义一个宏，用于检查指定的日志级别是否处于开启状态
#define VLOG_IS_ON(verboselevel) (CAFFE2_LOG_THRESHOLD <= -(verboselevel))

// 定义一个宏，用于在指定的文件和行号记录日志消息，只有当日志级别高于阈值时才有效
#define LOG_AT_FILE_LINE(n, file, line)        \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(file, line, ::c10::GLOG_##n).stream()

// 定义一个宏，仅当条件满足时记录日志，否则不产生任何输出
#define FATAL_IF(condition)            \
  condition ? (void)0                  \
            : ::c10::LoggerVoidify() & \
          ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_FATAL).stream()

// 如果处于调试模式（NDEBUG未定义），定义一个调试版本的检查宏，否则定义为不生成任何代码
#ifndef NDEBUG
#define DCHECK(condition) FATAL_IF(condition) << "Check failed: " #condition " "
#define DLOG(severity) LOG(severity)
#else // NDEBUG
#define DCHECK(condition) \
  while (false)           \
  CHECK(condition)

#define DLOG(n)                   \
  true ? (void)0                  \
       : ::c10::LoggerVoidify() & \
          ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()
#endif // NDEBUG

// 定义一个宏，用于检查给定的操作关系是否成立，否则产生一个致命错误
#define TORCH_CHECK_OP(val1, val2, op)                                        \
  FATAL_IF(((val1)op(val2))) << "Check failed: " #val1 " " #op " " #val2 " (" \
                             << (val1) << " vs. " << (val2) << ") "

// 定义一系列用于比较操作的宏，基于TORCH_CHECK_OP宏实现
#define TORCH_CHECK_EQ(val1, val2) TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_CHECK_NE(val1, val2) TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_CHECK_LE(val1, val2) TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_CHECK_LT(val1, val2) TORCH_CHECK_OP(val1, val2, <)
#define TORCH_CHECK_GE(val1, val2) TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_CHECK_GT(val1, val2) TORCH_CHECK_OP(val1, val2, >)

// 如果处于调试模式（NDEBUG未定义），定义一系列调试版本的比较宏，否则定义为不生成任何代码
#ifndef NDEBUG
#define TORCH_DCHECK_EQ(val1, val2) TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) TORCH_CHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) TORCH_CHECK_OP(val1, val2, >)
#else // !NDEBUG
// 这些版本在优化模式下不生成任何代码
#define TORCH_DCHECK_EQ(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >)
#endif // NDEBUG
// 定义一个宏，用于在调试模式下比较两个值是否满足小于关系
#define TORCH_DCHECK_LT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, <)

// 定义一个宏，用于在调试模式下比较两个值是否满足大于等于关系
#define TORCH_DCHECK_GE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >=)

// 定义一个宏，用于在调试模式下比较两个值是否满足大于关系
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >)

// 如果不处于调试模式，则结束条件编译指令
#endif // NDEBUG

// 检查指针是否非空，如果为空则触发检查失败信息
#define TORCH_CHECK_NOTNULL(val) \
  ::c10::CheckNotNull(           \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// 调试模式下的 TORCH_CHECK_NOTNULL 宏的定义
#define TORCH_DCHECK_NOTNULL(val) \
  ::c10::CheckNotNull(            \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
#else // !NDEBUG
// 非调试模式下的 TORCH_DCHECK_NOTNULL 宏定义，生成无代码
#define TORCH_DCHECK_NOTNULL(val) \
  while (false)                   \
  TORCH_CHECK_NOTNULL(val)
#endif // NDEBUG

// ---------------------- 支持 STL 对象的打印 --------------------------

// 这些函数用于支持对 STL 容器对象的有限日志记录功能，是从 glog 改编而来。

namespace std {
// 首先声明这两个函数，然后在所有的容器流操作符之后定义它们，
// 以便可以正确地递归从 pair -> container -> container -> pair。
template <class First, class Second>
std::ostream& operator<<(std::ostream& out, const std::pair<First, Second>& p);
} // namespace std

namespace c10 {
// 打印容器中元素的函数模板，参数为迭代器范围
template <class Iter>
void PrintSequence(std::ostream& ss, Iter begin, Iter end);
} // namespace c10

namespace std {
// 实例化容器的流输出操作符模板，用于 std::vector 的实例化
#define INSTANTIATE_FOR_CONTAINER(container)               \
  template <class... Types>                                \
  std::ostream& operator<<(                                \
      std::ostream& out, const container<Types...>& seq) { \
    c10::PrintSequence(out, seq.begin(), seq.end());       \
    return out;                                            \
  }

INSTANTIATE_FOR_CONTAINER(std::vector)
INSTANTIATE_FOR_CONTAINER(std::map)
INSTANTIATE_FOR_CONTAINER(std::set)
#undef INSTANTIATE_FOR_CONTAINER

// 定义 std::pair 的流输出操作符，输出格式为 (first, second)
template <class First, class Second>
inline std::ostream& operator<<(
    std::ostream& out,
    const std::pair<First, Second>& p) {
  out << '(' << p.first << ", " << p.second << ')';
  return out;
}

// 定义对 nullptr_t 类型的流输出操作符，输出为 "(null)"
inline std::ostream& operator<<(std::ostream& out, const std::nullptr_t&) {
  out << "(null)";
  return out;
}
} // namespace std

namespace c10 {
// 打印序列的函数模板，用于打印迭代器范围内的序列，最多打印 100 个元素
template <class Iter>
inline void PrintSequence(std::ostream& out, Iter begin, Iter end) {
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0)
      out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
}
} // namespace c10

#endif // C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_


这些注释完整地解释了每行代码的作用和功能，符合给定的要求和注意事项。
```