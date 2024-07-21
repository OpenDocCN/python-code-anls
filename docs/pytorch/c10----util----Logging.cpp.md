# `.\pytorch\c10\util\Logging.cpp`

```
// 引入必要的头文件
#include <c10/util/Backtrace.h>
#include <c10/util/Flags.h>
#include <c10/util/Lazy.h>
#include <c10/util/Logging.h>
#ifdef FBCODE_CAFFE2
#include <folly/synchronization/SanitizeThread.h>
#endif

// 如果不是在 Windows 平台下，引入系统时间相关的头文件
#ifndef _WIN32
#include <sys/time.h>
#endif

#include <algorithm>  // 引入算法相关的头文件
#include <cstdlib>    // 引入标准库的通用工具函数
#include <iostream>   // 引入输入输出流相关的头文件

// 无论是否使用 glog，这里是通用的代码。

// 定义一个布尔类型的命令行标志，用于控制 CAFFE_ENFORCE 的行为
C10_DEFINE_bool(
    caffe2_use_fatal_for_enforce,
    false,
    "If set true, when CAFFE_ENFORCE is not met, abort instead "
    "of throwing an exception.");

namespace c10 {

namespace {
// 获取获取堆栈跟踪的函数
std::function<::c10::Backtrace()>& GetFetchStackTrace() {
  // 使用 lambda 表达式初始化静态局部变量，返回获取堆栈跟踪的函数
  static std::function<::c10::Backtrace()> func = []() {
    return get_lazy_backtrace(/*frames_to_skip=*/1);
  };
  return func;
}
} // namespace

// 设置获取堆栈跟踪的函数
void SetStackTraceFetcher(std::function<::c10::Backtrace()> fetcher) {
  GetFetchStackTrace() = std::move(fetcher);
}

// 设置获取堆栈跟踪的函数（重载版本，用于字符串类型的 fetcher）
void SetStackTraceFetcher(std::function<string()> fetcher) {
  // 调用前一个函数，用于设置获取堆栈跟踪的函数
  SetStackTraceFetcher([fetcher = std::move(fetcher)] {
    return std::make_shared<PrecomputedLazyValue<std::string>>(fetcher());
  });
}

// 抛出 CAFFE_ENFORCE 未满足的异常
void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
  // 创建异常对象 e，并记录文件名、行号、条件、消息、堆栈跟踪和调用者信息
  c10::Error e(file, line, condition, msg, GetFetchStackTrace()(), caller);
  // 如果设置了使用致命错误模式，则记录异常信息并终止程序
  if (FLAGS_caffe2_use_fatal_for_enforce) {
    LOG(FATAL) << e.msg();
  }
  // 抛出异常 e
  throw std::move(e);
}

// 抛出 CAFFE_ENFORCE 未满足的异常（重载版本，用于字符串类型的 msg）
void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceNotMet(file, line, condition, std::string(msg), caller);
}

// 抛出 CAFFE_ENFORCE_FINITE 未满足的异常
void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
  // 抛出 CAFFE_ENFORCE_FINITE 未满足的异常，记录文件名、行号、条件、消息、堆栈跟踪和调用者信息
  throw c10::EnforceFiniteError(
      file, line, condition, msg, GetFetchStackTrace()(), caller);
}

// 抛出 CAFFE_ENFORCE_FINITE 未满足的异常（重载版本，用于字符串类型的 msg）
void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceFiniteNotMet(file, line, condition, std::string(msg), caller);
}

namespace {

// PyTorch 风格的堆栈跟踪类，继承自 OptimisticLazyValue<std::string>
class PyTorchStyleBacktrace : public OptimisticLazyValue<std::string> {
 public:
  // 构造函数，初始化堆栈跟踪和源位置信息
  PyTorchStyleBacktrace(SourceLocation source_location)
      : backtrace_(GetFetchStackTrace()()), source_location_(source_location) {}

 private:
  // 重写计算函数，返回异常信息和堆栈跟踪的字符串表示
  std::string compute() const override {
    return str(
        "Exception raised from ",
        source_location_,
        " (most recent call first):\n",
        backtrace_->get());
  }

  ::c10::Backtrace backtrace_;
  SourceLocation source_location_;
};

} // namespace

// PyTorch 风格的错误消息构造函数的实现
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          std::make_shared<PyTorchStyleBacktrace>(source_location)) {}

// APIUsageLoggerType 类型的别名，表示一个接收字符串参数的函数指针
using APIUsageLoggerType = std::function<void(const std::string&)>;
using APIUsageMetadataLoggerType = std::function<void(
    const std::string&,
    const std::map<std::string, std::string>& metadata_map)>;
// 定义 API 使用元数据记录器类型，它是一个函数类型，接受事件字符串和元数据映射作为参数

using DDPUsageLoggerType = std::function<void(const DDPLoggingData&)>;
// 定义 DDP 使用记录器类型，它是一个函数类型，接受 DDPLoggingData 结构体作为参数

namespace {
bool IsAPIUsageDebugMode() {
  const char* val = getenv("PYTORCH_API_USAGE_STDERR");
  // 获取环境变量 PYTORCH_API_USAGE_STDERR 的值

  return val && *val; // 返回是否存在且非空值
}

void APIUsageDebug(const string& event) {
  // 使用 stderr 输出事件字符串，避免与 glog 混淆
  std::cerr << "PYTORCH_API_USAGE " << event << std::endl;
}

APIUsageLoggerType* GetAPIUsageLogger() {
  static APIUsageLoggerType func =
      IsAPIUsageDebugMode() ? &APIUsageDebug : [](const string&) {};
  // 获取 API 使用记录器函数指针，根据调试模式决定使用 APIUsageDebug 或空lambda函数

  return &func; // 返回 API 使用记录器函数指针
}

APIUsageMetadataLoggerType* GetAPIUsageMetadataLogger() {
  static APIUsageMetadataLoggerType func =
      [](const std::string&,
         const std::map<std::string, std::string>& metadata_map) {};
  // 获取 API 使用元数据记录器函数指针，始终使用空lambda函数

  return &func; // 返回 API 使用元数据记录器函数指针
}

DDPUsageLoggerType* GetDDPUsageLogger() {
  static DDPUsageLoggerType func = [](const DDPLoggingData&) {};
  // 获取 DDP 使用记录器函数指针，始终使用空lambda函数

  return &func; // 返回 DDP 使用记录器函数指针
}
} // namespace

void SetAPIUsageLogger(std::function<void(const std::string&)> logger) {
  TORCH_CHECK(logger);
  // 检查传入的 API 使用记录器函数是否有效

  *GetAPIUsageLogger() = std::move(logger);
  // 将传入的 API 使用记录器函数移动赋值给全局的 API 使用记录器函数指针
}

void SetAPIUsageMetadataLogger(
    std::function<void(
        const std::string&,
        const std::map<std::string, std::string>& metadata_map)> logger) {
  TORCH_CHECK(logger);
  // 检查传入的 API 使用元数据记录器函数是否有效

  *GetAPIUsageMetadataLogger() = std::move(logger);
  // 将传入的 API 使用元数据记录器函数移动赋值给全局的 API 使用元数据记录器函数指针
}

void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger) {
  TORCH_CHECK(logger);
  // 检查传入的 DDP 使用记录器函数是否有效

  *GetDDPUsageLogger() = std::move(logger);
  // 将传入的 DDP 使用记录器函数移动赋值给全局的 DDP 使用记录器函数指针
}

static int64_t GLOBAL_RANK = -1;
// 静态全局变量，初始化为 -1

int64_t GetGlobalRank() {
  return GLOBAL_RANK;
  // 返回全局排名变量的值
}

void SetGlobalRank(int64_t rank) {
  GLOBAL_RANK = rank;
  // 设置全局排名变量的值为传入的排名值
}

void LogAPIUsage(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  // 尝试记录 API 使用事件，如果存在有效的 API 使用记录器函数指针

} catch (std::bad_function_call&) {
  // 静态析构器竞争异常处理
}

void LogAPIUsageMetadata(
    const std::string& context,
    const std::map<std::string, std::string>& metadata_map) try {
  if (auto logger = GetAPIUsageMetadataLogger())
    (*logger)(context, metadata_map);
  // 尝试记录 API 使用元数据，如果存在有效的 API 使用元数据记录器函数指针

} catch (std::bad_function_call&) {
  // 静态析构器竞争异常处理
}

void LogPyTorchDDPUsage(const DDPLoggingData& ddpData) try {
  if (auto logger = GetDDPUsageLogger())
    (*logger)(ddpData);
  // 尝试记录 PyTorch DDP 使用，如果存在有效的 DDP 使用记录器函数指针

} catch (std::bad_function_call&) {
  // 静态析构器竞争异常处理
}

namespace detail {
bool LogAPIUsageFakeReturn(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  // 尝试记录 API 使用事件，并返回 true，如果存在有效的 API 使用记录器函数指针

  return true;

} catch (std::bad_function_call&) {
  // 静态析构器竞争异常处理
  return true;
}

namespace {

void setLogLevelFlagFromEnv();

} // namespace
} // namespace detail
} // namespace c10

#if defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)
// 当 C10 使用 GFLAGS 和 GLOG 时，通过 GFLAGS 的定义直接在 GLOG 中定义这些变量，
// 因此使用 DECLARE_* 来声明它们，并在 Caffe2 中使用它们。
// GLOG 的最小日志级别
DECLARE_int32(minloglevel);
// 声明和定义 GLOG 的 verbose 日志级别
DECLARE_int32(v);
// 声明和定义 GLOG 的 logtostderr 标志
DECLARE_bool(logtostderr);
#endif // defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)

#if !defined(C10_USE_GLOG)
// 当 Caffe2 没有使用 glog 编译时，提供的兼容性标志
// 这些标志用于处理即使没有使用 glog 编译，但仍然有一些初始化标志传递的情况
C10_DEFINE_int32(minloglevel, 0, "Equivalent to glog minloglevel");
C10_DEFINE_int32(v, 0, "Equivalent to glog verbose");
C10_DEFINE_bool(logtostderr, false, "Equivalent to glog logtostderr");
#endif // !defined(c10_USE_GLOG)

#ifdef C10_USE_GLOG

// 提供方便访问上述变量的命名空间，不管是否依赖于 GFLAGS
// 注意命名空间 (fLI, fLB) 在 GLOG 和 GFLAGS 之间是一致的，因此可以一致地声明如下
namespace c10 {
using fLB::FLAGS_logtostderr;
using fLI::FLAGS_minloglevel;
using fLI::FLAGS_v;
} // namespace c10

// 定义 Caffe2 的日志级别，这是 caffe2 将输出的最低日志级别
C10_DEFINE_int(
    caffe2_log_level,
    google::GLOG_WARNING,
    "The minimum log level that caffe2 will output.");

// Google glog 没有外部函数可以检查 glog 是否已初始化，但有一个内部函数
// 在这里声明这个内部函数，虽然这是一个 hack，但已被其他一些库使用（如 Torch）
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace glog_internal_namespace_
} // namespace google

namespace c10 {
namespace {

// 初始化 Google logging 的函数
void initGoogleLogging(char const* name) {
#if !defined(_MSC_VER)
  // 这个技巧只能在 UNIX 平台上使用
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
#endif
  {
    // 初始化 Google logging
    ::google::InitGoogleLogging(name);
#if !defined(_MSC_VER)
    // 这在 Windows 上不会定义
    ::google::InstallFailureSignalHandler();
#endif
  }
}

} // namespace

// 初始化 logging
void initLogging() {
  detail::setLogLevelFlagFromEnv();

  // 从标志更新 logging 级别
  UpdateLoggingLevelsFromFlags();
}

// 初始化 Caffe logging
bool InitCaffeLogging(int* argc, char** argv) {
  if (*argc == 0) {
    return true;
  }

  // 初始化 Google logging
  initGoogleLogging(argv[0]);

  // 从标志更新 logging 级别
  UpdateLoggingLevelsFromFlags();

  return true;
}

// 根据标志更新 logging 级别
void UpdateLoggingLevelsFromFlags() {
#ifdef FBCODE_CAFFE2
  // TODO(T82645998): 修复 TSAN 暴露的数据竞争。
  folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
#endif
  // 如果 caffe2_log_level 设置并且低于 glog 的最小日志级别，将其设置到 glog 中以覆盖该值
  FLAGS_minloglevel = std::min(FLAGS_caffe2_log_level, FLAGS_minloglevel);
  // 如果 caffe2_log_level 显式设置，也打开 logtostderr
  if (FLAGS_caffe2_log_level < google::GLOG_WARNING) {
    FLAGS_logtostderr = 1;
  }
  // 同样，将 caffe2_log_level 的 verbose 设置传递给 glog
  if (FLAGS_caffe2_log_level < 0) {
    FLAGS_v = std::min(FLAGS_v, -FLAGS_caffe2_log_level);
  }
}
void ShowLogInfoToStderr() {
  // 将 FLAGS_logtostderr 设置为 1，表示将日志输出到 stderr
  FLAGS_logtostderr = 1;
  // 设置 FLAGS_minloglevel 为 FLAGS_minloglevel 和 GLOG_INFO 之间的较小值，控制最小输出日志级别
  FLAGS_minloglevel = std::min(FLAGS_minloglevel, google::GLOG_INFO);
}
} // namespace c10

#else // !C10_USE_GLOG

#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID

C10_DEFINE_int(
    caffe2_log_level,
    c10::GLOG_WARNING,
    "The minimum log level that caffe2 will output.");

namespace c10 {

void initLogging() {
  // 从环境变量设置日志级别标志
  detail::setLogLevelFlagFromEnv();
}

bool InitCaffeLogging(int* argc, char** argv) {
  // 当调用 InitCaffeLogging 时，假设 caffe 的命令行标志解析已经完成
  if (*argc == 0)
    return true;
  // 如果还没有解析命令行标志，则打印错误信息
  if (!c10::CommandLineFlagsHasBeenParsed()) {
    std::cerr << "InitCaffeLogging() has to be called after "
                 "c10::ParseCommandLineFlags. Modify your program to make sure "
                 "of this."
              << std::endl;
    return false;
  }
  // 如果 FLAGS_caffe2_log_level 大于 GLOG_FATAL，则将其限制为 GLOG_FATAL
  if (FLAGS_caffe2_log_level > GLOG_FATAL) {
    std::cerr << "The log level of Caffe2 has to be no larger than GLOG_FATAL("
              << GLOG_FATAL << "). Capping it to GLOG_FATAL." << std::endl;
    FLAGS_caffe2_log_level = GLOG_FATAL;
  }
  return true;
}

void UpdateLoggingLevelsFromFlags() {}

void ShowLogInfoToStderr() {
  // 将 FLAGS_caffe2_log_level 设置为 GLOG_INFO，以输出 INFO 级别的日志到 stderr
  FLAGS_caffe2_log_level = GLOG_INFO;
}

MessageLogger::MessageLogger(const char* file, int line, int severity)
    : severity_(severity) {
  // 如果该日志的严重程度小于 FLAGS_caffe2_log_level，则不需要记录日志
  if (severity_ < FLAGS_caffe2_log_level) {
    // 不需要记录日志，直接返回
    return;
  }
#ifdef ANDROID
  tag_ = "native";
#else // !ANDROID
  tag_ = "";
#endif // ANDROID

  time_t rawtime = 0;
  time(&rawtime);

#ifndef _WIN32
  struct tm raw_timeinfo = {0};
  struct tm* timeinfo = &raw_timeinfo;
  // 获取本地时间信息
  localtime_r(&rawtime, timeinfo);
#else
  // 在 Windows 上是线程安全的
  struct tm* timeinfo = localtime(&rawtime);
#endif

#ifndef _WIN32
  // 获取当前纳秒数（自从 epoch 以来）
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long ns = ts.tv_nsec;
#else
  long ns = 0;
#endif

  if (GLOBAL_RANK != -1) {
    // 如果 GLOBAL_RANK 不为 -1，则记录日志中包含该信息
    stream_ << "[rank" << GLOBAL_RANK << "]:";
  }
  // 构建日志的时间戳和格式化信息
  stream_ << "[" << CAFFE2_SEVERITY_PREFIX[std::min(4, GLOG_FATAL - severity_)]
          << (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
          << std::setfill('0') << " " << std::setw(2) << timeinfo->tm_hour
          << ":" << std::setw(2) << timeinfo->tm_min << ":" << std::setw(2)
          << timeinfo->tm_sec << "." << std::setw(9) << ns << " "
          << c10::detail::StripBasename(std::string(file)) << ":" << line
          << "] ";
}

// 在对象销毁时将流中的内容输出到适当的通道
MessageLogger::~MessageLogger() {
  // 如果该日志的严重程度小于 FLAGS_caffe2_log_level，则不需要记录日志
  if (severity_ < FLAGS_caffe2_log_level) {
    // 不需要记录日志，直接返回
    return;
  }
  // 将流中的内容换行输出
  stream_ << "\n";
#ifdef ANDROID
  // 定义一个静态常量数组，包含与 Android 日志级别对应的值
  static const int android_log_levels[] = {
      ANDROID_LOG_FATAL, // LOG_FATAL
      ANDROID_LOG_ERROR, // LOG_ERROR
      ANDROID_LOG_WARN, // LOG_WARNING
      ANDROID_LOG_INFO, // LOG_INFO
      ANDROID_LOG_DEBUG, // VLOG(1)
      ANDROID_LOG_VERBOSE, // VLOG(2) .. VLOG(N)
  };
  // 计算当前日志级别在 android_log_levels 数组中的索引
  int android_level_index = GLOG_FATAL - std::min(GLOG_FATAL, severity_);
  // 根据 android_level_index 获取对应的 Android 日志级别
  int level = android_log_levels[std::min(android_level_index, 5)];
  // 将日志字符串输出到 Android 日志中，使用适当的日志级别
  __android_log_print(level, tag_, "%s", stream_.str().c_str());
  // 如果是致命错误，输出终止信息到 Android 日志
  if (severity_ == GLOG_FATAL) {
    __android_log_print(ANDROID_LOG_FATAL, tag_, "terminating.\n");
  }
#else // !ANDROID
  // 如果不是在 Android 平台上
  if (severity_ >= FLAGS_caffe2_log_level) {
    // 如果日志级别高于或等于 FLAGS_caffe2_log_level
    // 将所有输出日志信息打印到 std::cerr
    std::cerr << stream_.str();
    // 模拟 glog 的默认行为：如果日志级别高于 INFO，
    // 则刷新流，以便立即在 std::cerr 上显示输出。
    // 这在某些测试中是预期的行为。
    if (severity_ > GLOG_INFO) {
      std::cerr << std::flush;
    }
  }
#endif // ANDROID
  // 如果是致命错误，执行致命错误处理函数
  if (severity_ == GLOG_FATAL) {
    DealWithFatal();
  }
}

} // namespace c10

#endif // !C10_USE_GLOG

namespace c10::detail {
namespace {

void setLogLevelFlagFromEnv() {
  // 从环境变量 TORCH_CPP_LOG_LEVEL 获取日志级别字符串
  const char* level_str = std::getenv("TORCH_CPP_LOG_LEVEL");

  // 如果未设置环境变量，则使用默认级别（警告级别）
  std::string level{level_str != nullptr ? level_str : ""};
  if (level.empty()) {
    return;
  }

  // 将日志级别字符串转换为大写
  std::transform(
      level.begin(), level.end(), level.begin(), [](unsigned char c) {
        return toupper(c);
      });

  // 根据转换后的日志级别字符串设置 FLAGS_caffe2_log_level 的值
  if (level == "0" || level == "INFO") {
    FLAGS_caffe2_log_level = 0;
    return;
  }
  if (level == "1" || level == "WARNING") {
    FLAGS_caffe2_log_level = 1;
    return;
  }
  if (level == "2" || level == "ERROR") {
    FLAGS_caffe2_log_level = 2;
    return;
  }
  if (level == "3" || level == "FATAL") {
    FLAGS_caffe2_log_level = 3;
    return;
  }

  // 如果无法解析 TORCH_CPP_LOG_LEVEL 环境变量的值，则输出错误信息
  std::cerr
      << "`TORCH_CPP_LOG_LEVEL` environment variable cannot be parsed. Valid values are "
         "`INFO`, `WARNING`, `ERROR`, and `FATAL` or their numerical equivalents `0`, `1`, "
         "`2`, and `3`."
      << std::endl;
}

} // namespace
} // namespace c10::detail
```