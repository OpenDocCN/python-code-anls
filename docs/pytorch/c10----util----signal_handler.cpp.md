# `.\pytorch\c10\util\signal_handler.cpp`

```
#include <c10/util/Backtrace.h>
#include <c10/util/Logging.h>
#include <c10/util/signal_handler.h>

#if defined(C10_SUPPORTS_SIGNAL_HANDLER)

// Normal signal handler implementation.
#include <dirent.h>  // For directory operations
#include <fmt/core.h>  // For string formatting
#include <sys/syscall.h>  // For system call definitions
#include <unistd.h>  // POSIX operating system API

#include <atomic>  // For atomic operations
#include <chrono>  // For time-related operations
#include <condition_variable>  // For thread synchronization
#include <cstdint>  // For fixed-width integer types
#include <cstdio>  // For C standard input/output operations
#include <cstdlib>  // For standard library functions
#include <iostream>  // For standard input/output streams
#include <mutex>  // For mutual exclusion primitives

#ifdef C10_ANDROID
#ifndef SYS_gettid
#define SYS_gettid __NR_gettid  // Android-specific system call number
#endif
#ifndef SYS_tgkill
#define SYS_tgkill __NR_tgkill  // Android-specific system call number
#endif
#endif

namespace {

struct sigaction previousSighup;  // Previous handler for SIGHUP signal
struct sigaction previousSigint;  // Previous handler for SIGINT signal
std::atomic<int> sigintCount(0);  // Atomic counter for SIGINT occurrences
std::atomic<int> sighupCount(0);  // Atomic counter for SIGHUP occurrences
std::atomic<int> hookedUpCount(0);  // Atomic counter for handler hookups

// Signal handler function
void handleSignal(int signal) {
  switch (signal) {
    // Handle SIGHUP signal
    case SIGHUP:
      sighupCount += 1;
      // Call previous SIGHUP handler if exists
      if (previousSighup.sa_handler) {
        previousSighup.sa_handler(signal);
      }
      break;
    // Handle SIGINT signal
    case SIGINT:
      sigintCount += 1;
      // Call previous SIGINT handler if exists
      if (previousSigint.sa_handler) {
        previousSigint.sa_handler(signal);
      }
      break;
  }
}

// Function to hook up signal handlers
void hookupHandler() {
  if (hookedUpCount++) {
    return;  // Already hooked up
  }
  struct sigaction sa {};
  // Setup the handler function
  sa.sa_handler = &handleSignal;
  // Restart the system call if interrupted by handler
  sa.sa_flags = SA_RESTART;
  // Block every signal during the handler execution
  sigfillset(&sa.sa_mask);
  // Install handler for SIGHUP signal
  if (sigaction(SIGHUP, &sa, &previousSighup) == -1) {
    LOG(FATAL) << "Cannot install SIGHUP handler.";
  }
  // Install handler for SIGINT signal
  if (sigaction(SIGINT, &sa, &previousSigint) == -1) {
    LOG(FATAL) << "Cannot install SIGINT handler.";
  }
}

// Function to unhook signal handlers
void unhookHandler() {
  if (--hookedUpCount > 0) {
    return;  // Still hooked up elsewhere
  }
  struct sigaction sa {};
  // Set handler to default behavior (SIG_DFL)
  sa.sa_handler = SIG_DFL;
  // Restart the system call if interrupted by handler
  sa.sa_flags = SA_RESTART;
  // Block every signal during the handler execution
  sigfillset(&sa.sa_mask);
  // Restore default handler for SIGHUP signal
  if (sigaction(SIGHUP, &previousSighup, nullptr) == -1) {
    LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
  }
  // Restore default handler for SIGINT signal
  if (sigaction(SIGINT, &previousSigint, nullptr) == -1) {
    LOG(FATAL) << "Cannot uninstall SIGINT handler.";
  }
}

} // namespace

namespace c10 {

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

// Singleton instance getter for FatalSignalHandler
FatalSignalHandler& FatalSignalHandler::getInstance() {
  // Create a singleton instance to avoid destructor race
  static FatalSignalHandler* handler = new FatalSignalHandler();
  return *handler;
}

// Destructor for FatalSignalHandler
FatalSignalHandler::~FatalSignalHandler() = default;

// Constructor for FatalSignalHandler
FatalSignalHandler::FatalSignalHandler()
    : fatalSignalHandlersInstalled(false),
      fatalSignalReceived(false),
      fatalSignalName("<UNKNOWN>"),
      writingCond(),
      writingMutex(),
      signalReceived(false) {}
// 定义静态数组 kSignalHandlers，包含各种致命信号的名称、信号值和先前的信号处理器
FatalSignalHandler::signal_handler FatalSignalHandler::kSignalHandlers[] = {
    {"SIGABRT", SIGABRT, {}},
    {"SIGINT", SIGINT, {}},
    {"SIGILL", SIGILL, {}},
    {"SIGFPE", SIGFPE, {}},
    {"SIGBUS", SIGBUS, {}},
    {"SIGSEGV", SIGSEGV, {}},
    {nullptr, 0, {}}};

// 获取给定信号的先前信号处理器的地址
struct sigaction* FatalSignalHandler::getPreviousSigaction(int signum) {
  // 遍历 kSignalHandlers 数组，寻找匹配给定信号的 handler，返回其 previous 字段的地址
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return &handler->previous;
    }
  }
  // 如果未找到匹配的信号处理器，则返回 nullptr
  return nullptr;
}

// 获取给定信号的名称
const char* FatalSignalHandler::getSignalName(int signum) {
  // 遍历 kSignalHandlers 数组，寻找匹配给定信号的 handler，返回其 name 字段
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return handler->name;
    }
  }
  // 如果未找到匹配的信号名称，则返回 nullptr
  return nullptr;
}

// 调用先前注册的信号处理器
void FatalSignalHandler::callPreviousSignalHandler(
    struct sigaction* action,
    int signum,
    siginfo_t* info,
    void* ctx) {
  // 如果没有注册的处理器，直接返回
  if (!action->sa_handler) {
    return;
  }
  // 根据 sa_flags 的设置调用相应的信号处理函数
  if ((action->sa_flags & SA_SIGINFO) == SA_SIGINFO) {
    action->sa_sigaction(signum, info, ctx);
  } else {
    action->sa_handler(signum);
  }
}

// 处理堆栈跟踪信号处理器，需要锁定写入互斥量的写操作
void FatalSignalHandler::stacktraceSignalHandler(bool needsLock) {
  // 创建独占锁 ul，根据 needsLock 决定是否加锁
  std::unique_lock<std::mutex> ul(writingMutex, std::defer_lock);
  if (needsLock) {
    ul.lock();
    // 设置信号接收标志为 true
    signalReceived = true;
  }
  // 获取当前线程 ID
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  // 格式化输出堆栈跟踪信息
  std::string backtrace = fmt::format(
      "{}({}), PID: {}, Thread {}: \n {}",
      fatalSignalName,
      fatalSignum,
      ::getpid(),
      tid,
      c10::get_backtrace());
  // 输出堆栈跟踪信息到标准错误流
  std::cerr << backtrace << std::endl;
  if (needsLock) {
    ul.unlock();
    // 唤醒所有等待的写入条件变量
    writingCond.notify_all();
  }
}

// 空函数，用于致命信号处理器后处理
void FatalSignalHandler::fatalSignalHandlerPostProcess() {}

// 静态函数，用于处理致命信号的静态入口点
void FatalSignalHandler::fatalSignalHandlerStatic(int signum) {
  // 调用实例化对象的致命信号处理方法
  getInstance().fatalSignalHandler(signum);
}

// 处理致命信号的入口点
void FatalSignalHandler::fatalSignalHandler(int signum) {
  // 检查给定信号是否是预定义的信号
  const char* name = getSignalName(signum);
  if (!name) {
    return;
  }
  // 如果已经收到致命信号，则直接返回
  if (fatalSignalReceived) {
    return;
  }
  // 设置致命信号接收标志为 true
  fatalSignalReceived = true;
  // 设置致命信号的信号值和名称
  fatalSignum = signum;
  fatalSignalName = name;
  // 打开进程的任务目录，获取当前线程 ID
  DIR* procDir = opendir("/proc/self/task");
  if (procDir) {
    // 获取当前进程 ID 和当前线程 ID
    pid_t pid = getpid();
    pid_t currentTid = static_cast<pid_t>(syscall(SYS_gettid));
    struct dirent* entry = nullptr;
    // 加锁写入互斥量
    std::unique_lock<std::mutex> ul(writingMutex);
    while ((entry = readdir(procDir)) != nullptr) {
      // 遍历读取 /proc/self/task 目录下的每一个条目，直到结束
      if (entry->d_name[0] == '.') {
        // 跳过以点开头的条目（通常表示当前目录或父目录）
        continue;
      }
      // 将条目名称转换为线程 ID
      pid_t tid = atoi(entry->d_name);
      // 如果找到的线程 ID 不是当前线程的 ID
      if (tid != currentTid) {
        // 标记未接收到信号
        signalReceived = false;
        // 向特定进程的特定线程发送 SIGUSR2 信号
        syscall(SYS_tgkill, pid, tid, SIGUSR2);
        // 获取当前时间点
        auto now = std::chrono::system_clock::now();
        using namespace std::chrono_literals;
        // 在指定时间内等待写入条件变量，用 wait_until 替代 wait，以避免在 ROCm 平台上的特定问题
        if (std::cv_status::timeout == writingCond.wait_until(ul, now + 2s)) {
          // 如果超时且仍未接收到信号
          if (!signalReceived) {
            // 输出信号丢失的错误信息
            std::cerr << "signal lost waiting for stacktrace " << pid << ":"
                      << tid << std::endl;
            // 退出循环
            break;
          }
        }
      } else {
        // 如果找到当前线程的 ID，则调用栈跟踪信号处理函数
        stacktraceSignalHandler(false);
      }
    }
  } else {
    // 如果无法打开 /proc/self/task，则输出错误信息
    perror("Failed to open /proc/self/task");
  }
  // 执行致命信号处理后处理操作
  fatalSignalHandlerPostProcess();
  // 恢复信号处理程序到先前的状态
  sigaction(signum, getPreviousSigaction(signum), nullptr);
  // 触发指定信号
  raise(signum);
// Our SIGUSR2 entry point
void FatalSignalHandler::stacktraceSignalHandlerStatic(
    int signum,
    siginfo_t* info,
    void* ctx) {
  getInstance().stacktraceSignalHandler(signum, info, ctx);
}

void FatalSignalHandler::stacktraceSignalHandler(
    int signum,
    siginfo_t* info,
    void* ctx) {
  if (fatalSignalReceived) {
    // 如果已经接收到致命信号，调用堆栈跟踪信号处理器
    stacktraceSignalHandler(true);
  } else {
    // 如果尚未接收到致命信号，调用前一个 SIGUSR2 信号处理器
    // 保持当前信号处理器以便稍后接收到 usr2 信号
    callPreviousSignalHandler(&previousSigusr2, signum, info, ctx);
  }
}

// Installs SIGABRT signal handler so that we get stack traces
// from every thread on SIGABRT caused exit. Also installs SIGUSR2 handler
// so that threads can communicate with each other (be sure if you use SIGUSR2)
// to install your handler before initing caffe2 (we properly fall back to
// the previous handler if we didn't initiate the SIGUSR2).
void FatalSignalHandler::installFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = true;
  struct sigaction sa {};
  sigemptyset(&sa.sa_mask);
  // Since we'll be in an exiting situation it's possible there's memory
  // corruption, so make our own stack just in case.
  sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
  sa.sa_handler = FatalSignalHandler::fatalSignalHandlerStatic;
  // 遍历预定义的信号处理器数组，为每个信号注册处理函数
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &sa, &handler->previous)) {
      std::string str("Failed to add ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    }
  }
  // 设置 SIGUSR2 的信号处理函数为 stacktraceSignalHandlerStatic
  sa.sa_sigaction = FatalSignalHandler::stacktraceSignalHandlerStatic;
  if (sigaction(SIGUSR2, &sa, &previousSigusr2)) {
    perror("Failed to add SIGUSR2 handler!");
  }
}

void FatalSignalHandler::uninstallFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (!fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = false;
  // 移除所有已安装的信号处理器
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &handler->previous, nullptr)) {
      std::string str("Failed to remove ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    } else {
      handler->previous = {};
    }
  }
  // 恢复 SIGUSR2 的默认信号处理函数
  if (sigaction(SIGUSR2, &previousSigusr2, nullptr)) {
    perror("Failed to add SIGUSR2 handler!");
  } else {
    previousSigusr2 = {};
  }
}
#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

SignalHandler::SignalHandler(
    SignalHandler::Action SIGINT_action,
    SignalHandler::Action SIGHUP_action)
    : SIGINT_action_(SIGINT_action),
      SIGHUP_action_(SIGHUP_action),
      my_sigint_count_(sigintCount),
      my_sighup_count_(sighupCount) {
  // 注册信号处理器
  hookupHandler();
}
// 析构函数：解除信号处理器
SignalHandler::~SignalHandler() {
  unhookHandler();  // 调用私有方法解除信号处理器
}

// 返回是否自上次调用以来接收到 SIGINT 信号
bool SignalHandler::GotSIGINT() {
  uint64_t count = sigintCount;  // 读取全局计数器的值
  uint64_t localCount = my_sigint_count_.exchange(count);  // 原子操作更新本地计数器
  return (localCount != count);  // 比较更新前后本地计数器的值
}

// 返回是否自上次调用以来接收到 SIGHUP 信号
bool SignalHandler::GotSIGHUP() {
  uint64_t count = sighupCount;  // 读取全局计数器的值
  uint64_t localCount = my_sighup_count_.exchange(count);  // 原子操作更新本地计数器
  return (localCount != count);  // 比较更新前后本地计数器的值
}

// 检查是否有信号待处理，并返回相应的动作
SignalHandler::Action SignalHandler::CheckForSignals() {
  if (GotSIGHUP()) {  // 如果接收到 SIGHUP 信号
    return SIGHUP_action_;  // 返回预设的处理动作
  }
  if (GotSIGINT()) {  // 如果接收到 SIGINT 信号
    return SIGINT_action_;  // 返回预设的处理动作
  }
  return SignalHandler::Action::NONE;  // 如果没有信号待处理，则返回 NONE
}

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
// 设置是否在致命信号时打印堆栈跟踪
void FatalSignalHandler::setPrintStackTracesOnFatalSignal(bool print) {
  if (print) {
    installFatalSignalHandlers();  // 安装致命信号处理器
  } else {
    uninstallFatalSignalHandlers();  // 卸载致命信号处理器
  }
}

// 返回是否在致命信号时打印堆栈跟踪
bool FatalSignalHandler::printStackTracesOnFatalSignal() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);  // 加锁以保护共享状态
  return fatalSignalHandlersInstalled;  // 返回致命信号处理器是否已安装的状态
}
#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

} // namespace c10

#else // defined(C10_SUPPORTS_SIGNAL_HANDLER)

// TODO: 目前在非 Linux 平台上我们还不支持信号处理 - 下面是一个最小的实现，仅仅让代码编译通过。
namespace c10 {
// 构造函数：初始化信号处理器的动作及计数器
SignalHandler::SignalHandler(
    SignalHandler::Action SIGINT_action,
    SignalHandler::Action SIGHUP_action) {
  SIGINT_action_ = SIGINT_action;  // 设置 SIGINT 的处理动作
  SIGHUP_action_ = SIGHUP_action;  // 设置 SIGHUP 的处理动作
  my_sigint_count_ = 0;  // 初始化 SIGINT 的计数器
  my_sighup_count_ = 0;  // 初始化 SIGHUP 的计数器
}

// 析构函数：无操作
SignalHandler::~SignalHandler() {}

// 返回 false，表示未接收到 SIGINT 信号
bool SignalHandler::GotSIGINT() {
  return false;
}

// 返回 false，表示未接收到 SIGHUP 信号
bool SignalHandler::GotSIGHUP() {
  return false;
}

// 返回 NONE，表示没有信号需要处理
SignalHandler::Action SignalHandler::CheckForSignals() {
  return SignalHandler::Action::NONE;
}
} // namespace c10

#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)
```