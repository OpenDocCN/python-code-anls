# `.\pytorch\c10\util\signal_handler.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <atomic>
// 原子操作库，提供线程安全的操作

#include <condition_variable>
// 条件变量，用于线程间的同步和通信

#include <csignal>
// C 信号处理的相关定义，用于处理系统信号

#include <cstdint>
// 标准整数类型定义，如 uint64_t

#include <mutex>
// 互斥量，用于保护共享数据的并发访问

#include <c10/macros/Export.h>
// 导出宏定义，用于符号导出或导入控制

#if defined(__APPLE__)
#define C10_SUPPORTS_SIGNAL_HANDLER
// 定义支持信号处理的宏，针对苹果系统
#elif defined(__linux__) && !defined(C10_DISABLE_SIGNAL_HANDLERS)
#define C10_SUPPORTS_FATAL_SIGNAL_HANDLERS
#define C10_SUPPORTS_SIGNAL_HANDLER
// 定义支持致命信号处理的宏，并支持通用信号处理，针对 Linux 平台
#endif

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
#include <pthread.h>
// 如果支持致命信号处理，则包含 POSIX 线程头文件
#endif

namespace c10 {

class C10_API SignalHandler {
 public:
  enum class Action { NONE, STOP };
  // 枚举类型 Action，定义了信号处理器的动作：无动作或停止动作

  // 构造函数。指定接收到信号时的动作。
  SignalHandler(Action SIGINT_action, Action SIGHUP_action);

  // 析构函数，释放资源
  ~SignalHandler();

  // 检查是否有信号待处理，返回采取的动作
  Action CheckForSignals();

  // 返回是否接收到 SIGINT 信号
  bool GotSIGINT();

  // 返回是否接收到 SIGHUP 信号
  bool GotSIGHUP();

  Action SIGINT_action_; // SIGINT 信号的处理动作
  Action SIGHUP_action_; // SIGHUP 信号的处理动作
  std::atomic<uint64_t> my_sigint_count_; // 接收到 SIGINT 信号的计数器
  std::atomic<uint64_t> my_sighup_count_; // 接收到 SIGHUP 信号的计数器
};

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
// 如果支持致命信号处理
class C10_API FatalSignalHandler {
  // This class manages fatal signal handlers for handling fatal signals in C10 API.
  // Previous fatal signal handlers will still be invoked when a signal is raised.

 public:
  // Sets whether to print stack traces on receiving fatal signals.
  C10_API void setPrintStackTracesOnFatalSignal(bool print);

  // Retrieves the current setting for printing stack traces on fatal signals.
  C10_API bool printStackTracesOnFatalSignal();

  // Returns a singleton instance of FatalSignalHandler.
  static FatalSignalHandler& getInstance();

  // Destructor for FatalSignalHandler.
  virtual ~FatalSignalHandler();

 protected:
  // Protected constructor for initializing FatalSignalHandler.
  explicit FatalSignalHandler();

 private:
  // Installs fatal signal handlers.
  void installFatalSignalHandlers();

  // Uninstalls fatal signal handlers.
  void uninstallFatalSignalHandlers();

  // Static handler for fatal signals (entry point).
  static void fatalSignalHandlerStatic(int signum);

  // Handler for fatal signals.
  void fatalSignalHandler(int signum);

  // Additional processing after handling a fatal signal.
  virtual void fatalSignalHandlerPostProcess();

  // Retrieves the previous signal action for a given signal number.
  struct sigaction* getPreviousSigaction(int signum);

  // Retrieves the name of a signal based on its signal number.
  const char* getSignalName(int signum);

  // Calls the previous signal handler for a given signal.
  void callPreviousSignalHandler(struct sigaction* action, int signum,
                                 siginfo_t* info, void* ctx);

  // Handler for stacktrace signals, with locking option.
  void stacktraceSignalHandler(bool needsLock);

  // Static handler for stacktrace signals (entry point).
  static void stacktraceSignalHandlerStatic(int signum, siginfo_t* info,
                                            void* ctx);

  // Handler for stacktrace signals.
  void stacktraceSignalHandler(int signum, siginfo_t* info, void* ctx);

  // Mutex to protect critical sections related to installation of signal handlers.
  std::mutex fatalSignalHandlersInstallationMutex;

  // Flag indicating if fatal signal handlers are installed.
  bool fatalSignalHandlersInstalled;

  // Holds the previous SIGUSR2 signal action for fallback purposes.
  struct sigaction previousSigusr2 {};

  // Atomic flag indicating if a fatal signal has been received.
  std::atomic<bool> fatalSignalReceived;

  // Name of the fatal signal received.
  const char* fatalSignalName;

  // Signal number of the fatal signal received.
  int fatalSignum = -1;

  // Condition variable to wait for completion of writing stack traces by other threads.
  std::condition_variable writingCond;

  // Mutex to protect critical sections related to writing stack traces.
  std::mutex writingMutex;

  // Flag indicating if a signal has been received and processed by other threads.
  bool signalReceived;

  // Structure to store information about each signal handler.
  struct signal_handler {
    const char* name;
    int signum;
    struct sigaction previous;
  };

  // Array of signal_handler structures defining supported signal handlers.
  // NOLINTNEXTLINE(*c-arrays*)
  static signal_handler kSignalHandlers[];
};

#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)

} // namespace c10
```