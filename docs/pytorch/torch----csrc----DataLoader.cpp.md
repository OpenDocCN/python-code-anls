# `.\pytorch\torch\csrc\DataLoader.cpp`

```
// Torch 数据加载模块的头文件，用于实现数据加载器的相关功能
#include <torch/csrc/DataLoader.h>

// 与 `torch/utils/data/_utils/signal_handling.py` 文件一起，以下代码尝试在工作进程由于错误或关键信号而终止时向用户提供错误消息。
// 更多细节请参见“NOTE [ Signal handling in multiprocessing data loading ]”。

// TODO: 以下功能在 Windows 平台不起作用。具体包括 sigaction、waitid 调用和 SIGCHLD 处理程序。目前为 Windows 提供了虚拟实现。

#ifndef _WIN32

// 引入异常处理相关的 Torch 头文件
#include <torch/csrc/Exceptions.h>
// 引入 Python 数字处理相关的 Torch 头文件
#include <torch/csrc/utils/python_numbers.h>

// 引入 C++ 标准库中范围迭代器相关的头文件
#include <c10/util/irange.h>
// 引入 fmt 库中格式化输出相关的头文件
#include <fmt/format.h>

// 引入 POSIX 标准中信号处理相关的头文件
#include <sys/wait.h>
#include <csignal>
// 引入 C++ 标准库中映射和集合相关的头文件
#include <map>
#include <set>
// 引入 C++ 标准库中字符串流相关的头文件
#include <sstream>

// 使用 Torch 命名空间
using namespace torch;

// 在工作进程执行实际工作之前，应该注册关键信号处理程序。
// 该处理程序将调用默认处理程序，以便从主进程中获取终止信息。
// Python 中相应的处理函数是 _set_worker_signal_handlers()。
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                    \
  static void HANDLER_NAME(int sig, siginfo_t* info, void* ctx) {          \
    auto _w =                                                              \
        write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char)); \
    (void)_w;                                                              \
    struct sigaction sa {};                                                \
    sa.sa_handler = SIG_DFL;                                               \
    sa.sa_flags = 0;                                                       \
    if (sigemptyset(&sa.sa_mask) != 0 ||                                   \
        sigaction(SIGNAL, &sa, nullptr) != 0) {                            \
      _exit(EXIT_FAILURE);                                                 \
    } else {                                                               \
      raise(SIGNAL);                                                       \
    }                                                                      \
  }

// signal(2) 在不同系统上的实现不一致，因此使用 sigaction 进行信号处理。
// 参考文档：http://man7.org/linux/man-pages/man2/signal.2.html
static inline void setSignalHandler(
    int signal,
    void (*handler)(int, siginfo_t*, void*),
    struct sigaction* old_sa_ptr) {
  struct sigaction sa {};
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 ||
      sigaction(signal, &sa, old_sa_ptr) != 0) {
    // 如果设置信号处理程序失败，则抛出运行时异常
    std::ostringstream oss;
    oss << "An error occurred while setting handler for " << strsignal(signal)
        << ".";
    throw std::runtime_error(oss.str());
  }
}

// 定义处理 SIGBUS 信号的处理函数及其相关错误消息
SIGNAL_HANDLER(
    SIGBUS,
    handler_SIGBUS,
    "ERROR: Unexpected bus error encountered in worker. "
    "This might be caused by insufficient shared memory (shm).\n");

// 定义处理 SIGSEGV 信号的处理函数
SIGNAL_HANDLER(
    SIGSEGV,
    handler_SIGSEGV,
    "ERROR: Segmentation fault encountered in worker. "
    "This might be caused by invalid memory access.\n");
    # 打印错误消息，指示在工作进程中遇到意外的分段错误
        "ERROR: Unexpected segmentation fault encountered in worker.\n");
// 设置信号处理函数 handler_SIGFPE 处理 SIGFPE 信号，打印浮点异常信息
SIGNAL_HANDLER(
    SIGFPE,
    handler_SIGFPE,
    "ERROR: Unexpected floating-point exception encountered in worker.\n");

// 当 DataLoader 方法发生错误且 Python 开始退出时，错误跟踪将保持 loader 活动状态，
// Python 可能会在删除 loader 对象之前先终止子进程。此时 DataLoader.__del__ 中的清理方法尚未调用，
// 因此 SIGCHILD 可能会打印错误，指示 worker 被 SIGTERM 终止。为了避免这种情况，
// 我们在此处抑制主 loader 进程的 SIGTERM，通过 _exit(EXIT_SUCCESS)。请注意，
// 如果我们以非零代码退出，则 loader 的 SIGCHLD 处理程序可能会再次报告 RuntimeError，
// 从而打破整个目的。
static void handler_SIGTERM(int sig, siginfo_t* info, void* ctx) {
  // 如果信号来源于父进程，直接退出程序
  if (info->si_pid == getppid()) {
    _exit(EXIT_SUCCESS);
  }
  struct sigaction sa {};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  // 清空 sa.sa_mask 并将 SIGTERM 信号设置为默认处理程序
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGTERM, &sa, nullptr) != 0) {
    _exit(EXIT_FAILURE);
  } else {
    // 发送 SIGTERM 信号给当前进程
    raise(SIGTERM);
  }
}

// 设置 worker 进程的信号处理函数，处理 SIGBUS、SIGSEGV、SIGTERM、SIGFPE 信号
static PyObject* THPModule_setWorkerSignalHandlers(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 设置 SIGBUS 信号处理函数为 handler_SIGBUS
  setSignalHandler(SIGBUS, &handler_SIGBUS, nullptr);
  // 设置 SIGSEGV 信号处理函数为 handler_SIGSEGV
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, nullptr);
  // 设置 SIGTERM 信号处理函数为 handler_SIGTERM
  setSignalHandler(SIGTERM, &handler_SIGTERM, nullptr);
  // 设置 SIGFPE 信号处理函数为 handler_SIGFPE
  setSignalHandler(SIGFPE, &handler_SIGFPE, nullptr);
  // 调用 setDataLoaderSignalHandlers 函数，但此处为弱引用，实际上不执行任何操作
  setDataLoaderSignalHandlers();
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 存储 worker 进程的 PID 和对应的 PID 集合的映射关系
static std::map<int64_t, std::set<pid_t>> worker_pids = {};

// 检查是否有任何 worker 进程失败的错误处理函数
static PyObject* THPModule_errorIfAnyWorkerFails(
    PyObject* module,
    PyObject* noargs) {
  HANDLE_TH_ERRORS

  // 只检查我们关心的 PID 集合
  for (auto& w : worker_pids) {
    auto& pid_set = w.second;
    for (auto worker_pid : pid_set) {
        // 使用 waitid 而不是 waitpid，以便可以设置 NOWAIT，使得 Python 和其他处理程序可以获取有关子进程的任何所需信息。
        siginfo_t infop{};
        infop.si_pid = 0;
        auto error =
            waitid(P_PID, worker_pid, &infop, WEXITED | WNOHANG | WNOWAIT);
        // 忽略错误和没有可等待子进程的情况
        if (error < 0 || infop.si_pid == 0)
            continue;
        if (infop.si_code == CLD_EXITED &&
            infop.si_status != EXIT_SUCCESS) { // 子进程以错误状态退出
            std::ostringstream oss;
            oss << "DataLoader worker (pid " << worker_pid << ") exited "
                << "unexpectedly with exit code " << infop.si_status << ". "
                << "Details are lost due to multiprocessing. Rerunning with "
                << "num_workers=0 may give better error trace.";
            // 这是必要的。否则，运行时错误会终止其他工作进程，并再次触发此异常。
            pid_set.clear();
            throw std::runtime_error(oss.str());
        } else if (
            infop.si_code == CLD_KILLED ||
            infop.si_code == CLD_DUMPED) { // 子进程被信号终止
            std::ostringstream oss;
            oss << "DataLoader worker (pid " << worker_pid << ") is killed "
                << "by signal: " << strsignal(infop.si_status) << ". ";
            if (infop.si_status == SIGBUS) {
                oss << "It is possible that dataloader's workers are out of shared memory. "
                    << "Please try to raise your shared memory limit.";
            }
            // 这是必要的。否则，运行时错误会终止其他工作进程，并再次触发此异常。
            pid_set.clear();
            throw std::runtime_error(oss.str());
        }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
// DataLoaderMethods 定义了一个 PyMethodDef 结构体数组，用于定义与数据加载器相关的 Python 方法
PyMethodDef DataLoaderMethods[] = {
    // 定义名为 _set_worker_signal_handlers 的 Python 方法，对应 C++ 中的 THPModule_setWorkerSignalHandlers 函数
    {"_set_worker_signal_handlers",
     THPModule_setWorkerSignalHandlers,
     METH_NOARGS,  // 方法不需要参数
     nullptr},     // 没有额外的文档字符串

    // 定义名为 _set_worker_pids 的 Python 方法，对应 C++ 中的 THPModule_setWorkerPIDs 函数
    {"_set_worker_pids",
     THPModule_setWorkerPIDs,
     METH_VARARGS,  // 方法接受一个元组作为参数
     nullptr},      // 没有额外的文档字符串

    // 定义名为 _remove_worker_pids 的 Python 方法，对应 C++ 中的 THPModule_removeWorkerPIDs 函数
    {"_remove_worker_pids",
     THPModule_removeWorkerPIDs,
     METH_O,   // 方法接受一个对象作为参数
     nullptr}, // 没有额外的文档字符串

    // 定义名为 _error_if_any_worker_fails 的 Python 方法，对应 C++ 中的 THPModule_errorIfAnyWorkerFails 函数
    {"_error_if_any_worker_fails",
     THPModule_errorIfAnyWorkerFails,
     METH_NOARGS,  // 方法不需要参数
     nullptr},     // 没有额外的文档字符串

    {nullptr, nullptr, 0, nullptr}  // 结束方法定义的标志
};
```