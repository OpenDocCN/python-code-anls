# `.\pytorch\c10\util\AbortHandler.h`

```py
namespace c10 {
// AbortHandlerHelper 类用于管理终止处理程序
class AbortHandlerHelper {
 public:
  // 返回 AbortHandlerHelper 的单例实例
  static AbortHandlerHelper& getInstance() {
#ifdef _WIN32
    // 在 Windows 下，使用线程本地存储来确保线程安全
    thread_local
#endif // _WIN32
        static AbortHandlerHelper instance;
    return instance;
  }

  // 设置终止处理程序
  void set(std::terminate_handler handler) {
    std::lock_guard<std::mutex> lk(mutex);
    // 如果还未初始化，则设置新的终止处理程序
    if (!inited) {
      prev = std::set_terminate(handler);
      curr = std::get_terminate();
      inited = true;
    }
  }

  // 获取先前的终止处理程序
  std::terminate_handler getPrev() const {
    return prev;
  }

 private:
  std::terminate_handler prev = nullptr; // 先前的终止处理程序
  std::terminate_handler curr = nullptr; // 当前的终止处理程序
  bool inited = false; // 是否已经初始化
  std::mutex mutex; // 互斥锁，用于保护设置操作
  AbortHandlerHelper() = default; // 构造函数，默认实现
  ~AbortHandlerHelper() {
    // 析构函数，仅在当前实例为当前终止处理程序时才恢复先前的处理程序
    if (inited && curr == std::get_terminate()) {
      std::set_terminate(prev);
    }
  }

 public:
  AbortHandlerHelper(AbortHandlerHelper const&) = delete; // 禁用拷贝构造函数
  void operator=(AbortHandlerHelper const&) = delete; // 禁用赋值运算符
};

namespace detail {
// 内部命名空间 detail，定义了终止处理程序的具体实现
C10_ALWAYS_INLINE void terminate_handler() {
  // 打印未处理异常的信息和堆栈跟踪
  std::cout << "Unhandled exception caught in c10/util/AbortHandler.h" << '\n';
  auto backtrace = get_backtrace(); // 获取堆栈跟踪信息
  std::cout << backtrace << '\n' << std::flush; // 打印堆栈跟踪信息
  auto prev_handler = AbortHandlerHelper::getInstance().getPrev(); // 获取先前的终止处理程序
  if (prev_handler) {
    prev_handler(); // 调用先前的终止处理程序
  } else {
    std::abort(); // 否则调用标准库的终止处理函数
  }
}
} // namespace detail

// 设置自定义的终止处理程序
C10_ALWAYS_INLINE void set_terminate_handler() {
  bool use_custom_terminate = false;
  // 在 Windows 下，默认启用自定义终止处理程序
#ifdef _WIN32
  use_custom_terminate = true;
#endif // _WIN32
  auto result = c10::utils::check_env("TORCH_CUSTOM_TERMINATE"); // 检查环境变量 TORCH_CUSTOM_TERMINATE
  if (result != std::nullopt) {
    use_custom_terminate = result.value(); // 根据环境变量设置是否使用自定义终止处理程序
  }
  if (use_custom_terminate) {
    // 如果需要使用自定义终止处理程序，则设置为 detail 命名空间中定义的处理函数
    AbortHandlerHelper::getInstance().set(detail::terminate_handler);
  }
}
} // namespace c10
```