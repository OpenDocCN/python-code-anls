# `.\pytorch\torch\csrc\jit\backends\backend_exception.h`

```
#pragma once
// 包含 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 定义 c10 命名空间
namespace c10 {

// BackendRuntimeException 类继承自 c10::Error
class TORCH_API BackendRuntimeException : public c10::Error {
 public:
  // 构造函数，用于抛出异常
  BackendRuntimeException(
      SourceLocation loc,  // 异常发生位置信息
      std::string msg,     // 异常信息
      int64_t debug_handle // 调试句柄
      ) : c10::Error(loc, msg) {
    // 将 debug_handle 添加到 debug_handles 向量中
    debug_handles.push_back(debug_handle);
  }
  
  // 添加调试句柄到异常堆栈
  // 用于在重新抛出异常时推入另一个调试句柄，用于完整堆栈追踪
  void pushDebugHandle(int64_t debug_handle) {
    debug_handles.push_back(debug_handle);
  }
  
  // 获取异常堆栈的所有调试句柄
  const std::vector<int64_t>& getDebugHandles() {
    return debug_handles;
  }

 private:
  // 存储调试句柄堆栈
  std::vector<int64_t> debug_handles;
};

} // namespace c10

// 定义宏 TORCH_DELEGATED_BACKEND_THROW，用于抛出 BackendRuntimeException 异常
#define TORCH_DELEGATED_BACKEND_THROW(cond, msg, debug_handle) \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                        \
    throw ::c10::BackendRuntimeException(                      \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        msg,                                                   \
        debug_handle);                                         \
  }

// 定义宏 TORCH_DELEGATED_BACKEND_RETHROW，用于重新抛出异常并添加调试句柄
#define TORCH_DELEGATED_BACKEND_RETHROW(e, debug_handle) \
  do {                                                   \
    e.pushDebugHandle(debug_handle);                     \
    throw;                                               \
  } while (false)

// 定义调试句柄常量
#define DEBUG_HANDLE_UNKNOWN -1
```