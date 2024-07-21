# `.\pytorch\torch\csrc\jit\mobile\model_tracer\OperatorCallTracer.h`

```
#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>

namespace torch {
namespace jit {
namespace mobile {

/* OperatorCallTracer 类处理回调函数的附加和移除，用于追踪调用 ATen（和其他）PyTorch 操作符，这些操作符通过 Dispatcher 被调用。

   可以通过 getCalledOperators() 函数获取被调用的操作符集合（op_name.overload_name）。

   注意：这个类不是线程安全的，也不是可重入的，不应该跨多个执行线程使用。
*/
struct OperatorCallTracer final {
  at::CallbackHandle handle_;

  // 构造函数，用于初始化 OperatorCallTracer 对象
  OperatorCallTracer();

  // 静态函数，返回被调用的操作符集合的引用
  static c10::Synchronized<std::set<std::string>>& getCalledOperators() {
    // 静态局部变量，用于保存被调用操作符的集合，保证唯一性和同步访问
    static c10::Synchronized<std::set<std::string>> called_operators_;
    return called_operators_;
  }

  // 析构函数，移除回调函数，确保资源释放
  ~OperatorCallTracer() {
    at::removeCallback(handle_);
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
```