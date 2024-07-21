# `.\pytorch\c10\core\impl\GPUTrace.cpp`

```py
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/CallOnce.h>

namespace c10::impl {

// 定义静态原子指针，用于存储 GPU 追踪状态
std::atomic<const PyInterpreter*> GPUTrace::gpuTraceState{nullptr};

// 定义静态布尔变量，标识是否存在 GPU 追踪状态
bool GPUTrace::haveState{false};

// 设置 GPU 追踪状态的静态方法
void GPUTrace::set_trace(const PyInterpreter* trace) {
  // 定义静态的 call_once 标志
  static c10::once_flag flag;
  // 保证以下代码只会执行一次：存储追踪状态并更新 haveState 标志
  c10::call_once(flag, [&]() {
    // 使用 release 内存顺序存储追踪状态
    gpuTraceState.store(trace, std::memory_order_release);
    // 设置 haveState 为 true，表示存在 GPU 追踪状态
    haveState = true;
  });
}

} // namespace c10::impl
```