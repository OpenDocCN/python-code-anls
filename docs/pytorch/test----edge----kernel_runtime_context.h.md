# `.\pytorch\test\edge\kernel_runtime_context.h`

```
#pragma once

#include "event_tracer.h"  // 包含事件跟踪器头文件

namespace torch {
namespace executor {

/**
 * 表示一个桶类型的抽象，包含许多运行时状态元素，内核作者可能希望其可用，但否则无法访问。
 *
 * 在精简模式下运行时，此类将传递给所有操作符。注意：如果在 ATen 模式下运行，将不会传递给操作符，
 * 因为这些操作符不期望接收 KernelRuntimeContext 并且不会使用它。
 *
 * 这些状态包括设置错误状态、为需要超出常量空间的操作符提供临时分配器，以及为动态形状张量提供
 * TensorResizer，使程序能够更灵活地处理张量形状。
 */
class KernelRuntimeContext {
  public:
  /**
   * 构造一个新的内核运行时上下文，可选择传入一个事件跟踪器。
   */
  KernelRuntimeContext(EventTracer* event_tracer = nullptr)
      : event_tracer_(event_tracer) {}

  /**
   * 仅供内部使用
   *
   * 返回一个指向 EventTracer 实例的指针，用于在代码生成层进行性能分析和调试日志记录。
   * 这仅用于代码生成层内部使用，用户不应访问此方法。
   */
  EventTracer* internal_event_tracer() {
    return event_tracer_;
  }

  private:
  EventTracer* event_tracer_;  // 指向事件跟踪器的指针
};

} // namespace executor
} // namespace torch
```