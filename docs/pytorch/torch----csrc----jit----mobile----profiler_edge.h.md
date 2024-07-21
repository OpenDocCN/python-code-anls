# `.\pytorch\torch\csrc\jit\mobile\profiler_edge.h`

```
#pragma once
// 使用预处理指令#pragma once确保头文件只被编译一次

#include <torch/csrc/autograd/profiler_kineto.h>
// 包含Kineto性能分析器相关头文件

#include <torch/csrc/jit/mobile/module.h>
// 包含移动端模块相关头文件

namespace torch {
namespace jit {
namespace mobile {

// 如果没有可用的Kineto，那么边缘分析器将无法工作，因为它依赖于Kineto
#ifdef USE_KINETO
};
// 如果定义了USE_KINETO宏，则进入torch::jit::mobile命名空间

// 定义一个TORCH_API函数getCurrentEdgeProfiler()，返回KinetoEdgeCPUProfiler指针
TORCH_API KinetoEdgeCPUProfiler* getCurrentEdgeProfiler();

// 定义一个宏RECORD_BACKEND_EVENT_TO_EDGE_PROFILER，用于记录后端事件到Edge Profiler中
#define RECORD_BACKEND_EVENT_TO_EDGE_PROFILER(                               \
    start_time_us, end_time_us, debug_handle, event_name, backend_name)      \
  if (mobile::getCurrentEdgeProfiler()) {                                    \
    mobile::getCurrentEdgeProfiler()->recordBackendEvent(                    \
        start_time_us, end_time_us, debug_handle, event_name, backend_name); \
  }
// 如果当前有Edge Profiler实例可用，则调用其recordBackendEvent方法记录后端事件

// 定义一个宏RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER，用于记录后端内存事件到Edge Profiler中
#define RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER(              \
    ptr, alloc_size, total_allocated, total_reserved, device)      \
  if (mobile::getCurrentEdgeProfiler()) {                          \
    mobile::getCurrentEdgeProfiler()->recordBackendMemoryEvent(    \
        ptr, alloc_size, total_allocated, total_reserved, device); \
  }
// 如果当前有Edge Profiler实例可用，则调用其recordBackendMemoryEvent方法记录后端内存事件

#else

// 如果没有定义USE_KINETO宏，则以下宏定义为空

// 定义一个空的宏RECORD_BACKEND_EVENT_TO_EDGE_PROFILER
#define RECORD_BACKEND_EVENT_TO_EDGE_PROFILER( \
    start_time_us, end_time_us, debug_handle, event_name, backend_name)

// 定义一个空的宏RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER
#define RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER( \
    ptr, alloc_size, total_allocated, total_reserved, device)
#endif

} // namespace mobile
} // namespace jit
} // namespace torch
```