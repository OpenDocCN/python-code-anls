# `.\pytorch\torch\csrc\profiler\standalone\execution_trace_observer.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/macros/Export.h>
// 包含 c10 库中的 Export.h 头文件

#include <string>
// 包含标准字符串库

namespace torch::profiler::impl {
// 命名空间 torch::profiler::impl 开始

// 添加执行跟踪观察器作为全局回调函数，数据将写入指定的输出文件路径。
TORCH_API bool addExecutionTraceObserver(const std::string& output_file_path);
// 声明 addExecutionTraceObserver 函数，接受一个输出文件路径参数，返回一个布尔值

// 从全局回调函数中移除执行跟踪观察器。
TORCH_API void removeExecutionTraceObserver();
// 声明 removeExecutionTraceObserver 函数，无返回值

// 启用执行跟踪观察器。
TORCH_API void enableExecutionTraceObserver();
// 声明 enableExecutionTraceObserver 函数，无返回值

// 禁用执行跟踪观察器。
TORCH_API void disableExecutionTraceObserver();
// 声明 disableExecutionTraceObserver 函数，无返回值

} // namespace torch::profiler::impl
// 命名空间 torch::profiler::impl 结束
```