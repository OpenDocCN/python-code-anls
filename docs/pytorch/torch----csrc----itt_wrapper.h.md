# `.\pytorch\torch\csrc\itt_wrapper.h`

```py
#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H
// 包含 C10 库中的导出宏定义
#include <c10/macros/Export.h>

// 定义 torch::profiler 命名空间
namespace torch::profiler {
    // 声明一个函数，用于检查 ITT 是否可用，返回布尔值
    TORCH_API bool itt_is_available();
    // 声明一个函数，用于推送一个 ITT 范围，参数为消息字符串
    TORCH_API void itt_range_push(const char* msg);
    // 声明一个函数，用于弹出当前 ITT 范围
    TORCH_API void itt_range_pop();
    // 声明一个函数，用于标记一个 ITT 事件，参数为消息字符串
    TORCH_API void itt_mark(const char* msg);
} // namespace torch::profiler

// 结束 ifndef 包含指令
#endif // PROFILER_ITT_H
```