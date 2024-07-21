# `.\pytorch\torch\csrc\profiler\standalone\itt_observer.h`

```
#include <torch/csrc/profiler/api.h>

# 包含 Torch 的性能分析器 API 头文件


namespace torch::profiler::impl {

# 定义命名空间 `torch::profiler::impl`


void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

# 声明 `pushITTCallbacks` 函数，接受 `ProfilerConfig` 类型的参数 `config` 和无序集合 `scopes`，用于推入 Intel ITT (Instrumentation and Tracing Technology) 回调函数。


} // namespace torch::profiler::impl

# 命名空间 `torch::profiler::impl` 结束
```