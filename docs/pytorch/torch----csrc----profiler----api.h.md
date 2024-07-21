# `.\pytorch\torch\csrc\profiler\api.h`

```
#pragma once

// 包含了 torch/csrc/profiler/orchestration/observer.h 头文件
#include <torch/csrc/profiler/orchestration/observer.h>

// 有些组件使用了这些符号。在我们迁移它们之前，我们必须在旧的 autograd 命名空间中镜像它们。

// 定义了命名空间 torch::autograd::profiler
namespace torch::autograd::profiler {
    // 使用了 torch::profiler::impl::ActivityType 符号
    using torch::profiler::impl::ActivityType;
    // 使用了 torch::profiler::impl::getProfilerConfig 符号
    using torch::profiler::impl::getProfilerConfig;
    // 使用了 torch::profiler::impl::ProfilerConfig 符号
    using torch::profiler::impl::ProfilerConfig;
    // 使用了 torch::profiler::impl::profilerEnabled 符号
    using torch::profiler::impl::profilerEnabled;
    // 使用了 torch::profiler::impl::ProfilerState 符号
    using torch::profiler::impl::ProfilerState;
} // namespace torch::autograd::profiler
```