# `.\pytorch\torch\csrc\profiler\standalone\nvtx_observer.h`

```
#include <torch/csrc/profiler/api.h>
// 引入 Torch 的性能分析工具头文件

namespace torch::profiler::impl {
// 进入 torch::profiler::impl 命名空间

void pushNVTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);
// 声明一个函数 pushNVTXCallbacks，接受 ProfilerConfig 和无序集合 at::RecordScope 的参数

} // namespace torch::profiler::impl
// 退出 torch::profiler::impl 命名空间
```