# `.\pytorch\torch\csrc\distributed\rpc\metrics\registry.cpp`

```py
#include <torch/csrc/distributed/rpc/metrics/RpcMetricsHandler.h> // 引入RpcMetricsHandler.h头文件，用于RPC指标处理

namespace torch {
namespace distributed {
namespace rpc {
// 定义RpcMetricsHandlerRegistry注册表，注册RpcMetricsHandler类型的处理器
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);
} // namespace rpc
} // namespace distributed
} // namespace torch
```