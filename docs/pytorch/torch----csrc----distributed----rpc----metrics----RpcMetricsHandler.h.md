# `.\pytorch\torch\csrc\distributed\rpc\metrics\RpcMetricsHandler.h`

```py
#pragma once
#include <c10/util/Registry.h>
#include <string>

namespace torch {
namespace distributed {
namespace rpc {

// 所有指标均以以下键值前缀开头。
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char kRpcMetricsKeyPrefix[] = "torch.distributed.rpc.";

// RPC-based分布式训练的时间序列指标记录API。
// 实现此类的类应提供线程安全性，以便可以从多个线程记录指标而无需用户协调序列化。
class RpcMetricsHandler {
 public:
  // 累积指定名称的指标值，用于随时间计算聚合统计信息。
  virtual void accumulateMetric(const std::string& name, double value) = 0;
  // 递增名称给定的指标的计数。
  virtual void incrementMetric(const std::string& name) = 0;
  virtual ~RpcMetricsHandler() = default;
};

// 指标处理的配置结构体。
struct RpcMetricsConfig {
  explicit RpcMetricsConfig(std::string handlerName, bool enabled)
      : handlerName_(std::move(handlerName)), enabled_(enabled) {}

  // 处理器名称
  std::string handlerName_;
  // 是否启用指标导出。
  bool enabled_;
};

// RpcMetricsHandler的不同实现的注册表。实现上述接口的类应使用此注册表注册实现。
C10_DECLARE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);

} // namespace rpc
} // namespace distributed
} // namespace torch
```