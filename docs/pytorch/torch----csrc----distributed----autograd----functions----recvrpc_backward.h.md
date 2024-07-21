# `.\pytorch\torch\csrc\distributed\autograd\functions\recvrpc_backward.h`

```
#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace autograd {

// 前向声明，声明 DistAutogradContext 类
class DistAutogradContext;

// 在我们的分布式自动求导实现中，每当我们从节点接收到 RPC 时，
// 我们将向自动求导图中添加一个 'RecvRpcBackward' 自动求导函数。
// 这基本上是一个占位符函数，用于在反向传播期间将梯度传递到远程主机。
// RPC 函数的输入是这个自动求导函数的输入。
class TORCH_API RecvRpcBackward : public torch::autograd::Node {
 public:
  // 构造函数，初始化接收到的自动求导元数据、自动求导上下文、来源 worker 的 worker id 和设备映射
  explicit RecvRpcBackward(
      const AutogradMetadata& autogradMetadata,
      std::shared_ptr<DistAutogradContext> autogradContext,
      rpc::worker_id_t fromWorkerId,
      rpc::DeviceMap deviceMap);

  // 应用函数，实现自动求导的计算过程
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;

 private:
  // 接收到的自动求导元数据
  const AutogradMetadata autogradMetadata_;

  // 弱引用指向自动求导上下文，避免与上下文之间的循环依赖
  std::weak_ptr<DistAutogradContext> autogradContext_;

  // 发送 RPC 的 worker id。在反向传播期间，需要将梯度传播到这个 workerId。
  rpc::worker_id_t fromWorkerId_;

  // 用于在 RPC 上发送的张量的设备映射
  const rpc::DeviceMap deviceMap_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```