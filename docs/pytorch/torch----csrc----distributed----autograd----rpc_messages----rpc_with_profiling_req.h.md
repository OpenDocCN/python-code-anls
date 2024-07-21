# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_profiling_req.h`

```py
#pragma once
// 包含 Torch 的自动微分库中的性能分析器头文件
#include <torch/csrc/autograd/profiler.h>
// 包含 Torch 的分布式 RPC 中的消息定义头文件
#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 的分布式 RPC 中的 RPC 代理头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含 Torch 的分布式 RPC 中的 RPC 命令基类头文件
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 Torch 的分布式 RPC 中的类型定义头文件
#include <torch/csrc/distributed/rpc/types.h>

// Torch 的命名空间
namespace torch {
// Torch 分布式命名空间
namespace distributed {
// Torch 自动微分命名空间
namespace autograd {

// 定义 TORCH_API 的 RpcWithProfilingReq 类，继承自 rpc::RpcCommandBase
class TORCH_API RpcWithProfilingReq : public rpc::RpcCommandBase {
 public:
  // 构造函数，用于创建发送 RPC 的实例
  RpcWithProfilingReq(
      rpc::MessageType messageType, // RPC 消息类型
      c10::intrusive_ptr<rpc::Message> wrappedMessage, // 封装的消息指针
      torch::autograd::profiler::ProfilerConfig&& profilerConfig, // 性能分析器配置
      rpc::ProfilingId profilingKeyId); // 性能分析键 ID

  // 构造函数，用于接收 RPC 的实例
  // 从消息中创建
  RpcWithProfilingReq(
      rpc::MessageType messageType, // RPC 消息类型
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc, // 封装的 RPC 命令指针
      rpc::MessageType wrappedMessageType, // 封装的 RPC 消息类型
      std::vector<torch::Tensor> tensors, // Torch 张量向量
      torch::autograd::profiler::ProfilerConfig&& profilerConfig, // 性能分析器配置
      rpc::ProfilingId profilingKeyId); // 性能分析键 ID

  // 转换为可以通过网络发送的消息对象
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;

  // 从消息中创建 RpcWithProfilingReq 对象的静态方法
  static std::unique_ptr<RpcWithProfilingReq> fromMessage(const rpc::Message& message);

  // 获取与此命令相关的性能分析配置
  torch::autograd::profiler::ProfilerConfig getProfilingConfig() const;

  // 获取与此命令相关的全局唯一性能分析 ID
  const rpc::ProfilingId& getProfilingId() const;

  // 获取此 ProfilingRPC 包装的原始 RPC 命令
  RpcCommandBase& wrappedRpc();

  // 移动封装的 RPC 命令对象
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // 获取封装的 RPC 的消息类型
  rpc::MessageType wrappedMessageType() const;

  // 设置封装的 RPC 命令对象
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // RPC 消息类型
  const rpc::MessageType messageType_;

  // 封装的消息指针
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;

  // 封装的 RPC 命令对象指针
  std::unique_ptr<RpcCommandBase> wrappedRpc_;

  // 封装的 RPC 消息类型
  rpc::MessageType wrappedMessageType_;

  // Torch 张量向量
  std::vector<torch::Tensor> tensors_;

  // 性能分析器配置
  const torch::autograd::profiler::ProfilerConfig profilerConfig_;

  // 性能分析键 ID
  const rpc::ProfilingId profilingKeyId_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```