# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_profiling_resp.h`

```
#pragma once
// 包含 Torch 自动求导模块中的性能分析器头文件
#include <torch/csrc/autograd/profiler.h>
// 包含 Torch 分布式 RPC 模块中的消息定义头文件
#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 分布式 RPC 模块中的 RPC Agent 头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含 Torch 分布式 RPC 模块中的 RPC 命令基类头文件
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 Torch 分布式 RPC 模块中的类型定义头文件
#include <torch/csrc/distributed/rpc/types.h>

// 定义 Torch 命名空间
namespace torch {
// 定义分布式命名空间
namespace distributed {
// 定义自动求导命名空间
namespace autograd {

// 定义一个派生自 RpcCommandBase 的 RpcWithProfilingResp 类
class TORCH_API RpcWithProfilingResp : public rpc::RpcCommandBase {
 public:
  // 构造函数，用于发送经过性能分析的 RPC
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);

  // 构造函数，用于接收 RPC。在接收到的消息转换时使用。
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);

  // 覆盖基类方法，将对象转换为消息
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;

  // 从消息中创建 RpcWithProfilingResp 对象的静态方法
  static std::unique_ptr<RpcWithProfilingResp> fromMessage(
      const rpc::Message& message);

  // 获取远程事件列表
  std::vector<torch::autograd::profiler::LegacyEvent> getProfiledEvents() const;

  // 获取与此命令对应的全局唯一性能分析 ID
  const rpc::ProfilingId& getProfilingId() const;

  // 获取此 ProfilingRPC 封装的原始 RPC
  RpcCommandBase& wrappedRpc();

  // 移动封装的 RPC
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // 获取封装的 RPC 的消息类型
  rpc::MessageType wrappedMessageType() const;

  // 设置此 RPC 的封装 RPC
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // RPC 消息类型
  const rpc::MessageType messageType_;
  // 封装的消息
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;
  // 封装的 RPC 对象
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  // 封装的 RPC 消息类型
  rpc::MessageType wrappedMessageType_;
  // 相关张量
  std::vector<torch::Tensor> tensors_;
  // 性能分析事件列表
  const std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents_;
  // 性能分析 ID
  const rpc::ProfilingId profilingId_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```