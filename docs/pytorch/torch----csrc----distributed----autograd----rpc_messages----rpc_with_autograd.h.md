# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_autograd.h`

```
#pragma once
// 预处理指令：指示编译器只包含该头文件一次

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
// 包含 Torch 分布式自动求导模块的 RPC 消息的自动求导元数据头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含 Torch 分布式 RPC 的 RPC Agent 头文件
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 Torch 分布式 RPC 的 RPC 命令基类头文件

namespace torch {
namespace distributed {
namespace autograd {

// 命名空间 torch::distributed::autograd，用于包含与 Torch 分布式自动求导相关的内容

// Represents an RPC that includes autograd information. This class basically
// wraps another `RpcCommandBase` object which represents the actual RPC and has
// additional autograd information associated with that RPC.
// 表示包含自动求导信息的 RPC。这个类基本上包装了另一个 `RpcCommandBase` 对象，
// 该对象表示实际的 RPC 并且具有与该 RPC 相关的额外自动求导信息。
class TORCH_API RpcWithAutograd final : public rpc::RpcCommandBase {
 public:
  // 当我们通过网络发送 RPC 时使用。
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      rpc::DeviceMap deviceMap = {});

  // 当我们接收到网络中的 RPC 时使用。
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      rpc::DeviceMap deviceMap = {});

  // 实现转换为消息的方法，使用右值引用。
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;

  // 从消息中创建 RpcWithAutograd 的静态方法。
  static std::unique_ptr<RpcWithAutograd> fromMessage(
      const rpc::Message& message);

  // 返回此 RPC 中涉及的张量，这些张量需要用于自动求导计算。
  std::vector<torch::Tensor>& tensors();

  // 返回自动求导元数据的引用。
  const AutogradMetadata& autogradMetadata() const;

  // 返回包装的 RPC 命令。
  RpcCommandBase& wrappedRpc();

  // 设置包装的 RPC 命令。
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

  // 移动包装的 RPC 命令。
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // 返回包装的 RPC 的消息类型。
  rpc::MessageType wrappedMessageType() const;

  // 返回发起此 RPC 的 worker id。
  rpc::worker_id_t fromWorkerId() const;

  // 返回设备映射。
  const rpc::DeviceMap& deviceMap();

 private:
  // 发起此 RPC 的 worker id，用于在反向传播过程中确定需要联系的 worker。
  rpc::worker_id_t fromWorkerId_;

  // 此调用的消息类型。
  rpc::MessageType messageType_;

  AutogradMetadata autogradMetadata_;

  // wrappedMessage_ 是从 wrappedRpc_ 构造而来的，它们互斥有效。
  // 在构造接收的 rpcWithAutograd 时使用 wrappedRpc_；
  // 在构造发送的 rpcWithAutograd 之前使用 wrappedMessage_；
  std::unique_ptr<RpcCommandBase> wrappedRpc_;

  // 表示 wrappedRpc_ 的序列化消息。主要用作缓存，避免重复序列化请求。
  // 在从消息构造接收的 rpcWithAutograd 时为 nullptr；
  // 在构造发送的 rpcWithAutograd 之前为有效。
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;

  // wrappedMessage 的消息类型，单独存储，因为不保证 wrappedMessage_ 总是有值。
  rpc::MessageType wrappedMessageType_;

  // 包含在 wrappedRpc 中需要用于自动求导的张量。
  std::vector<torch::Tensor> tensors_;

  // 传递到另一个节点的张量的设备映射。
  rpc::DeviceMap deviceMap_;
};
};

} // namespace autograd
} // namespace distributed
} // namespace torch


注释：

};  

This curly brace closes the namespace block for the `torch` namespace.


} // namespace autograd

This curly brace closes the namespace block for the `distributed` namespace, which is nested inside the `autograd` namespace.


} // namespace distributed

This curly brace closes the namespace block for the `autograd` namespace, which is nested inside the global `torch` namespace.


} // namespace torch

This curly brace closes the global `torch` namespace, which encapsulates all definitions within the `torch` library or framework.
```