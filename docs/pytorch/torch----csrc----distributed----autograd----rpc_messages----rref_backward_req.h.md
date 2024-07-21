# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rref_backward_req.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 分布式 RPC 消息头文件

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 Torch 分布式 RPC 命令基类头文件

#include <torch/csrc/distributed/rpc/types.h>
// 包含 Torch 分布式 RPC 类型定义头文件

namespace torch {
namespace distributed {
namespace autograd {

// Torch 分布式自动求导命名空间

// Internal system RPC to invoke distributed backward pass on remote nodes when
// 'rref.backward()' is invoked.
// 内部系统 RPC，在调用 'rref.backward()' 时在远程节点上触发分布式反向传播。

class TORCH_API RRefBackwardReq : public rpc::RpcCommandBase {
  // RRefBackwardReq 类继承自 rpc::RpcCommandBase

 public:
  RRefBackwardReq(
      const rpc::RRefId& rrefId,
      int64_t autogradContextId,
      bool retainGraph = false);
  // 构造函数，用于初始化 RRefBackwardReq 对象的成员变量

  const rpc::RRefId& getRRefId() const;
  // 获取 RRefId 成员变量的常引用

  int64_t getAutogradContextId() const;
  // 获取 autogradContextId 成员变量的值

  bool retainGraph() const;
  // 获取 retainGraph 成员变量的值

  // Serialization and deserialization methods.
  // 序列化和反序列化方法。

  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  // 实现序列化为消息的方法，使用 C++11 右值引用语法

  static std::unique_ptr<RRefBackwardReq> fromMessage(
      const rpc::Message& message);
  // 从消息中反序列化出 RRefBackwardReq 对象的静态方法

 private:
  const rpc::RRefId rrefId_;
  const int64_t autogradContextId_;
  const bool retainGraph_;
  // 成员变量，分别存储 RRef 的 ID、自动求导上下文 ID、是否保留计算图的标志
};

} // namespace autograd
} // namespace distributed
} // namespace torch
// Torch 分布式自动求导命名空间结束
```