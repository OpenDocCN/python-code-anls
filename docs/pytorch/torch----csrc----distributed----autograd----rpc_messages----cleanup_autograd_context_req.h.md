# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\cleanup_autograd_context_req.h`

```py
#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

// 声明命名空间 torch::distributed::autograd

namespace torch {
namespace distributed {
namespace autograd {

// 用于请求其他工作节点清理其自动求导上下文的类。
class TORCH_API CleanupAutogradContextReq : public rpc::RpcCommandBase {
 public:
  // 构造函数，接受一个 context_id 参数，用于指定要清理的自动求导上下文的ID。
  explicit CleanupAutogradContextReq(int64_t context_id);
  
  // 实现消息序列化和反序列化的方法。
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  
  // 从给定消息中创建 CleanupAutogradContextReq 对象的静态方法。
  static std::unique_ptr<CleanupAutogradContextReq> fromMessage(
      const rpc::Message& message);

  // 返回当前对象关联的上下文ID。
  int64_t getContextId();

 private:
  // 内部存储的上下文ID。
  int64_t context_id_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```