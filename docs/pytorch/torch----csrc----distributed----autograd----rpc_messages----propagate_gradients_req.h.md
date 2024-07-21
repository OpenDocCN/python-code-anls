# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\propagate_gradients_req.h`

```py
#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace torch {
namespace distributed {
namespace autograd {

// 用于在分布式反向传播过程中从一个节点传播梯度到另一个节点。
// 当在反向传播执行期间遇到 `recv` 自动求导函数时，会调用此 RPC 调用。
class TORCH_API PropagateGradientsReq : public rpc::RpcCommandBase {
 public:
  // 构造函数，初始化梯度传播请求
  PropagateGradientsReq(
      const AutogradMetadata& autogradMetadata,  // 自动求导元数据
      std::vector<torch::autograd::Variable> grads,  // 梯度向量
      bool retainGraph = false);  // 是否保留自动求导图

  // 获取自动求导元数据
  const AutogradMetadata& getAutogradMetadata();

  // 获取梯度向量
  const std::vector<torch::autograd::Variable>& getGrads();

  // 序列化和反序列化方法
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<PropagateGradientsReq> fromMessage(
      const rpc::Message& message);

  // 是否保留自动求导图
  bool retainGraph();

 private:
  AutogradMetadata autogradMetadata_;  // 自动求导元数据
  std::vector<torch::autograd::Variable> grads_;  // 梯度向量
  bool retainGraph_;  // 是否保留自动求导图
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```