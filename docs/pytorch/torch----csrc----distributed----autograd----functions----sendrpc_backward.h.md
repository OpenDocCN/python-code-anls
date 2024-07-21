# `.\pytorch\torch\csrc\distributed\autograd\functions\sendrpc_backward.h`

```
#pragma once

#include <torch/csrc/autograd/function.h>

namespace torch {
namespace distributed {
namespace autograd {

// 在我们的分布式自动求导实现中，每当我们从一个节点发送RPC到另一个节点时，
// 我们会向自动求导图中添加一个'SendRpcBackward'自动求导函数。
// 这个函数实际上是一个占位符，用于在反向传播时在当前工作节点上启动自动求导引擎。
// 该函数的边是RPC方法的输入。
//
// 在反向传播过程中，这个函数被排队以在自动求导引擎中执行，最终运行其余的自动求导图。
struct TORCH_API SendRpcBackward : public torch::autograd::Node {
 public:
  // 应用函数，接受输入并返回变量列表
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& inputs) override;

  // SendRpcBackward实际上是本地节点上自动求导图的根。因此，它不接收任何“inputs”，
  // 而是RPC框架将梯度传递给这个函数以启动本地自动求导计算。
  void setGrads(const torch::autograd::variable_list& grads);

  // 获取函数的梯度。
  const torch::autograd::variable_list& getGrads() const;

 private:
  // 存储梯度的变量列表
  torch::autograd::variable_list grads_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```