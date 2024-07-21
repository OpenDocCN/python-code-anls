# `.\pytorch\torch\csrc\distributed\autograd\functions\recvrpc_backward.cpp`

```
// 包含头文件，用于 ATen 核心功能、C10 实用工具和 Torch 分布式自动求导相关模块
#include <ATen/core/functional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

// 定义命名空间 torch::distributed::autograd，用于分布式自动求导相关代码
namespace torch {
namespace distributed {
namespace autograd {

// 使用 torch::autograd 命名空间下的 Variable 和 variable_list 类型
using torch::autograd::Variable;
using torch::autograd::variable_list;

// RecvRpcBackward 类的构造函数，接受自动求导元数据、自动求导上下文、发送方 worker ID 和设备映射
RecvRpcBackward::RecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    ContextPtr autogradContext,
    rpc::worker_id_t fromWorkerId,
    rpc::DeviceMap deviceMap)
    : autogradMetadata_(autogradMetadata), // 初始化成员变量 autogradMetadata_
      autogradContext_(std::move(autogradContext)), // 初始化成员变量 autogradContext_
      fromWorkerId_(fromWorkerId), // 初始化成员变量 fromWorkerId_
      deviceMap_(std::move(deviceMap)) {} // 初始化成员变量 deviceMap_

// apply 方法，接受一个变量列表 grads，用于处理接收到的梯度信息
variable_list RecvRpcBackward::apply(variable_list&& grads) {
  std::vector<Variable> outputGrads; // 创建存储输出梯度的变量向量

  // 遍历传入的 grads 变量列表
  for (const auto i : c10::irange(grads.size())) {
    const auto& grad = grads[i]; // 获取第 i 个梯度变量的引用

    // 如果梯度变量 grad 已定义（非空）
    if (grad.defined()) {
      outputGrads.emplace_back(grad); // 将其添加到 outputGrads 中
    } else {
      // 否则，对于没有梯度的张量，放入与输入元数据相同形状的零张量
      outputGrads.emplace_back(input_metadata(i).zeros_like());
    }
  }

  auto sharedContext = autogradContext_.lock(); // 获取自动求导上下文的弱引用

  // 检查自动求导上下文是否有效，如果无效则抛出错误
  TORCH_CHECK(
      sharedContext,
      c10::str(
          "Autograd context no longer valid! This usually ",
          "means the autograd context was cleaned up by a different thread due ",
          "to an error before RecvRcpBackward had a chance to run"));

  // 创建 PropagateGradientsReq 对象，用于传播梯度请求
  PropagateGradientsReq gradCall(
      autogradMetadata_,
      outputGrads,
      sharedContext->retrieveGraphTask()->keep_graph_);

  // 获取当前的 RPC 代理对象
  auto rpcAgent = rpc::RpcAgent::getCurrentRpcAgent();

  // 发送梯度数据到指定节点，并将未来对象记录在自动求导上下文中
  auto jitFuture = rpcAgent->send(
      rpcAgent->getWorkerInfo(fromWorkerId_),
      std::move(gradCall).toMessage(),
      rpc::kUnsetRpcTimeout,
      deviceMap_);

  // 在上下文中添加未完成的 RPC 请求
  sharedContext->addOutstandingRpc(jitFuture);

  // 'recv' 函数通过 RPC 发送梯度数据，不需要返回任何东西给下游自动求导函数
  return variable_list(); // 返回空的变量列表
}

} // namespace autograd
} // namespace distributed
} // namespace torch
```