# `.\pytorch\torch\csrc\distributed\autograd\utils.h`

```
#pragma once

#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

// 此方法用于在使用 RPC 时将 'send' 自动微分函数附加到自动微分图中。
// 方法创建一个新的 'send' 自动微分函数，并将提供的张量作为 'send' 函数的下一步边缘附加。
// 此外，它还在提供的自动微分上下文中注册 send 函数。
// 最后，更新 RPC 消息，为接收者添加适当的自动微分信息。
TORCH_API void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors);

// 此方法用于在使用 RPC 时将 'recv' 自动微分函数附加到自动微分图中。
// 方法创建一个新的 'recv' 自动微分函数，并将提供的张量作为 'recv' 函数的输入附加。
// 如果需要，它会创建一个新的自动微分上下文，并将 'recv' 函数注册到该上下文中。
//
// 返回指向创建的自动微分上下文的指针。
TORCH_API ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const rpc::DeviceMap& deviceMap);

// 此方法是一个内部使用的包装工具，用于根据 RPC 调用的类型包装自动微分信息并附加自动微分函数。
// 如果具有有效的上下文并且张量需要梯度，或者 forceGradRecording 为 true，则返回 RpcWithAutograd 消息；
// 否则返回原始的 RPC 消息。
// 注意：当请求不包含任何张量但相应包含时，forceGradRecording 非常有用。
TORCH_API c10::intrusive_ptr<rpc::Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<rpc::Message> wrappedRpcMsg,
    rpc::MessageType msgType,
    bool forceGradRecording = false,
    const rpc::DeviceMap& deviceMap = {});

// 在进行自动微分检查后发送消息
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> sendMessageWithAutograd(
    rpc::RpcAgent& agent,
    const rpc::WorkerInfo& dst,
    c10::intrusive_ptr<rpc::Message> wrappedRpcMsg,
    bool forceGradRecording = false,
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
    bool forceDisableProfiling = false);

} // namespace autograd
} // namespace distributed
} // namespace torch
```