# `.\pytorch\torch\csrc\distributed\rpc\utils.h`

```py
#pragma once

#include <c10/core/Device.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace rpc {

// 根据错误消息解析并返回相应的 RPC 错误类型。
TORCH_API RPCErrorType getRPCErrorType(const JitFuture& jitFuture);

// 根据错误描述和错误类型创建一个错误字符串。
TORCH_API std::string makeRPCError(
    const std::string& rpcErrorStr,
    RPCErrorType errorType);

// 将从网络接收的 RPC 请求消息反序列化为相应的 'RpcCommandBase' 类型。
TORCH_API std::unique_ptr<RpcCommandBase> deserializeRequest(
    const Message& request);

// 将从网络接收的 RPC 响应消息反序列化为相应的 'RpcCommandBase' 类型。
// 如果响应类型为 FORWARD_AUTOGRAD_RESP，则解封它，并附加 recvBackward() 函数到接收到的张量上，并设置 wrappedMsgType 为其包装消息类型。
TORCH_API std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType);

// 将从网络接收的 RPC 响应消息反序列化为有效的 IValue，如果消息是用于脚本 RPC 结果，则进行反序列化，否则反序列化为永远不会使用的虚拟 IValue。
// 在此反序列化过程中，如果需要，还会附加接收 RPC 反向传播函数。
TORCH_API IValue deserializeResptoIValue(const Message& message);

// 注意：此格式可能会更改，用于 RPC。
// 对于持久保存到磁盘，请使用 torch::save()。
TORCH_API std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors);

// 反序列化 wire 格式的数据为一对 payload 和 tensors。
TORCH_API std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,
    size_t data_size);

// 我们使用 vector<char> 作为 blob 的类型，因为它是 rpc::Message 用于其 payload 的类型，
// 即使它有一个缺点，即它不能使用未初始化的内存分配：它总是被清零。

// 一些张量实际上是大张量的视图，其中只引用了存储数据的小子集。
// 这通常很好，可以在本地保留而不复制，但如果我们简单地将整个存储传输到网络上，将会产生过多的网络流量。
// 这个改动会在能够节省至少一半数据时克隆张量，并超过一个最小障碍。
TORCH_API c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors);

// 将原始 payload 和包装 payload 合并到原始 payload 中。
// 用于生成包装 RPC 的整体 payload。
TORCH_API void writeWrappedPayload(
    // 接受两个引用参数：originalPayload 和 additionalPayload，它们都是 std::vector<char> 类型的引用
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload);
    // 函数声明结束
// 从输入的 payload 中读取额外的包装负载，使得 payload 现在包含未包装的原始 RPC 的负载。
TORCH_API std::vector<at::IValue> readWrappedPayload(
    std::vector<char>& payload,
    const rpc::Message& message);

// 将来自 autograd 分析器的事件列表填充到 profiledEvents 中，以便通过 RPC 传输。
TORCH_API void populateRemoteProfiledEvents(
    std::vector<torch::autograd::profiler::LegacyEvent>& profiledEvents,
    const torch::autograd::profiler::ProfilerConfig& profilerConfig,
    const std::vector<std::vector<torch::autograd::profiler::LegacyEvent>>&
        eventLists);

} // namespace rpc
} // namespace distributed
} // namespace torch
```