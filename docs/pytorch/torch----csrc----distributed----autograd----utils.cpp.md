# `.\pytorch\torch\csrc\distributed\autograd\utils.cpp`

```py
// 包含 ATen 库的线程本地状态头文件
#include <ATen/ThreadLocalState.h>
// 包含 C10 工具库的线程本地调试信息头文件
#include <c10/util/ThreadLocalDebugInfo.h>
// 包含 Torch 自动微分函数工具头文件
#include <torch/csrc/autograd/functions/utils.h>
// 包含 Torch 自动微分性能分析头文件
#include <torch/csrc/autograd/profiler.h>
// 包含 Torch 分布式自动微分上下文容器头文件
#include <torch/csrc/distributed/autograd/context/container.h>
// 包含 Torch 分布式自动微分发送 RPC 反向传播函数头文件
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
// 包含 Torch 分布式自动微分接收 RPC 反向传播函数头文件
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
// 包含 Torch 分布式自动微分工具头文件
#include <torch/csrc/distributed/autograd/utils.h>
// 包含 Torch 分布式 RPC 性能分析远程管理器头文件
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
// 包含 Torch 分布式 RPC 代理头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含 Torch 分布式 RPC 类型头文件
#include <torch/csrc/distributed/rpc/types.h>

// Torch 命名空间开始
namespace torch {
// 分布式命名空间开始
namespace distributed {
// 自动微分命名空间开始
namespace autograd {

// 使用语句定义一些类型别名，简化代码中的类型使用
using torch::distributed::autograd::AutogradMetadata;
using torch::distributed::autograd::RpcWithAutograd;
using torch::distributed::rpc::JitFuture;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::WorkerInfo;

// 定义函数 addSendRpcBackward，用于添加发送 RPC 的反向传播函数
void addSendRpcBackward(
    const ContextPtr& autogradContext,                    // 自动微分上下文指针
    const AutogradMetadata& autogradMetadata,             // 自动微分元数据
    std::vector<torch::Tensor>& tensors) {                // 包含张量的向量

  // 仅为需要梯度的张量附加自动微分信息
  std::vector<torch::Tensor> tensors_with_grad;
  std::copy_if(
      tensors.begin(),
      tensors.end(),
      std::back_inserter(tensors_with_grad),
      [](const torch::Tensor& t) { return t.requires_grad(); });

  // 创建发送 RPC 的反向传播函数对象
  auto grad_fn = std::make_shared<SendRpcBackward>();
  grad_fn->set_next_edges(
      torch::autograd::collect_next_edges(tensors_with_grad));

  // 为反向传播函数添加输入元数据
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);
  }

  // 将发送的自动微分函数记录到当前上下文中
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

// 定义函数 addRecvRpcBackward，用于添加接收 RPC 的反向传播函数
ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,             // 自动微分元数据
    std::vector<torch::Tensor>& tensors,                  // 包含张量的向量
    rpc::worker_id_t fromWorkerId,                        // 发送方工作器 ID
    const rpc::DeviceMap& deviceMap) {                    // 设备映射表

  // 如果需要，初始化自动微分上下文
  auto& autogradContainer = DistAutogradContainer::getInstance();
  auto autogradContext =
      autogradContainer.getOrCreateContext(autogradMetadata.autogradContextId);

  // 如果张量不为空且至少有一个张量需要梯度
  if (!tensors.empty() && torch::autograd::compute_requires_grad(tensors)) {
    // 创建接收 RPC 的反向传播函数对象
    auto grad_fn = std::make_shared<RecvRpcBackward>(
        autogradMetadata, autogradContext, fromWorkerId, deviceMap);
    // 为需要梯度的张量设置历史记录
    for (auto& tensor : tensors) {
      if (tensor.requires_grad()) {
        torch::autograd::set_history(tensor, grad_fn);
      }
    }

    // 更新自动微分上下文，添加接收的自动微分函数
    autogradContext->addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }

  return autogradContext;  // 返回更新后的自动微分上下文
}

// 定义静态函数 getMessageWithProfiling，用于获取带有性能分析的消息
static c10::intrusive_ptr<Message> getMessageWithProfiling(
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMessage,
    // 参数：包装后的 RPC 消息对象，使用引用计数的指针管理
    MessageType msgType,
    // 参数：消息类型，用于 RPC 请求的类型标识
    torch::autograd::profiler::ProfilerConfig&& profilerConfig) {
  auto& remoteProfilerManager =
      // 获取远程分析器管理器的单例实例
      torch::distributed::rpc::RemoteProfilerManager::getInstance();

  auto key = remoteProfilerManager.getCurrentProfilingKey();
  // 获取当前的分析器键（key）

  auto globallyUniqueProfilingId = remoteProfilerManager.getNextProfilerId();
  // 生成全局唯一的分析器 ID

  remoteProfilerManager.saveRPCKey(globallyUniqueProfilingId, key);
  // 将 ID 和当前的 RPC 分析键（key）保存到管理器中，并取消当前线程的分析键（key）

  auto wrappedProfilingMsg = RpcWithProfilingReq(
      // 创建包含分析信息的 RPC 请求
      msgType,
      std::move(wrappedRpcMessage),
      std::move(profilerConfig),
      globallyUniqueProfilingId);

  return std::move(wrappedProfilingMsg).toMessage();
  // 返回包含分析信息的 RPC 消息
}
// 结束当前命名空间 'autograd'

c10::intrusive_ptr<Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording,
    const rpc::DeviceMap& deviceMap) {
  // 获取全局唯一的 DistAutogradContainer 实例
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // 如果没有有效的上下文且没有张量需要梯度，则发送原始的 rpc 消息
  // 否则，附加梯度信息和梯度函数，发送 rpcWithAutograd 消息
  auto tensorsRequireGrad =
      torch::autograd::compute_requires_grad(wrappedRpcMsg->tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return wrappedRpcMsg;
  }

  // 获取当前要修改的自动求导上下文
  auto autogradContext = autogradContainer.currentContext();

  // 使用自动求导信息包装原始的 rpc 消息
  AutogradMetadata autogradMetadata(
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg),
      deviceMap);

  if (tensorsRequireGrad) {
    // 为 'send' 操作记录自动求导信息
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // 记录目标 worker 的 workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();
}

c10::intrusive_ptr<JitFuture> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    bool forceGradRecording,
    const float rpcTimeoutSeconds,
    bool forceDisableProfiling) {
  // 调用 getMessageWithAutograd 获取包含自动求导信息的消息
  auto msg = getMessageWithAutograd(
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording,
      agent.getDeviceMap(dst));

  // 如果启用了分析器，则使用分析元数据包装此消息，告知远程端启用分析器处理此请求
  if (!forceDisableProfiling) {
    switch (torch::profiler::impl::profilerType()) {
      case torch::profiler::impl::ActiveProfilerType::LEGACY: {
        auto profilerConfig = torch::autograd::profiler::getProfilerConfig();
        auto msgWithProfiling = getMessageWithProfiling(
            std::move(msg),
            rpc::MessageType::RUN_WITH_PROFILING_REQ,
            std::move(profilerConfig));
        // 使用代理发送包含分析信息的消息，设置超时时间
        return agent.send(dst, std::move(msgWithProfiling), rpcTimeoutSeconds);
      }
      case torch::profiler::impl::ActiveProfilerType::KINETO:
        // 警告：使用 Kineto 分析器对分布式调用进行分析将会分析调用方，而不是工作节点
        TORCH_WARN_ONCE(
            "Profiling a distributed call with the Kineto profiler will profile "
            "the caller, but not the worker.");
        break;
      default:
        break;
    }
  }

  // 使用代理发送消息，设置超时时间
  return agent.send(dst, std::move(msg), rpcTimeoutSeconds);
  ;
}

// 结束当前命名空间 'autograd'
} // 结束 torch 命名空间
} // 结束 distributed 命名空间
```