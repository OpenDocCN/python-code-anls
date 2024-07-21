# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_profiling_req.cpp`

```py
// 包含必要的头文件，用于实现 RPC 消息的序列化和反序列化
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <vector>

// 定义命名空间 torch::distributed::autograd，用于放置自动求导相关的分布式功能
namespace torch {
namespace distributed {
namespace autograd {

// 定义常量 kProfilingResponseElementExpectedSize，期望的分析响应元素数量为 3
constexpr auto kProfilingResponseElementExpectedSize = 3;

// 使用命名空间 rpc 中的 RpcCommandBase 类
using rpc::RpcCommandBase;

// 构造函数：RpcWithProfilingReq 的客户端版本，用于创建 RpcWithProfilingReq 对象
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,                                // 消息类型
    c10::intrusive_ptr<rpc::Message> wrappedMessage,              // 封装的消息对象
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,   // 分析器配置对象
    rpc::ProfilingId profilingKeyId)                              // 分析键 ID
    : messageType_(messageType),                                  // 初始化消息类型
      wrappedMessage_(std::move(wrappedMessage)),                 // 初始化封装的消息对象
      tensors_(wrappedMessage_->tensors()),                       // 初始化消息中的张量列表
      profilerConfig_(profilerConfig),                            // 初始化分析器配置对象
      profilingKeyId_(profilingKeyId) {                           // 初始化分析键 ID
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_REQ,   // 断言消息类型为运行带分析请求
      c10::str(
          "Incorrect message type, expected message type ",
          rpc::MessageType::RUN_WITH_PROFILING_REQ));             // 错误消息断言
  wrappedMessageType_ = wrappedMessage_->type();                  // 初始化封装消息的类型
}

// 构造函数：RpcWithProfilingReq 的远程版本，用于在远程重建 RpcWithProfilingReq 对象
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,                        // 消息类型
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,     // 封装的 RPC 命令对象
    rpc::MessageType wrappedMessageType,                 // 封装消息的类型
    std::vector<torch::Tensor> tensors,                  // 张量列表
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,   // 分析器配置对象
    rpc::ProfilingId profilingKeyId)                     // 分析键 ID
    : messageType_(messageType),                          // 初始化消息类型
      wrappedRpc_(std::move(wrappedRpc)),                // 初始化封装的 RPC 命令对象
      wrappedMessageType_(wrappedMessageType),            // 初始化封装消息的类型
      tensors_(std::move(tensors)),                       // 初始化张量列表
      profilerConfig_(profilerConfig),                    // 初始化分析器配置对象
      profilingKeyId_(profilingKeyId) {                   // 初始化分析键 ID
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cant be null"); // 断言封装的 RPC 命令对象非空
}

// 返回封装消息的类型
rpc::MessageType RpcWithProfilingReq::wrappedMessageType() const {
  return wrappedMessageType_;
}

// 设置封装的 RPC 命令对象
void RpcWithProfilingReq::setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

// 命名空间 autograd 的结束
} // namespace autograd
} // namespace distributed
} // namespace torch
// 将 RpcWithProfilingReq 对象转换为 rpc::Message 对象，并移动其资源
c10::intrusive_ptr<rpc::Message> RpcWithProfilingReq::toMessageImpl() && {
  // 保存移动前的原始消息 ID 和类型
  auto wrappedMsgId = wrappedMessage_->id();
  auto wrappedMsgType = wrappedMessage_->type();
  // 移动 wrappedMessage 并获取其负载。此时 wrappedMessage 的负载将不再处于有效状态。
  auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
  // 检查 wrapped 负载不为空
  TORCH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload should not be empty.");
  // 创建要发送的 IValue 向量，包括原始消息类型和 ID，以及一些分析元数据
  std::vector<at::IValue> ivalues{
      wrappedMsgType, profilerConfig_.toIValue(), profilingKeyId_.toIValue()};
  // 将其序列化为 char 负载以便通过网络发送
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  // 将分析负载添加到 wrapped 负载中
  rpc::writeWrappedPayload(wrappedPayload, profilingPayload);
  // 将 wrapped 负载放入要返回的消息中
  auto returnMsg = c10::make_intrusive<rpc::Message>(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);

  return returnMsg;
}

// 返回封装的 RpcCommandBase 对象的引用
RpcCommandBase& RpcWithProfilingReq::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

// 返回分析配置 ProfilerConfig 的副本
torch::autograd::profiler::ProfilerConfig RpcWithProfilingReq::
    getProfilingConfig() const {
  return profilerConfig_;
}

// 返回分析 ID 的常量引用
const rpc::ProfilingId& RpcWithProfilingReq::getProfilingId() const {
  return profilingKeyId_;
}
    // 获取消息的类型
    const rpc::Message& message) {
  // 获取原始消息类型
  rpc::MessageType origMsgType = message.type();
  // 获取消息中的张量列表
  std::vector<torch::Tensor> tensors = message.tensors();
  // 获取消息的ID
  int64_t msgId = message.id();
  // 获取消息的有效负载
  auto payload = message.payload();
  // 解析有效负载中的元组元素
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // 确保元组中的元素数量符合预期
  TORCH_INTERNAL_ASSERT(
      tupleElements.size() == kProfilingResponseElementExpectedSize,
      c10::str(
          "Expected payload of size ",
          kProfilingResponseElementExpectedSize,
          " but got ",
          tupleElements.size()));
  // 从元组中获取包装消息的类型
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  // 根据元组中的配置创建性能分析器配置对象
  torch::autograd::profiler::ProfilerConfig cfg =
      torch::autograd::profiler::ProfilerConfig::fromIValue(tupleElements[1]);
  // 从元组中获取性能分析器ID
  rpc::ProfilingId profilerId = rpc::ProfilingId::fromIValue(tupleElements[2]);

  // 创建新的消息对象，用于包装原始消息的RPC请求
  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  // 确保包装后的消息是请求类型
  TORCH_INTERNAL_ASSERT(
      wrappedMessage->isRequest(),
      "Messages wrapped with profiling requests must be requests.");
  // 反序列化包装后消息中的RPC请求
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeRequest(*wrappedMessage);

  // 创建带有性能分析请求的RPC对象并返回
  return std::make_unique<RpcWithProfilingReq>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(cfg),
      profilerId);
}
// 结束 torch 命名空间
} // namespace torch
// 结束 distributed 命名空间
} // namespace distributed
// 结束 autograd 命名空间
} // namespace autograd
```