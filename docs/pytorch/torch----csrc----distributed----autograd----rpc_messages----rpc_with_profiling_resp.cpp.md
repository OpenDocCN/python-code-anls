# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_profiling_resp.cpp`

```
namespace torch {
namespace distributed {
namespace autograd {

// 引用命名空间中定义的基类 RpcCommandBase
using rpc::RpcCommandBase;

// 定义用于标识事件起始索引的常量
constexpr auto kProfileEventsStartIdx = 3;

// 构造函数：创建 RpcWithProfilingResp 对象，用于发送消息
// messageType: 消息类型
// wrappedMessage: 包装的消息对象
// profiledEvents: 事件列表，用于性能分析
// profilingId: 分析 ID
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      tensors_(wrappedMessage_->tensors()),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  
  // 断言消息类型正确
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_RESP,
      "Incorrect Message type");

  // 记录包装消息的类型
  wrappedMessageType_ = wrappedMessage_->type();
}

// 构造函数：从消息中恢复 RPC 命令时调用
// messageType: 消息类型
// wrappedRpc: 包装的 RPC 命令对象
// wrappedMessageType: 包装的消息类型
// tensors: Tensor 列表
// profiledEvents: 事件列表，用于性能分析
// profilingId: 分析 ID
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  
  // 断言包装的 RPC 命令对象不为空
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrapped RPC cannot be null");
}

// 移动包装的 RPC 命令对象，并返回唯一指针
std::unique_ptr<RpcCommandBase> RpcWithProfilingResp::moveWrappedRpc() && {
  // 断言包装的 RPC 命令对象不为空
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  // 移动并返回包装的 RPC 命令对象
  return std::move(wrappedRpc_);
}

// 获取包装消息的消息类型
rpc::MessageType RpcWithProfilingResp::wrappedMessageType() const {
  return wrappedMessageType_;
}

// 获取性能分析事件列表
std::vector<torch::autograd::profiler::LegacyEvent> RpcWithProfilingResp::
    getProfiledEvents() const {
  return profiledEvents_;
}

// 获取性能分析 ID 的引用
const rpc::ProfilingId& RpcWithProfilingResp::getProfilingId() const {
  return profilingId_;
}

// 设置包装的 RPC 命令对象
void RpcWithProfilingResp::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

} // namespace autograd
} // namespace distributed
} // namespace torch
// 移动语义：获取包裹消息的ID
auto wrappedMsgId = wrappedMessage_->id();
// 移动语义：获取包裹消息的类型
auto wrappedMsgType = wrappedMessage_->type();
// 移动语义：获取包裹消息的有效负载并移动所有权
auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
// 检查包裹的有效负载不为空
TORCH_INTERNAL_ASSERT(
    !wrappedPayload.empty(), "Wrapped payload cannot be empty");
// 创建要发送的IValues向量
std::vector<at::IValue> ivalues{wrappedMsgType, profilingId_.toIValue()};
// 将序列化的事件附加到IValues向量中
ivalues.emplace_back(static_cast<int32_t>(profiledEvents_.size()));
for (const auto& e : profiledEvents_) {
  ivalues.emplace_back(e.toIValue());
}
// 创建用于传输的张量表
std::vector<torch::Tensor> tensorTable;
// 使用pickle序列化IValues向量，并填充张量表
std::vector<char> profilingPayload =
    jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
// 将包裹的有效负载和分析负载写入RPC消息
rpc::writeWrappedPayload(wrappedPayload, profilingPayload);

// 创建返回消息，使用包裹消息的有效负载、张量表、消息类型和消息ID
auto returnMsg = c10::make_intrusive<rpc::Message>(
    std::move(wrappedPayload),
    std::move(tensors_),
    messageType_,
    wrappedMsgId);
// 返回创建的消息
return returnMsg;



// 返回包裹的RPC命令，确保不为null
RpcCommandBase& RpcWithProfilingResp::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}



// 从消息反序列化时在客户端运行
std::unique_ptr<RpcWithProfilingResp> RpcWithProfilingResp::fromMessage(
    const rpc::Message& message) {
  // 获取原始消息类型
  rpc::MessageType origMsgType = message.type();
  // 获取消息中的张量
  std::vector<torch::Tensor> tensors = message.tensors();
  // 获取消息ID
  int64_t msgId = message.id();
  // 获取消息有效负载
  auto payload = message.payload();
  // 读取包裹的有效负载，返回元组元素
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // 确保元组元素数量符合预期
  TORCH_INTERNAL_ASSERT(
      tupleElements.size() >= kProfileEventsStartIdx,
      c10::str(
          "Expected payload size of at least ",
          kProfileEventsStartIdx,
          " but got size ",
          tupleElements.size()));
  // 获取包裹消息类型
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  // 获取分析ID
  rpc::ProfilingId profilingId = rpc::ProfilingId::fromIValue(tupleElements[1]);
  // 获取分析事件数量
  int profiledEventsSize = tupleElements[2].toInt();
  // 创建远程事件向量，并预留空间
  std::vector<torch::autograd::profiler::LegacyEvent> remoteEvents;
  remoteEvents.reserve(profiledEventsSize);
  // 遍历分析事件索引范围，重构远程事件
  for (const auto i : c10::irange(
           kProfileEventsStartIdx,
           kProfileEventsStartIdx + profiledEventsSize)) {
    // 确保索引在元组元素范围内
    TORCH_CHECK(static_cast<size_t>(i) < tupleElements.size());
    // 从IValues重构远程事件
    torch::autograd::profiler::LegacyEvent fromIvalueEvent =
        torch::autograd::profiler::LegacyEvent::fromIValue(tupleElements[i]);
    // 将 fromIvalueEvent 移动(push_back)到 remoteEvents 向量的末尾
    remoteEvents.push_back(std::move(fromIvalueEvent));
  }

  // 创建一个包含给定参数的 rpc::Message 对象
  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  
  // 断言 wrappedMessage 是一个响应消息
  TORCH_INTERNAL_ASSERT(
      wrappedMessage->isResponse(),
      "Messages wrapped with profiling response must be responses.");

  // 使用 deserializeResponse 函数反序列化 wrappedMessage，并返回包装它的 RpcCommandBase 对象的唯一指针
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeResponse(*wrappedMessage, wrappedMsgType);
  
  // 创建并返回一个包含各种参数的 RpcWithProfilingResp 对象的唯一指针
  return std::make_unique<RpcWithProfilingResp>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(remoteEvents),
      profilingId);
} // end of namespace autograd
} // end of namespace distributed
} // end of namespace torch
```