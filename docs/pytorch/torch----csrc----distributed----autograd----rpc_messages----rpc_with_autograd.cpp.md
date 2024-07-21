# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\rpc_with_autograd.cpp`

```
// 包含需要的头文件：RPC 消息定义、RPC 代理、RPC 工具、序列化相关工具、字节序工具
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

// 命名空间声明：分布式自动求导相关功能
namespace torch {
namespace distributed {
namespace autograd {

// 使用语句别名：RPC 消息、消息类型、RPC 命令基类、工作进程 ID
using rpc::Message;
using rpc::MessageType;
using rpc::RpcCommandBase;
using rpc::worker_id_t;

// 构造函数实现：构造带自动求导信息的 RPC 对象
RpcWithAutograd::RpcWithAutograd(
    worker_id_t fromWorkerId,                            // 发送者工作进程 ID
    MessageType messageType,                             // 消息类型
    const AutogradMetadata& autogradMetadata,             // 自动求导元数据
    c10::intrusive_ptr<rpc::Message> wrappedMessage,      // 封装的 RPC 消息对象
    rpc::DeviceMap deviceMap)                             // 设备映射
    : fromWorkerId_(fromWorkerId),                        // 初始化成员变量：发送者工作进程 ID
      messageType_(messageType),                         // 初始化成员变量：消息类型
      autogradMetadata_(autogradMetadata),               // 初始化成员变量：自动求导元数据
      wrappedMessage_(std::move(wrappedMessage)),        // 初始化成员变量：封装的 RPC 消息对象
      deviceMap_(std::move(deviceMap)) {                 // 初始化成员变量：设备映射
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||  // 断言：消息类型为前向自动求导请求或响应
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
  tensors_ = wrappedMessage_->tensors();                // 获取封装消息中的张量列表
  wrappedMessageType_ = wrappedMessage_->type();        // 获取封装消息的消息类型
}

// 构造函数实现：构造带自动求导信息的 RPC 对象（另一种重载）
RpcWithAutograd::RpcWithAutograd(
    worker_id_t fromWorkerId,                            // 发送者工作进程 ID
    MessageType messageType,                             // 消息类型
    const AutogradMetadata& autogradMetadata,             // 自动求导元数据
    std::unique_ptr<RpcCommandBase> wrappedRpc,          // 封装的 RPC 命令对象
    MessageType wrappedMessageType,                      // 封装消息的消息类型
    std::vector<torch::Tensor> tensors,                  // 张量列表
    rpc::DeviceMap deviceMap)                             // 设备映射
    : fromWorkerId_(fromWorkerId),                        // 初始化成员变量：发送者工作进程 ID
      messageType_(messageType),                         // 初始化成员变量：消息类型
      autogradMetadata_(autogradMetadata),               // 初始化成员变量：自动求导元数据
      wrappedRpc_(std::move(wrappedRpc)),                // 初始化成员变量：封装的 RPC 命令对象
      wrappedMessageType_(wrappedMessageType),           // 初始化成员变量：封装消息的消息类型
      tensors_(std::move(tensors)),                      // 初始化成员变量：张量列表
      deviceMap_(std::move(deviceMap)) {                 // 初始化成员变量：设备映射
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");  // 断言：封装的 RPC 命令对象不能为空
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||  // 断言：消息类型为前向自动求导请求或响应
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
}

// 转换为消息实现方法：移动语义
c10::intrusive_ptr<Message> RpcWithAutograd::toMessageImpl() && {
  auto messageId = wrappedMessage_->id();                // 获取封装消息的 ID
  auto wrappedMessageType = wrappedMessage_->type();     // 获取封装消息的消息类型

  auto payload = std::move(*wrappedMessage_).movePayload();  // 移动封装消息的有效载荷
  TORCH_INTERNAL_ASSERT(!payload.empty());               // 断言：有效载荷不应为空

  // 将设备映射转换为 c10::Dict 以进行序列化
  c10::Dict<std::string, std::string> deviceMap;
  for (const auto& mapEntry : deviceMap_) {
  // 将mapEntry中的键值对插入到deviceMap中
  deviceMap.insert(mapEntry.first.str(), mapEntry.second.str());
}

// 创建一个包含多个at::IValue对象的向量
std::vector<at::IValue> ivalues{
    wrappedMessageType,                              // 消息类型的包装
    autogradMetadata_.autogradContextId,              // 自动求导上下文ID
    autogradMetadata_.autogradMessageId,              // 自动求导消息ID
    fromWorkerId_,                                    // 来自工作进程的ID
    deviceMap                                         // 设备映射表
};

// 使用JIT pickler对ivalues进行序列化为字节流，并填充tensorTable
std::vector<torch::Tensor> tensorTable;
std::vector<char> additionalPayload =
    jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

// tensorTable应当为空
TORCH_INTERNAL_ASSERT(tensorTable.empty());

// 将additionalPayload封装到payload中，并处理大小调整和编码
rpc::writeWrappedPayload(payload, additionalPayload);

// 创建并返回一个Message对象，包含payload、tensors_、messageType_和messageId
return c10::make_intrusive<Message>(
    std::move(payload), std::move(tensors_), messageType_, messageId);
}

// 结束 autograd 命名空间
namespace autograd {

// 结束 distributed 命名空间
namespace distributed {

// 结束 torch 命名空间
namespace torch {

std::unique_ptr<RpcWithAutograd> RpcWithAutograd::fromMessage(
    const Message& message) {
  // 获取消息的原始类型
  MessageType originalMessageType = message.type();
  // 断言消息类型为 FORWARD_AUTOGRAD_REQ 或 FORWARD_AUTOGRAD_RESP
  TORCH_INTERNAL_ASSERT(
      MessageType::FORWARD_AUTOGRAD_REQ == originalMessageType ||
      MessageType::FORWARD_AUTOGRAD_RESP == originalMessageType);

  // 获取消息中的张量列表
  std::vector<torch::Tensor> tensors = message.tensors();
  // 获取消息的 ID
  int64_t messageId = message.id();
  
  // 解码消息类型、自动求导上下文 ID、自动求导消息 ID 和发送该消息的 worker ID
  auto payload = message.payload();
  auto tupleElements = rpc::readWrappedPayload(payload, message);

  // 收集所有字段
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 5);
  // 解析封装的消息类型
  MessageType wrappedMessageType =
      static_cast<MessageType>(tupleElements[0].toInt());
  // 解析自动求导元数据
  AutogradMetadata autogradMetadata(
      tupleElements[1].toInt(), tupleElements[2].toInt());
  // 解析发送消息的 worker ID
  worker_id_t workerId = tupleElements[3].toInt();
  // 将 C10 设备映射转换为普通映射
  auto c10DeviceMap =
      tupleElements[4].to<c10::Dict<std::string, std::string>>();
  
  // 将 C10 设备映射转换为普通映射
  rpc::DeviceMap deviceMap;
  for (const auto& mapEntry : c10DeviceMap) {
    deviceMap.insert({mapEntry.key(), mapEntry.value()});
  }

  // 创建新的消息类型并构建封装的 RPC
  auto wrappedMessage = c10::make_intrusive<Message>(
      std::move(payload), std::move(tensors), wrappedMessageType, messageId);

  std::unique_ptr<RpcCommandBase> wrappedRpc;
  // 根据原始消息类型反序列化请求或响应
  if (originalMessageType == MessageType::FORWARD_AUTOGRAD_REQ) {
    wrappedRpc = deserializeRequest(*wrappedMessage);
  } else {
    wrappedRpc = deserializeResponse(*wrappedMessage, wrappedMessageType);
  }

  // 创建包含自动求导信息的 RpcWithAutograd 对象并返回
  return std::make_unique<RpcWithAutograd>(
      workerId,
      originalMessageType,
      autogradMetadata,
      std::move(wrappedRpc),
      wrappedMessageType,
      wrappedMessage->tensors(),
      deviceMap);
}

// 返回 RpcWithAutograd 对象中的张量引用
std::vector<torch::Tensor>& RpcWithAutograd::tensors() {
  return tensors_;
}

// 返回 RpcWithAutograd 对象中的自动求导元数据
const AutogradMetadata& RpcWithAutograd::autogradMetadata() const {
  return autogradMetadata_;
}

// 返回 RpcWithAutograd 对象中的封装的 RPC 命令
RpcCommandBase& RpcWithAutograd::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

// 设置 RpcWithAutograd 对象中的封装的 RPC 命令
void RpcWithAutograd::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

// 将 RpcWithAutograd 对象中的封装的 RPC 命令以右值形式返回
std::unique_ptr<RpcCommandBase> RpcWithAutograd::moveWrappedRpc() && {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return std::move(wrappedRpc_);
}

// 返回 RpcWithAutograd 对象中封装的消息类型
MessageType RpcWithAutograd::wrappedMessageType() const {
  return wrappedMessageType_;
}

// 返回 RpcWithAutograd 对象中发送消息的 worker ID
rpc::worker_id_t RpcWithAutograd::fromWorkerId() const {
  return fromWorkerId_;
}

// 返回 RpcWithAutograd 对象中的设备映射
const rpc::DeviceMap& RpcWithAutograd::deviceMap() {
  return deviceMap_;
}

} // namespace torch
} // namespace distributed
} // namespace autograd
```