# `.\pytorch\torch\csrc\distributed\rpc\testing\faulty_tensorpipe_agent.cpp`

```py
// 如果定义了预处理器标记 USE_TENSORPIPE，则编译以下代码块

#include <torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>

// torch 命名空间开始
namespace torch {
// distributed 命名空间开始
namespace distributed {
// rpc 命名空间开始
namespace rpc {

// 定义一个静态函数，将字符向量转换为字符串
static std::string fromVecToString(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

// FaultyTensorPipeAgent 构造函数的实现
FaultyTensorPipeAgent::FaultyTensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,  // 分布式存储的智能指针
    std::string selfName,                           // 代理自身名称
    worker_id_t selfId,                             // 代理自身 ID
    int worldSize,                                  // 世界大小
    FaultyTensorPipeRpcBackendOptions opts,          // 故障 TensorPipe RPC 后端选项
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,  // 设备映射表
    std::vector<c10::Device> devices,                // 设备列表
    std::unique_ptr<RequestCallback> callback)       // 请求回调的唯一指针
    : TensorPipeAgent(                              // 调用基类 TensorPipeAgent 的构造函数
          store,
          std::move(selfName),
          selfId,
          worldSize,
          std::move(opts),
          std::move(reverseDeviceMaps),
          std::move(devices),
          std::move(callback)),
      numFailSends_(opts.numFailSends),             // 初始化成员变量 numFailSends_
      messageTypesToFail_(parseMessagesToFailInput(opts.messagesToFail)),  // 调用函数解析故障消息类型
      messageTypesToDelay_(parseMessagesToDelay(opts.messagesToDelay)) {}  // 调用函数解析延迟消息类型

// 解析要故障的消息类型输入，将字符串列表解析为消息类型向量
std::vector<MessageType> FaultyTensorPipeAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  std::vector<MessageType> messageTypesToFail;
  messageTypesToFail.reserve(messagesToFail.size());
  for (const auto& msgString : messagesToFail) {
    messageTypesToFail.push_back(messageStringToType(msgString));  // 调用函数将字符串转换为消息类型
  }
  return messageTypesToFail;
}

// 解析要延迟的消息类型输入，将字符串到延迟值的映射解析为消息类型到延迟时间的映射
std::unordered_map<MessageType, float, std::hash<int>> FaultyTensorPipeAgent::
    parseMessagesToDelay(const std::unordered_map<std::string, float>&
                             messageTypesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messageTypesToDelay) {
    float delay = messagePair.second;
    TORCH_CHECK(
        delay >= 0,
        "Delays passed to FaultyTensorPipeAgent must be non-negative.")  // 检查延迟时间必须为非负数
    delayMessages.insert({messageStringToType(messagePair.first), delay});  // 调用函数将字符串转换为消息类型，插入映射
  }
  return delayMessages;
}

// 发送消息的函数实现
c10::intrusive_ptr<JitFuture> FaultyTensorPipeAgent::send(
    const WorkerInfo& to,                           // 接收消息的工作节点信息
    c10::intrusive_ptr<Message> message,            // 要发送的消息指针
    const float rpcTimeoutSeconds,                  // RPC 超时时间
    const DeviceMap& /* unused */) {                // 设备映射（未使用）
  // 只会故障在测试用例中指定的控制消息，对于其他消息，发送而不故障
  if (!shouldFailMessage(message->type())) {
  // 调用 TensorPipeAgent 的 send 函数，发送消息给目标节点 to，包含移动语义的 message 和超时时间 rpcTimeoutSeconds
  return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
}

// 这个 send 函数检查 failMessageCountMap_ 是否需要失败下一次发送。
// 如果需要失败发送，立即在返回的 future 上设置错误，并在映射中增加计数器。
// 否则，调用 TensorPipeAgent 的 send 函数继续正常发送。
const auto key = fromVecToString(message->payload());
std::unique_lock<std::mutex> lock(failMapMutex_);
auto it = failMessageCountMap_.find(key);
if (it == failMessageCountMap_.end()) {
  failMessageCountMap_[key] = 0;
}
if (failMessageCountMap_[key] < numFailSends_) {
  failMessageCountMap_[key]++;
  lock.unlock();
  // 创建一个 JitFuture 对象，并设置发送失败的异常错误信息
  auto jitFuture = c10::make_intrusive<JitFuture>(at::AnyClassType::get());
  jitFuture->setError(std::make_exception_ptr(std::runtime_error(makeRPCError(
      c10::str("Send attempt failed intentionally for ", key),
      RPCErrorType::INTENTIONAL_FAILURE))));
  return jitFuture;
} else {
  lock.unlock();
  // 调用 TensorPipeAgent 的 send 函数，继续正常发送消息给目标节点 to
  return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
}
} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE


注释：


// 结束 namespace torch，这里是对应于 USE_TENSORPIPE 宏定义的尾部
```