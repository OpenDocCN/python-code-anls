# `.\pytorch\torch\csrc\distributed\rpc\testing\faulty_tensorpipe_agent.h`

```py
#pragma once
// 如果定义了 USE_TENSORPIPE 宏，则包含以下头文件
#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

// torch 命名空间开始
namespace torch {
// distributed 命名空间开始
namespace distributed {
// rpc 命名空间开始
namespace rpc {

// FaultyTensorPipeRpcBackendOptions 结构体，继承自 TensorPipeRpcBackendOptions
struct TORCH_API FaultyTensorPipeRpcBackendOptions
    : public TensorPipeRpcBackendOptions {
  // 构造函数定义，接收多个参数
  FaultyTensorPipeRpcBackendOptions(
      int num_worker_threads,  // 工作线程数
      float rpc_timeout,        // RPC 超时时间
      std::string init_method,  // 初始化方法
      std::vector<std::string> messages_to_fail,  // 要失败的消息列表
      std::unordered_map<std::string, float> messages_to_delay,  // 要延迟的消息映射
      int num_fail_sends = 0)  // 发送失败次数，默认为 0
      // 调用基类的构造函数，初始化基类成员变量
      : TensorPipeRpcBackendOptions(
            num_worker_threads,
            optional<std::vector<std::string>>(),
            optional<std::vector<std::string>>(),
            rpc_timeout,
            std::move(init_method)),
        messagesToFail(std::move(messages_to_fail)),  // 初始化 messagesToFail
        messagesToDelay(std::move(messages_to_delay)),  // 初始化 messagesToDelay
        numFailSends(num_fail_sends) {  // 初始化 numFailSends
    // 检查 numFailSends 是否为非负数，若为负数则抛出异常
    TORCH_CHECK(numFailSends >= 0, "numFailSends should be non-negative");
  }

  // 要失败的消息列表
  std::vector<std::string> messagesToFail;
  // 要延迟的消息映射
  std::unordered_map<std::string, float> messagesToDelay;
  // 发送失败次数
  int numFailSends;
};

// rpc 命名空间结束
} // namespace rpc
// distributed 命名空间结束
} // namespace distributed
// torch 命名空间结束
} // namespace torch

// 结束条件编译块
#endif  // USE_TENSORPIPE
class TORCH_API FaultyTensorPipeAgent : public TensorPipeAgent {
 public:
  // 构造函数，初始化 FaultyTensorPipeAgent 实例
  FaultyTensorPipeAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string selfName,
      worker_id_t selfId,
      int worldSize,
      FaultyTensorPipeRpcBackendOptions opts,
      std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
      std::vector<c10::Device> devices,
      std::unique_ptr<RequestCallback> callback);

  // 重写父类的 send 函数，用于发送消息
  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
      const DeviceMap& deviceMap = {}) override;

  // 在写入过程中添加延迟的函数
  void pipeWrite(
      const std::shared_ptr<tensorpipe::Pipe>& pipe,
      c10::intrusive_ptr<Message> rpcMessage,
      std::vector<c10::Device>&& devices,
      std::vector<c10::Stream> streams,
      std::function<void(const tensorpipe::Error&)> fn) noexcept override;

 protected:
  // 检查 messageTypesToFail_ 是否包含指定的消息类型，确定是否使用故障发送
  bool shouldFailMessage(MessageType type) const;

 private:
  // 解析由 Python 测试传入的字符串列表，确定必须使用故障发送的消息类型
  std::vector<MessageType> parseMessagesToFailInput(
      const std::vector<std::string>& messagesToFail) const;

  // 返回指定消息类型的发送延迟时间（秒）
  float getDelayForMessage(MessageType type) const;

  // 解析需要为其注入任意延迟的消息类型
  std::unordered_map<MessageType, float, std::hash<int>> parseMessagesToDelay(
      const std::unordered_map<std::string, float>& messageTypesToDelay) const;

  // 故障发送前允许失败的发送次数
  const int numFailSends_;

  // 必须使用故障发送的消息类型的向量，根据 Python 测试传入的字符串列表解析得到
  const std::vector<MessageType> messageTypesToFail_;

  // 消息类型到发送延迟时间的映射
  std::unordered_map<MessageType, float, std::hash<int>> messageTypesToDelay_;

  // 记录每个 RPC 的发送失败次数的映射
  std::unordered_map<std::string, int> failMessageCountMap_;

  // 保护 failMessageCountMap_ 的互斥锁
  std::mutex failMapMutex_;

  // 根据消息字符串返回对应的 MessageType 枚举值
  MessageType messageStringToType(const std::string& messageString) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
```