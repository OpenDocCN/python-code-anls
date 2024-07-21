# `.\pytorch\torch\csrc\distributed\rpc\rpc_agent.h`

```
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace torch {
namespace distributed {
namespace rpc {

// 使用 unordered_map 存储从源设备到目标设备的映射关系
using DeviceMap = std::unordered_map<c10::Device, c10::Device>;

// 默认 RPC 超时时间，单位为秒
constexpr float kDefaultRpcTimeoutSeconds = 60;
// 未设置的 RPC 超时时间，表示 agent::send() 如果未指定超时时间，则使用默认超时时间
constexpr float kUnsetRpcTimeout = -1;
// 默认初始化方法，使用环境变量 'env://'
constexpr auto kDefaultInitMethod = "env://";
// 秒转毫秒的转换因子
constexpr float kSecToMsConversion = 1000;
// RPC 超时错误信息模板
constexpr auto kRpcTimeoutErrorStr =
    "RPC ran for more than set timeout ({} ms) and will now be marked with an error";

// 使用 steady_clock 记录的时间点类型别名
using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

// 输入为限定名称字符串，输出为 JIT StrongTypePtr 的函数类型
// 类似 jit::TypeResolver，但此处未导入 jit::TypeResolver 避免循环依赖问题
using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

// RPC 后端选项的配置结构体
struct TORCH_API RpcBackendOptions {
  RpcBackendOptions()
      : RpcBackendOptions(kDefaultRpcTimeoutSeconds, kDefaultInitMethod) {}

  RpcBackendOptions(float rpcTimeoutSeconds, std::string initMethod)
      : rpcTimeoutSeconds(rpcTimeoutSeconds),
        initMethod(std::move(initMethod)) {
    TORCH_CHECK(rpcTimeoutSeconds >= 0, "RPC Timeout must be non-negative");
  }

  float rpcTimeoutSeconds;  // RPC 超时时间
  std::string initMethod;   // 初始化方法
};

// 表示一个全局唯一的 RpcAgent 标识符的结构体
struct TORCH_API WorkerInfo : torch::CustomClassHolder {
  WorkerInfo(std::string name, int64_t id);

  WorkerInfo(std::string name, worker_id_t id);

  // 比较运算符重载，用于比较两个 WorkerInfo 结构体是否相等
  bool operator==(const WorkerInfo& rhs) {
    return (id_ == rhs.id_) && (name_ == rhs.name_);
  }

  static constexpr size_t MAX_NAME_LEN = 128;  // 最大名称长度

  const std::string name_;  // 工作节点名称
  const worker_id_t id_;    // 工作节点 ID
};

// 用于在全局范围内注册 WorkerInfo 的辅助结构体
struct TORCH_API RegisterWorkerInfoOnce {
  RegisterWorkerInfoOnce();
};

// 重载运算符 <<，用于将 WorkerInfo 结构体输出到流中
TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const WorkerInfo& workerInfo);

// 用于配置 RPC 重试协议选项的结构体
struct TORCH_API RpcRetryOptions {
  // 默认构造函数，与 RPC 代码库中其他选项结构体一致，输入验证由 sendWithRetries 函数处理
  RpcRetryOptions() = default;
  // 最大重试次数
  int maxRetries{5};
  // 连续 RPC 发送尝试之间的初始间隔时间
  std::chrono::milliseconds rpcRetryDuration{std::chrono::milliseconds(1000)};
  // 指数退避常数，用于计算未来等待持续时间时使用
  float retryBackoff{1.5};
};

// 存储重试给定 RPC 所需的所有元数据的结构体
// (未完成，未提供下文)
// 定义 RpcRetryInfo 结构体，用于管理 RPC 重试信息
struct TORCH_API RpcRetryInfo {
  // 构造函数，初始化 RpcRetryInfo 对象
  RpcRetryInfo(
      const WorkerInfo& to,  // 目标 WorkerInfo 对象的引用
      c10::intrusive_ptr<Message> message,  // 消息的智能指针
      c10::intrusive_ptr<JitFuture> originalFuture,  // 原始 Future 的智能指针
      int retryCount,  // 重试次数
      RpcRetryOptions options)  // RPC 重试选项对象
      : to_(to),  // 初始化目标 WorkerInfo 对象
        message_(std::move(message)),  // 移动构造消息
        originalFuture_(std::move(originalFuture)),  // 移动构造原始 Future
        retryCount_(retryCount),  // 初始化重试次数
        options_(options) {}  // 初始化 RPC 重试选项

  const WorkerInfo& to_;  // 目标 WorkerInfo 对象的常量引用
  c10::intrusive_ptr<Message> message_;  // 消息的智能指针
  // 用于返回给 sendWithRetries() 调用者的 Future
  c10::intrusive_ptr<JitFuture> originalFuture_;
  // 已完成的发送尝试次数
  int retryCount_;
  RpcRetryOptions options_;  // RPC 重试选项
};

// ``RpcAgent`` 是发送和接收 RPC 消息的基类。它提供统一的 ``send`` API 用于发送请求和响应消息，并且会调用给定的 ``RequestCallback`` 处理接收到的请求。在构造后，它应立即准备好服务请求并接受响应。

// 获取当前 RPC 超时时间
return rpcTimeout_.load();
}

// 设置所有 RPC 的超时时间
inline void setRpcTimeout(const std::chrono::milliseconds& rpcTimeout) {
  // 这里使用的指数退避算法是:
  // newTime = timeNow + (retryDuration * (backoffConstant ^ retryCount)).
  std::chrono::milliseconds timedelta =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          options.rpcRetryDuration * pow(options.retryBackoff, retryCount));
  return std::chrono::time_point_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() + timedelta);
}

// 条件变量，用于在填充 rpcRetryMap_ 后发出信号
std::condition_variable rpcRetryMapCV_;

// 保护 RpcRetryMap_ 的互斥锁
std::mutex rpcRetryMutex_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch

namespace std {
template <>
// 自定义 std::hash 结构体，用于 torch::distributed::rpc::WorkerInfo 类型
struct hash<torch::distributed::rpc::WorkerInfo> {
  std::size_t operator()(
      const torch::distributed::rpc::WorkerInfo& worker_info) const noexcept {
    return worker_info.id_;  // 返回 WorkerInfo 对象的 id_ 作为哈希值
  }
};
} // namespace std
```