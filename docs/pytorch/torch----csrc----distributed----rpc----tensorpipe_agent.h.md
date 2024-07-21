# `.\pytorch\torch\csrc\distributed\rpc\tensorpipe_agent.h`

```
#pragma once

#ifdef USE_TENSORPIPE

#include <atomic>
#include <thread>

#include <c10/core/thread_pool.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

// Forward-declare the TensorPipe classes we need, to avoid including its
// headers in PyTorch's ones and thus have it become a public dependency.

// 前向声明我们需要的 TensorPipe 类，避免在 PyTorch 的头文件中包含它们，从而避免成为公共依赖项。
namespace tensorpipe {

class Context;
class Error;
class Listener;
class Message;
class Pipe;

namespace transport {
class Context;
} // namespace transport

namespace channel {
class Context;
} // namespace channel

} // namespace tensorpipe

namespace torch {
namespace distributed {
namespace rpc {

// These priorities instruct TensorPipe on which transport/channel to pick
// during handshake. Higher priorities will take precedence over lower ones.
// The transport with lowest priority will be the one used to bootstrap pipes.

// 这些优先级指导 TensorPipe 在握手期间选择传输/通道的顺序。较高优先级将优先于较低优先级。
// 具有最低优先级的传输将用于引导管道。

constexpr int64_t kShmTransportPriority = 200;
constexpr int64_t kIbvTransportPriority = 100;
// The UV transport just uses TCP and should work everywhere, thus keep it last.
constexpr int64_t kUvTransportPriority = 0;

constexpr int64_t kCmaChannelPriority = 1200;
constexpr int64_t kMultiplexedUvChannelPriority = 1100;
// The basic channel reuses a transport as a channel, and is thus our fallback.
constexpr int64_t kBasicChannelPriority = 1000;

// CPU channel have higher priority than CUDA channels, since the latter might
// handle CPU-to-CPU transfers, but will always be less efficient than their
// CPU-only counterparts.
// CPU 通道的优先级高于 CUDA 通道，因为后者可能处理 CPU 到 CPU 的传输，
// 但始终不如其仅限于 CPU 的对应部分效率高。
constexpr int64_t kCudaIpcChannelPriority = 300;
constexpr int64_t kCudaGdrChannelPriority = 200;
constexpr int64_t kCudaXthChannelPriority = 400;
constexpr int64_t kCudaBasicChannelPriority = 0;

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

struct TORCH_API TransportRegistration {
  std::shared_ptr<tensorpipe::transport::Context> transport;
  int64_t priority;
  std::string address;
};

C10_DECLARE_REGISTRY(TensorPipeTransportRegistry, TransportRegistration);

struct TORCH_API ChannelRegistration {
  std::shared_ptr<tensorpipe::channel::Context> channel;
  int64_t priority;
};

C10_DECLARE_REGISTRY(TensorPipeChannelRegistry, ChannelRegistration);

constexpr auto kDefaultNumWorkerThreads = 16;

struct TORCH_API TensorPipeRpcBackendOptions : public RpcBackendOptions {
  TensorPipeRpcBackendOptions(
      int numWorkerThreads,
      optional<std::vector<std::string>> transports,
      optional<std::vector<std::string>> channels,
      float rpc_timeout,
      std::string init_method,
      std::unordered_map<std::string, DeviceMap> device_maps = {},
      std::vector<c10::Device> devices = {})
      : RpcBackendOptions(rpc_timeout, init_method),
        numWorkerThreads(numWorkerThreads),
        transports(std::move(transports)),
        channels(std::move(channels)),
        deviceMaps(std::move(device_maps)),
        devices(std::move(devices)) {
    # 检查工作线程数是否大于零，如果不是则抛出异常信息
    TORCH_CHECK(
        numWorkerThreads > 0,
        "num_worker_threads must be positive, got ",
        numWorkerThreads);

    # 如果存在传输方式列表
    if (this->transports.has_value()) {
      # 遍历每种传输方式名称
      for (const std::string& transportName : this->transports.value()) {
        # 检查传输方式是否在注册表中，如果不在则抛出异常信息
        TORCH_CHECK(
            TensorPipeTransportRegistry()->Has(transportName),
            "Unknown transport: ",
            transportName);
      }
    }

    # 如果存在通道列表
    if (this->channels.has_value()) {
      # 遍历每种通道名称
      for (const std::string& channelName : this->channels.value()) {
        # 检查通道是否在注册表中，如果不在则抛出异常信息
        TORCH_CHECK(
            TensorPipeChannelRegistry()->Has(channelName),
            "Unknown channel: ",
            channelName);
      }
    }
  }

  # 设置特定工作节点的设备映射关系
  void setDeviceMap(const std::string& workerName, const DeviceMap& deviceMap) {
    # 查找指定工作节点的设备映射关系是否已存在
    auto iter = deviceMaps.find(workerName);
    # 如果不存在，则直接插入新的映射关系
    if (iter == deviceMaps.end()) {
      deviceMaps[workerName] = deviceMap;
    } else {
      # 如果存在，则更新现有映射关系中的每个条目
      for (auto& entry : deviceMap) {
        # 检查设备是否已存在于映射中，如果不存在则直接插入
        auto entryIter = iter->second.find(entry.first);
        if (entryIter == iter->second.end()) {
          iter->second.emplace(entry.first, entry.second);
        } else {
          # 如果存在则更新现有设备的映射关系
          entryIter->second = entry.second;
        }
      }
    }
  }

  # 工作线程数
  int numWorkerThreads;
  # 可选的传输方式列表
  const optional<std::vector<std::string>> transports;
  # 可选的通道列表
  const optional<std::vector<std::string>> channels;
  # 工作节点到设备映射的哈希表
  std::unordered_map<std::string, DeviceMap> deviceMaps;
  # 设备列表
  std::vector<c10::Device> devices;
// 结构体，用于跟踪网络数据源的指标信息
struct TORCH_API NetworkSourceInfo {
  worker_id_t srcRank;                 // 源排名，表示数据源的等级
  std::vector<uint8_t> srcMachineAddr; // 源机器地址，存储源的机器地址信息
};

// 结构体，用于跟踪聚合网络指标数据
struct TORCH_API AggregatedNetworkData {
  uint64_t numCalls{0};          // 调用次数统计
  uint64_t totalSentBytes{0};    // 总发送字节数统计
  uint64_t totalRecvBytes{0};    // 总接收字节数统计
  uint64_t totalErrors{0};       // 总错误数统计
};

// TensorPipeAgent 利用 TensorPipe（https://github.com/pytorch/tensorpipe）
// 在最快可用的传输通道中透明地移动张量和有效负载。它像一个混合 RPC 传输，
// 提供共享内存（Linux）和 TCP 支持（Linux 和 macOS）。CUDA 支持正在进行中。
explicit AtomicJitFuture(const std::vector<c10::Device>& devices) {
  // 构造函数，创建一个包含设备的 IntrusivePtr 类型的 JIT Future 对象
  jitFuture = c10::make_intrusive<at::ivalue::Future>(
      at::AnyClassType::get(), devices);
}

std::atomic_flag isComplete = ATOMIC_FLAG_INIT; // 原子标志，用于表示操作是否完成
c10::intrusive_ptr<JitFuture> jitFuture;        // JIT Future 对象的指针

// 每个客户端管道维护状态，用于跟踪待处理的响应消息和错误状态。
// pendingResponseMessage_ 应该由互斥锁保护，因为它可能与用户的 send() 调用竞争。
// TODO: 为了实现更好的性能，我们可以使用 RpcBackendOptions 配置每个客户端的管道池。
struct ClientPipe {
  explicit ClientPipe(std::shared_ptr<tensorpipe::Pipe> pipe)
      : pipe_(std::move(pipe)) {}
  std::shared_ptr<tensorpipe::Pipe> pipe_; // 共享指针，指向 TensorPipe 管道对象
  mutable std::mutex mutex_;               // 可变的互斥锁，用于保护管道的状态
  bool inError_{false};                    // 表示管道是否处于错误状态的布尔值
  // 映射，从消息请求 ID 到相应的 futures。
    // 声明一个无序映射，用于存储待处理的响应消息，键为 uint64_t 类型，值为 AtomicJitFuture 的共享指针
    std::unordered_map<uint64_t, std::shared_ptr<AtomicJitFuture>> pendingResponseMessage_;
    
    const c10::intrusive_ptr<::c10d::Store> store_;
    
    const TensorPipeRpcBackendOptions opts_;
    
    // 存储逆设备映射的无序映射。对于动态 RPC，在新的排名加入或离开组时更新这些映射。
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps_;
    
    // 由此代理使用的本地设备列表。如果应用程序未指定此字段，则使用 opts_.deviceMaps 和 reverseDeviceMaps_ 中对应的本地设备进行初始化。
    std::vector<c10::Device> devices_;
    
    // 线程池对象
    ThreadPool threadPool_;
    
    // TensorPipe 上下文的共享指针
    std::shared_ptr<tensorpipe::Context> context_;
    
    // TensorPipe 监听器的共享指针
    std::shared_ptr<tensorpipe::Listener> listener_;
    
    // 互斥锁，用于保护 connectedPipes_ 的访问
    mutable std::mutex connectedPipesMutex_;
    
    // 存储已连接管道的映射，键为 worker_id_t 类型，值为 ClientPipe 对象
    std::unordered_map<worker_id_t, ClientPipe> connectedPipes_;
    
    // 用于通过名称和 ID 快速查找 WorkerInfo 的无序映射
    std::unordered_map<worker_id_t, WorkerInfo> workerIdToInfo_;
    
    // 用于通过名称快速查找 WorkerInfo 的无序映射
    std::unordered_map<std::string, WorkerInfo> workerNameToInfo_;
    
    // 用于通过名称查找 URL 的无序映射
    std::unordered_map<std::string, std::string> workerNameToURL_;
    
    // 用于存储排名到名称的前缀存储对象
    ::c10d::PrefixStore rankToNameStore_;
    
    // 用于存储名称到地址的前缀存储对象
    ::c10d::PrefixStore nameToAddressStore_;
    
    // 用于在关闭过程中计算加入进程和活动调用数的前缀存储对象
    ::c10d::PrefixStore shutdownStore_;
    
    // 整数，表示通信组中的进程总数，默认为 0
    int worldSize_ = 0;
    
    // 下一个消息的唯一标识符，使用原子操作保证线程安全
    std::atomic<uint64_t> nextMessageID_{0};
    
    // 用于跟踪某些 RPC 是否超时的元数据结构
    struct TimeoutMessageMetadata {
      TimeoutMessageMetadata(
          uint64_t messageId_,
          std::shared_ptr<AtomicJitFuture> responseFuture_,
          std::chrono::milliseconds timeout_)
          : messageId(messageId_),
            responseFuture(std::move(responseFuture_)),
            timeout(timeout_) {}
      uint64_t messageId;
      std::shared_ptr<AtomicJitFuture> responseFuture;
      std::chrono::milliseconds timeout;
    };
    
    // 用于存储每条消息的到期时间的映射，键为 steady_clock_time_point，值为 TimeoutMessageMetadata 的向量
    std::map<steady_clock_time_point, std::vector<TimeoutMessageMetadata>> timeoutMap_;
    
    // 用于存储消息 ID 到到期时间的映射，键为 uint64_t，值为 steady_clock_time_point
    std::unordered_map<uint64_t, steady_clock_time_point> messageIdToTimeout_;
    
    // 用于轮询 timeoutMap_ 的线程对象，以检测超时的 RPC 并相应标记错误
    std::thread timeoutThread_;
    
    // timeoutMap_ 的互斥锁，用于保护其访问
    std::mutex timeoutMapMutex_;
    
    // 用于信号 timeoutThread_ 线程进行 timeoutMap_ 填充的条件变量
    std::condition_variable timeoutThreadCV_;
    
    // 根据传入的超时值计算 RPC 消息的到期时间点，返回一个 steady_clock_time_point 对象
    inline steady_clock_time_point computeRpcMessageExpiryTime(
        std::chrono::milliseconds timeout) const {
    // 返回当前时刻加上超时时长后的时间点，精确到毫秒
    return std::chrono::time_point_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() + timeout);
    }
    
    // 处理传出管道的错误情况
    void handleClientError(
        ClientPipe& clientPipe,
        const tensorpipe::Error& error);
    
    // 这是一个用于捕获时间序列指标的通用结构。它维护数据点（观测值）的累加和和计数，
    // 可以返回到目前为止观察到的数据点的平均值。目前仅用于跟踪 RPC Agent 中的 GIL 等待时间，
    // 但也可用于其他指标的跟踪。
    struct TimeSeriesMetricsTracker {
      // 到目前为止观察到的数据点的累加和
      uint64_t currentSum_;
      // 到目前为止观察到的数据点的计数
      uint64_t currentCount_;
    
      explicit TimeSeriesMetricsTracker(
          uint64_t currentSum = 0,
          uint64_t currentCount = 0);
    
      // 添加一个数据点（基本上是正在跟踪的指标的一个观察结果），更新累加和和计数。
      void addData(uint64_t dataPoint);
      // 返回到目前为止观察到的所有数据点的平均值。
      float computeAverage() const;
    };
    
    // RPC Agent 跟踪的时间序列指标映射
    std::unordered_map<std::string, TimeSeriesMetricsTracker> timeSeriesMetrics_;
    // 保护 timeSeriesMetrics_ 的互斥锁
    std::mutex metricsMutex_;
    
    // 自定义的锁保护，用于检查 RPC 组是否是动态的，并在是动态组时锁定互斥锁
    struct GroupMembershipLockGuard {
      GroupMembershipLockGuard(std::mutex& mutex, bool isStaticGroup)
          : ref_(mutex), isStaticGroup_(isStaticGroup) {
        if (isStaticGroup_) {
          ref_.lock();
        }
      }
    
      ~GroupMembershipLockGuard() {
        if (isStaticGroup_) {
          ref_.unlock();
        }
      }
    
      GroupMembershipLockGuard(const GroupMembershipLockGuard&) = delete;
    
     private:
      std::mutex& ref_;
    // 是否为静态分组的标志
    bool isStaticGroup_;
  };

  // 用于保护对群组成员数据的访问的互斥锁
  // 例如更新 workerIdToInfo_, workerNameToInfo_, workerNameToURL_ 的操作
  mutable std::mutex groupMembershipMutex_;

  // 用于跟踪网络数据的字典
  NetworkDataDict networkData_;
  // 保护 networkData_ 的互斥锁
  std::mutex networkDataMutex_;

  // 用于保护调用计数并监听其变化的互斥锁和条件变量
  std::mutex callCountMutex_;
  std::condition_variable callCountCV_;
  // 当前未处理且未出错的 RPC 调用总数
  int32_t clientActiveCalls_{0};
  // 当前未处理的 RPC 请求接收总数
  int32_t serverActiveCalls_{0};
  // 将异步完成的 RPC 请求总数
  int32_t serverActiveAsyncCalls_{0};

  // 全局优雅关闭是否已启动的原子布尔标志，
  // 如果是，则会静音远程工作进程关闭管道时产生的错误消息
  std::atomic<bool> shuttingDown_{false};

  // 增加调用计数的辅助函数，正确处理互斥锁和条件变量
  void increaseCallCount(int32_t& count);
  // 减少调用计数的辅助函数，正确处理互斥锁和条件变量
  void decreaseCallCount(int32_t& count);

  // 标记将来完成的异步 JIT 任务
  void markFutureAsComplete(
      std::shared_ptr<AtomicJitFuture> atomicFuture,
      c10::intrusive_ptr<Message> message,
      std::vector<c10::Stream> streams);
  // 标记将来发生错误的异步 JIT 任务
  void markFutureWithError(
      std::shared_ptr<AtomicJitFuture> atomicFuture,
      std::string errorMsg);
};

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE


注释：


// 结束了名为torch的命名空间
};

// 结束了名为rpc的命名空间，位于torch命名空间内部
} // namespace rpc

// 结束了名为distributed的命名空间，位于rpc命名空间内部
} // namespace distributed

// 结束了名为torch的顶层命名空间
} // namespace torch

// 如果定义了宏USE_TENSORPIPE，则结束整个文件的条件编译
#endif // USE_TENSORPIPE
```