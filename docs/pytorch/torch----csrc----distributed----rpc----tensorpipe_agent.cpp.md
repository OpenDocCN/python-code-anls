# `.\pytorch\torch\csrc\distributed\rpc\tensorpipe_agent.cpp`

```
// 包含TensorPipe代理的头文件
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

// 如果使用TensorPipe，继续包含必要的头文件
#ifdef USE_TENSORPIPE

// 包含一些标准库头文件
#include <limits>
#include <tuple>
#include <utility>

// 使用fmt库进行格式化输出
#include <fmt/format.h>

// 忽略特定的编译警告，因为TensorPipe库中可能包含已弃用的功能
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>
C10_DIAGNOSTIC_POP()

// 包含一些RPC相关的实用函数和工具
#include <torch/csrc/distributed/rpc/agent_utils.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>
#include <torch/csrc/distributed/rpc/utils.h>

// 包含C10库中的流保护和整数范围工具
#include <c10/core/StreamGuard.h>
#include <c10/util/irange.h>

// 定义在命名空间torch::distributed::rpc中的匿名命名空间，封装了私有的实用函数和常量

namespace {

// 环境变量，类似于GLOO_和NCCL_SOCKET_IFNAME，允许用户指定绑定的设备
const std::string kSocketIfnameEnvVar = "TP_SOCKET_IFNAME";
// 默认的UV地址
const std::string kDefaultUvAddress = "127.0.0.1";

// 一些与代理相关的度量和配置项的名称
const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";
const std::string kClientActiveCalls = "agent.client_active_calls";
const std::string kServerActiveCalls = "agent.server_active_calls";
const std::string kServerActiveAsyncCalls = "agent.server_active_async_calls";

// 根据输入张量、设备映射和远程名称获取用于TensorPipe的设备列表
std::vector<c10::Device> getDevicesForTensors(
    const std::vector<torch::Tensor>& tensors,
    const DeviceMap& deviceMap,
    const std::string& remoteName) {
  // 如果设备映射被覆盖，则使用覆盖后的设备映射
  const auto errStr = c10::str(
      "TensorPipe RPC backend only supports CPU tensors by default, please "
      "move your tensors to CPU before sending them over RPC, or call "
      "`set_device_map` on `TensorPipeRpcBackendOptions` to explicitly "
      "configure device mapping. ",
      "Request device mapping is not available for destination ",
      remoteName);
  std::vector<c10::Device> devices;
  devices.reserve(tensors.size());
  bool hasMappedDevice = false;
  for (const auto& t : tensors) {
    if (t.device().is_cpu()) {
      // 如果张量在CPU上，则根据设备映射决定使用的设备
      const auto deviceIter = deviceMap.find(c10::kCPU);
      if (deviceIter == deviceMap.end()) {
        devices.emplace_back(c10::kCPU);
      } else {
        devices.emplace_back(deviceIter->second);
        hasMappedDevice = true;
      }
    } else {
      // 如果张量不在CPU上，则根据其设备查找设备映射并检查是否存在
      const auto deviceIter = deviceMap.find(t.device());
      TORCH_CHECK(
          deviceIter != deviceMap.end(),
          errStr,
          " for device ",
          t.device(),
          " but received a tensor on that device.");
      devices.push_back(deviceIter->second);
      hasMappedDevice = true;
    }
  }
  // 如果没有找到映射的设备，清空设备列表
  if (!hasMappedDevice) {
    devices.clear();
  }
  return devices;
}

// 根据设备列表从池中获取流对象
std::vector<c10::Stream> getStreamsFromPoolForDevices(
    const std::vector<c10::Device>& devices) {
  // 如果设备列表为空，则返回空的流对象列表
  if (devices.empty()) {
    // 返回一个空的字典，表示没有找到匹配条件的结果
    return {};
  }
  // 创建一个 VirtualGuardImpl 对象，传入设备列表中的第一个设备类型作为参数
  c10::impl::VirtualGuardImpl impl(devices[0].type());
  // 创建一个空的 Stream 向量，预留足够的空间以容纳所有设备
  std::vector<c10::Stream> streams;
  streams.reserve(devices.size());
  // 遍历设备列表中的每一个设备
  for (const c10::Device& device : devices) {
    // 内部断言，确保每个设备的类型与 impl 对象的类型相匹配
    TORCH_INTERNAL_ASSERT(device.type() == impl.type());
    // 从全局池中获取与当前设备相关的流对象，并将其添加到 streams 向量中
    streams.push_back(impl.getStreamFromGlobalPool(device));
  }
  // 返回包含所有设备流对象的 streams 向量
  return streams;
}

// 获取给定设备列表的当前流
std::vector<c10::Stream> getCurrentStreamsForDevices(
    const std::vector<c10::Device>& devices) {
  // 如果设备列表为空，则返回空的流列表
  if (devices.empty()) {
    return {};
  }
  // 创建一个虚拟保护实现对象，使用设备列表中第一个设备的类型
  c10::impl::VirtualGuardImpl impl(devices[0].type());
  // 创建一个流向量，预留足够空间以容纳所有设备的流
  std::vector<c10::Stream> streams;
  streams.reserve(devices.size());
  // 遍历每个设备，确保设备类型与虚拟保护实现对象类型相符，并获取相应的流添加到流列表中
  for (const c10::Device& device : devices) {
    TORCH_INTERNAL_ASSERT(device.type() == impl.type());
    streams.push_back(impl.getStream(device));
  }
  // 返回所有设备的流列表
  return streams;
}

// 获取给定张量列表的设备
std::vector<c10::Device> getDevicesOfTensors(
    const std::vector<torch::Tensor>& tensors) {
  // 可选的虚拟保护实现对象
  std::optional<c10::impl::VirtualGuardImpl> impl;
  // 设备计数
  size_t deviceCount = 0;
  // 索引位集合，用于跟踪设备是否已经添加过
  std::vector<bool> indexBitset;
  // 遍历每个张量
  for (const torch::Tensor& tensor : tensors) {
    // 如果张量不是在 CPU 上，则获取其设备
    if (!tensor.is_cpu()) {
      c10::Device device = tensor.device();
      // 如果虚拟保护实现对象尚未创建，则创建并调整索引位集合的大小
      if (!impl.has_value()) {
        impl.emplace(device.type());
        indexBitset.resize(impl->deviceCount());
      }
      // 确保设备类型与虚拟保护实现对象类型相符，并且设备有有效索引
      TORCH_INTERNAL_ASSERT(device.type() == impl->type());
      TORCH_INTERNAL_ASSERT(device.has_index());
      // 如果该设备索引位尚未设置，则增加设备计数并标记该索引位为已设置
      if (!indexBitset[device.index()]) {
        deviceCount++;
        indexBitset[device.index()] = true;
      }
    }
  }
  // 创建设备列表，预留足够的空间以容纳计数好的设备
  std::vector<c10::Device> devices;
  devices.reserve(deviceCount);
  // 遍历索引位集合，将已设置的设备索引添加到设备列表中
  for (const auto idx : c10::irange(indexBitset.size())) {
    if (indexBitset[idx]) {
      devices.emplace_back(impl->type(), static_cast<c10::DeviceIndex>(idx));
    }
  }
  // 返回包含所有设备的列表
  return devices;
}

// 使消费者流等待生产者流
void makeStreamsWaitOnOthers(
    const std::vector<c10::Stream>& consumers,
    const std::vector<c10::Stream>& producers) {
  // 遍历所有生产者流
  for (const c10::Stream& producer : producers) {
    // 获取与生产者设备对应的消费者流
    const c10::Stream& consumer =
        getStreamForDevice(consumers, producer.device());
    // 创建事件对象并记录生产者流
    c10::Event event(producer.device_type());
    event.record(producer);
    // 阻塞消费者流直至事件完成
    event.block(consumer);
  }
}

} // namespace

// 定义没有警告的 TensorPipeTransportRegistry 注册表
C10_DEFINE_REGISTRY_WITHOUT_WARNING(
    TensorPipeTransportRegistry,
    TransportRegistration);

// 定义没有警告的 TensorPipeChannelRegistry 注册表
C10_DEFINE_REGISTRY_WITHOUT_WARNING(
    TensorPipeChannelRegistry,
    ChannelRegistration);

// 猜测 TensorPipeAgent 的地址
const std::string& TensorPipeAgent::guessAddress() {
  // 静态变量 uvAddress，根据环境变量或主机名查找并返回地址
  static const std::string uvAddress = []() {
    char* ifnameEnv = std::getenv(kSocketIfnameEnvVar.c_str());
    // 如果环境变量存在，则根据接口名查找地址
    if (ifnameEnv != nullptr) {
      auto [error, result] =
          tensorpipe::transport::uv::lookupAddrForIface(ifnameEnv);
      if (error) {
        // 如果查找失败，记录警告并返回默认地址
        LOG(WARNING) << "Failed to look up the IP address for interface "
                     << ifnameEnv << " (" << error.what() << "), defaulting to "
                     << kDefaultUvAddress;
        return kDefaultUvAddress;
      }
      // 返回查找到的地址
      return result;
    }
    // 否则根据主机名查找地址
    auto [error, result] = tensorpipe::transport::uv::lookupAddrForHostname();
    if (error) {
      // 如果查找失败，记录警告并返回默认地址
      LOG(WARNING) << "Failed to look up the IP address for the hostname ("
                   << error.what() << "), defaulting to " << kDefaultUvAddress;
      return kDefaultUvAddress;
    }
    // 返回查找到的地址
    return result;
  }();
  // 返回最终确定的地址
  return uvAddress;
}

namespace {
// 创建一个基于 UV transport 的 TransportRegistration 实例的工厂函数
std::unique_ptr<TransportRegistration> makeUvTransport() {
  // 使用 tensorpipe::transport::uv::create() 创建一个 UV transport context
  auto context = tensorpipe::transport::uv::create();
  // 获取 TensorPipeAgent 猜测的地址作为字符串
  std::string address = TensorPipeAgent::guessAddress();
  // 创建并返回一个 TransportRegistration 实例，包含 context、kUvTransportPriority 和 address
  return std::make_unique<TransportRegistration>(TransportRegistration{
      std::move(context), kUvTransportPriority, std::move(address)});
}

// UV transport 使用标准的 TCP 连接实现，利用 libuv 实现跨平台功能。
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, uv, makeUvTransport);

#if TENSORPIPE_HAS_SHM_TRANSPORT

// 创建一个基于 SHM transport 的 TransportRegistration 实例的工厂函数
std::unique_ptr<TransportRegistration> makeShmTransport() {
  // 使用 tensorpipe::transport::shm::create() 创建一个 SHM transport context
  auto context = tensorpipe::transport::shm::create();
  // 创建并返回一个 TransportRegistration 实例，包含 context、kShmTransportPriority 和空字符串地址
  return std::make_unique<TransportRegistration>(
      TransportRegistration{std::move(context), kShmTransportPriority, ""});
}

// SHM transport 使用匿名共享内存中的环形缓冲区实现连接（还利用 UNIX 域套接字来引导连接和交换文件描述符）。
// 由于使用了一些高级特性（如 O_TMPFILE、eventfd 等），因此仅在 Linux 环境下可用。
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, shm, makeShmTransport);

#endif // TENSORPIPE_HAS_SHM_TRANSPORT

#if TENSORPIPE_HAS_IBV_TRANSPORT

// 创建一个基于 IBV transport 的 TransportRegistration 实例的工厂函数
std::unique_ptr<TransportRegistration> makeIbvTransport() {
  // 使用 tensorpipe::transport::ibv::create() 创建一个 IBV transport context
  auto context = tensorpipe::transport::ibv::create();
  // 获取 TensorPipeAgent 猜测的地址作为字符串
  std::string address = TensorPipeAgent::guessAddress();
  // 创建并返回一个 TransportRegistration 实例，包含 context、kIbvTransportPriority 和 address
  return std::make_unique<TransportRegistration>(TransportRegistration{
      std::move(context), kIbvTransportPriority, std::move(address)});
}

// IBV transport 使用 InfiniBand 队列对发送数据，本地复制数据到和从一个注册到 libibverbs 的缓冲区，并发出 RDMA 写操作以在机器间传输数据（并发出发送以确认）。
// 初始启动时使用标准的 TCP 连接来交换设置信息。仅在 Linux 环境下可用。
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, ibv, makeIbvTransport);

#endif // TENSORPIPE_HAS_IBV_TRANSPORT

// 创建一个基本通道的 ChannelRegistration 实例的工厂函数
std::unique_ptr<ChannelRegistration> makeBasicChannel() {
  // 使用 tensorpipe::channel::basic::create() 创建一个基本通道 context
  auto context = tensorpipe::channel::basic::create();
  // 创建并返回一个 ChannelRegistration 实例，包含 context 和 kBasicChannelPriority
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kBasicChannelPriority});
}

// 基本通道只是一个简单的适配器包装，允许任何传输作为通道使用。
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, basic, makeBasicChannel);

#if TENSORPIPE_HAS_CMA_CHANNEL

// 创建一个基于 CMA channel 的 ChannelRegistration 实例的工厂函数
std::unique_ptr<ChannelRegistration> makeCmaChannel() {
  // 使用 tensorpipe::channel::cma::create() 创建一个 CMA channel context
  auto context = tensorpipe::channel::cma::create();
  // 创建并返回一个 ChannelRegistration 实例，包含 context 和 kCmaChannelPriority
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCmaChannelPriority});
}

// CMA channel 使用 Linux 跨内存附加系统调用（process_vm_readv 和 _writev），允许一个进程访问另一个进程的私有内存（只要它们属于同一用户和其他安全性限制）
#endif // TENSORPIPE_HAS_CMA_CHANNEL
// 定义了一个常量 kNumUvThreads，表示使用的 UV 线程数量为 16
constexpr static int kNumUvThreads = 16;

// 创建并返回一个多路复用 UV 通道的唯一指针
std::unique_ptr<ChannelRegistration> makeMultiplexedUvChannel() {
  // 创建存储共享指针的容器，用于存储 UV 传输的上下文和监听器
  std::vector<std::shared_ptr<tensorpipe::transport::Context>> contexts;
  std::vector<std::shared_ptr<tensorpipe::transport::Listener>> listeners;

  // 使用循环创建 kNumUvThreads 个 UV 上下文，并将它们的地址推测给地址字符串
  for (const auto laneIdx C10_UNUSED : c10::irange(kNumUvThreads)) {
    auto context = tensorpipe::transport::uv::create();
    std::string address = TensorPipeAgent::guessAddress();
    contexts.push_back(std::move(context));  // 将创建的上下文移动到 contexts 容器中
    listeners.push_back(contexts.back()->listen(address));  // 监听器监听指定地址
  }

  // 使用创建的上下文和监听器创建多路复用 UV 通道的上下文对象
  auto context = tensorpipe::channel::mpt::create(
      std::move(contexts), std::move(listeners));
  
  // 返回包含通道上下文和优先级的通道注册对象的唯一指针
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kMultiplexedUvChannelPriority});
}

// 注册一个多路复用 UV 通道的创建函数到 TensorPipeChannelRegistry 中
C10_REGISTER_CREATOR(
    TensorPipeChannelRegistry,
    mpt_uv,
    makeMultiplexedUvChannel);
    messageIdToTimeout_.erase(messageId);


// 从名为 messageIdToTimeout_ 的关联容器中删除键为 messageId 的条目
}

void TensorPipeAgent::prepareNames(bool isStaticGroup) {
  // 创建一个空的哈希映射，用于存储工作节点名称到节点 ID 的映射关系
  std::unordered_map<std::string, worker_id_t> nameToId;
  
  if (isStaticGroup) {
    // 如果是静态群组，调用 collectNames 函数收集节点名称和 ID 的映射关系
    nameToId = collectNames(rankToNameStore_, workerInfo_.id_, workerInfo_.name_, worldSize_);
  } else {
    // 如果不是静态群组，调用 collectCurrentNames 函数收集当前节点名称和 ID 的映射关系
    nameToId = collectCurrentNames(rankToNameStore_, workerInfo_.id_, workerInfo_.name_);
  }

  // 将收集到的每个工作节点的名称和 ID 添加到 workerIdToInfo_ 和 workerNameToInfo_ 中
  for (const auto& entry : nameToId) {
    const auto& workerName = entry.first;
    const auto& workerId = entry.second;
    // 将 workerId 映射到 WorkerInfo 对象并添加到 workerIdToInfo_
    workerIdToInfo_.emplace(workerId, WorkerInfo(workerName, workerId));
    // 将 workerName 映射到 WorkerInfo 对象并添加到 workerNameToInfo_
    workerNameToInfo_.emplace(workerName, WorkerInfo(workerName, workerId));
  }
}

void TensorPipeAgent::checkAndSetStaticGroup(
    const c10::intrusive_ptr<::c10d::Store>& store) {
  // 定义键值字符串，表示是否为静态群组的属性键
  std::string isStaticGroupKey("rpcIsStaticGroup");

  // 将布尔值 isStaticGroup_ 转换为字符串形式
  std::string isStaticGroupStr = isStaticGroup_ ? "true" : "false";
  
  // 将 isStaticGroupStr 转换为字节向量形式
  std::vector<uint8_t> isStaticGroupVec(
      (uint8_t*)isStaticGroupStr.c_str(),
      (uint8_t*)isStaticGroupStr.c_str() + isStaticGroupStr.length());
  
  // 调用 store 的 compareSet 方法，将 isStaticGroupKey 和 isStaticGroupVec 作为参数进行比较和设置
  std::vector<uint8_t> returnedVec;
  returnedVec = store->compareSet(
      isStaticGroupKey, std::vector<uint8_t>(), isStaticGroupVec);
  
  // 将返回的字节向量转换为字符串形式
  std::string returnedVal = std::string(returnedVec.begin(), returnedVec.end());
  
  // 检查返回的值是否与 isStaticGroupStr 相等，如果不相等则抛出异常
  TORCH_CHECK(
      returnedVal == isStaticGroupStr,
      fmt::format(
          "RPC group mixes statically and dynamically initialized members which is not supported. ",
          "Static group property is initialized as {} and is trying to be set as {} ",
          isStaticGroup_,
          returnedVal));
}

TensorPipeAgent::TensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string selfName,
    worker_id_t selfId,
    optional<int> worldSize,
    TensorPipeRpcBackendOptions opts,
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
    std::vector<c10::Device> devices,
    std::unique_ptr<RequestCallback> cb)
    : RpcAgent(
          WorkerInfo(std::move(selfName), selfId),
          std::move(cb),
          std::chrono::milliseconds(
              (long)(opts.rpcTimeoutSeconds * kSecToMsConversion))),  // 初始化 RpcAgent 基类
      isStaticGroup_(worldSize.has_value()),  // 根据 worldSize 是否有值设置 isStaticGroup_
      store_(store),  // 初始化存储 store
      opts_(std::move(opts)),  // 初始化选项 opts
      reverseDeviceMaps_(std::move(reverseDeviceMaps)),  // 初始化反向设备映射 reverseDeviceMaps
      devices_(std::move(devices)),  // 初始化设备列表 devices
      threadPool_(opts_.numWorkerThreads),  // 初始化线程池 threadPool
      context_(std::make_shared<tensorpipe::Context>(  // 使用 workerInfo_.name_ 创建 TensorPipe 上下文
          tensorpipe::ContextOptions().name(workerInfo_.name_))),
      rankToNameStore_("names", store),  // 初始化 rankToNameStore_
      nameToAddressStore_("addrs", store),  // 初始化 nameToAddressStore_
      shutdownStore_("shutdown", store) {  // 初始化 shutdownStore_
  
  if (isStaticGroup_) {
    // 如果是静态群组，执行以下代码块
    worldSize_ = worldSize.value();
  }

  // 将 worldSize 的值赋给 worldSize_ 变量
  worldSize_ = worldSize.value();

  // 检查静态组属性与存储中的匹配情况
  checkAndSetStaticGroup(store);

  // 准备工作进程的名称列表
  prepareNames(isStaticGroup_);

  // 初始化时间序列指标跟踪映射
  timeSeriesMetrics_.emplace(kGilAverageWaitTime, TimeSeriesMetricsTracker());
}

// TensorPipeAgent 析构函数
TensorPipeAgent::~TensorPipeAgent() {
  // 打印销毁信息，显示代理所属的工作器名称
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is being destroyed";
  // 执行关闭操作
  shutdown();
}

// TensorPipeAgent 类的 startImpl 方法
void TensorPipeAgent::startImpl() {
  // 打印启动信息，显示代理所属的工作器名称
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is starting";

  // 存储所有地址的向量
  std::vector<std::string> addresses;
  // 初始化最低优先级为最大整数
  int lowestPriority = std::numeric_limits<int>::max();
  // 存储优先级最低的传输方式的字符串
  std::string lowestPriorityTransport;

  // 注册传输方式
  for (auto& key : TensorPipeTransportRegistry()->Keys()) {
    // 初始化优先级为 -1
    int64_t priority = -1;
    // 如果指定了传输方式选项
    if (opts_.transports.has_value()) {
      // 在传输方式选项中查找当前 key
      auto iter =
          std::find(opts_.transports->begin(), opts_.transports->end(), key);
      // 如果未找到，则跳过当前传输方式
      if (iter == opts_.transports->end()) {
        continue;
      }
      // 按照在选项中的倒序分配优先级，确保先出现的传输方式获得更高的优先级
      priority =
          opts_.transports->size() - 1 - (iter - opts_.transports->begin());
    }
    // 创建传输方式注册对象
    std::unique_ptr<TransportRegistration> reg =
        TensorPipeTransportRegistry()->Create(key);
    // 如果传输方式不可用，则跳过
    if (!reg->transport->isViable()) {
      continue;
    }
    // 如果优先级仍为 -1，则使用注册对象的默认优先级
    if (priority == -1) {
      priority = reg->priority;
    }
    // 更新最低优先级和对应的传输方式字符串
    if (priority < lowestPriority) {
      lowestPriority = priority;
      lowestPriorityTransport = key;
    }
    // 构造地址字符串并添加到地址向量中
    addresses.push_back(c10::str(key, "://", reg->address));
    // 在上下文中注册传输方式
    context_->registerTransport(
        priority, std::move(key), std::move(reg->transport));
  }

  // 注册通道
  for (auto& key : TensorPipeChannelRegistry()->Keys()) {
    // 初始化优先级为 -1
    int64_t priority = -1;
    // 如果指定了通道选项
    if (opts_.channels.has_value()) {
      // 在通道选项中查找当前 key
      auto iter =
          std::find(opts_.channels->begin(), opts_.channels->end(), key);
      // 如果未找到，则跳过当前通道
      if (iter == opts_.channels->end()) {
        continue;
      }
      // 按照在选项中的倒序分配优先级，确保先出现的通道获得更高的优先级
      priority = opts_.channels->size() - 1 - (iter - opts_.channels->begin());
    }
    // 创建通道注册对象
    std::unique_ptr<ChannelRegistration> reg =
        TensorPipeChannelRegistry()->Create(key);
    // 如果通道不可用，则跳过
    if (!reg->channel->isViable()) {
      continue;
    }
    // 如果优先级仍为 -1，则使用注册对象的默认优先级
    if (priority == -1) {
      priority = reg->priority;
    }
    // 在上下文中注册通道
    context_->registerChannel(
        priority, std::move(key), std::move(reg->channel));
  }

  // 在上下文中监听所有地址
  listener_ = context_->listen(addresses);

  // 存储代理使用的地址
  const auto address = listener_->url(lowestPriorityTransport);
  nameToAddressStore_.set(workerInfo_.name_, address);

  // 打印使用的地址信息
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is using address "
          << address;

  // 遍历所有工作器信息
  for (const auto& p : workerNameToInfo_) {
    // 获取工作器名称
    const auto& name = p.first;
    // 获取工作器对应的地址数据
    auto nodeAddrData = nameToAddressStore_.get(name);
    // 将地址数据转换为字符串
    auto nodeAddrStr =
        std::string((const char*)nodeAddrData.data(), nodeAddrData.size());
    workerNameToURL_.insert({name, nodeAddrStr});
  }

  // Start the Timeout Thread
  // 启动超时线程，调用 pollTimeoutRpcs 方法
  timeoutThread_ = std::thread(&TensorPipeAgent::pollTimeoutRpcs, this);

  // Listen for incoming connections
  // 监听并接受传入的连接
  listener_->accept([this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    // Handle listener acceptance callback
    // 处理监听器接受连接后的回调函数
    onListenerAccepted(error, pipe);
  });
}

// 当监听器接受连接时的回调函数
void TensorPipeAgent::onListenerAccepted(
    const tensorpipe::Error& error,  // 错误对象，指示连接是否成功或失败
    std::shared_ptr<tensorpipe::Pipe>& pipe) {  // 共享指针，表示接受到的管道
  if (error) {  // 如果有错误发生
    if (error.isOfType<tensorpipe::ListenerClosedError>() &&
        !rpcAgentRunning_.load()) {
      // 如果是监听器关闭错误且 RPC Agent 没有运行，这是预期的情况。
    } else {
      LOG(WARNING) << "RPC agent for " << workerInfo_.name_
                   << " encountered error when accepting incoming pipe: "
                   << error.what();  // 记录警告日志，显示错误信息
    }
    return;  // 返回，不继续处理
  }

  // 接受下一个连接请求
  listener_->accept([this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    onListenerAccepted(error, pipe);  // 递归调用自身，处理新的连接
  });

  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " accepted incoming pipe from " << pipe->getRemoteName();  // 记录信息日志，显示接受到的管道远程名称

  // 准备进行服务器端读取操作
  respond(pipe);  // 调用 respond 函数，进行后续处理
}

// 管道读取操作
void TensorPipeAgent::pipeRead(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,  // 共享指针，表示要读取的管道
    std::function<void(
        const tensorpipe::Error&,
        c10::intrusive_ptr<Message>,
        std::vector<c10::Stream>)> fn) noexcept {  // 回调函数，用于处理读取结果
  pipe->readDescriptor([this, fn{std::move(fn)}, pipe](
                           const tensorpipe::Error& error,
                           tensorpipe::Descriptor tpDescriptor) mutable {
    if (error) {  // 如果读取描述符时发生错误
      fn(error, c10::intrusive_ptr<Message>(), {});  // 调用回调函数通知错误
      return;
    }

    std::vector<c10::Stream> streams;
    {
      GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
      streams = getStreamsFromPoolForDevices(devices_);  // 获取用于设备的流
    }
    auto [tpAllocation, tpBuffers] = tensorpipeAllocate(tpDescriptor, streams);  // 分配用于读取的内存和缓冲区

    pipe->read(
        std::move(tpAllocation),
        [tpDescriptor{std::move(tpDescriptor)},
         tpBuffers{
             std::make_shared<TensorpipeReadBuffers>(std::move(tpBuffers))},
         fn{std::move(fn)},
         streams{std::move(streams)}](const tensorpipe::Error& error) mutable {
          if (error) {  // 如果读取数据时发生错误
            fn(error, c10::intrusive_ptr<Message>(), {});  // 调用回调函数通知错误
            return;
          }

          // FIXME This does some unpickling, which could be a bit expensive:
          // perhaps it would be best to perform it inside the worker threads?
          // 对数据进行反序列化，生成 RPC 消息
          c10::intrusive_ptr<Message> rpcMessage = tensorpipeDeserialize(
              std::move(tpDescriptor), std::move(*tpBuffers));

          fn(error, std::move(rpcMessage), std::move(streams));  // 调用回调函数，传递反序列化后的消息和流
        });
  });
}

// 管道写入操作
void TensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,  // 共享指针，表示要写入的管道
    c10::intrusive_ptr<Message> rpcMessage,  // RPC 消息的智能指针
    std::vector<c10::Device>&& devices,  // 设备列表的右值引用
    std::vector<c10::Stream> streams,  // 流列表
    std::function<void(const tensorpipe::Error&)>&& fn) {  // 回调函数，处理写入结果
    std::function<void(const tensorpipe::Error&)> fn) noexcept {

# 定义一个名为 `fn` 的参数，类型为 `std::function`，用于接收处理 `tensorpipe::Error` 的回调函数。


  auto [tpMessage, tpBuffers] =
      tensorpipeSerialize(std::move(rpcMessage), std::move(devices), streams);

# 调用 `tensorpipeSerialize` 函数，传入 `rpcMessage`, `devices` 和 `streams` 参数，并将返回的结果解构为 `tpMessage` 和 `tpBuffers` 变量。


  pipe->write(
      std::move(tpMessage),
      [tpBuffers{
           std::make_shared<TensorpipeWriteBuffers>(std::move(tpBuffers))},
       fn{std::move(fn)},
       streams{std::move(streams)}](const tensorpipe::Error& error) {
        fn(error);
      });

# 调用 `pipe` 对象的 `write` 方法，传入 `tpMessage` 和一个 lambda 表达式作为回调函数。Lambda 表达式捕获了 `tpBuffers`、`fn` 和 `streams` 变量，并定义了一个 `const tensorpipe::Error& error` 参数用于接收错误信息。在 lambda 函数体内部，调用传入的 `fn` 函数，并传入 `error` 参数。
// 当前方法用于在TensorPipe管道上发送已完成的响应消息
void TensorPipeAgent::sendCompletedResponseMessage(
    std::shared_ptr<tensorpipe::Pipe>& pipe,  // TensorPipe管道的共享指针
    JitFuture& futureResponseMessage,          // 表示未来响应消息的JitFuture对象
    uint64_t messageId,                        // 消息ID
    std::vector<c10::Stream> streams) {        // 流向量
  // 如果RPC代理未运行，则记录警告并返回
  if (!rpcAgentRunning_.load()) {
    LOG(WARNING) << "RPC agent for " << workerInfo_.name_
                 << " won't send response to request #" << messageId << " to "
                 << pipe->getRemoteName() << ", as the agent is shutting down";
    return;
  }

  // 记录日志，表明正在发送响应消息
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " is sending response to request #" << messageId << " to "
          << pipe->getRemoteName();

  // 检查未来响应消息是否有错误
  if (!futureResponseMessage.hasError()) {
    // 将未来响应消息转换为自定义类Message的指针
    c10::intrusive_ptr<Message> responseMessage =
        futureResponseMessage.value().toCustomClass<Message>();
    responseMessage->setId(messageId);  // 设置消息ID

    std::vector<c10::Device> devices;
    try {
      // 获取用于远程端口的设备列表
      devices = getDevicesForRemote(pipe->getRemoteName(), *responseMessage);
    } catch (const std::exception& e) {
      // 如果出现异常，则创建异常响应消息
      responseMessage = createExceptionResponse(e.what(), messageId);
    }

    // 遍历响应消息中的张量
    for (const auto& tensor : responseMessage->tensors()) {
      const auto device = tensor.device();
      // 如果设备不是CPU
      if (!device.is_cpu()) {
        // 获取组成员锁，检查设备是否属于当前设备组
        GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
        if (std::find(devices_.begin(), devices_.end(), device) ==
            devices_.end()) {
          // 如果设备不在允许列表中，则创建异常响应消息
          std::ostringstream oss;
          std::copy(
              devices_.begin(),
              devices_.end(),
              std::ostream_iterator<c10::Device>(oss, ", "));
          responseMessage = createExceptionResponse(
              c10::str(
                  "RPC detected that a user-function output tensor on device ",
                  device,
                  ". This device is not one of the input tensor devices: ",
                  oss.str(),
                  "which is not yet supported. Please file a feature request "
                  "issue in PyTorch GitHub repo."),
              messageId);
          break;
        }
      }
    }

    // 使用管道发送消息
    pipeWrite(
        pipe,
        std::move(responseMessage),
        std::move(devices),
        std::move(streams),
        [this, pipe, messageId](const tensorpipe::Error& error) {
          if (error) {
            // 记录发送过程中的错误日志
            LOG(WARNING)
                << "RPC agent for " << workerInfo_.name_
                << " encountered error when sending response to request #"
                << messageId << " to " << pipe->getRemoteName() << ": "
                << error.what();
            return;
          }

          // 记录消息发送完成的日志
          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " done sending response to request #" << messageId
                  << " to " << pipe->getRemoteName();
        });
  } else {
    // 调用管道写入函数，将异常响应、设备信息、流数据和错误处理函数传入
    pipeWrite(
        // 管道对象
        pipe,
        // 创建异常响应消息，使用尝试检索的错误消息和消息ID
        createExceptionResponse(
            futureResponseMessage.tryRetrieveErrorMessage(), messageId),
        // 设备信息暂未提供，使用空对象
        /* devices */ {},
        // 将流数据移动至函数内部
        std::move(streams),
        // 错误处理函数，捕获并记录发送过程中的错误信息
        [this, pipe, messageId](const tensorpipe::Error& error) {
          // 如果有错误发生，则记录警告日志
          if (error) {
            LOG(WARNING)
                << "RPC agent for " << workerInfo_.name_
                << " encountered error when sending response to request #"
                << messageId << " to " << pipe->getRemoteName() << ": "
                << error.what();
            return;
          }

          // 否则，记录调试日志表示发送成功
          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " done sending response to request #" << messageId
                  << " to " << pipe->getRemoteName();
        });
  }

  // 结束之前的if语句块，与第一个if语句块对应，保证语句的正确性和完整性
}

c10::intrusive_ptr<JitFuture> TensorPipeAgent::send(
    const WorkerInfo& toWorkerInfo,  // 发送消息的目标工作节点信息
    c10::intrusive_ptr<Message> requestMessage,  // 要发送的消息
    const float rpcTimeoutSeconds,  // RPC超时时间
    const DeviceMap& deviceMap) {  // 设备映射表

  TORCH_CHECK(
      requestMessage->isRequest(),
      "TensorPipeAgent::send(..) is only for sending requests.");  // 检查消息是否为请求类型

  if (!rpcAgentRunning_.load()) {  // 如果RPC代理未运行
    auto err = c10::str(
        "Node ",
        RpcAgent::getWorkerInfo().id_,
        "tried to send() a message of type ",
        requestMessage->type(),
        " but RPC is no longer running on this node.");
    TORCH_CHECK(false, err);  // 抛出错误，RPC在该节点上不再运行
  }

  const auto& url = findWorkerURL(toWorkerInfo);  // 根据目标工作节点信息查找对应的URL地址

  decltype(connectedPipes_)::iterator it;
  {
    std::unique_lock<std::mutex> lock(connectedPipesMutex_);

    // 检查是否已经存在与目标地址的连接
    it = connectedPipes_.find(toWorkerInfo.id_);
    if (it == connectedPipes_.end()) {
      // 如果不存在连接，则创建一个新的ClientPipe对象并与目标地址建立连接
      it = connectedPipes_
               .emplace(
                   std::piecewise_construct,
                   std::forward_as_tuple(toWorkerInfo.id_),
                   std::forward_as_tuple(context_->connect(
                       url,
                       tensorpipe::PipeOptions().remoteName(
                           toWorkerInfo.name_))))
               .first;
    }
  }
  ClientPipe& clientPipe = it->second;  // 获取或创建的ClientPipe对象的引用

  std::shared_ptr<torch::distributed::rpc::TensorPipeAgent::AtomicJitFuture>
      futureResponseMessage;
  {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
    futureResponseMessage = std::make_shared<AtomicJitFuture>(devices_);
  }
  uint64_t messageId = nextMessageID_++;
  requestMessage->setId(messageId);  // 设置请求消息的ID

  {
    std::unique_lock<std::mutex> lock(clientPipe.mutex_);
    clientPipe.pendingResponseMessage_[messageId] = futureResponseMessage;  // 将未来的响应消息存储在ClientPipe中
  }

  // 获取请求消息中张量的设备信息。如果设备映射未正确配置，可能会抛出异常。
  std::vector<c10::Device> devices;
  if (deviceMap.empty()) {
    devices =
        getDevicesForRemote(clientPipe.pipe_->getRemoteName(), *requestMessage);  // 根据远程名称获取设备信息
  } else {
    // 如果指定了deviceMap，则使用指定的设备映射
    devices = getDevicesForTensors(
        requestMessage->tensors(),
        deviceMap,
        clientPipe.pipe_->getRemoteName());
  }


  // 获取用于张量的设备列表，使用请求消息中的张量，设备映射和客户端管道的远程名称
  devices = getDevicesForTensors(
      requestMessage->tensors(),
      deviceMap,
      clientPipe.pipe_->getRemoteName());



  futureResponseMessage->jitFuture->addCallback(
      [this](JitFuture& /* unused */) {
        TORCH_INTERNAL_ASSERT(
            this->threadPool_.inThreadPool(),
            "Future marked complete from outside the thread pool");
      });


  // 将回调函数添加到 jitFuture 对象，用于在 future 完成时执行
  futureResponseMessage->jitFuture->addCallback(
      [this](JitFuture& /* unused */) {
        // 断言：确保回调函数在正确的线程池中执行
        TORCH_INTERNAL_ASSERT(
            this->threadPool_.inThreadPool(),
            "Future marked complete from outside the thread pool");
      });



  increaseCallCount(clientActiveCalls_);


  // 增加客户端活跃调用计数
  increaseCallCount(clientActiveCalls_);



  // Use the default RPC timeout if no timeout is specified for this send call
  auto timeout = rpcTimeoutSeconds == kUnsetRpcTimeout
      ? getRpcTimeout()
      : std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * kSecToMsConversion));


  // 如果在发送调用时未指定超时时间，则使用默认的 RPC 超时时间
  auto timeout = rpcTimeoutSeconds == kUnsetRpcTimeout
      ? getRpcTimeout()
      : std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * kSecToMsConversion));



  // We only add to the timeoutMap_ if the timeout is not 0. Per our
  // documentation, a user-provided timeout of 0 indicates the RPC should never
  // expire (infinite timeout), so there is no need to track it in the
  // timeoutMap_.
  steady_clock_time_point expirationTime;
  if (timeout.count() != 0) {
    // Compute the expiration time for this message based on the timeout
    expirationTime = computeRpcMessageExpiryTime(timeout);

    // Add the Future to the right vector in the timeoutMap_
    {
      std::unique_lock<std::mutex> lock(timeoutMapMutex_);
      auto& timeoutFuturesVector = timeoutMap_[expirationTime];
      messageIdToTimeout_.emplace(messageId, expirationTime);
      timeoutFuturesVector.emplace_back(
          messageId, futureResponseMessage, timeout);
    }
    timeoutThreadCV_.notify_one();
  }


  // 只有当超时时间不为 0 时，才将其添加到 timeoutMap_ 中。根据文档说明，用户提供的超时时间为 0 表示 RPC 永不过期（无限超时），
  // 因此无需在 timeoutMap_ 中跟踪它。
  steady_clock_time_point expirationTime;
  if (timeout.count() != 0) {
    // 根据超时时间计算此消息的到期时间
    expirationTime = computeRpcMessageExpiryTime(timeout);

    // 将 Future 添加到 timeoutMap_ 中合适的向量中
    {
      std::unique_lock<std::mutex> lock(timeoutMapMutex_);
      auto& timeoutFuturesVector = timeoutMap_[expirationTime];
      messageIdToTimeout_.emplace(messageId, expirationTime);
      timeoutFuturesVector.emplace_back(
          messageId, futureResponseMessage, timeout);
    }
    // 通知超时线程有新的超时任务
    timeoutThreadCV_.notify_one();
  }



  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is sending request #"
          << messageId << " to " << clientPipe.pipe_->getRemoteName();


  // 记录日志：RPC 代理（对应的工作器名称）正在向指定的远程客户端管道发送请求 #
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is sending request #"
          << messageId << " to " << clientPipe.pipe_->getRemoteName();



  std::vector<c10::Stream> streams;
  {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);


  // 创建空的流向量
  std::vector<c10::Stream> streams;
  {
    // 在 groupMembershipMutex_ 上使用 GroupMembershipLockGuard 进行加锁，如果是静态群组则加锁
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
// 处理客户端管道发生错误的情况
void TensorPipeAgent::handleClientError(
    ClientPipe& clientPipe,
    const tensorpipe::Error& error) {
  // 当管道发生错误时，所有挂起的操作将被中止，并且所有回调函数将被调用并传入错误信息，
  // 因此我们立即清空属于该管道的所有未来消息。
  decltype(clientPipe.pendingResponseMessage_) pendingMsgs;
  {
    // 锁住客户端管道的互斥锁，以确保线程安全地访问管道状态
    std::lock_guard<std::mutex> lock(clientPipe.mutex_);
    // 交换挂起的响应消息，将其移至局部变量 pendingMsgs 中
    std::swap(clientPipe.pendingResponseMessage_, pendingMsgs);
    // 标记客户端管道处于错误状态
    clientPipe.inError_ = true;
  }
  // 获取错误信息的字符串表示
  std::string errorMsg = error.what();
  // 遍历所有挂起的消息
  for (auto& p : pendingMsgs) {
    // 标记将来的消息为错误状态，传入错误信息
    markFutureWithError(std::move(p.second), errorMsg);

    // 从超时映射中移除对应的条目
    removeFromTimeoutMap(p.first);
  }
}

// 轮询超时的 RPC 请求
void TensorPipeAgent::pollTimeoutRpcs() {
  // 在 RPC 代理运行时循环执行
  while (rpcAgentRunning_.load()) {
    // 使用独占锁定超时映射的互斥锁
    std::unique_lock<std::mutex> lock(timeoutMapMutex_);

    // 等待直到超时映射中最早超时的 RPC 请求。需要确保在映射为空时也继续睡眠，
    // 并在 RPC 代理已关闭时退出睡眠。
    for (;;) {
      if (!rpcAgentRunning_.load()) {
        return;
      }

      if (!timeoutMap_.empty()) {
        // 获取超时映射中最早超时的时间点
        steady_clock_time_point earliestTimeout = timeoutMap_.begin()->first;
        // 如果当前时间已经超过最早超时时间，则跳出循环
        if (std::chrono::steady_clock::now() >= earliestTimeout) {
          break;
        }
        // 在超时时间点前等待，直到被唤醒
        timeoutThreadCV_.wait_until(lock, earliestTimeout);
      } else {
        // 当超时映射为空时，等待被唤醒
        timeoutThreadCV_.wait(lock);
      }
    }

    // 将所有超时的未来消息移至一个独立的向量中，以便在锁外处理
    std::vector<TimeoutMessageMetadata> timedOutFutures =
        std::move(timeoutMap_.begin()->second);

    // 可以安全地从超时映射中删除该键，因为所有这些未来消息都将被处理
    timeoutMap_.erase(timeoutMap_.begin());

    // 遍历所有超时的未来消息的元数据
    for (auto& timeoutMetadata : timedOutFutures) {
      // 从 messageIdToTimeout 映射中移除相应的条目
      messageIdToTimeout_.erase(timeoutMetadata.messageId);
    }
    lock.unlock();

    // 为超时的未来消息设置错误状态。在锁外执行此操作，以避免由 setError 调用触发的
    // 回调可能引发的锁定顺序反转问题。
    for (auto& timeoutMetadata : timedOutFutures) {
      // 格式化 RPC 超时错误信息字符串
      std::string errorMsg =
          fmt::format(kRpcTimeoutErrorStr, timeoutMetadata.timeout.count());
      // 创建 RPC 错误对象并标记未来消息为错误状态
      auto err = makeRPCError(errorMsg, RPCErrorType::TIMEOUT);
      markFutureWithError(
          std::move(timeoutMetadata.responseFuture), std::move(err));
    }
  }
}
// 获得对 callCountMutex_ 的独占锁，确保线程安全
std::unique_lock<std::mutex> lock(callCountMutex_);
// 在此处，local worker 的 ActiveCallCount 为 0，表示将要关闭（任何后续调用将被丢弃）
callCountCV_.wait(lock, [this] { return clientActiveCalls_ == 0; });

// 从存储中移除当前 Agent 的 WorkerInfo
removeCurrentName(rankToNameStore_, workerInfo_.id_, workerInfo_.name_);

// 设置用于析构期间的内部变量
shuttingDown_ = true;
// 离开组的操作
void TensorPipeAgent::leaveGroup() {

// This method behaves like a barrier, as it
    {
      // 使用互斥锁确保对 callCountMutex_ 的唯一访问
      std::unique_lock<std::mutex> lock(callCountMutex_);
    
      // 此时，客户端调用的计数可能再次变为非零。
      // 我们不能等待这些调用完成，因为其他工作线程正在等待我们执行 allreduce 操作，
      // 如果阻塞它们的话会有问题。因此，即使计数非零，我们也会发送我们的计数。
      // 如果有任何人（无论是我们还是其他工作线程）有非零计数，我们将进行另一轮处理。
      VLOG(1) << "RPC agent for " << workerInfo_.name_
              << " exited the barrier and found " << clientActiveCalls_
              << " active client calls";
    
      // 调用 syncCallCount 函数，同步获取客户端活动调用的总数
      int totalClientActiveCalls =
          syncCallCount(shutdownStore_, worldSize_, clientActiveCalls_);
    
      VLOG(1) << "RPC agent for " << workerInfo_.name_
              << " completed sync call counts and got a total of "
              << totalClientActiveCalls
              << " active client calls across all workers";
    
      // 如果总的客户端活动调用数为零，执行以下操作
      if (totalClientActiveCalls == 0) {
        // 如果设置了 shutdown 标志，将 shuttingDown_ 标记为 true，
        // 然后调用 syncCallCount 函数同步更新所有 worker 的状态
        if (shutdown) {
          shuttingDown_ = true;
          syncCallCount(shutdownStore_, worldSize_);
        }
        // 跳出循环，结束执行
        break;
      }
    }
    // 输出日志，指示当前 RPC agent 完成了加入操作
    VLOG(1) << "RPC agent for " << workerInfo_.name_ << " done joining";
}

// 关闭 TensorPipeAgent 的实现
void TensorPipeAgent::shutdownImpl() {
  // FIXME 库在正常操作中打印日志是否太冗长了？
  LOG(INFO) << "RPC agent for " << workerInfo_.name_ << " is shutting down";

  // 等待超时线程结束
  timeoutThreadCV_.notify_one();
  if (timeoutThread_.joinable()) {
    timeoutThread_.join();
  }
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for timeout thread to join";

  // 关闭 TensorPipe 的上下文，关闭所有管道和监听器，调用所有错误回调，
  // 关闭 I/O 事件循环并等待所有线程终止
  context_->join();
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for TensorPipe context to join";

  // 注意：我们需要在关闭所有监听器后调用 waitWorkComplete。
  // 这是为了清空线程池中已接受的工作。如果在关闭监听器之前调用，
  // 可能会导致在系统关闭期间添加额外的工作，并在关闭监听器之前继续执行。
  threadPool_.waitWorkComplete();
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for thread pool to complete work";
}

// 根据 workerName 获取 WorkerInfo 对象的常量引用
const WorkerInfo& TensorPipeAgent::getWorkerInfo(
    const std::string& workerName) const {
  std::unordered_map<std::string, WorkerInfo>::const_iterator it;
  {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
    it = workerNameToInfo_.find(workerName);
  }
  TORCH_CHECK(
      it != workerNameToInfo_.end(),
      fmt::format(
          "name:{},rank:{} could not find destination name {}",
          workerInfo_.name_,
          workerInfo_.id_,
          workerName));
  return it->second;
}

// 根据 workerId 获取 WorkerInfo 对象的常量引用
const WorkerInfo& TensorPipeAgent::getWorkerInfo(worker_id_t workerId) const {
  std::unordered_map<worker_id_t, WorkerInfo>::const_iterator it;
  {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
    it = workerIdToInfo_.find(workerId);
  }
  TORCH_CHECK(
      it != workerIdToInfo_.end(),
      fmt::format(
          "name:{},rank:{} could not find destination id {}",
          workerInfo_.name_,
          workerInfo_.id_,
          workerId));
  return it->second;
}

// 获取所有 WorkerInfo 对象的向量
std::vector<WorkerInfo> TensorPipeAgent::getWorkerInfos() const {
  std::vector<WorkerInfo> workerInfos;
  workerInfos.reserve(workerNameToInfo_.size());
  for (auto& item : workerNameToInfo_) {
    workerInfos.emplace_back(item.second);
  }
  return workerInfos;
}

// 根据 WorkerInfo 查找对应的 WorkerURL
const std::string& TensorPipeAgent::findWorkerURL(
    const WorkerInfo& worker) const {
  std::unordered_map<std::string, std::string>::const_iterator it;
  {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
    it = workerURLMap_.find(worker.name_);
  }
  // 返回找到的 WorkerURL
  return it->second;
}
    // 在 workerNameToURL_ 中查找 worker.name_ 对应的迭代器
    it = workerNameToURL_.find(worker.name_);
  }
  // 使用 TORCH_CHECK 确保找到了 worker.name_ 对应的 URL
  TORCH_CHECK(
      it != workerNameToURL_.end(),
      // 如果未找到，则抛出错误信息，包含 workerInfo_.name_、workerInfo_.id_ 和 worker.name_
      fmt::format(
          "name:{},rank:{} could not find destination url for name {}",
          workerInfo_.name_,
          workerInfo_.id_,
          worker.name_));
  // 返回找到的 worker.name_ 对应的 URL
  return it->second;
}

// 更新工作组成员资格
void TensorPipeAgent::updateGroupMembership(
    const WorkerInfo& workerInfo,                           // 接收工作信息
    const std::vector<c10::Device>& devices,                // 设备列表
    const std::unordered_map<std::string, DeviceMap>& reverseDeviceMaps, // 反向设备映射
    bool isJoin) {                                          // 是否加入标志

  std::string name = workerInfo.name_;                      // 获取工作名
  worker_id_t id = workerInfo.id_;                          // 获取工作 ID

  // 如果是加入操作，更新内部映射
  if (isJoin) {
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);  // 获取锁
    workerIdToInfo_.emplace(id, workerInfo);                 // 将工作 ID 到信息的映射添加到容器中
    workerNameToInfo_.emplace(name, workerInfo);             // 将工作名到信息的映射添加到容器中

    // TODO: 在加入过程中应获取 nodeAddrStr，而不是每次从存储获取
    auto nodeAddrData = nameToAddressStore_.get(name);       // 获取节点地址数据
    auto nodeAddrStr = std::string((const char*)nodeAddrData.data(), nodeAddrData.size());  // 转换为字符串
    workerNameToURL_.insert({name, nodeAddrStr});            // 将工作名到 URL 的映射插入容器中

    // 将未在反向设备映射中的项目添加进去
    for (const auto& it : reverseDeviceMaps) {
      if (reverseDeviceMaps_.find(it.first) == reverseDeviceMaps_.end()) {
        reverseDeviceMaps_[it.first] = it.second;
      }
    }

    // TODO: 清理设备使用的互斥锁
    // 将尚未添加的设备添加进来
    for (const auto& it : devices) {
      if (std::find(devices_.begin(), devices_.end(), it) == devices_.end()) {
        devices_.push_back(it);
      }
    }
  } else {
    workerIdToInfo_.erase(id);                               // 从容器中删除工作 ID 的信息
    workerNameToInfo_.erase(name);                           // 从容器中删除工作名的信息
    workerNameToURL_.erase(name);                            // 从容器中删除工作名到 URL 的映射

    // 移除不再使用的反向设备映射
    for (auto it = reverseDeviceMaps_.begin(); it != reverseDeviceMaps_.end();) {
      if (reverseDeviceMaps.find(it->first) == reverseDeviceMaps.end()) {
        it = reverseDeviceMaps_.erase(it);
      } else {
        it++;
      }
    }

    // 移除不再使用的设备
    for (auto it = devices_.begin(); it != devices_.end();) {
      if (std::find(devices.begin(), devices.end(), *it) == devices.end()) {
        it = devices_.erase(it);
      } else {
        it++;
      }
    }
  }
}

// 获取度量指标
std::unordered_map<std::string, std::string> TensorPipeAgent::getMetrics() {
  std::unordered_map<std::string, std::string> metrics;     // 初始化度量指标容器
  metrics[kThreadPoolSize] = std::to_string(threadPool_.size());  // 记录线程池大小
  metrics[kNumIdleThreads] = std::to_string(threadPool_.numAvailable());  // 记录空闲线程数

  {
    std::unique_lock<std::mutex> lock(callCountMutex_);      // 获取调用计数互斥锁
    metrics[kClientActiveCalls] = std::to_string(clientActiveCalls_);  // 记录客户端活动调用数
    metrics[kServerActiveCalls] = std::to_string(serverActiveCalls_);  // 记录服务器活动调用数
    metrics[kServerActiveAsyncCalls] = std::to_string(serverActiveAsyncCalls_);  // 记录服务器异步活动调用数
  }

  if (isGILProfilingEnabled()) {
    {
      std::unique_lock<std::mutex> lock(metricsMutex_);      // 获取度量指标互斥锁
      auto averageGilWaitTime = timeSeriesMetrics_[kGilAverageWaitTime].computeAverage();  // 计算 GIL 平均等待时间
      lock.unlock();                                         // 解锁互斥锁
      metrics[kGilAverageWaitTime] = std::to_string(averageGilWaitTime);  // 记录 GIL 平均等待时间
    }
  }



    # 结束内部的 for 循环并返回到外部的 while 循环
    }
  }




  return metrics;



  # 返回函数最终计算得到的 metrics 变量作为结果
  return metrics;
}

// 将 GIL 等待时间添加到时间序列指标中
void TensorPipeAgent::addGilWaitTime(
    const std::chrono::microseconds gilWaitTime) {
  // 加锁以确保线程安全
  std::lock_guard<std::mutex> lock(metricsMutex_);
  // 将 GIL 等待时间添加到时间序列指标中
  timeSeriesMetrics_[kGilAverageWaitTime].addData(gilWaitTime.count());
}

// 获取网络数据
TensorPipeAgent::NetworkDataDict TensorPipeAgent::getNetworkData() {
  // 加锁以确保线程安全
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  return networkData_;
}

// 获取网络源信息
NetworkSourceInfo TensorPipeAgent::getNetworkSourceInfo() {
  // 获取网络源信息
  NetworkSourceInfo info = {
      RpcAgent::getWorkerInfo().id_,
      nameToAddressStore_.get(RpcAgent::getWorkerInfo().name_)};

  return info;
}

// 跟踪网络数据
void TensorPipeAgent::trackNetworkData(
    uint64_t requestSize,
    uint64_t responseSize,
    const std::string& destWorkerName) {
  // 加锁以确保线程安全
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  // 更新网络数据
  networkData_[destWorkerName].numCalls++;
  networkData_[destWorkerName].totalSentBytes += requestSize;
  networkData_[destWorkerName].totalRecvBytes += responseSize;
}

// 跟踪网络错误
void TensorPipeAgent::trackNetworkError(
    uint64_t requestSize,
    const std::string& destWorkerName) {
  // 加锁以确保线程安全
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  // 更新网络数据
  networkData_[destWorkerName].numCalls++;
  networkData_[destWorkerName].totalSentBytes += requestSize;
  networkData_[destWorkerName].totalErrors++;
}

// 增加调用计数
void TensorPipeAgent::increaseCallCount(int32_t& count) {
  {
    // 加锁以确保线程安全
    std::unique_lock<std::mutex> lock(callCountMutex_);
    // 增加调用计数
    ++count;
  }
  // 通知所有等待的线程
  callCountCV_.notify_all();
}

// 减少调用计数
void TensorPipeAgent::decreaseCallCount(int32_t& count) {
  {
    // 加锁以确保线程安全
    std::unique_lock<std::mutex> lock(callCountMutex_);
    // 减少调用计数
    --count;
  }
  // 通知所有等待的线程
  callCountCV_.notify_all();
}

// 标记未来任务为完成
void TensorPipeAgent::markFutureAsComplete(
    std::shared_ptr<AtomicJitFuture> atomicFuture,
    c10::intrusive_ptr<Message> message,
    std::vector<c10::Stream> streams) {
  if (!atomicFuture->isComplete.test_and_set()) {
    // 完成未来任务将运行其回调函数，可能执行任意用户代码。为防止阻塞或停滞 TensorPipe 事件循环，我们将其推迟到工作线程中。
    threadPool_.run([this,
                     atomicFuture{std::move(atomicFuture)},
                     message{std::move(message)},
                     streams{std::move(streams)}]() mutable {
      c10::MultiStreamGuard guard(streams);
      std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages =
          message->getStorages();
      atomicFuture->jitFuture->markCompleted(
          std::move(message), std::move(storages));
      // 未来任务的回调函数可能安排进一步的 RPC，增加计数。因此，在完成未来任务后必须减少计数，否则可能会短暂地降至零，并误导 join 认为所有工作已完成。
      decreaseCallCount(clientActiveCalls_);
    });
  }
}

// 标记未来任务为错误
void TensorPipeAgent::markFutureWithError(
    std::shared_ptr<AtomicJitFuture> atomicFuture,
    std::string errorMsg) {
  if (!atomicFuture->isComplete.test_and_set()) {
    // 完成 future 将运行其回调函数，这些回调函数可能执行任意用户代码。
    // 为了避免阻塞或使 TensorPipe 事件循环停滞，我们将此操作推迟到一个工作线程中执行。
    threadPool_.run([this,
                     atomicFuture{std::move(atomicFuture)},
                     errorMsg{std::move(errorMsg)}]() mutable {
      // 设置 atomicFuture 所关联的 jitFuture 的错误状态为运行时异常，并使用 errorMsg 提供错误消息。
      atomicFuture->jitFuture->setError(
          std::make_exception_ptr(std::runtime_error(errorMsg)));
      // 未来的回调函数可能会安排进一步的远程过程调用（RPC），增加计数。
      // 因此，在完成未来后，我们必须减少计数，否则它可能会短暂地降为零，并误导 join 认为所有工作都已完成。
      decreaseCallCount(clientActiveCalls_);
    });
  }
}

std::vector<c10::Device> TensorPipeAgent::getDevicesForRemote(
    const std::string& remoteName,
    const Message& message) const {
  // 定义一个无序映射，用于存储设备映射关系
  std::unordered_map<std::string, DeviceMap> deviceMaps;
  {
    // 获取群组成员锁的守卫，确保在静态群组中
    GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
    // 根据消息类型选择设备映射表或反向设备映射表
    deviceMaps = message.isRequest() ? opts_.deviceMaps : reverseDeviceMaps_;
  }

  // 构建错误字符串，用于报告不支持的设备映射
  const auto errStr = c10::str(
      "TensorPipe RPC backend only supports CPU tensors by default, please "
      "move your tensors to CPU before sending them over RPC, or call "
      "`set_device_map` on `TensorPipeRpcBackendOptions` to explicitly "
      "configure device mapping. ",
      message.isRequest() ? "Request" : "Response",
      " device mapping is not available for destination ",
      remoteName);

  // 查找远程目标的设备映射
  const auto& iter = deviceMaps.find(remoteName);
  if (iter == deviceMaps.end()) {
    // 如果找不到映射，检查消息中的张量是否在CPU上，并报告错误
    for (const auto& t : message.tensors()) {
      TORCH_CHECK(
          t.device().is_cpu(),
          errStr,
          ", but found tensor on device: ",
          t.device());
    }
    // 返回空向量，表示没有可用设备
    return {};
  } else {
    // 根据远程目标的设备映射，获取相应的设备列表
    return getDevicesForTensors(message.tensors(), iter->second, errStr);
  }
}

DeviceMap TensorPipeAgent::getDeviceMap(const WorkerInfo& dst) const {
  // 查找给定目标的设备映射
  auto it = opts_.deviceMaps.find(dst.name_);
  if (it == opts_.deviceMaps.end()) {
    // 如果找不到映射，返回空映射
    return {};
  }
  // 返回找到的设备映射
  return it->second;
}

const c10::intrusive_ptr<::c10d::Store> TensorPipeAgent::getStore() const {
  // 返回存储指针
  return store_;
}

TensorPipeRpcBackendOptions TensorPipeAgent::getBackendOptions() const {
  // 返回后端选项
  return opts_;
}

const std::vector<c10::Device>& TensorPipeAgent::getDevices() const {
  // 获取设备列表时加锁，确保线程安全
  GroupMembershipLockGuard guard(groupMembershipMutex_, isStaticGroup_);
  // 返回设备列表
  return devices_;
}

size_t TensorPipeAgent::timeoutMapSize() {
  // 获取超时映射的大小时加锁，确保线程安全
  std::unique_lock<std::mutex> lock(timeoutMapMutex_);
  return timeoutMap_.size();
}

size_t TensorPipeAgent::numPendingResponses() {
  // 获取待处理响应的数量时加锁，确保线程安全
  std::unique_lock<std::mutex> lock(callCountMutex_);
  return clientActiveCalls_;
}

size_t TensorPipeAgent::messageIdToTimeoutMapSize() {
  // 获取消息ID到超时映射的大小时加锁，确保线程安全
  std::unique_lock<std::mutex> lock(timeoutMapMutex_);
  return messageIdToTimeout_.size();
}

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
```