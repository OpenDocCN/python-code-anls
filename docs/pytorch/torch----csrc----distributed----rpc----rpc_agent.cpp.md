# `.\pytorch\torch\csrc\distributed\rpc\rpc_agent.cpp`

```
// 包含必要的头文件
#include <c10/util/DeadlockDetection.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

// 命名空间声明
namespace torch {
namespace distributed {
namespace rpc {

// RegisterWorkerInfoOnce 类的构造函数定义
RegisterWorkerInfoOnce::RegisterWorkerInfoOnce() {
  // WorkerInfo 需要确保只注册一次。由于操作注册发生在 libtorch_python 中，
  // 我们在这里使用一个辅助函数封装类的注册，以确保在像 torch::deploy 这样使用多个 Python 实例时仅注册一次。
  static auto workerInfo = torch::class_<WorkerInfo>("dist_rpc", "WorkerInfo")
                               .def(torch::init<std::string, int64_t>());
}

// WorkerInfo 类的构造函数定义，使用字符串名称和 int64_t 类型的 id 参数
WorkerInfo::WorkerInfo(std::string name, int64_t id)
    : WorkerInfo(std::move(name), (worker_id_t)id) {
  // 检查 id 是否在 worker_id_t 的限制范围内
  TORCH_CHECK(
      id <= std::numeric_limits<worker_id_t>::max(),
      "RPC worker id ",
      id,
      " out of bound of int16_t.");
}

// WorkerInfo 类的构造函数定义，使用字符串名称和 worker_id_t 类型的 id 参数
WorkerInfo::WorkerInfo(std::string name, worker_id_t id)
    : name_(std::move(name)), id_(id) {
  // 检查名称长度和字符的有效性
  bool validSize = name_.length() < MAX_NAME_LEN && name_.length() > 0;
  bool validChar =
      std::find_if(name_.begin(), name_.end(), [](char c) {
        return !(std::isalnum(c) || c == '-' || c == '_' || c == ':');
      }) == name_.end();
  TORCH_CHECK(
      validSize && validChar,
      "Worker name must match ^[A-Za-z0-9-_:]*$, "
      "and must be non-empty and shorter than ",
      MAX_NAME_LEN,
      " chars, "
      "but got ",
      name_);
}

// 等待条件变量直到映射完成的大时间间隔。由于已知的与溢出相关的 bug，不能使用 std::chrono::time_point<std::chrono::steady_clock>::max()。
constexpr auto kLargeTimeDuration = std::chrono::hours(10000);

// RpcAgent 类的构造函数定义，使用 WorkerInfo、RequestCallback 的唯一指针和 rpc 超时时长作为参数
RpcAgent::RpcAgent(
    WorkerInfo workerId,
    std::unique_ptr<RequestCallback> cb,
    std::chrono::milliseconds rpcTimeout)
    : workerInfo_(std::move(workerId)),
      cb_(std::move(cb)),
      rpcTimeout_(rpcTimeout),
      profilingEnabled_(false),
      rpcAgentRunning_(false) {}

// RpcAgent 类的析构函数定义
RpcAgent::~RpcAgent() {
  // 如果 rpcAgentRunning_ 标志为 true，则执行 shutdown 操作
  if (rpcAgentRunning_.load()) {
    shutdown();
  }
}

// RpcAgent 类的 start 方法定义
void RpcAgent::start() {
  // 将 rpcAgentRunning_ 标志设置为 true
  rpcAgentRunning_.store(true);
  // 启动 rpcRetryThread_ 线程，调用 retryExpiredRpcs 方法
  rpcRetryThread_ = std::thread(&RpcAgent::retryExpiredRpcs, this);
  // 调用 startImpl 方法启动实现
  startImpl();
}

// RpcAgent 类的 shutdown 方法定义
void RpcAgent::shutdown() {
  // 断言在无 Python 依赖时没有全局解释器锁
  TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP();
  // 获取 rpcRetryMutex_ 的独占锁
  std::unique_lock<std::mutex> lock(rpcRetryMutex_);
  // 将 rpcAgentRunning_ 标志设置为 false
  rpcAgentRunning_.store(false);
  // 解锁 mutex
  lock.unlock();
  // 通知一个等待在 rpcRetryMapCV_ 上的线程
  rpcRetryMapCV_.notify_one();
  // 如果 rpcRetryThread_ 是可加入的，则等待其结束
  if (rpcRetryThread_.joinable()) {
    rpcRetryThread_.join();
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.PureVirtualCall)
  // 调用 shutdownImpl 方法
  shutdownImpl();
}

// RpcAgent 类的 sendWithRetries 方法定义，发送带重试的消息给指定 WorkerInfo
c10::intrusive_ptr<JitFuture> RpcAgent::sendWithRetries(
    const WorkerInfo& to,
    c10::intrusive_ptr<Message> message,
    // 检查重试选项中的最大重试次数是否为非负数
    TORCH_CHECK(retryOptions.maxRetries >= 0, "maxRetries cannot be negative.");
    // 检查重试选项中的重试间隔是否大于等于1，确保不是指数衰减
    TORCH_CHECK(
        retryOptions.retryBackoff >= 1,
        "maxRetries cannot be exponentially decaying.");
    // 检查重试选项中的重试持续时间是否为非负数
    TORCH_CHECK(
        retryOptions.rpcRetryDuration.count() >= 0,
        "rpcRetryDuration cannot be negative.");

    // 创建一个新的 JitFuture 对象，代表最初的异步操作
    auto originalFuture =
        c10::make_intrusive<JitFuture>(at::AnyClassType::get(), getDevices());
    // 计算第一次重试的时间点
    steady_clock_time_point newTime =
        computeNewRpcRetryTime(retryOptions, /* retryCount */ 0);
    // 创建 RpcRetryInfo 对象，用于跟踪重试相关信息
    auto firstRetryRpc = std::make_shared<RpcRetryInfo>(
        to,
        message,
        originalFuture,
        /* retryCount */ 0,
        retryOptions);
    // 发送 RPC 消息并获取对应的 JitFuture 对象
    auto jitFuture = send(to, std::move(message));
    // 添加回调函数，处理 RPC 消息发送后的异步操作
    jitFuture->addCallback([this, newTime, firstRetryRpc](JitFuture& future) {
        rpcRetryCallback(future, newTime, firstRetryRpc);
    });

    // 返回最初创建的 JitFuture 对象，用于异步操作的追踪和处理
    return originalFuture;
}

void RpcAgent::retryExpiredRpcs() {
  // 存储被重试的 futures，以便在锁之外添加回调
  std::vector<
      std::pair<c10::intrusive_ptr<JitFuture>, std::shared_ptr<RpcRetryInfo>>>
      futures;
  // 存储非可重试错误 futures 和异常消息
  std::vector<std::pair<c10::intrusive_ptr<JitFuture>, std::string>>
      errorFutures;

  while (rpcAgentRunning_.load()) {
    std::unique_lock<std::mutex> lock(rpcRetryMutex_);

    // 只要 RpcAgent 在运行并且重试映射为空或最早过期的 RPC 被设置为将来重试，就继续休眠
    steady_clock_time_point earliestTimeout =
        std::chrono::steady_clock::now() + kLargeTimeDuration;

    for (;;) {
      if (!rpcAgentRunning_.load())
        return;
      if (std::chrono::steady_clock::now() >= earliestTimeout)
        break;
      if (!rpcRetryMap_.empty()) {
        earliestTimeout = rpcRetryMap_.begin()->first;
      }
      // 等待直到最早超时或者唤醒条件变量
      rpcRetryMapCV_.wait_until(lock, earliestTimeout);
    }

    // 更新这些值，因为在线程休眠时可能已向映射中添加了条目
    earliestTimeout = rpcRetryMap_.begin()->first;
    auto& earliestRpcList = rpcRetryMap_.begin()->second;

    // 遍历当前时间点设置为重试的所有 RPC，重新发送这些 RPC，并将它们的 futures 添加到列表中以便稍后附加回调
    // 这些回调会根据发送结果，要么将 RPC 安排在将来重试，要么标记为成功/失败
    for (auto it = earliestRpcList.begin(); it != earliestRpcList.end();
         /* no increment */) {
      auto& earliestRpc = *it;
      c10::intrusive_ptr<JitFuture> jitFuture;

      // 如果在代理关闭时重试 RPC，send() 将抛出异常。我们必须捕获此异常，并标记原始 future 为错误，
      // 因为此 RPC 从未成功，也不再能重试。
      try {
        jitFuture = send(earliestRpc->to_, earliestRpc->message_);
        futures.emplace_back(jitFuture, earliestRpc);
      } catch (std::exception& e) {
        // 必须在释放锁之后存储 futures 和异常消息，并且只在之后标记 futures 为错误
        errorFutures.emplace_back(earliestRpc->originalFuture_, e.what());
      }

      // 对于此列表中的所有 futures，将附加回调。因此，它们要么被安排在将来重试，要么被标记为完成。
      // 可以安全地从重试映射中删除它们。
      it = earliestRpcList.erase(it);
    }

    // 如果当前时间点没有更多要重试的 RPC，则可以从重试映射中移除对应的 unordered_set。
    // 如果 earliestRpcList 是空的，表示没有最早的 RPC 列表，则从 rpcRetryMap_ 中移除对应的项
    if (earliestRpcList.empty()) {
      rpcRetryMap_.erase(earliestTimeout);
    }

    // 解锁当前的互斥锁
    lock.unlock();

    // 我们在锁之外为 futures 添加回调，以防止潜在的死锁情况发生
    for (const auto& it : futures) {
      auto jitFuture = it.first;
      auto earliestRpc = it.second;
      // 计算新的 RPC 重试时间点
      steady_clock_time_point newTime = computeNewRpcRetryTime(
          earliestRpc->options_, earliestRpc->retryCount_);
      // 增加 RPC 的重试计数
      earliestRpc->retryCount_++;

      // 为 jitFuture 添加回调函数，以在未来完成时执行 RPC 重试回调函数
      jitFuture->addCallback([this, newTime, earliestRpc](JitFuture& future) {
        rpcRetryCallback(future, newTime, earliestRpc);
      });
    }
    // 清空 futures 容器
    futures.clear();

    // 对于在上面重试 RPC 时捕获的异常，现在在释放锁后设置这些 futures 的错误状态
    for (const auto& it : errorFutures) {
      auto errorFuture = it.first;
      auto errorMsg = it.second;
      // 设置 errorFuture 的错误状态为捕获到的异常
      errorFuture->setError(
          std::make_exception_ptr(std::runtime_error(errorMsg)));
    }
    // 清空 errorFutures 容器
    errorFutures.clear();
}

// RPC代理类的rpcRetryCallback方法，处理RPC发送失败重试逻辑
void RpcAgent::rpcRetryCallback(
    JitFuture& jitFuture,  // 传入的JitFuture对象引用
    steady_clock_time_point newTime,  // 新的时间点，用于重试调度
    std::shared_ptr<RpcRetryInfo> earliestRpc) {  // 最早的RPC重试信息的智能指针

  if (jitFuture.hasError()) {  // 如果JitFuture对象有错误发生
    // 输出日志信息，指示发送尝试失败次数
    LOG(INFO) << "Send try " << (earliestRpc->retryCount_ + 1) << " failed";

    if (!rpcAgentRunning_.load()) {  // 如果RPC代理未运行
      // 如果RPC代理已经关闭，则标记原始Future为错误状态，因为RPC未成功完成
      std::string errorMessage = c10::str(
          "RPC Agent is no longer running on Node ",
          RpcAgent::getWorkerInfo().id_,
          ". Cannot retry message.");
      earliestRpc->originalFuture_->setError(jitFuture.exception_ptr());
    } else if (earliestRpc->retryCount_ < earliestRpc->options_.maxRetries) {
      // 如果之前的Future完成时有错误，并且未达到最大重试次数，则将earliestRpc结构移到重试映射的新时间点
      {
        std::lock_guard<std::mutex> retryMapLock(rpcRetryMutex_);
        rpcRetryMap_[newTime].emplace(std::move(earliestRpc));
      }
      // 通知重试线程有新的项目已添加到映射中
      rpcRetryMapCV_.notify_one();
    } else {
      // 已达到最大重试次数，标记Future为错误状态
      std::string errorMessage = c10::str(
          "The RPC has not succeeded after the specified number of max retries (",
          earliestRpc->options_.maxRetries,
          ").");
      earliestRpc->originalFuture_->setError(
          std::make_exception_ptr(std::runtime_error(errorMessage)));
    }
  } else {
    // 如果没有错误发生，标记原始Future为已完成状态
    earliestRpc->originalFuture_->markCompleted(
        jitFuture.value(), jitFuture.storages());
  }
}

// 返回当前RPC代理的WorkerInfo对象引用
const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

// 返回当前线程中的RPC代理智能指针是否已设置
std::shared_ptr<RpcAgent> RpcAgent::currentRpcAgent_ = nullptr;

bool RpcAgent::isCurrentRpcAgentSet() {
  return std::atomic_load(&currentRpcAgent_) != nullptr;
}

// 返回当前线程中的RPC代理的智能指针
std::shared_ptr<RpcAgent> RpcAgent::getCurrentRpcAgent() {
  std::shared_ptr<RpcAgent> agent = std::atomic_load(&currentRpcAgent_);
  TORCH_CHECK(
      agent,
      "Current RPC agent is not set! Did you initialize the RPC "
      "framework (e.g. by calling `rpc.init_rpc`)?");
  return agent;
}

// 设置当前线程中的RPC代理的智能指针
void RpcAgent::setCurrentRpcAgent(std::shared_ptr<RpcAgent> rpcAgent) {
  if (rpcAgent) {
    std::shared_ptr<RpcAgent> previousAgent;
    // 使用compare_exchange确保只在需要时进行代理的交换
    // 避免触发下面的断言条件，参考文档：https://en.cppreference.com/w/cpp/atomic/atomic_compare_exchange
    // 使用原子操作尝试将 currentRpcAgent_ 的值与 previousAgent 比较并替换为 std::move(rpcAgent)
    std::atomic_compare_exchange_strong(
        &currentRpcAgent_, &previousAgent, std::move(rpcAgent));
    
    // 断言，验证 previousAgent 是否为 nullptr，如果不是则输出错误信息 "Current RPC agent is set!"
    TORCH_INTERNAL_ASSERT(
        previousAgent == nullptr, "Current RPC agent is set!");
  } else {
    // 如果无法使用 compare_exchange（因为无法预期期望的值），但我们不需要，因为触发断言的唯一情况是我们用 nullptr 替换 nullptr，这种情况下不会产生影响。
    // 使用原子操作将 currentRpcAgent_ 的值替换为 std::move(rpcAgent)，并将之前的值存储在 previousAgent 中
    std::shared_ptr<RpcAgent> previousAgent =
        std::atomic_exchange(&currentRpcAgent_, std::move(rpcAgent));
    
    // 断言，验证 previousAgent 是否不为 nullptr，如果是则输出错误信息 "Current RPC agent is not set!"
    TORCH_INTERNAL_ASSERT(
        previousAgent != nullptr, "Current RPC agent is not set!");
  }
}

// 设置类型解析器，接受一个类型解析器的智能指针作为参数
void RpcAgent::setTypeResolver(std::shared_ptr<TypeResolver> typeResolver) {
  // 将传入的类型解析器移动到成员变量中
  typeResolver_ = std::move(typeResolver);
}

// 获取当前的类型解析器
std::shared_ptr<TypeResolver> RpcAgent::getTypeResolver() {
  // 断言类型解析器已设置，如果未设置则输出错误信息
  TORCH_INTERNAL_ASSERT(typeResolver_, "Type resolver is not set!");
  return typeResolver_;
}

// 启用或禁用 GIL（全局解释器锁）分析
void RpcAgent::enableGILProfiling(bool flag) {
  // 设置 GIL 分析标志
  profilingEnabled_ = flag;
}

// 检查当前是否启用 GIL 分析
bool RpcAgent::isGILProfilingEnabled() {
  return profilingEnabled_.load();
}

// 获取设备映射信息，对于给定的 WorkerInfo 参数，默认实现不返回任何设备映射
DeviceMap RpcAgent::getDeviceMap(const WorkerInfo& /* unused */) const {
  // 默认实现没有设备映射
  return {};
}

// 获取当前代理的设备列表，默认情况下代理是仅限 CPU 的
const std::vector<c10::Device>& RpcAgent::getDevices() const {
  // 默认情况下，代理没有设备
  static const std::vector<c10::Device> noDevices = {};
  return noDevices;
}

// 获取调试信息，返回一个包含指标的无序映射，未来可能包含线程堆栈等更多信息
std::unordered_map<std::string, std::string> RpcAgent::getDebugInfo() {
  /* This would later include more info other than metrics for eg: may include
     stack traces for the threads owned by the agent */
  // 默认实现：返回 getMetrics() 的结果
  return getMetrics();
}

// 重载输出流操作符，用于打印 WorkerInfo 对象的信息
std::ostream& operator<<(std::ostream& os, const WorkerInfo& workerInfo) {
  return os << "WorkerInfo(id=" << workerInfo.id_
            << ", name=" << workerInfo.name_ << ")";
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```