# `.\pytorch\torch\csrc\distributed\autograd\context\container.cpp`

```py
// 包含头文件 torch/csrc/distributed/autograd/context/container.h
#include <torch/csrc/distributed/autograd/context/container.h>

// 包含异常处理头文件
#include <c10/util/Exception.h>
// 包含清理自动求导上下文请求的消息头文件
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>

// 命名空间 torch 中的 distributed 子命名空间中的 autograd 子命名空间
namespace torch {
namespace distributed {
namespace autograd {

// 常量定义：自动递增位数
constexpr int kAutoIncrementBits = 48;
// 常量定义：自动递增掩码
constexpr int64_t kAutoIncrementMask = (1LL << kAutoIncrementBits) - 1;
// 常量定义：最大工作器 ID
constexpr int kMaxWorkerId = 65535;
// 常量定义：清理上下文重试次数
constexpr int kNumCleanupContextRetries = 20;

// 常量定义：无效上下文 ID
constexpr int64_t kInvalidContextId = -1;

// 每个线程在任何时候只有一个有效的自动求导上下文 ID
static thread_local int64_t current_context_id_ = kInvalidContextId;

// 锁，用于确保 DistAutogradContainer 仅初始化一次
static std::mutex dist_container_init_lock_;

// DistAutogradContainer 类的实现
DistAutogradContainer::DistAutogradContainer(uint32_t num_shards)
    : next_context_id_(0),
      worker_id_(0),
      initialized_(false),
      autograd_contexts_(num_shards),
      num_shards_(num_shards),
      next_autograd_message_id_(0),
      max_id_(0) {
  // num_shards 必须是 2 的幂，以便 'getShard' 中的模数技巧能够工作
  TORCH_INTERNAL_ASSERT((num_shards & (num_shards - 1)) == 0);
}

// 初始化 DistAutogradContainer 的实例
DistAutogradContainer& DistAutogradContainer::init(int64_t worker_id) {
  // 使用互斥锁保护初始化过程
  std::lock_guard<std::mutex> guard(dist_container_init_lock_);

  // 检查 worker_id 是否在有效范围内
  TORCH_CHECK(
      worker_id >= 0 && worker_id <= kMaxWorkerId,
      "worker_id needs to be in the range [0, 65535]")

  auto& container = getInstanceInternal();

  // 检查容器是否已经初始化，或者是否尝试使用不同的 worker_id 重新初始化
  TORCH_CHECK(
      !container.initialized_ || (worker_id == container.worker_id_),
      "Container is already initialized with worker_id: ",
      container.worker_id_,
      ", cannot initialize with different worker_id: ",
      worker_id);

  // 如果容器已经初始化，直接返回
  if (container.initialized_) {
    LOG(INFO) << "DistAutogradContainer is already initialized";
    return container;
  }

  // 设置 worker_id，并计算初始的上下文 ID 和消息 ID
  container.worker_id_ = worker_id;
  container.next_context_id_ = static_cast<int64_t>(worker_id)
      << kAutoIncrementBits;
  container.next_autograd_message_id_ = static_cast<int64_t>(worker_id)
      << kAutoIncrementBits;
  container.max_id_ =
      (kAutoIncrementMask |
       (static_cast<int64_t>(worker_id) << kAutoIncrementBits));
  container.initialized_ = true;

  // 返回初始化后的容器实例
  return container;
}

// 计算 DistAutogradContainer 的分片数量
uint32_t DistAutogradContainer::computeNumShards() {
  uint32_t num_shards = 1;
  auto num_hw_threads = std::thread::hardware_concurrency();

  // 根据硬件线程数计算下一个大于两倍硬件线程数的2的幂次方作为分片数量
  if (num_hw_threads == 0) {
    num_shards = kNumDefaultShards;
  } else {
    while (num_shards < num_hw_threads * 2) {
      num_shards <<= 1;
    }
  }

  // 输出日志，显示 DistAutogradContainer 的分片数量
  VLOG(1) << "Number of shards for DistAutogradContainer: " << num_shards;

  // 返回计算出的分片数量
  return num_shards;
}

// 获取指定上下文 ID 对应的上下文分片
inline DistAutogradContainer::ContextsShard& DistAutogradContainer::getShard(
    int64_t context_id) {
  // num_shards_ 必须是2的幂，以使此模数技巧工作（在初始化过程中验证）
  return autograd_contexts_[context_id & (num_shards_ - 1)];
}

// 命名空间 autograd 的结束
} // namespace autograd
} // namespace distributed
} // namespace torch
DistAutogradContainer& DistAutogradContainer::getInstance() {
  // 获取单例实例（内部方法）
  auto& instance = getInstanceInternal();
  // 检查实例是否已初始化，若未初始化则抛出错误
  TORCH_CHECK(
      instance.initialized_,
      "Need to initialize distributed autograd using "
      "torch.distributed.autograd.init()");
  // 返回单例实例
  return instance;
}

DistAutogradContainer& DistAutogradContainer::getInstanceInternal() {
  // 避免模块析构函数竞争的泄漏单例实现
  static DistAutogradContainer* container =
      new DistAutogradContainer(computeNumShards());
  // 返回单例实例的引用
  return *container;
}

int64_t DistAutogradContainer::newAutogradMessageId() {
  // 检查是否会溢出到 workerId_ 区段
  TORCH_INTERNAL_ASSERT(next_autograd_message_id_ < max_id_);
  // 返回新的自动求导消息 ID，并自增
  return next_autograd_message_id_++;
}

ContextPtr DistAutogradContainer::getOrCreateContext(int64_t context_id) {
  // 获取与指定 context_id 相关的分片
  auto& shard = getShard(context_id);
  // 锁定分片的互斥锁
  std::lock_guard<std::mutex> guard(shard.lock);
  // 查找是否已存在该 context_id 对应的上下文
  auto it = shard.contexts.find(context_id);
  if (it != shard.contexts.end()) {
    // 如果已存在，则返回现有上下文
    return it->second;
  }

  // 如果不存在，则创建新的分布式自动求导上下文，并将其插入到分片的上下文映射中
  auto& context =
      shard.contexts
          .emplace(
              std::piecewise_construct,
              std::forward_as_tuple(context_id),
              std::forward_as_tuple(
                  std::make_shared<DistAutogradContext>(context_id)))
          .first->second;
  return context;
}

rpc::worker_id_t DistAutogradContainer::getWorkerId() const {
  // 返回 worker_id_
  return worker_id_;
}

const ContextPtr DistAutogradContainer::newContext() {
  // 检查当前线程是否已有有效的自动求导上下文 ID
  TORCH_CHECK(
      current_context_id_ == kInvalidContextId,
      "Already have an autograd context id for this thread.");

  // 生成新的自动求导上下文 ID
  auto context_id = next_context_id_++;
  current_context_id_ = context_id;

  // 检查是否会溢出到 workerId_ 区段
  TORCH_INTERNAL_ASSERT(context_id < max_id_);

  // 获取与新生成的 context_id 相关的分片
  auto& shard = getShard(context_id);
  // 锁定分片的互斥锁
  std::lock_guard<std::mutex> guard(shard.lock);
  // 创建新的分布式自动求导上下文，并将其插入到分片的上下文映射中
  auto& context =
      shard.contexts
          .emplace(
              std::piecewise_construct,
              std::forward_as_tuple(context_id),
              std::forward_as_tuple(
                  std::make_shared<DistAutogradContext>(context_id)))
          .first->second;

  return context;
}

bool DistAutogradContainer::hasValidContext() const {
  // 检查当前线程是否有有效的自动求导上下文 ID
  return current_context_id_ != kInvalidContextId;
}

ContextPtr DistAutogradContainer::currentContext() {
  // 检查当前线程是否有有效的自动求导上下文 ID，若无效则抛出错误
  TORCH_CHECK(
      hasValidContext(),
      "Current thread doesn't have a valid autograd context. Please wrap your "
      "code using: `with torch.distributed.autograd.context() as context_id` "
      "to generate a valid context");

  // 获取与当前 context_id 相关的分片
  auto& shard = getShard(current_context_id_);
  // 锁定分片的互斥锁
  std::lock_guard<std::mutex> guard(shard.lock);
  // 查找是否存在当前 context_id 对应的上下文，并返回其引用
  auto it = shard.contexts.find(current_context_id_);
  TORCH_CHECK(
      it != shard.contexts.end(),
      "Couldn't find autograd context "
      "data for current autograd context id");
  return it->second;
}
// 如果存在上下文 ID，则释放其上下文数据
void DistAutogradContainer::releaseContextIfPresent(int64_t context_id) {
  // 获取特定上下文 ID 对应的分片
  auto& shard = getShard(context_id);
  // 获取分片锁，并设置为独占锁
  std::unique_lock<std::mutex> lock(shard.lock);
  // 在分片的上下文映射中查找指定的上下文 ID
  auto it = shard.contexts.find(context_id);

  // 如果在当前线程中找不到上下文，则什么也不做。这可能发生在正在进行的 RPC 中，
  // 当前线程上的上下文已经被释放。
  if (it == shard.contexts.end()) {
    return;
  }

  // 获取已知的工作节点 ID，以备后用
  auto knownWorkerIds = it->second->getKnownWorkerIds();
  // 从分片中删除上下文 ID 并重置相关数据
  eraseContextIdAndReset(shard, context_id);

  // 由于不再需要锁定，解锁当前的独占锁
  lock.unlock();
  // 发送释放上下文的 RPC 请求给已知的工作节点
  sendReleaseContextRpc(knownWorkerIds, context_id);
}

// 释放指定上下文 ID 的上下文数据
void DistAutogradContainer::releaseContext(int64_t context_id) {
  // 获取特定上下文 ID 对应的分片
  auto& shard = getShard(context_id);
  // 获取分片锁，并设置为独占锁
  std::unique_lock<std::mutex> lock(shard.lock);
  // 在分片的上下文映射中查找指定的上下文 ID
  auto it = shard.contexts.find(context_id);

  // 如果找不到指定上下文 ID 的自动求导上下文，则抛出错误信息
  TORCH_CHECK(
      it != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);

  // 获取已知的工作节点 ID，以备后用
  auto knownWorkerIds = it->second->getKnownWorkerIds();
  // 从分片中删除上下文 ID 并重置相关数据
  eraseContextIdAndReset(shard, context_id);

  // 由于不再需要锁定，解锁当前的独占锁
  lock.unlock();
  // 发送释放上下文的 RPC 请求给已知的工作节点
  sendReleaseContextRpc(knownWorkerIds, context_id);
}

// 发送释放上下文的 RPC 请求给指定的工作节点集合
void DistAutogradContainer::sendReleaseContextRpc(
    const std::unordered_set<rpc::worker_id_t>& workerIds,
    int64_t context_id) {
  // 尽力通知其他工作节点清理其分布式自动求导上下文，以减少内存使用量。
  // agent.send() 或 getCurrentRpcAgent 可能会在不正常关闭时抛出错误，
  // 比如 RPC 关闭同时在另一个线程中处理此消息的情况。这时候不应该抛出错误。
  std::shared_ptr<rpc::RpcAgent> agent;
  try {
    // 尝试获取当前的 RPC 代理
    agent = rpc::RpcAgent::getCurrentRpcAgent();
  } catch (const std::exception& e) {
    // 如果失败，记录发送 RPC 失败的信息，但不抛出异常
    LOG(INFO)
        << "Failed to send RPC to clear Dist Autograd context to all workers: "
        << e.what();
    return;
  }

  // 断言 RPC 代理不为空
  TORCH_INTERNAL_ASSERT(agent, "RPC Agent should be set.");

  // 设置 RPC 重试选项
  rpc::RpcRetryOptions options;
  options.maxRetries = kNumCleanupContextRetries;
  // 遍历每个工作节点 ID，尝试发送清理自动求导上下文的 RPC 请求
  for (const auto& worker_id : workerIds) {
    try {
      // 发送带有重试的 RPC 请求
      auto cleanupFuture = agent->sendWithRetries(
          agent->getWorkerInfo(worker_id),
          CleanupAutogradContextReq(context_id).toMessage(),
          options);

      // 添加回调函数处理 RPC 请求的结果
      cleanupFuture->addCallback([worker_id](rpc::JitFuture& future) {
        if (future.hasError()) {
          // 如果 RPC 请求发生错误，记录错误信息
          std::string errorMsg = c10::str(
              "Could not release Dist Autograd Context on node ",
              worker_id,
              ": ",
              future.tryRetrieveErrorMessage());
          LOG(ERROR) << errorMsg;
          return;
        }
      });
    } catch (const std::exception& e) {
      // 记录发送 RPC 请求失败的信息
      LOG(INFO)
          << "Failed to send RPC to clear Dist Autograd context to worker id: "
          << worker_id << " : " << e.what();
    }
  }
}

// 从指定的分片中删除上下文 ID 并重置相关数据
void DistAutogradContainer::eraseContextIdAndReset(
    DistAutogradContainer::ContextsShard& shard,
    int64_t context_id) {
    int64_t context_id) {
  // 在此处已经获取了分片锁。

  // 从contexts中移除指定context_id对应的记录。
  shard.contexts.erase(context_id);

  // 检查当前的context_id是否等于被删除的context_id。
  if (current_context_id_ == context_id) {
    // 如果相等，表示当前线程的上下文ID已经失效，需要重置为无效的上下文ID。
    current_context_id_ = kInvalidContextId;
  }
} // 结束 DistAutogradContainer 类的定义

void DistAutogradContainer::isValidContext(int64_t context_id) {
  // 获取特定 context_id 对应的 shard 引用
  auto& shard = getShard(context_id);
  // 对 shard 中的锁进行加锁，保证线程安全性
  std::lock_guard<std::mutex> guard(shard.lock);
  // 检查是否能在 shard 的 contexts 中找到指定的 context_id
  TORCH_CHECK(
      shard.contexts.find(context_id) != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);
}

ContextPtr DistAutogradContainer::retrieveContext(int64_t context_id) {
  // 获取特定 context_id 对应的 shard 引用
  auto& shard = getShard(context_id);
  // 对 shard 中的锁进行加锁，保证线程安全性
  std::lock_guard<std::mutex> guard(shard.lock);
  // 在 shard 的 contexts 中查找指定的 context_id
  auto it = shard.contexts.find(context_id);
  // 如果找不到指定的 context_id，则抛出错误信息
  TORCH_CHECK(
      it != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);
  // 返回找到的 context_id 对应的上下文指针
  return it->second;
}

int64_t DistAutogradContainer::getMaxId() {
  // 返回最大的 context_id
  return max_id_;
}

void DistAutogradContainer::forceCurrentContextId(int64_t contextId) {
  // 强制设置当前线程的 context_id
  current_context_id_ = contextId;
}

void DistAutogradContainer::setCurrentContextId(int64_t contextId) {
  // 断言当前线程没有已经设置的 context_id，确保线程安全
  TORCH_INTERNAL_ASSERT(
      current_context_id_ == kInvalidContextId,
      "Already have an autograd context id for this thread.");
  // 设置当前线程的 context_id
  current_context_id_ = contextId;
}

void DistAutogradContainer::clearCurrentContext() {
  // 清除当前线程的 context_id
  current_context_id_ = -1;
}

size_t DistAutogradContainer::numAutogradContexts() const {
  // 统计所有 shard 中的 autograd 上下文数量
  size_t ret = 0;
  for (const auto& shard : autograd_contexts_) {
    // 对每个 shard 中的锁进行加锁，保证线程安全性
    std::lock_guard<std::mutex> guard(shard.lock);
    // 累加当前 shard 中的 autograd 上下文数量
    ret += shard.contexts.size();
  }
  // 返回总的 autograd 上下文数量
  return ret;
}

int64_t DistAutogradContainer::currentContextId() {
  // 返回当前线程的 context_id
  return current_context_id_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
```