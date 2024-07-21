# `.\pytorch\torch\csrc\distributed\rpc\profiler\remote_profiler_manager.cpp`

```py
// 包含远程分析器管理器的头文件
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
// 包含RPC代理的头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含RPC命令基类的头文件
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含序列化pickle的头文件
#include <torch/csrc/jit/serialization/pickle.h>
// 包含字节顺序处理的头文件
#include <torch/csrc/utils/byte_order.h>

// Torch命名空间
namespace torch {
// 分布式命名空间
namespace distributed {
// RPC命名空间
namespace rpc {

// 远程分析器的键前缀
const std::string REMOTE_PROFILING_KEY_PREFIX = "#remote_op: ";
// 自动增量位数
constexpr int kAutoIncrementBits = 48;

// 当前线程局部键的静态线程局部变量定义
/*static */ thread_local std::optional<std::string> RemoteProfilerManager::currentThreadLocalKey_ = c10::nullopt;

// 获取远程分析器管理器的单例实例
/*static */ RemoteProfilerManager& RemoteProfilerManager::getInstance() {
  // 创建远程分析器管理器的静态实例
  static RemoteProfilerManager* handler = new RemoteProfilerManager();
  // 返回实例的引用
  return *handler;
}

// 设置当前线程局部键
void RemoteProfilerManager::setCurrentKey(std::string key) {
  // 禁止覆盖当前键，必须显式调用writeKey()来提交
  if (RemoteProfilerManager::currentThreadLocalKey_) {
    // 如果当前键已经设置，则抛出错误
    TORCH_CHECK(
        false,
        "Cannot call RemoteProfilerManager::setCurrentKey when current key is already set.");
  }
  // 移动赋值当前线程局部键
  currentThreadLocalKey_ = std::move(key);
}

// 检查当前键是否已设置
bool RemoteProfilerManager::isCurrentKeySet() const {
  // 返回当前线程局部键是否有值
  return currentThreadLocalKey_ ? true : false;
}

// 清除当前线程局部键
void RemoteProfilerManager::unsetCurrentKey() {
  // 将当前线程局部键置为null
  currentThreadLocalKey_ = c10::nullopt;
}

// 删除给定全局唯一ID对应的键
void RemoteProfilerManager::eraseKey(const ProfilingId& globallyUniqueId) {
  // 加锁保护
  std::lock_guard<std::mutex> guard(mutex_);
  // 查找并删除给定全局唯一ID对应的键
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  profiledRpcKeys_.erase(it);
}

// 检索给定全局唯一ID对应的RPC分析键
std::string RemoteProfilerManager::retrieveRPCProfilingKey(
    const ProfilingId& globallyUniqueId) {
  // 加锁保护
  std::lock_guard<std::mutex> guard(mutex_);
  // 查找给定全局唯一ID对应的RPC分析键
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  // 返回找到的RPC分析键
  return it->second;
}

// 获取下一个分析器ID
ProfilingId RemoteProfilerManager::getNextProfilerId() {
  // 获取下一个本地ID
  auto localId = getNextLocalId();
  // 获取当前RPC代理的工作器ID
  auto localWorkerId = RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_;
  // 创建全局唯一ID
  auto globallyUniqueId = torch::distributed::rpc::ProfilingId(localWorkerId, localId);
  // 返回全局唯一ID
  return globallyUniqueId;
}

// 获取下一个本地ID
local_id_t RemoteProfilerManager::getNextLocalId() {
  // 加锁保护
  std::lock_guard<std::mutex> guard(mutex_);
  // 返回当前本地ID并递增
  return currentLocalId_++;
}

// 获取当前分析键的引用
std::string& RemoteProfilerManager::getCurrentProfilingKey() {
  // 检查当前线程局部键是否已设置
  TORCH_CHECK(
      RemoteProfilerManager::currentThreadLocalKey_,
      "Must set currentThreadLocalKey_ before calling getCurrentProfilingKey");
  // 返回当前线程局部键的引用
  return *currentThreadLocalKey_;
}

// 保存RPC分析键
void RemoteProfilerManager::saveRPCKey(
    ProfilingId globallyUniqueId,
    const std::string& rpcProfilingKey) {
  // 加锁保护
  std::lock_guard<std::mutex> guard(mutex_);
  // 插入给定全局唯一ID和RPC分析键到map中
  profiledRpcKeys_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(globallyUniqueId),
      std::forward_as_tuple(rpcProfilingKey));
}
// 定义 RemoteProfilerManager 类的构造函数
RemoteProfilerManager::RemoteProfilerManager() {
    // 获取当前 RPC 代理的工作信息，并获取其 ID
    auto workerId = static_cast<int64_t>(RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_);
    // 根据工作 ID 左移 kAutoIncrementBits 位，生成当前本地 ID
    currentLocalId_ = workerId << kAutoIncrementBits;
}
// 结束命名空间 rpc
} // namespace rpc
// 结束命名空间 distributed
} // namespace distributed
// 结束命名空间 torch
} // namespace torch
```