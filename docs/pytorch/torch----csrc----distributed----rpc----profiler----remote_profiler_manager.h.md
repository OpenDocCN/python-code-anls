# `.\pytorch\torch\csrc\distributed\rpc\profiler\remote_profiler_manager.h`

```
#pragma once
// 引入必要的头文件
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <mutex>  // 互斥量
#include <unordered_map>  // 无序映射

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 远程性能分析的键前缀常量
extern const std::string REMOTE_PROFILING_KEY_PREFIX;

// 远程性能分析管理器类
class TORCH_API RemoteProfilerManager {
 public:
  // 获取 RemoteProfilerManager 的单例实例（延迟初始化）
  static RemoteProfilerManager& getInstance();
  
  // 设置当前线程的性能分析键
  void setCurrentKey(std::string key);
  
  // 检查当前是否已设置性能分析键
  bool isCurrentKeySet() const;
  
  // 清除当前线程的性能分析键，以允许其他 RPC 重置它
  void unsetCurrentKey();
  
  // 将全局唯一 ID 和性能分析键对保存到内存映射中，
  // 用于在 RPC 反序列化时为远程分析事件添加正确的键前缀
  void saveRPCKey(
      ProfilingId globallyUniqueId,
      const std::string& rpcProfilingKey);
  
  // 根据给定的全局唯一 ID 检索相应的性能分析键，如果未找到则抛出异常
  std::string retrieveRPCProfilingKey(const ProfilingId& globallyUniqueId);
  
  // 生成下一个全局唯一的性能分析 ID
  ProfilingId getNextProfilerId();
  
  // 检索当前线程设置的性能分析键，如果未设置则抛出异常
  std::string& getCurrentProfilingKey();
  
  // 从映射中删除指定的全局唯一 ID，这有助于在进行大量 RPC 性能分析时节省内存
  void eraseKey(const ProfilingId& globallyUniqueId);

  // 禁用复制和移动构造函数及赋值运算符
  RemoteProfilerManager(const RemoteProfilerManager& other) = delete;
  RemoteProfilerManager operator=(const RemoteProfilerManager& other) = delete;
  RemoteProfilerManager(RemoteProfilerManager&&) = delete;
  RemoteProfilerManager& operator=(RemoteProfilerManager&&) = delete;

 private:
  // 构造函数和析构函数
  RemoteProfilerManager();
  ~RemoteProfilerManager() = default;

  // 获取下一个本地 ID
  local_id_t getNextLocalId();

  // 用于存储全局唯一 ID 到性能分析键的映射
  std::unordered_map<ProfilingId, std::string, ProfilingId::Hash> profiledRpcKeys_;

  // 当前线程的线程局部存储，存储当前的性能分析键
  static thread_local std::optional<std::string> currentThreadLocalKey_;

  // 互斥量，用于保护对成员变量的并发访问
  std::mutex mutex_;

  // 当前本地 ID
  local_id_t currentLocalId_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```