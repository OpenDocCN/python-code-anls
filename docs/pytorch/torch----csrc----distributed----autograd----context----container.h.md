# `.\pytorch\torch\csrc\distributed\autograd\context\container.h`

```py
// 预处理指令，确保头文件在编译时只包含一次
#pragma once

// 包含互斥量的标准库头文件
#include <mutex>
// 包含无序映射的标准库头文件
#include <unordered_map>

// 包含 Torch 的分布式自动求导上下文头文件
#include <torch/csrc/distributed/autograd/context/context.h>

// Torch 命名空间
namespace torch {
// 分布式命名空间
namespace distributed {
// 自动求导命名空间
namespace autograd {

// 每个工作进程的单例类，负责存储分布式自动求导上下文，并在自动求导完成后清理数据
class DistAutogradContainer {
    // 可变互斥量，用于对当前分片加锁
    mutable std::mutex lock;

    // 存储该分片的自动求导上下文映射
    std::unordered_map<int64_t, ContextPtr> contexts;

  public:
    // 禁用默认构造函数
    DistAutogradContainer() = delete;
    // 默认析构函数
    ~DistAutogradContainer() = default;

    // 获取单例实例的内部方法
    static DistAutogradContainer& getInstanceInternal();

    // 根据上下文 ID 获取分片
    ContextsShard& getShard(int64_t context_id);

    // 发送 RPC 到具有指定上下文 ID 的工作进程
    // 此函数需要在加锁状态下调用
    void sendReleaseContextRpc(
        const std::unordered_set<rpc::worker_id_t>& workerIds,
        int64_t context_id);

    // 从自动求导上下文映射中擦除指定的上下文 ID，并在当前线程的上下文 ID 匹配时重置
    // 此函数需要在加锁状态下调用
    void eraseContextIdAndReset(ContextsShard& shard, int64_t context_id);

    // 计算自动求导上下文映射的分片数量
    static uint32_t computeNumShards();

    // 自动递增的上下文 ID，用于标识唯一的自动求导过程
    // 初始值的前 16 位为 worker_id
    std::atomic<int64_t> next_context_id_;

    // 在分布式环境中唯一标识工作进程的 ID
    int16_t worker_id_;

    // 容器是否已适当初始化的标志
    bool initialized_;

    // 分片化的自动求导上下文映射
    std::vector<ContextsShard> autograd_contexts_;

    // sharded autograd_contexts_ 映射的分片数量
    uint32_t num_shards_;

    // 用于标识唯一的发送/接收自动求导函数对的自动求导消息 ID
    std::atomic<int64_t> next_autograd_message_id_;

    // autograd_context_id 或 autograd_message_id 的最大允许值
    int64_t max_id_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```