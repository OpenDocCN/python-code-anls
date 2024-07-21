# `.\pytorch\torch\csrc\distributed\rpc\agent_utils.h`

```
// 防止头文件重复包含，只在首次引入时起作用
#pragma once

// 包含 Torch 分布式模块中使用的 PrefixStore 头文件
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
// 包含 RPC 模块的工具函数头文件
#include <torch/csrc/distributed/rpc/utils.h>

// Torch 的命名空间开始
namespace torch {
// 分布式命名空间开始
namespace distributed {
// RPC 命名空间开始
namespace rpc {

// 所有 RPC 对等体应同时调用此函数。每个对等体提供自己的 id 和名称，此函数使用给定的 Store
// 在所有对等体上收集全局的名称到 id 的映射。
TORCH_API std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,            // 使用的前缀存储对象
    const worker_id_t selfId,             // 调用方自身的 id
    const std::string& selfName,          // 调用方自身的名称
    const int worldSize);                 // RPC 所涵盖的对等体总数

// 动态 RPC 组中的等级最初会调用此函数，以建立当前组中对等体的名称到 id 的映射。
// 当前等级将自己的工作信息放入存储中，并发现所有在其之前的等级。
// 注意：需要在持有动态 RPC 组成员管理令牌的情况下调用此函数。
TORCH_API std::unordered_map<std::string, worker_id_t> collectCurrentNames(
    ::c10d::PrefixStore store,            // 使用的前缀存储对象
    const worker_id_t selfId,             // 调用方自身的 id
    const std::string& selfName);         // 调用方自身的名称

// 从存储中移除名称，用于动态 RPC 组。
// 注意：需要在持有动态 RPC 组成员管理令牌的情况下调用此函数。
TORCH_API void removeCurrentName(
    ::c10d::PrefixStore store,            // 使用的前缀存储对象
    const worker_id_t selfId,             // 调用方自身的 id
    const std::string& selfName);         // 调用方自身的名称

// 使用存储执行所有调用计数的同步。
// 所有 RPC 对等体等待其他对等体加入以同时退出。
TORCH_API int syncCallCount(
    ::c10d::PrefixStore store,            // 使用的前缀存储对象
    const int worldSize,                  // RPC 所涵盖的对等体总数
    int activeCalls = 0);                 // 活跃调用的初始计数，默认为0

} // namespace rpc
} // namespace distributed
} // namespace torch
```