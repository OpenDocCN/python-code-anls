# `.\pytorch\torch\csrc\distributed\c10d\GroupRegistry.hpp`

```py
// 防止头文件被多次引用
#pragma once

// 包含 Torch 分布式库的进程组定义头文件
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// 定义 c10d 命名空间
namespace c10d {

// 设置线程隔离模式的函数声明
C10_EXPORT void set_thread_isolation_mode(bool enable);

// 获取线程隔离模式的函数声明
bool get_thread_isolation_mode();

// 注册给定名称的进程组到全局映射中
C10_EXPORT void register_process_group(
    const std::string& group_name,
    c10::intrusive_ptr<c10d::ProcessGroup> group);

// 解析并返回给定名称的进程组实例
C10_EXPORT c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name);

// 从全局映射中注销给定名称的进程组
C10_EXPORT void unregister_process_group(const std::string& group_name);

// 注销所有已注册的进程组
C10_EXPORT void unregister_all_process_groups();

} // namespace c10d
```