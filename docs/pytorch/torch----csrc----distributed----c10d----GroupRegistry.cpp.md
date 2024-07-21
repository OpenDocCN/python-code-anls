# `.\pytorch\torch\csrc\distributed\c10d\GroupRegistry.cpp`

```py
// 包含 Torch 分布式库中的 GroupRegistry 头文件
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

// 包含 Torch 分布式库中的 RankLocal 头文件
#include <torch/csrc/distributed/c10d/RankLocal.hpp>

// 匿名命名空间，用于限定 GroupRegistry 类和其相关函数的作用域
namespace {

// 每个进程操作同一逻辑进程组的不同 c10d::ProcessGroup 实例。
// 使用 RankLocal<GroupRegistry>::get() 确保每个进程获得唯一的注册表。
class GroupRegistry {
 public:
  // 注册一个进程组，将进程组名映射到对应的进程组指针
  void register_group(
      const std::string& group_name,
      c10::intrusive_ptr<c10d::ProcessGroup> group) {
    std::unique_lock write_lock(lock_);
    // 尝试将进程组名和进程组指针插入到注册表中
    auto [_, inserted] = registry_.try_emplace(group_name, std::move(group));
    // 检查是否插入成功，否则报错进程组名已经注册
    TORCH_CHECK(
        inserted,
        "A process group is already registered under the name",
        group_name);
  }

  // 解析进程组，根据进程组名返回对应的进程组指针
  c10::intrusive_ptr<c10d::ProcessGroup> resolve_group(
      const std::string& group_name) {
    std::shared_lock read_lock(lock_);
    // 在注册表中查找指定进程组名的项
    auto it = registry_.find(group_name);
    // 如果找不到对应进程组，则报错
    TORCH_CHECK(
        it != registry_.end(),
        "Could not resolve the process group registered under the name ",
        group_name);

    // 获取进程组弱引用指针
    auto group = it->second.lock();
    // 检查获取的进程组指针是否为空，如果为空则报错进程组已经销毁
    TORCH_CHECK(
        group != nullptr,
        "Process group registered under the name ",
        group_name,
        " has already been destroyed.");
    return group;
  }

  // 注销指定进程组名的进程组
  void unregister_group(const std::string& group_name) {
    std::unique_lock write_lock(lock_);
    // 从注册表中删除指定进程组名的项
    registry_.erase(group_name);
  }

  // 注销所有进程组
  void unregister_all_groups() {
    std::unique_lock write_lock(lock_);
    // 清空注册表
    registry_.clear();
  }

 private:
  // 使用 map 存储进程组名到进程组弱引用指针的映射关系
  std::map<std::string, c10::weak_intrusive_ptr<c10d::ProcessGroup>> registry_;
  // 使用 shared_mutex 提供读写并发控制
  std::shared_mutex lock_;
};

} // namespace

namespace c10d {

// 静态变量，表示当前是否处于线程隔离模式，默认为 false
static bool thread_isolation_mode = false;
// 静态变量，表示进程注册表的实例
static GroupRegistry process_registry;

// 设置线程隔离模式的函数
void set_thread_isolation_mode(bool enable) {
  thread_isolation_mode = enable;
}

// 获取当前线程隔离模式的函数
bool get_thread_isolation_mode() {
  return thread_isolation_mode;
}

// 注册进程组的函数，根据线程隔离模式选择不同的注册方式
void register_process_group(
    const std::string& group_name,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  if (thread_isolation_mode) {
    // 如果处于线程隔离模式，则使用 RankLocal<::GroupRegistry>::get() 注册进程组
    RankLocal<::GroupRegistry>::get().register_group(
        group_name, std::move(group));
  } else {
    // 否则使用全局进程注册表注册进程组
    process_registry.register_group(group_name, std::move(group));
  }
}

// 解析进程组的函数，根据线程隔离模式选择不同的解析方式
c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name) {
  if (thread_isolation_mode) {
    // 如果处于线程隔离模式，则使用 RankLocal<::GroupRegistry>::get() 解析进程组
    return RankLocal<::GroupRegistry>::get().resolve_group(group_name);
  } else {
    // 否则使用全局进程注册表解析进程组
    return process_registry.resolve_group(group_name);
  }
}

// 注销进程组的函数，根据线程隔离模式选择不同的注销方式
void unregister_process_group(const std::string& group_name) {
  if (thread_isolation_mode) {
    // 如果处于线程隔离模式，则使用 RankLocal<::GroupRegistry>::get() 注销进程组
    RankLocal<::GroupRegistry>::get().unregister_group(group_name);
  } else {
    // 否则使用全局进程注册表注销进程组
    process_registry.unregister_group(group_name);
  }
}

// 注销所有进程组的函数，根据线程隔离模式选择不同的注销方式
void unregister_all_process_groups() {
  if (thread_isolation_mode) {
    // 如果处于线程隔离模式，则使用 RankLocal<::GroupRegistry>::get() 注销所有进程组
    RankLocal<::GroupRegistry>::get().unregister_all_groups();
  } else {
    // 否则使用全局进程注册表注销所有进程组
    process_registry.unregister_all_groups();
  }
}

} // namespace c10d
```