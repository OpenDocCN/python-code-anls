# `.\pytorch\c10\util\ThreadLocalDebugInfo.cpp`

```
// 包含 C10 库中的异常处理、线程局部存储和调试信息相关的头文件
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <utility>  // 包含一些通用的实用函数和类模板

namespace c10 {

// 定义一个静态线程局部存储的指针，用于存储线程本地调试信息的共享指针
C10_DEFINE_TLS_static(std::shared_ptr<ThreadLocalDebugInfo>, tls_debug_info);
#define debug_info (tls_debug_info.get())  // 定义一个宏，用于获取当前线程的调试信息指针

/* static */
// 根据调试信息类型获取当前线程的调试信息
DebugInfoBase* ThreadLocalDebugInfo::get(DebugInfoKind kind) {
  // 获取当前线程的调试信息
  ThreadLocalDebugInfo* cur = debug_info.get();
  // 遍历当前线程的调试信息链表，直到找到指定类型的调试信息或链表结束
  while (cur) {
    if (cur->kind_ == kind) {
      return cur->info_.get();  // 返回找到的调试信息指针
    }
    cur = cur->parent_info_.get();  // 移动到下一个父调试信息
  }
  return nullptr;  // 如果未找到指定类型的调试信息，则返回空指针
}

/* static */
// 返回当前线程的调试信息的共享指针
std::shared_ptr<ThreadLocalDebugInfo> ThreadLocalDebugInfo::current() {
  return debug_info;  // 返回当前线程的调试信息共享指针
}

/* static */
// 强制设置当前线程的调试信息为给定的调试信息共享指针
void ThreadLocalDebugInfo::_forceCurrentDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfo> info) {
  debug_info = std::move(info);  // 强制设置当前线程的调试信息
}

/* static */
// 将给定的调试信息推入当前线程的调试信息链表
void ThreadLocalDebugInfo::_push(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  auto prev_info = debug_info;  // 保存当前线程的调试信息共享指针
  debug_info = std::make_shared<ThreadLocalDebugInfo>();  // 创建新的线程局部调试信息对象
  debug_info->parent_info_ = prev_info;  // 设置新调试信息的父调试信息为之前保存的指针
  debug_info->kind_ = kind;  // 设置新调试信息的类型
  debug_info->info_ = std::move(info);  // 设置新调试信息的详细信息
}

/* static */
// 弹出当前线程调试信息链表中指定类型的调试信息
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_pop(DebugInfoKind kind) {
  TORCH_CHECK(
      debug_info && debug_info->kind_ == kind,
      "Expected debug info of type ",
      (size_t)kind);  // 检查当前线程是否存在并且调试信息类型是否匹配
  auto res = debug_info;  // 保存当前线程的调试信息共享指针
  debug_info = debug_info->parent_info_;  // 设置当前线程的调试信息为父调试信息
  return res->info_;  // 返回之前保存的调试信息详细信息共享指针
}

/* static */
// 获取当前线程调试信息链表中指定类型的调试信息，但不弹出
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_peek(DebugInfoKind kind) {
  TORCH_CHECK(
      debug_info && debug_info->kind_ == kind,
      "Expected debug info of type ",
      (size_t)kind);  // 检查当前线程是否存在并且调试信息类型是否匹配
  return debug_info->info_;  // 返回当前线程的调试信息详细信息共享指针
}

// 调试信息保护类的构造函数，用于推入新的调试信息到线程调试信息链表中
DebugInfoGuard::DebugInfoGuard(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  if (!info) {
    return;  // 如果调试信息为空，则直接返回
  }
  prev_info_ = debug_info;  // 保存当前线程的调试信息共享指针
  ThreadLocalDebugInfo::_push(kind, std::move(info));  // 推入新的调试信息到线程调试信息链表
  active_ = true;  // 设置保护类为激活状态
}

// 调试信息保护类的析构函数，用于在对象销毁时恢复之前的调试信息状态
DebugInfoGuard::~DebugInfoGuard() {
  if (active_) {
    debug_info = prev_info_;  // 如果保护类为激活状态，则恢复之前保存的调试信息
  }
}

// 仅用于在跨线程边界后设置调试信息的保护类构造函数；
// 在这种情况下，我们假设线程池的线程没有活动的调试信息
DebugInfoGuard::DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info) {
  if (!info) {
    return;  // 如果调试信息为空，则直接返回
  }
  prev_info_ = std::move(debug_info);  // 保存当前线程的调试信息共享指针
  debug_info = std::move(info);  // 设置当前线程的调试信息为给定的调试信息共享指针
  active_ = true;  // 设置保护类为激活状态
}

} // namespace c10
```