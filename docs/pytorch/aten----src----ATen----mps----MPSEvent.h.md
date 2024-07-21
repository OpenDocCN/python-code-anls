# `.\pytorch\aten\src\ATen\mps\MPSEvent.h`

```
// 版权声明，标明代码版权归 Apple Inc. 所有
#pragma once

// 包含 ATen 库中的 MPSStream.h 头文件，提供了与 MPS 相关的流处理功能
#include <ATen/mps/MPSStream.h>

// 包含时间处理相关的头文件
#include <ctime>
// 包含标准库中的 stack 容器头文件
#include <stack>

// 定义了 at::mps 命名空间
namespace at::mps {

// MPSEvent 类的声明，不要直接创建该类的实例，应使用 MPSEventPool 类获取 MPSEvent 实例
class MPSEvent {
public:
  // 构造函数，初始化 MPSEvent 实例
  explicit MPSEvent(id_t ID, MPSStream* stream, bool enable_timing);
  // 析构函数，清理 MPSEvent 实例
  ~MPSEvent();

  // 记录事件在流上的发生
  void record(bool needsLock, bool syncEvent = false);
  // 使流中未来提交的工作等待此事件
  bool wait(bool needsLock, bool syncEvent = false);
  // 调度事件的 notifyListener 回调
  bool notify(bool needsLock, MTLSharedEventNotificationBlock block);
  // 检查事件是否已经被信号通知
  bool query() const;
  // 阻塞 CPU 线程，直到所有在记录此事件之前提交的 GPU 工作完成
  bool synchronize();
  // 重置事件，以便从事件池中重新使用
  void reset(MPSStream* stream, bool enable_timing);
  // 返回事件实例的唯一 ID
  id_t getID() const { return m_id; }
  // 返回事件完成的时间戳
  uint64_t getCompletionTime() const { return m_completion_time; }
  // 如果已记录，等待 cpu_sync_cv 被信号通知
  void waitForCpuSync();

private:
  // 事件的唯一 ID
  id_t m_id;
  // 是否启用测量事件完成时间
  bool m_enable_timing;
  // 信号计数器
  uint64_t m_signalCounter = 0;
  // 关联的 MPS 流对象
  MPSStream* m_stream = nullptr;
  // Metal 共享事件对象
  MTLSharedEvent_t m_event = nullptr;
  // Metal 共享事件监听器
  MTLSharedEventListener* m_listener = nullptr;
  // 用于同步在该流上创建的事件与 CPU 的互斥量
  std::mutex m_cpu_sync_mutex{};
  // 用于同步在该流上创建的事件与 CPU 的条件变量
  std::condition_variable m_cpu_sync_cv{};
  // 用于同步在该流上创建的事件与 CPU 的标志
  bool m_cpu_sync_completed = false;
  // 用于计算完成时间的时间戳
  uint64_t m_completion_time = 0;

  // 在锁定状态下记录事件的私有方法
  void recordLocked(bool syncEvent);
  // 在锁定状态下等待事件的私有方法
  bool waitLocked(bool syncEvent);
  // 在锁定状态下通知事件的私有方法
  bool notifyLocked(MTLSharedEventNotificationBlock block);
  // 通知 CPU 同步完成的私有方法
  void notifyCpuSync();
  // 获取当前时间的静态方法
  static uint64_t getTime() {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
  }
};

// MPSEventPtr 类型定义，是 MPSEvent 的智能指针，带有自定义删除器
typedef std::unique_ptr<MPSEvent, std::function<void(MPSEvent*)>> MPSEventPtr;

// MPSEventPool 类声明，用于管理 MPSEvent 实例的对象池
class MPSEventPool {
public:
  // 构造函数，初始化事件池，并关联默认流对象
  explicit MPSEventPool(MPSStream* default_stream);
  // 析构函数，清理事件池资源
  ~MPSEventPool();

  // 获取一个 MPSEvent 实例，可以指定是否启用计时和关联的流对象
  MPSEventPtr acquireEvent(bool enable_timing, MPSStream* stream);
  // 清空事件缓存，释放所有事件实例
  void emptyCache();

  // 以下方法主要用于 MPSHooks 和 torch.mps.Event() 的绑定

  // 获取一个事件的唯一 ID，可以指定是否启用计时
  id_t acquireEvent(bool enable_timing);
  // 释放指定 ID 的事件实例
  void releaseEvent(id_t event_id);
  // 记录指定 ID 的事件，可以指定是否同步
  void recordEvent(id_t event_id, bool syncEvent);
  // 等待指定 ID 的事件，可以指定是否同步
  void waitForEvent(id_t event_id, bool syncEvent);
  // 同步指定 ID 的事件
  void synchronizeEvent(id_t event_id);
  // 查询指定 ID 的事件是否已经完成
  bool queryEvent(id_t event_id);
  // 返回两个记录事件之间的经过时间，单位为毫秒
  double elapsedTime(id_t start_event_id, id_t end_event_id);
private:
  // 默认流对象指针，初始化为空指针
  MPSStream* m_default_stream = nullptr;
  // 递归互斥锁，用于保护共享资源
  std::recursive_mutex m_mutex;
  // 事件对象的堆栈，用于事件对象的重用
  std::stack<std::unique_ptr<MPSEvent>> m_pool{};
  // 用于将事件ID与事件对象关联的字典
  // 用于保留在池外使用的事件，供 torch.mps.Event() 绑定使用
  std::unordered_map<id_t, MPSEventPtr> m_in_use_events{};
  // 事件计数器，用于生成唯一的事件ID
  uint64_t m_event_counter = 0;
  // 默认的事件对象删除函数对象
  std::function<void(MPSEvent*)> m_default_deleter;

  // 获取正在使用的事件对象的指针
  MPSEvent* getInUseEvent(id_t event_id, bool locked = true);
};

// 获取 MPSEventPool 的共享指针，确保在依赖实例销毁后销毁 MPSEventPool
std::shared_ptr<MPSEventPool> getMPSEventPool();

} // namespace at::mps
```