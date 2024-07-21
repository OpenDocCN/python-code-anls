# `.\pytorch\c10\xpu\impl\XPUGuardImpl.h`

```
#pragma once
// 预处理指令，确保此头文件只被包含一次

#include <c10/core/DeviceGuard.h>
// 包含 C10 库的设备保护功能头文件
#include <c10/core/impl/DeviceGuardImplInterface.h>
// 包含 C10 库的设备保护实现接口头文件
#include <c10/core/impl/GPUTrace.h>
// 包含 C10 库的 GPU 跟踪功能头文件
#include <c10/xpu/XPUCachingAllocator.h>
// 包含 C10 XPU 缓存分配器头文件
#include <c10/xpu/XPUFunctions.h>
// 包含 C10 XPU 功能头文件
#include <c10/xpu/XPUStream.h>
// 包含 C10 XPU 流处理头文件

#include <vector>
// 包含 C++ 标准库的向量容器头文件

namespace c10::xpu::impl {

struct XPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  // 定义 XPUGuardImpl 结构体，继承自 DeviceGuardImplInterface 接口

  static constexpr DeviceType static_type = kXPU;
  // 静态常量，指定设备类型为 XPU

  XPUGuardImpl() = default;
  // 默认构造函数

  explicit XPUGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == kXPU);
    // 显式构造函数，确保设备类型为 XPU
  }

  DeviceType type() const override {
    return kXPU;
    // 返回设备类型为 XPU
  }

  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    // 断言设备类型为 XPU
    const auto old_device_index = c10::xpu::exchange_device(d.index());
    // 调用 XPU 设备交换函数，获取旧设备索引
    return Device(kXPU, old_device_index);
    // 返回新设备对象
  }

  Device getDevice() const override {
    const auto device = c10::xpu::current_device();
    // 获取当前 XPU 设备
    return Device(kXPU, device);
    // 返回当前设备对象
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    // 断言设备类型为 XPU
    c10::xpu::set_device(d.index());
    // 设置当前 XPU 设备
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    c10::xpu::set_device(d.index());
    // 不检查设置当前 XPU 设备
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentXPUStream(d.index()).unwrap();
    // 获取当前 XPU 设备的流对象
  }

  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
    // 获取新的 XPU 流对象，可以设置优先级
  }

  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
    // 从全局池获取 XPU 流对象，可以设置是否高优先级
  }

  // NB: These do NOT set the current device
  // 注意：这些函数不会设置当前设备
  Stream exchangeStream(Stream s) const noexcept override {
    const XPUStream stream(s);
    // 创建 XPUStream 对象
    const auto old_stream = getCurrentXPUStream(s.device().index());
    // 获取当前 XPU 设备的当前流对象
    setCurrentXPUStream(stream);
    // 设置当前 XPU 设备的当前流对象
    return old_stream.unwrap();
    // 返回原来的流对象
  }

  DeviceIndex deviceCount() const noexcept override {
    return c10::xpu::device_count();
    // 返回 XPU 设备数量
  }

  // Event-related functions
  // 事件相关函数

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    // 如果事件为空则返回

    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 获取 GPU 跟踪对象的指针
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          c10::kXPU, reinterpret_cast<uintptr_t>(event));
      // 在 GPU 跟踪对象中记录 GPU 事件的删除
    }

    delete reinterpret_cast<sycl::event*>(event);
    // 删除事件对象
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");
    // 检查事件设备索引是否匹配记录流的设备索引

    auto* xpu_event = reinterpret_cast<sycl::event*>(*event);
    // 获取 XPU 事件对象指针
    const XPUStream xpu_stream{stream};
    // 创建 XPU 流对象

    // Delete the event previously recorded.
    // 删除先前记录的事件
    if (xpu_event)
      delete xpu_event;
    // 创建一个新的 SYCL 事件，并使用队列的扩展方法 ext_oneapi_submit_barrier() 进行提交，将其转换为 void 指针类型并赋值给 event
    xpu_event = new sycl::event(xpu_stream.queue().ext_oneapi_submit_barrier());
    // 将 xpu_event 的地址转换为 void 指针并存储在 event 指向的位置

    // 获取 GPUTrace 的跟踪器实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 interp 不为空，调用其 trace_gpu_event_record 方法记录 GPU 事件
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kXPU,
          reinterpret_cast<uintptr_t>(xpu_event),
          reinterpret_cast<uintptr_t>(&xpu_stream.queue()));
    }
  }

  // 实现基类函数 block
  void block(void* event, const Stream& stream) const override {
    // 如果 event 为空指针，则直接返回
    if (!event)
      return;
    // 将 event 强制转换为 sycl::event* 类型并赋值给 xpu_event
    auto* xpu_event = reinterpret_cast<sycl::event*>(event);
    // 创建事件列表并初始化为 xpu_event
    std::vector<sycl::event> event_list{*xpu_event};
    // 创建 XPUStream 对象 xpu_stream，用于调用其队列的 ext_oneapi_submit_barrier 方法
    const XPUStream xpu_stream(stream);
    xpu_stream.queue().ext_oneapi_submit_barrier(event_list);
    // 获取 GPUTrace 的跟踪器实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 interp 不为空，调用其 trace_gpu_event_wait 方法等待 GPU 事件完成
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          c10::kXPU,
          reinterpret_cast<uintptr_t>(xpu_event),
          reinterpret_cast<uintptr_t>(&xpu_stream.queue()));
    }
  }

  // 实现基类函数 queryEvent
  bool queryEvent(void* event) const override {
    // 如果 event 为空指针，返回 true
    if (!event)
      return true;
    // 将 event 强制转换为 sycl::event* 类型并赋值给 xpu_event
    auto* xpu_event = reinterpret_cast<sycl::event*>(event);
    // 查询 xpu_event 的执行状态是否为 complete，并返回相应结果
    return xpu_event->get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete;
  }

  // Stream 相关函数

  // 查询给定流 stream 是否有效
  bool queryStream(const Stream& stream) const override {
    // 创建 XPUStream 对象 xpu_stream 并调用其 query 方法返回结果
    const XPUStream xpu_stream{stream};
    return xpu_stream.query();
  }

  // 同步给定流 stream
  void synchronizeStream(const Stream& stream) const override {
    // 创建 XPUStream 对象 xpu_stream 并调用其 synchronize 方法
    const XPUStream xpu_stream{stream};
    xpu_stream.synchronize();
  }

  // 同步给定事件 event
  void synchronizeEvent(void* event) const override {
    // 如果 event 为空指针，直接返回
    if (!event)
      return;
    // 将 event 强制转换为 sycl::event* 类型并赋值给 xpu_event
    auto* xpu_event = reinterpret_cast<sycl::event*>(event);
    // 获取 GPUTrace 的跟踪器实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 interp 不为空，调用其 trace_gpu_event_synchronization 方法记录 GPU 事件同步
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_synchronization(
          c10::kXPU, reinterpret_cast<uintptr_t>(xpu_event));
    }
    // 等待 xpu_event 的执行并抛出任何异常
    xpu_event->wait_and_throw();
  }

  // 在给定流 stream 上记录数据指针 data_ptr
  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    // 创建 XPUStream 对象 xpu_stream 并调用 XPUCachingAllocator::recordStream 方法记录数据指针
    const XPUStream xpu_stream{stream};
    XPUCachingAllocator::recordStream(data_ptr, xpu_stream);
  }

  // 计算事件 event1 和 event2 之间的时间差，返回秒数
  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    // 报告错误，不支持 XPU 后端的 elapsedTime 函数
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "elapsedTime is not supported by XPU backend.");
  }
};

// 结束 c10::xpu::impl 命名空间
} // namespace c10::xpu::impl
```