# `.\pytorch\aten\src\ATen\xpu\XPUEvent.h`

```py
#pragma once
#include <ATen/xpu/XPUContext.h>

#include <optional>

namespace at::xpu {

/*
 * XPUEvent are movable not copyable wrappers around SYCL event. XPUEvent are
 * constructed lazily when first recorded. It has a device, and this device is
 * acquired from the first recording stream. Later streams that record the event
 * must match the same device.
 *
 * Currently, XPUEvent does NOT support to export an inter-process event from
 * another process via inter-process comunication(IPC). So it means that
 * inter-process communication for event handles between different processes is
 * not available. This could impact some applications that rely on cross-process
 * synchronization and communication.
 */
struct TORCH_XPU_API XPUEvent {
  // Constructors
  // XPUEvent 构造函数，默认不启用计时
  XPUEvent(bool enable_timing = false) noexcept
      : enable_timing_{enable_timing} {}

  // 析构函数，释放事件对象，并记录 GPU 事件的删除
  ~XPUEvent() {
    if (isCreated()) {
      // 获取 GPU 跟踪器实例
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 如果跟踪器存在，记录 GPU 事件的删除操作
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_deletion(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    }
  }

  // 禁用拷贝构造函数和赋值运算符
  XPUEvent(const XPUEvent&) = delete;
  XPUEvent& operator=(const XPUEvent&) = delete;

  // 移动构造函数和移动赋值运算符默认实现
  XPUEvent(XPUEvent&& other) = default;
  XPUEvent& operator=(XPUEvent&& other) = default;

  // 类型转换操作符，返回事件对象的引用
  operator sycl::event&() const {
    return event();
  }

  // 获取事件关联的设备信息，若未创建事件则返回空的 optional 对象
  std::optional<at::Device> device() const {
    if (isCreated()) {
      return at::Device(at::kXPU, device_index_);
    } else {
      return std::nullopt;
    }
  }

  // 判断事件是否已创建
  inline bool isCreated() const {
    return (event_.get() != nullptr);
  }

  // 获取事件关联的设备索引
  DeviceIndex device_index() const {
    return device_index_;
  }

  // 返回事件对象的引用
  sycl::event& event() const {
    return *event_;
  }

  // 查询事件的执行状态，若未创建事件则返回 true
  bool query() const {
    using namespace sycl::info;
    if (!isCreated()) {
      return true;
    }

    return event().get_info<event::command_execution_status>() ==
        event_command_status::complete;
  }

  // 在当前流中记录事件
  void record() {
    record(getCurrentXPUStream());
  }

  // 在指定流中记录事件，仅记录一次
  void recordOnce(const XPUStream& stream) {
    if (!isCreated()) {
      record(stream);
    }
  }

  // 在指定流中记录事件，允许多次记录
  void record(const XPUStream& stream) {
    if (!isCreated()) {
      device_index_ = stream.device_index();
      event_ = std::make_unique<sycl::event>(
          stream.queue().ext_oneapi_submit_barrier());
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 如果跟踪器存在，记录 GPU 事件的创建操作
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_creation(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    } else {
      // 检查流的设备索引与事件关联的设备索引是否匹配
      TORCH_CHECK(
          device_index_ == stream.device_index(),
          "Event device ",
          device_index_,
          " does not match recording stream's device ",
          stream.device_index(),
          ".");
      // 重置事件对象，然后在指定流中重新记录事件
      event_.reset();
      event_ = std::make_unique<sycl::event>(
          stream.queue().ext_oneapi_submit_barrier());
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 获取当前的 GPU 跟踪器对象
    if (C10_UNLIKELY(interp)) {
      // 如果跟踪器对象存在
      (*interp)->trace_gpu_event_record(
          at::kXPU,
          reinterpret_cast<uintptr_t>(event_.get()),
          reinterpret_cast<uintptr_t>(&stream.queue()));
      // 调用跟踪器记录 GPU 事件的方法，记录事件的类型、事件对象和队列对象的地址
    }
  }

  void block(const XPUStream& stream) {
    if (isCreated()) {
      // 如果事件对象已创建
      std::vector<sycl::event> event_list{event()};
      // 创建事件对象的向量列表
      // Make this stream wait until event_ is completed.
      // 让当前流等待事件对象完成
      stream.queue().ext_oneapi_submit_barrier(event_list);
      // 在流队列上提交一个 barrier，等待事件对象完成
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 获取当前的 GPU 跟踪器对象
      if (C10_UNLIKELY(interp)) {
        // 如果跟踪器对象存在
        (*interp)->trace_gpu_event_wait(
            at::kXPU,
            reinterpret_cast<uintptr_t>(event_.get()),
            reinterpret_cast<uintptr_t>(&stream.queue()));
        // 调用跟踪器等待 GPU 事件的方法，等待事件的类型、事件对象和队列对象的地址
      }
    }
  }

  float elapsed_time(const XPUEvent& other) const {
    TORCH_CHECK(
        isCreated() && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");
    TORCH_CHECK(
        enable_timing_ && other.enable_timing_,
        "Both events must be created with argument 'enable_timing=True'.");
    // TODO: provides the ability to time the execution of commands in a SYCL
    // queue without enabling profiling on the entire queue
    // 提供在 SYCL 队列上计时执行命令的能力，而无需在整个队列上启用性能分析
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "elapsed_time is not supported by XPUEvent.");
    // 报告未实现的错误，因为 XPUEvent 不支持 elapsed_time 方法
  }

  void synchronize() const {
    if (isCreated()) {
      // 如果事件对象已创建
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 获取当前的 GPU 跟踪器对象
      if (C10_UNLIKELY(interp)) {
        // 如果跟踪器对象存在
        (*interp)->trace_gpu_event_synchronization(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
        // 调用跟踪器执行 GPU 事件同步的方法，指定事件的类型和事件对象的地址
      }
      event().wait_and_throw();
      // 等待事件对象完成并抛出任何异常
    }
  }
};

} // namespace at::xpu
```