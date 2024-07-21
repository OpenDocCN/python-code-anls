# `.\pytorch\aten\src\ATen\cuda\CUDAEvent.h`

```
#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

namespace at::cuda {

/*
 * CUDAEvents are movable not copyable wrappers around CUDA's events.
 *
 * CUDAEvents are constructed lazily when first recorded unless it is
 * reconstructed from a cudaIpcEventHandle_t. The event has a device, and this
 * device is acquired from the first recording stream. However, if reconstructed
 * from a handle, the device should be explicitly specified; or if ipc_handle() is
 * called before the event is ever recorded, it will use the current device.
 * Later streams that record the event must match this device.
 */
struct TORCH_CUDA_CPP_API CUDAEvent {
  // Constructors
  // Default value for `flags` is specified below - it's cudaEventDisableTiming
  CUDAEvent() noexcept = default;
  CUDAEvent(unsigned int flags) noexcept : flags_{flags} {}

  CUDAEvent(
      DeviceIndex device_index, const cudaIpcEventHandle_t* handle) : device_index_(device_index) {
      // Acquire the CUDA context guard for the specified device index
      CUDAGuard guard(device_index_);

      // Open CUDA IPC event handle and assign to event_
      AT_CUDA_CHECK(cudaIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // CUDA context on other devices.
  ~CUDAEvent() {
    try {
      if (is_created_) {
        // Acquire the CUDA context guard for the device index of this event
        CUDAGuard guard(device_index_);

        // Get GPU trace if available and trace GPU event deletion
        const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
        if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_deletion(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
        }

        // Destroy the CUDA event
        AT_CUDA_CHECK(cudaEventDestroy(event_));
      }
    } catch (...) { /* No throw */ }
  }

  // Copy and assignment operations are deleted to enforce movability
  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  // Move constructor
  CUDAEvent(CUDAEvent&& other) noexcept { moveHelper(std::move(other)); }
  // Move assignment operator
  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  // Conversion operator to cudaEvent_t
  operator cudaEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.event_ < right.event_;
  }

  // Returns the optional device associated with the event
  optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kCUDA, device_index_);
    } else {
      return {};
    }
  }

  // Checks if the CUDA event has been created
  bool isCreated() const { return is_created_; }
  // Returns the device index associated with the event
  DeviceIndex device_index() const { return device_index_; }
  // Returns the CUDA event
  cudaEvent_t event() const { return event_; }

  // Queries the CUDA event status
  // Note: cudaEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err != cudaErrorNotReady) {
      // 如果错误不是 cudaErrorNotReady，则抛出异常
      C10_CUDA_CHECK(err);
    } else {
      // 如果错误是 cudaErrorNotReady，则忽略并清除该错误
      // 使用 (void)cudaGetLastError(); 来清除错误
      (void)cudaGetLastError();
    }

    // 返回 false 表示未完成
    return false;
  }

  // 记录当前 CUDA 流的事件
  void record() { record(getCurrentCUDAStream()); }

  // 仅在事件未被记录过时才记录
  void recordOnce(const CUDAStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: cudaEventRecord 必须在与事件相同的设备上调用。
  // 记录事件到指定的 CUDA 流
  void record(const CUDAStream& stream) {
    // 如果事件尚未创建，则创建它
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    // 检查事件的设备索引与流的设备索引是否匹配
    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
      " does not match recording stream's device ", stream.device_index(), ".");
    // 设置当前设备为事件所在的设备
    CUDAGuard guard(device_index_);
    // 记录事件到指定的 CUDA 流
    AT_CUDA_CHECK(cudaEventRecord(event_, stream));
    // 获取 GPU 跟踪器
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 GPU 跟踪器存在，则记录 GPU 事件
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(at::kCUDA,
          reinterpret_cast<uintptr_t>(event_),
          reinterpret_cast<uintptr_t>(stream.stream())
      );
    }
    // 标记事件已被记录
    was_recorded_ = true;
  }

  // Note: cudaStreamWaitEvent 必须在与流相同的设备上调用。
  // 该事件并不实际持有任何 GPU 资源。
  // 阻塞当前流，直到事件完成
  void block(const CUDAStream& stream) {
    // 如果事件已创建
    if (is_created_) {
      // 设置当前设备为流所在的设备
      CUDAGuard guard(stream.device_index());
      // 等待流执行完事件
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
      // 获取 GPU 跟踪器
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 如果 GPU 跟踪器存在，则等待 GPU 事件
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(at::kCUDA,
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream())
        );
      }
    }
  }

  // Note: cudaEventElapsedTime 可以安全地从任何设备调用。
  // 计算与另一个事件的经过时间（毫秒）
  float elapsed_time(const CUDAEvent& other) const {
    // 检查两个事件是否都已记录
    TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // 设置当前设备为事件所在的设备
    CUDAGuard guard(device_index_);
    // 获取两个事件之间的经过时间
    AT_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: cudaEventSynchronize 可以安全地从任何设备调用。
  // 同步等待事件完成
  void synchronize() const {
    // 如果事件已创建
    if (is_created_) {
      // 获取 GPU 跟踪器
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      // 如果 GPU 跟踪器存在，则同步 GPU 事件
      if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_synchronization(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
      }
      // 等待事件完成
      AT_CUDA_CHECK(cudaEventSynchronize(event_));

      }
    }
  // 结束了一个代码块
  }
}

// 注意：cudaIpcGetEventHandle 必须在与事件相同的设备上调用
void ipc_handle(cudaIpcEventHandle_t * handle) {
    // 如果事件尚未创建，则从当前 CUDA 流的设备索引创建事件
    if (!is_created_) {
      // 这个 CUDAEvent 对象最初是根据标志构造的，但是事件尚未创建。
      createEvent(getCurrentCUDAStream().device_index());
    }
    // 在特定的 CUDA 设备上操作
    CUDAGuard guard(device_index_);
    // 调用 CUDA 函数获取事件的 IPC 句柄
    AT_CUDA_CHECK(cudaIpcGetEventHandle(handle, event_));
}
# 初始化事件的标志位为禁用计时
unsigned int flags_ = cudaEventDisableTiming;
# 表示事件是否已创建的标志
bool is_created_ = false;
# 表示事件是否已记录的标志
bool was_recorded_ = false;
# 指定事件所在的设备索引，默认为未指定状态
DeviceIndex device_index_ = -1;
# CUDA 事件对象的初始化
cudaEvent_t event_{};

# 创建事件的私有方法，接受设备索引作为参数
void createEvent(DeviceIndex device_index) {
  # 将设备索引设置为传入的设备索引
  device_index_ = device_index;
  # 使用 CUDAGuard 保护设备上下文
  CUDAGuard guard(device_index_);
  # 使用指定的标志位创建 CUDA 事件对象
  AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
  # 获取 GPU 跟踪器的实例，并记录 GPU 事件的创建过程
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_event_creation(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
  }
  # 标记事件已成功创建
  is_created_ = true;
}

# 移动语义的辅助函数，用于移动构造 CUDAEvent 对象
void moveHelper(CUDAEvent&& other) {
  # 使用 std::swap 交换成员变量，实现对象的移动
  std::swap(flags_, other.flags_);
  std::swap(is_created_, other.is_created_);
  std::swap(was_recorded_, other.was_recorded_);
  std::swap(device_index_, other.device_index_);
  std::swap(event_, other.event_);
}
```