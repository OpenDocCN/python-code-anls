# `.\pytorch\c10\cuda\impl\CUDAGuardImpl.h`

```py
  #pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace c10::cuda::impl {

// CUDA设备守卫实现，继承自DeviceGuardImplInterface接口
struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  // 静态成员变量，表示设备类型为CUDA
  static constexpr DeviceType static_type = DeviceType::CUDA;

  // 默认构造函数
  CUDAGuardImpl() = default;
  // 显式构造函数，接受设备类型参数t，验证其为CUDA类型
  explicit CUDAGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::CUDA);
  }
  
  // 返回设备类型为CUDA
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  
  // 交换当前设备并返回旧设备，需验证设备为CUDA类型
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    auto old_device_index = c10::cuda::ExchangeDevice(d.index());
    return Device(DeviceType::CUDA, old_device_index);
  }
  
  // 获取当前设备，调用CUDA函数获取设备编号并返回CUDA设备对象
  Device getDevice() const override {
    DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  
  // 不安全地获取当前设备，使用CUDA函数获取设备编号，返回可能为空的CUDA设备对象
  std::optional<Device> uncheckedGetDevice() const noexcept {
    DeviceIndex device{-1};
    const auto err = C10_CUDA_ERROR_HANDLED(c10::cuda::GetDevice(&device));
    C10_CUDA_CHECK_WARN(err);
    if (err != cudaSuccess) {
      return c10::nullopt;
    }
    return Device(DeviceType::CUDA, device);
  }
  
  // 设置当前设备，需验证设备为CUDA类型，调用CUDA函数设置设备
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    C10_CUDA_CHECK(c10::cuda::SetDevice(d.index()));
  }
  
  // 不安全地设置当前设备，调用CUDA函数尝试设置设备
  void uncheckedSetDevice(Device d) const noexcept override {
    C10_CUDA_CHECK_WARN(c10::cuda::MaybeSetDevice(d.index()));
  }
  
  // 获取当前流对象，使用当前设备的CUDA流
  Stream getStream(Device d) const noexcept override {
    return getCurrentCUDAStream(d.index()).unwrap();
  }
  
  // 获取默认流对象，使用当前设备的默认CUDA流
  Stream getDefaultStream(Device d) const override {
    return getDefaultCUDAStream(d.index());
  }
  
  // 创建新流对象，使用当前设备的CUDA流池中的流
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }
  
  // 从全局流池获取流对象，使用当前设备的CUDA流池中的流
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }
  
  // 交换流对象，使用给定流对象替换当前设备的当前CUDA流对象，并返回旧流对象
  // 注意：不会更改当前设备
  Stream exchangeStream(Stream s) const noexcept override {
    CUDAStream cs(s);
    auto old_stream = getCurrentCUDAStream(s.device().index());
    setCurrentCUDAStream(cs);
    return old_stream.unwrap();
  }
  
  // 返回设备数量，使用CUDA函数获取设备数量
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event相关函数

  // 创建CUDA事件，使用给定的EventFlag映射为CUDA标志
  void createEvent(cudaEvent_t* cuda_event, const EventFlag flag) const {
    auto cuda_flag = cudaEventDefault;
    // 根据不同的事件标志设置相应的 CUDA 事件标志
    switch (flag) {
      // 如果事件标志为 PYTORCH_DEFAULT，使用 cudaEventDisableTiming 标志
      case EventFlag::PYTORCH_DEFAULT:
        cuda_flag = cudaEventDisableTiming;
        break;
      // 如果事件标志为 BACKEND_DEFAULT，使用 cudaEventDefault 标志
      case EventFlag::BACKEND_DEFAULT:
        cuda_flag = cudaEventDefault;
        break;
      // 对于未知的事件标志，抛出错误信息并终止程序
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }

    // 使用指定标志创建 CUDA 事件
    C10_CUDA_CHECK(cudaEventCreateWithFlags(cuda_event, cuda_flag));

    // 获取当前 GPU 跟踪器的实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 GPU 跟踪器实例存在，记录 GPU 事件的创建过程
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
  }

  // 销毁 CUDA 事件
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    // 如果事件为空指针，直接返回
    if (!event)
      return;

    // 将事件转换为 cudaEvent_t 类型
    auto cuda_event = static_cast<cudaEvent_t>(event);
    // 获取当前设备索引
    DeviceIndex orig_device{-1};
    C10_CUDA_CHECK_WARN(c10::cuda::GetDevice(&orig_device));
    // 将设备切换到指定的 device_index
    C10_CUDA_CHECK_WARN(c10::cuda::SetDevice(device_index));
    // 获取当前 GPU 跟踪器的实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 GPU 跟踪器实例存在，记录 GPU 事件的销毁过程
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    // 销毁 CUDA 事件
    C10_CUDA_CHECK_WARN(cudaEventDestroy(cuda_event));
    // 恢复到原始的设备索引
    C10_CUDA_CHECK_WARN(c10::cuda::SetDevice(orig_device));
  }

  // 记录 CUDA 事件
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    // 检查事件的设备索引是否与流的设备索引匹配
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    // 将事件指针转换为 cudaEvent_t 类型
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(*event);
    // 将流对象转换为 CUDAStream 类型
    CUDAStream cuda_stream{stream};

    // 获取当前的设备索引并保存为原始设备索引
    const auto orig_device = getDevice();
    // 设置设备到流的设备上
    setDevice(stream.device());

    // 如果事件为空，创建一个 CUDA 事件
    if (!cuda_event)
      createEvent(&cuda_event, flag);

    // 记录 CUDA 事件到指定流中
    C10_CUDA_CHECK(cudaEventRecord(cuda_event, cuda_stream));

    // 更新事件指针，使其指向可能刚刚分配的 CUDA 事件
    *event = cuda_event;

    // 获取当前 GPU 跟踪器的实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 GPU 跟踪器实例存在，记录 GPU 事件的记录过程
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }

    // 恢复到原始的设备索引
    setDevice(orig_device);
  }

  // 阻塞等待 CUDA 事件完成
  void block(void* event, const Stream& stream) const override {
    // 如果事件为空指针，直接返回
    if (!event)
      return;

    // 将事件转换为 cudaEvent_t 类型
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    // 将流对象转换为 CUDAStream 类型
    CUDAStream cuda_stream{stream};

    // 获取当前的设备索引并保存为原始设备索引
    const auto orig_device = getDevice();
    // 设置设备到流的设备上
    setDevice(stream.device());

    // 使用 CUDA 流等待 CUDA 事件完成
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        cuda_stream,
        cuda_event,
        /*flags (must be zero)=*/0));

    // 获取当前 GPU 跟踪器的实例
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    // 如果 GPU 跟踪器实例存在，记录 GPU 事件的阻塞过程
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_block(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }
    if (C10_UNLIKELY(interp)) {
      // 如果 interp 不为空，则执行以下操作：
      (*interp)->trace_gpu_event_wait(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }
    // 恢复原始设备
    setDevice(orig_device);
  }

  // 可以从任何设备调用
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    // 注意: cudaEventQuery 可以安全地从任何设备调用
    const cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(cuda_event));
    if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    } else {
      // 如果事件未准备好，则忽略并清除错误
      (void)cudaGetLastError();
    }
    return (err == cudaSuccess);
  }

  // 与流相关的函数
  bool queryStream(const Stream& stream) const override {
    CUDAStream cuda_stream{stream};
    return cuda_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    CUDAStream cuda_stream{stream};
    cuda_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      // 如果 interp 不为空，则执行以下操作：
      (*interp)->trace_gpu_event_synchronization(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    // 注意: cudaEventSynchronize 可以安全地从任何设备调用
    C10_CUDA_CHECK(cudaEventSynchronize(cuda_event));
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    CUDAStream cuda_stream{stream};
    // 记录数据指针在指定流上的信息
    CUDACachingAllocator::recordStream(data_ptr, cuda_stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // 即使 cudaEventElapsedTime 可以安全地从任何设备调用，如果当前设备未初始化，它将创建一个新的 CUDA 上下文，这会消耗大量内存。
    DeviceIndex orig_device{-1};
    C10_CUDA_CHECK(c10::cuda::GetDevice(&orig_device));
    C10_CUDA_CHECK(c10::cuda::SetDevice(device_index));
    cudaEvent_t cuda_event1 = static_cast<cudaEvent_t>(event1);
    cudaEvent_t cuda_event2 = static_cast<cudaEvent_t>(event2);
    float time_ms = 0;
    // 如果任一事件已记录但尚未完成，则引发 cudaErrorNotReady
    C10_CUDA_CHECK(cudaEventElapsedTime(&time_ms, cuda_event1, cuda_event2));
    C10_CUDA_CHECK(c10::cuda::SetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

// 结束 c10::cuda::impl 命名空间
} // namespace c10::cuda::impl
```