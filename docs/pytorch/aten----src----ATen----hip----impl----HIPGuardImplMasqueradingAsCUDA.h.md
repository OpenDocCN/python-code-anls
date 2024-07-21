# `.\pytorch\aten\src\ATen\hip\impl\HIPGuardImplMasqueradingAsCUDA.h`

```py
#pragma once
// 预处理指令，确保此头文件仅被包含一次

#include <ATen/hip/HIPConfig.h>
// 包含 ATen 库的 HIPConfig 头文件

// The includes of HIPGuard.h
// 包含 HIPGuard.h 的依赖头文件
#include <c10/hip/impl/HIPGuardImpl.h>
#include <c10/hip/HIPMacros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/util/Exception.h>

#include <c10/hip/impl/HIPGuardImpl.h>
// 再次包含 HIPGuardImpl.h 头文件，可能是为了确保定义一致性

#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
// 包含 HIP 下的 CachingAllocator 和 Stream 的 CUDA 伪装实现头文件

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
// 使用 c10::hip 命名空间使得 HIP 化更加简便，避免了额外的命名空间修复工作

namespace c10 { namespace hip {

// Note [Masquerading as CUDA]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 关于 CUDA 的伪装说明，用 HIP 替代 CUDA 的情况和解决方法的说明

// c10_hip is very easy to understand: it is HIPified from c10_cuda,
// and anywhere you said CUDA, the source code now says HIP.  HIPified
// PyTorch is much harder to understand: it is HIPified from regular
// PyTorch, yes, but NO source-to-source translation from CUDA to
// HIP occurs; instead, anywhere we see "CUDA", it actually means "HIP".
// For example, when you use HIPified PyTorch, you say x.cuda() to
// move a tensor onto ROCm device.  We call this situation "HIP
// masquerading as CUDA".
//
// This leads to a very awkward situation when we want to call c10_hip
// code from PyTorch, since c10_hip is expecting things to be called
// HIP, but PyTorch is calling them CUDA (masquerading as HIP).  To
// fix this impedance mismatch, we have MasqueradingAsCUDA variants
// for all c10_hip classes.  These translate between the "HIP" and "CUDA
// masquerading as HIP" worlds.  For example,
// HIPGuardImplMasqueradingAsCUDA (this file) provides something like a
// HIPGuardImpl, but it reports its DeviceType as CUDA (e.g., type()
// returns CUDA, getDevice() reports the current HIP device as a CUDA
// device.)
//
// We should be able to delete all of these classes entirely once
// we switch PyTorch to calling a HIP a HIP.
//
// When you add a new MasqueradingAsCUDA class/function, you need to
// also update the rewrite rules in torch/utils/hipify/cuda_to_hip_mappings.py
//
//
//
// By the way, note that the cpp file associated with this also
// *overwrites* the entry in the DeviceGuardImpl registry for CUDA with
// this HIP implementation.
//
// 关于 HIP masquerading as CUDA 的情况说明，解释了为什么需要伪装和如何处理伪装间的不匹配问题

struct HIPGuardImplMasqueradingAsCUDA final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::CUDA;
  // 类静态常量，标识设备类型为 CUDA

  HIPGuardImplMasqueradingAsCUDA() {}
  // 默认构造函数

  HIPGuardImplMasqueradingAsCUDA(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::CUDA);
  }
  // 带参数构造函数，断言传入的设备类型为 CUDA

  c10::DeviceType type() const override {
    return c10::DeviceType::CUDA;
  }
  // 返回当前设备类型为 CUDA

  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    // 断言传入的设备是 CUDA 设备
    Device old_device = getDevice();
    // 获取当前设备
    if (old_device.index() != d.index()) {
      C10_HIP_CHECK(hipSetDevice(d.index()));
    }
    // 设置新的 HIP 设备索引
    return old_device;
    // 返回旧的设备
  }

  Device getDevice() const override {
    int device;
    C10_HIP_CHECK(hipGetDevice(&device));
    // 获取当前 HIP 设备索引
    return Device(DeviceType::CUDA, device);
    // 返回当前设备类型为 CUDA 的设备
  }
};
// HIPGuardImplMasqueradingAsCUDA 结构体实现，提供类似 HIPGuardImpl 的功能，
// 但将其设备类型报告为 CUDA

}} // namespace c10::hip
// 命名空间 c10::hip 的结束
  return Device(c10::DeviceType::CUDA, device);


  // 返回一个表示 CUDA 设备的 Device 对象，使用给定的设备索引
  return Device(c10::DeviceType::CUDA, device);



  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }


  // 设置当前线程的设备为给定的 CUDA 设备
  void setDevice(Device d) const override {
    // 确保给定的设备是 CUDA 设备
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    // 调用 HIP 函数设置设备为给定设备的索引
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }



  void uncheckedSetDevice(Device d) const noexcept override {
    C10_HIP_CHECK_WARN(hipSetDevice(d.index()));
  }


  // 不安全地设置当前线程的设备为给定的 CUDA 设备，不抛出异常版本
  void uncheckedSetDevice(Device d) const noexcept override {
    // 调用 HIP 函数设置设备为给定设备的索引，如果出错则发出警告
    C10_HIP_CHECK_WARN(hipSetDevice(d.index()));
  }



  Stream getStream(Device d) const noexcept override {
    return getCurrentHIPStreamMasqueradingAsCUDA(d.index()).unwrap();
  }


  // 获取当前线程的 CUDA 流对象，对外表现为 Stream 对象
  Stream getStream(Device d) const noexcept override {
    // 获取当前设备对应的 HIP 流对象，然后将其包装成 CUDA 流对象返回
    return getCurrentHIPStreamMasqueradingAsCUDA(d.index()).unwrap();
  }



  Stream getDefaultStream(Device d) const override {
    return getDefaultHIPStreamMasqueradingAsCUDA(d.index());
  }


  // 获取默认的 CUDA 流对象
  Stream getDefaultStream(Device d) const override {
    // 获取默认的 HIP 流对象，并将其包装成 CUDA 流对象返回
    return getDefaultHIPStreamMasqueradingAsCUDA(d.index());
  }



  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPoolMasqueradingAsCUDA(priority, d.index());
  }


  // 获取一个新的 CUDA 流对象，带有给定的优先级
  Stream getNewStream(Device d, int priority = 0) const override {
    // 从池中获取对应设备和优先级的 HIP 流对象，然后将其包装成 CUDA 流对象返回
    return getStreamFromPoolMasqueradingAsCUDA(priority, d.index());
  }



  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false) const override {
    return getStreamFromPoolMasqueradingAsCUDA(isHighPriority, d.index());
  }


  // 从全局池中获取 CUDA 流对象，可选的指定是否高优先级
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false) const override {
    // 从池中获取对应设备和优先级的 HIP 流对象，然后将其包装成 CUDA 流对象返回
    return getStreamFromPoolMasqueradingAsCUDA(isHighPriority, d.index());
  }



  Stream exchangeStream(Stream s) const noexcept override {
    HIPStreamMasqueradingAsCUDA cs(s);
    auto old_stream = getCurrentHIPStreamMasqueradingAsCUDA(s.device().index());
    setCurrentHIPStreamMasqueradingAsCUDA(cs);
    return old_stream.unwrap();
  }


  // 交换流对象并返回之前的流对象
  Stream exchangeStream(Stream s) const noexcept override {
    // 将给定的 CUDA 流对象转换为 HIP 流对象
    HIPStreamMasqueradingAsCUDA cs(s);
    // 获取当前 HIP 流对象，用新 HIP 流对象替换为当前流对象
    auto old_stream = getCurrentHIPStreamMasqueradingAsCUDA(s.device().index());
    setCurrentHIPStreamMasqueradingAsCUDA(cs);
    // 返回之前的 CUDA 流对象
    return old_stream.unwrap();
  }



  DeviceIndex deviceCount() const noexcept override {
    int deviceCnt;
    hipError_t _err;
    _err = hipGetDeviceCount(&deviceCnt);
    if(_err != hipErrorNoDevice && _err != hipSuccess)
        C10_HIP_CHECK(_err);
    return deviceCnt;
  }


  // 返回当前系统中的 CUDA 设备数量
  DeviceIndex deviceCount() const noexcept override {
    int deviceCnt;
    hipError_t _err;
    // 调用 HIP 函数获取设备数量，并存储在 deviceCnt 变量中
    _err = hipGetDeviceCount(&deviceCnt);
    // 如果出错不是没有设备或者成功，则检查并报错
    if(_err != hipErrorNoDevice && _err != hipSuccess)
        C10_HIP_CHECK(_err);
    // 返回设备数量
    return deviceCnt;
  }



  // Event-related functions
  // Note: hipEventCreateWithFlags should be called on the same device as
  //  the recording stream's device.
  void createEvent(
    hipEvent_t* hip_event,
    const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to HIP flag
    auto hip_flag = hipEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        hip_flag = hipEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        hip_flag = hipEventDefault;
        break;
      default:
        TORCH_CHECK(false, "HIP event received unknown flag");
    }

    C10_HIP_CHECK(hipEventCreateWithFlags(hip_event, hip_flag));
  }


  // 创建一个 HIP 事件对象
  // 注意：hipEventCreateWithFlags 应该在与记录流的设备相同的设备上调用
  void createEvent(
    hipEvent_t* hip_event,
    const EventFlag flag) const {
    // 将 PyTorch 的事件标志映射到 HIP 的标志
    auto hip_flag = hipEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        hip_flag = hipEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        hip_flag = hipEventDefault;
        break;
      default:
        TORCH_CHECK(false, "HIP event received unknown flag");
    }

    // 调用 HIP 函数创建带有指定标志的 HIP 事件
    C10_HIP_CHECK(hipEventCreateWithFlags(hip_event, hip_flag));
  }



  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override {
    if (!event) return;
    auto hip_event = static_cast<hipEvent_t>(event);
    int orig_device;
    C10_HIP_CHECK_WARN(hipGetDevice(&orig_device));
    C10_HIP_CHECK_WARN(hipSetDevice(device_index));
    C10_HIP_CHECK_WARN(hipEventDestroy(hip_event));
    C10_HIP_CHECK_WARN(hipSetDevice(orig_device));
  }


  // 销毁指定设备上的 HIP 事件对象
  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override {
    // 如果事件为空，则直接返回
    if (!event) return;
    // 将事件转换为 HIP 事件对象
    auto hip_event = static_cast<hipEvent_t>(event);
    int orig_device;
    // 获取当前设备索引并保存
    C10_HIP_CHECK_WARN(hipGetDevice(&orig_device));
    // 设置设备为指定设备索引
    C10_HIP_CHECK_WARN(hipSetDevice(device_index));
    // 销毁 HIP 事件对象
    C10_HIP_CHECK_WARN(hipEventDestroy(hip_event));
    // 恢复原来的设备索引
    C10_HIP_CHECK_WARN(hipSetDevice(orig_device));
  }



  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
      "Event device index ",
      device_index,
      " does not match recording stream's device index ",
      stream.device_index(),
      ".");

    hipEvent_t hip_event = static_cast<hipEvent_t>(*event);
    HIPStreamMasqueradingAsCUDA hip_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();


  // 记录给定流上的事件
  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const Event
    setDevice(stream.device());

    // 设置当前设备为流的设备
    if (!hip_event) createEvent(&hip_event, flag);
    // 如果 HIP 事件尚未创建，则创建之
    C10_HIP_CHECK(hipEventRecord(hip_event, hip_stream));
    // 记录 HIP 事件的发生在 HIP 流上
    *event = hip_event;
    // 使 void* 指向刚刚分配的 HIP 事件

    // 恢复原始设备
    setDevice(orig_device);
    // 设置当前设备为原始设备
  }

  void block(
    void* event,
    const Stream& stream) const override {
    if (!event) return;
    // 如果事件为空，则直接返回
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    // 将事件指针转换为 HIP 事件类型
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    // 使用流初始化 HIPStreamMasqueradingAsCUDA 类型的对象
    const auto orig_device = getDevice();
    // 获取当前设备
    setDevice(stream.device());
    // 设置当前设备为流的设备
    C10_HIP_CHECK(hipStreamWaitEvent(
      hip_stream,
      hip_event,
      /*flags (must be zero)=*/ 0));
    // 等待 HIP 事件在 HIP 流上完成，标志必须为零

    setDevice(orig_device);
    // 恢复原始设备
  }

  bool queryEvent(void* event) const override {
    if (!event) return true;
    // 如果事件为空，则返回真
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    // 将事件指针转换为 HIP 事件类型
    const hipError_t err = hipEventQuery(hip_event);
    // 查询 HIP 事件状态
    if (err != hipErrorNotReady) C10_HIP_CHECK(err);
    else {
      // 如果事件尚未准备好，则忽略并清除错误
      (void)hipGetLastError();
    }
    return (err == hipSuccess);
    // 返回事件是否成功
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    // 使用流初始化 HIPStreamMasqueradingAsCUDA 类型的对象
    return hip_stream.query();
    // 查询流状态
  }

  void synchronizeStream(const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    // 使用流初始化 HIPStreamMasqueradingAsCUDA 类型的对象
    hip_stream.synchronize();
    // 同步流
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    // 如果事件为空，则直接返回
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    // 将事件指针转换为 HIP 事件类型
    C10_HIP_CHECK(hipEventSynchronize(hip_event));
    // 同步 HIP 事件
  }

  void recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    // 使用流初始化 HIPStreamMasqueradingAsCUDA 类型的对象
    HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA(data_ptr, hip_stream);
    // 记录数据指针在流上
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // 断言两个事件均已记录才能计算经过时间
    int orig_device;
    C10_HIP_CHECK(hipGetDevice(&orig_device));
    // 获取当前设备
    C10_HIP_CHECK(hipSetDevice(device_index));
    // 设置当前设备为指定设备
    hipEvent_t hip_event1 = static_cast<hipEvent_t>(event1);
    // 将事件指针转换为 HIP 事件类型
    hipEvent_t hip_event2 = static_cast<hipEvent_t>(event2);
    // 将事件指针转换为 HIP 事件类型
    float time_ms = 0;
    // 用于存储时间（毫秒）
    C10_HIP_CHECK(hipEventElapsedTime(&time_ms, hip_event1, hip_event2));
    // 计算两个 HIP 事件之间的经过时间
    C10_HIP_CHECK(hipSetDevice(orig_device));
    // 恢复原始设备
    return static_cast<double>(time_ms);
    // 返回经过时间（秒）
  }
};

// 所有使用 HIPGuardImpl 的守卫都需要增加对 HIPGuardImplMasqueradingAsCUDA 的变体。

/// 这段代码完全复制自 c10/cuda/HIPGuardMasqueradingAsCUDA.h，只是使用了正确的 InlineDeviceGuard。
/// 抱歉这里的复制粘贴。

// 定义 HIPGuardMasqueradingAsCUDA 结构体
struct HIPGuardMasqueradingAsCUDA {
  // 默认构造函数被删除
  explicit HIPGuardMasqueradingAsCUDA() = delete;
  // 带设备索引参数的构造函数，使用 guard_ 初始化
  explicit HIPGuardMasqueradingAsCUDA(DeviceIndex device_index) : guard_(device_index) {}
  // 带设备参数的构造函数，使用 guard_ 初始化
  explicit HIPGuardMasqueradingAsCUDA(Device device) : guard_(device) {}

  // 删除拷贝构造函数和赋值运算符重载
  HIPGuardMasqueradingAsCUDA(const HIPGuardMasqueradingAsCUDA&) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(const HIPGuardMasqueradingAsCUDA&) = delete;
  // 删除移动构造函数和移动赋值运算符
  HIPGuardMasqueradingAsCUDA(HIPGuardMasqueradingAsCUDA&& other) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(HIPGuardMasqueradingAsCUDA&& other) = delete;

  // 设置设备的方法
  void set_device(Device device) { guard_.set_device(device); }
  // 重置设备的方法
  void reset_device(Device device) { guard_.reset_device(device); }
  // 设置设备索引的方法
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  // 获取原始设备的方法
  Device original_device() const { return guard_.original_device(); }
  // 获取当前设备的方法
  Device current_device() const { return guard_.current_device(); }

 private:
  // 使用 HIPGuardImplMasqueradingAsCUDA 初始化的 InlineDeviceGuard 对象
  c10::impl::InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

// 定义 OptionalHIPGuardMasqueradingAsCUDA 结构体
struct OptionalHIPGuardMasqueradingAsCUDA {
  // 默认构造函数，使用 guard_ 初始化
  explicit OptionalHIPGuardMasqueradingAsCUDA() : guard_() {}
  // 带可选设备参数的构造函数，使用 guard_ 初始化
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<Device> device_opt) : guard_(device_opt) {}
  // 带可选设备索引参数的构造函数，使用 guard_ 初始化
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  // 删除拷贝构造函数和赋值运算符重载
  OptionalHIPGuardMasqueradingAsCUDA(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  // 删除移动构造函数和移动赋值运算符
  OptionalHIPGuardMasqueradingAsCUDA(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;

  // 设置设备的方法
  void set_device(Device device) { guard_.set_device(device); }
  // 重置设备的方法
  void reset_device(Device device) { guard_.reset_device(device); }
  // 设置设备索引的方法
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  // 获取原始设备的方法
  optional<Device> original_device() const { return guard_.original_device(); }
  // 获取当前设备的方法
  optional<Device> current_device() const { return guard_.current_device(); }
  // 重置守卫对象的方法
  void reset() { guard_.reset(); }

private:
  // 使用 HIPGuardImplMasqueradingAsCUDA 初始化的 InlineOptionalDeviceGuard 对象
  c10::impl::InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};
struct HIPStreamGuardMasqueradingAsCUDA {
  // 禁用默认构造函数
  explicit HIPStreamGuardMasqueradingAsCUDA() = delete;
  // 使用给定的流对象初始化 guard_
  explicit HIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  // 禁用拷贝构造函数
  HIPStreamGuardMasqueradingAsCUDA(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  // 禁用拷贝赋值运算符
  HIPStreamGuardMasqueradingAsCUDA& operator=(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  // 禁用移动构造函数
  HIPStreamGuardMasqueradingAsCUDA(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  // 禁用移动赋值运算符
  HIPStreamGuardMasqueradingAsCUDA& operator=(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  // 重设 guard_ 的流对象为给定的流对象
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  // 返回原始流对象的 HIPStreamMasqueradingAsCUDA 实例
  HIPStreamMasqueradingAsCUDA original_stream() const {
    return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.original_stream());
  }
  // 返回当前流对象的 HIPStreamMasqueradingAsCUDA 实例
  HIPStreamMasqueradingAsCUDA current_stream() const {
    return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.current_stream());
  }

  // 返回当前设备对象
  Device current_device() const { return guard_.current_device(); }
  // 返回原始设备对象
  Device original_device() const { return guard_.original_device(); }

private:
  // 内部使用的 HIPGuardImplMasqueradingAsCUDA 类型的 guard_
  c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct OptionalHIPStreamGuardMasqueradingAsCUDA {
  // 默认构造函数，初始化 guard_ 为默认值
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA() : guard_() {}
  // 使用给定的流对象初始化 guard_
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  // 使用可选的流对象初始化 guard_
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(optional<Stream> stream_opt) : guard_(stream_opt) {}

  // 禁用拷贝构造函数
  OptionalHIPStreamGuardMasqueradingAsCUDA(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  // 禁用拷贝赋值运算符
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  // 禁用移动构造函数
  OptionalHIPStreamGuardMasqueradingAsCUDA(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  // 禁用移动赋值运算符
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  // 重设 guard_ 的流对象为给定的流对象
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  // 返回原始流对象的可选 HIPStreamMasqueradingAsCUDA 实例
  optional<HIPStreamMasqueradingAsCUDA> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  // 返回当前流对象的可选 HIPStreamMasqueradingAsCUDA 实例
  optional<HIPStreamMasqueradingAsCUDA> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  // 重设 guard_ 的状态为默认状态
  void reset() { guard_.reset(); }

private:
  // 内部使用的 InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA> 类型的 guard_
  c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct HIPMultiStreamGuardMasqueradingAsCUDA {
  // 使用给定的 HIPStreamMasqueradingAsCUDA 数组初始化 guard_
  explicit HIPMultiStreamGuardMasqueradingAsCUDA(ArrayRef<HIPStreamMasqueradingAsCUDA> streams)
    // 使用 guard_ 初始化，其中 unwrapStreams(streams) 函数用于解封装 streams 对象，并返回 guard_ 对象
    : guard_(unwrapStreams(streams)) {}
    
    // 禁用拷贝构造函数，使其不能通过拷贝构造函数复制 HIPMultiStreamGuardMasqueradingAsCUDA 对象
    HIPMultiStreamGuardMasqueradingAsCUDA(const HIPMultiStreamGuardMasqueradingAsCUDA&) = delete;
    
    // 禁用赋值运算符重载，防止通过赋值操作符复制 HIPMultiStreamGuardMasqueradingAsCUDA 对象
    HIPMultiStreamGuardMasqueradingAsCUDA& operator=(const HIPMultiStreamGuardMasqueradingAsCUDA&) = delete;
    
    // 禁用移动构造函数，确保不能通过移动构造函数移动 HIPMultiStreamGuardMasqueradingAsCUDA 对象
    HIPMultiStreamGuardMasqueradingAsCUDA(HIPMultiStreamGuardMasqueradingAsCUDA&& other) = delete;
    
    // 禁用移动赋值运算符，防止通过移动赋值运算符移动 HIPMultiStreamGuardMasqueradingAsCUDA 对象
    HIPMultiStreamGuardMasqueradingAsCUDA& operator=(HIPMultiStreamGuardMasqueradingAsCUDA&& other) = delete;
// 声明一个私有成员变量 `guard_`，类型是 `c10::impl::InlineMultiStreamGuard<HIPGuardImplMasqueradingAsCUDA>`
private:
  c10::impl::InlineMultiStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;

// 定义一个静态函数 `unwrapStreams`，用于将 `hipStreams` 中的 `HIPStreamMasqueradingAsCUDA` 对象解包成 `Stream` 对象并返回一个 `std::vector<Stream>` 结果
static std::vector<Stream> unwrapStreams(ArrayRef<HIPStreamMasqueradingAsCUDA> hipStreams) {
  // 创建一个空的 `std::vector<Stream>` 对象 `streams`，预留足够的空间以容纳 `hipStreams` 中的所有元素
  std::vector<Stream> streams;
  streams.reserve(hipStreams.size());

  // 遍历 `hipStreams` 中的每个 `HIPStreamMasqueradingAsCUDA` 对象 `hipStream`
  for (const HIPStreamMasqueradingAsCUDA& hipStream : hipStreams) {
    // 将每个 `hipStream` 添加到 `streams` 后面，转换为 `Stream` 对象
    streams.push_back(hipStream);
  }

  // 返回转换后的 `std::vector<Stream>` 对象 `streams`
  return streams;
}
```