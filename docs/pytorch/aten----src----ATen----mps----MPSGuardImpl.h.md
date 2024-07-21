# `.\pytorch\aten\src\ATen\mps\MPSGuardImpl.h`

```
//  Copyright © 2022 Apple Inc.

#pragma once
#include <c10/core/impl/DeviceGuardImplInterface.h>  // 引入设备保护接口
#include <c10/macros/Macros.h>  // 引入宏定义
#include <c10/util/Exception.h>  // 引入异常处理工具
#include <ATen/Context.h>  // 引入 ATen 上下文
#include <ATen/mps/MPSStream.h>  // 引入 MPS 流
#include <ATen/mps/MPSEvent.h>  // 引入 MPS 事件

#ifdef __OBJC__
#include <Foundation/Foundation.h>  // 引入 Foundation 框架
#include <Metal/Metal.h>  // 引入 Metal 框架
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>  // 引入 MetalPerformanceShaders 框架
#endif

#include <ATen/Tensor.h>  // 引入 ATen 张量
#include <c10/core/MemoryFormat.h>  // 引入内存格式
#include <c10/core/Storage.h>  // 引入存储
#include <c10/core/TensorImpl.h>  // 引入张量实现
#include <sys/_types/_size_t.h>  // 引入 size_t 类型
#include <memory>  // 引入内存管理工具
#include <c10/core/UndefinedTensorImpl.h>  // 引入未定义张量实现
#include <c10/util/intrusive_ptr.h>  // 引入侵入式指针工具

// 进入 ATen 的 mps 命名空间
namespace at::mps {

typedef MPSEvent* mpsEvent_t;  // 定义 mpsEvent_t 类型为 MPSEvent 指针

// TODO: Move the MPSGuardImpl to inherit from NoOpDeviceGuardImpl
// https://github.com/pytorch/pytorch/issues/77170
// MPSGuardImpl 类继承自 DeviceGuardImplInterface 接口，用于 MPS 设备的管理
struct TORCH_API MPSGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::MPS;  // 静态常量设备类型为 MPS

  // 构造函数
  MPSGuardImpl() {}  // 默认构造函数
  explicit MPSGuardImpl(c10::DeviceType t) {  // 显式构造函数，验证设备类型为 MPS
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::MPS);
  }

  // 返回设备类型为 MPS
  c10::DeviceType type() const override {
    return c10::DeviceType::MPS;
  }

  // 交换设备为 MPS 设备
  Device exchangeDevice(Device d) const override {
    return Device(c10::DeviceType::MPS, 0);
  }

  // 获取当前设备为 MPS 设备
  Device getDevice() const override {
    return Device(c10::DeviceType::MPS, 0);
  }

  // 不安全地获取设备为 MPS 设备
  std::optional<Device> uncheckedGetDevice() const noexcept {
    return Device(c10::DeviceType::MPS, 0);
  }

  // 设置设备为 MPS 设备
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_mps());
  }

  // 不安全地设置设备为 MPS 设备，目前仅支持设备 0
  void uncheckedSetDevice(Device d) const noexcept override {
    // TODO: Currently setting only device 0
  }

  // 获取流对象，设备为 MPS 设备
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  // 获取默认流对象，设备为 MPS 设备
  Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  // 交换流对象，设备为 MPS 设备，不设置当前设备
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  // 获取设备数量，目前仅支持单设备，未来可扩展至多设备
  DeviceIndex deviceCount() const noexcept override {
    if (at::hasMPS()) {
      //TODO: extend it for multi-device case
      return 1;
    } else {
      return 0;
    }
  }

  // 事件相关函数声明
  void createEvent(
    mpsEvent_t* event,
    const EventFlag flag) const;  // 创建事件函数声明

  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override;  // 销毁事件函数声明

  void record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override;  // 记录事件函数声明

  void block(
    void* event,
    const Stream& stream) const override;  // 阻塞事件函数声明

  bool queryEvent(void* event) const override;  // 查询事件函数声明

};

/// A variant of OptionalDeviceGuard that is specialized for MPS.
/// MPS 特定的 OptionalDeviceGuard 变体
// 定义一个名为 OptionalMPSGuard 的结构体

explicit OptionalMPSGuard() : guard_() {}
// 无参数构造函数，初始化 guard_

explicit OptionalMPSGuard(std::optional<Device> device_opt)
    : guard_(device_opt) {}
// 接受 optional<Device> 类型参数的构造函数，使用 device_opt 初始化 guard_

/// Set the current MPS device to the passed device index, if it is not
/// nullopt
explicit OptionalMPSGuard(std::optional<DeviceIndex> device_index_opt)
    : guard_(device_index_opt) {}
// 接受 optional<DeviceIndex> 类型参数的构造函数，使用 device_index_opt 初始化 guard_

// Copy is not allowed
OptionalMPSGuard(const OptionalMPSGuard&) = delete;
OptionalMPSGuard& operator=(const OptionalMPSGuard&) = delete;
OptionalMPSGuard(OptionalMPSGuard&& other) = delete;
OptionalMPSGuard& operator=(OptionalMPSGuard&& other) = delete;
// 禁用拷贝构造函数和移动构造函数

/// Sets the MPS device to the given device, initializing the guard if it
/// is not already initialized.  Errors if the given device is not a MPS
/// device.
void set_device(Device device) {
  guard_.set_device(device);
}
// 设置 MPS 设备为给定的 device，如果 guard 尚未初始化则进行初始化。如果给定的 device 不是 MPS 设备则报错。

/// Sets the MPS device to the given device, initializing the guard if it is
/// not already initialized.  Errors if the given device is not a MPS device.
void reset_device(Device device) {
  guard_.reset_device(device);
}
// 将 MPS 设备重置为给定的 device，如果 guard 尚未初始化则进行初始化。如果给定的 device 不是 MPS 设备则报错。

/// Sets the MPS device to the given device index, initializing the guard if
/// it is not already initialized.
void set_index(DeviceIndex device_index) {
  guard_.set_index(device_index);
}
// 将 MPS 设备设置为给定的 device index，如果 guard 尚未初始化则进行初始化。

/// Returns the device that was set immediately prior to initialization of the
/// guard, or nullopt if the guard is uninitialized.
std::optional<Device> original_device() const {
  return guard_.original_device();
}
// 返回在初始化 guard 之前设置的设备，如果 guard 尚未初始化则返回 nullopt。

/// Returns the most recent device that was set using this device guard,
/// either from construction, or via set_device, if the guard is initialized,
/// or nullopt if the guard is uninitialized.
std::optional<Device> current_device() const {
  return guard_.current_device();
}
// 返回最近使用该设备 guard 设置的设备，可以是构造函数中设置的，也可以是通过 set_device 设置的。如果 guard 尚未初始化则返回 nullopt。

/// Restore the original MPS device, resetting this guard to uninitialized
/// state.
void reset() {
  guard_.reset();
}
// 恢复原始的 MPS 设备设置，将该 guard 重置为未初始化状态。

private:
c10::impl::InlineOptionalDeviceGuard<MPSGuardImpl> guard_;
// 私有成员变量，使用 c10::impl::InlineOptionalDeviceGuard<MPSGuardImpl> 类型的 guard_
```