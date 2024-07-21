# `.\pytorch\c10\core\impl\VirtualGuardImpl.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/core/impl/DeviceGuardImplInterface.h>
// 包含设备保护实现接口的头文件

namespace c10::impl {

/**
 * An implementation of DeviceGuardImplInterface which delegates
 * to virtual dispatch on the DeviceGuardImpl registry.
 */
// VirtualGuardImpl 类，实现 DeviceGuardImplInterface 接口，通过虚拟调度委托到 DeviceGuardImpl 注册表
class VirtualGuardImpl final : public DeviceGuardImplInterface {
 public:
  VirtualGuardImpl(DeviceType device_type)
      : impl_(getDeviceGuardImpl(device_type)) {}
  // 构造函数，根据设备类型获取相应的设备保护实现

  // This constructor exists purely for testing
  VirtualGuardImpl(const DeviceGuardImplInterface* impl) : impl_(impl) {}
  // 用于测试的构造函数，接受一个设备保护实现接口的指针作为参数

  // Copying and moving is OK!
  VirtualGuardImpl(const VirtualGuardImpl&) = default;
  VirtualGuardImpl& operator=(const VirtualGuardImpl&) = default;
  VirtualGuardImpl(VirtualGuardImpl&&) noexcept = default;
  VirtualGuardImpl& operator=(VirtualGuardImpl&&) noexcept = default;
  // 默认的复制和移动构造函数，使用默认实现

  DeviceType type() const override {
    return impl_->type();
  }
  // 获得设备类型的虚函数重载

  Device exchangeDevice(Device d) const override {
    return impl_->exchangeDevice(d);
  }
  // 交换设备的虚函数重载

  Device getDevice() const override {
    return impl_->getDevice();
  }
  // 获取设备的虚函数重载

  void setDevice(Device d) const override {
    impl_->setDevice(d);
  }
  // 设置设备的虚函数重载

  void uncheckedSetDevice(Device d) const noexcept override {
    impl_->uncheckedSetDevice(d);
  }
  // 不安全设置设备的虚函数重载

  Stream getStream(Device d) const noexcept override {
    return impl_->getStream(d);
  }
  // 获取流的虚函数重载

  Stream getNewStream(Device d, int priority = 0) const override {
    return impl_->getNewStream(d, priority);
  }
  // 获取新流的虚函数重载

  Stream getDefaultStream(Device d) const override {
    return impl_->getDefaultStream(d);
  }
  // 获取默认流的虚函数重载

  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return impl_->getStreamFromGlobalPool(d, isHighPriority);
  }
  // 从全局池中获取流的虚函数重载

  Stream exchangeStream(Stream s) const noexcept override {
    return impl_->exchangeStream(s);
  }
  // 交换流的虚函数重载

  DeviceIndex deviceCount() const noexcept override {
    return impl_->deviceCount();
  }
  // 获取设备计数的虚函数重载

  // Event functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    impl_->record(event, stream, device_index, flag);
  }
  // 记录事件的虚函数重载

  void block(void* event, const Stream& stream) const override {
    impl_->block(event, stream);
  }
  // 阻塞事件的虚函数重载

  bool queryEvent(void* event) const override {
    return impl_->queryEvent(event);
  }
  // 查询事件的虚函数重载

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    impl_->destroyEvent(event, device_index);
  }
  // 销毁事件的虚函数重载

  bool queryStream(const Stream& stream) const override {
    return impl_->queryStream(stream);
  }
  // 查询流的虚函数重载

  void synchronizeStream(const Stream& stream) const override {
    impl_->synchronizeStream(stream);
  }
  // 同步流的虚函数重载

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    impl_->recordDataPtrOnStream(data_ptr, stream);
  }
  // 在流上记录数据指针的虚函数重载

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    # 调用实现接口中的方法，计算两个事件之间的经过时间
    return impl_->elapsedTime(event1, event2, device_index);
  }

  # 重写接口方法，用于同步特定事件
  void synchronizeEvent(void* event) const override {
    # 调用实现接口中的方法，同步指定的事件
    return impl_->synchronizeEvent(event);
  }

 private:
  # 接口实现对象的指针，初始化为空指针
  const DeviceGuardImplInterface* impl_ = nullptr;
};

} // namespace c10::impl
```