# `.\pytorch\aten\src\ATen\native\metal\MetalGuardImpl.cpp`

```py
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at {
namespace detail {

// MetalGuardImpl 结构体实现了 c10::impl::DeviceGuardImplInterface 接口
struct MetalGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  // 默认构造函数
  MetalGuardImpl() = default;

  // 根据设备类型构造 MetalGuardImpl
  explicit MetalGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::Metal);
  }

  // 返回设备类型为 Metal
  DeviceType type() const override {
    return DeviceType::Metal;
  }

  // 交换设备，但在 MetalGuardImpl 中是无操作
  Device exchangeDevice(Device) const override {
    // no-op
    return Device(DeviceType::Metal, -1);
  }

  // 返回当前设备为 Metal
  Device getDevice() const override {
    return Device(DeviceType::Metal, -1);
  }

  // 设置设备，但在 MetalGuardImpl 中是无操作
  void setDevice(Device) const override {
    // no-op
  }

  // 不安全地设置设备，但在 MetalGuardImpl 中是无操作
  void uncheckedSetDevice(Device d) const noexcept override {
    // no-op
  }

  // 返回默认流，但在 MetalGuardImpl 中是无操作
  Stream getStream(Device d) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Metal, -1));
  }

  // 交换流，但在 MetalGuardImpl 中是无操作
  // 注意：这些方法不会设置当前设备
  Stream exchangeStream(Stream s) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Metal, -1));
  }

  // 返回设备数量为 1
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // 以下是与事件相关的函数，但在 Metal 后端中不支持事件
  // 记录事件，但在 MetalGuardImpl 中抛出异常
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(false, "Metal backend doesn't support events.");
  }

  // 阻塞事件，但在 MetalGuardImpl 中抛出异常
  void block(void* event, const Stream& stream) const override {
    TORCH_CHECK(false, "Metal backend doesn't support events.")
  }

  // 查询事件，但在 MetalGuardImpl 中抛出异常
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "Metal backend doesn't support events.")
  }

  // 销毁事件，但在 MetalGuardImpl 中是无操作
  void destroyEvent(void* event, const DeviceIndex device_index) const
      noexcept override {}
};

// 将 MetalGuardImpl 注册为 Metal 设备的设备保护实现
C10_REGISTER_GUARD_IMPL(Metal, MetalGuardImpl);

} // namespace detail
} // namespace at
```