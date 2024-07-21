# `.\pytorch\aten\src\ATen\native\vulkan\VulkanGuardImpl.cpp`

```py
namespace at {
namespace detail {

namespace {

// 定义 VulkanGuardImpl 结构，实现 c10::impl::DeviceGuardImplInterface 接口
struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  VulkanGuardImpl() = default;  // 默认构造函数
  
  // NOLINTNEXTLINE
  explicit VulkanGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::Vulkan);  // 断言设备类型必须是 Vulkan
  }

  // 返回设备类型为 Vulkan
  DeviceType type() const override {
    return DeviceType::Vulkan;
  }

  // 不执行任何操作，返回 Vulkan 设备
  Device exchangeDevice(Device) const override {
    // no-op
    return Device(DeviceType::Vulkan, -1);
  }

  // 返回 Vulkan 设备
  Device getDevice() const override {
    return Device(DeviceType::Vulkan, -1);
  }

  // 不执行任何操作，设置设备
  void setDevice(Device) const override {
    // no-op
  }

  // 不执行任何操作，设置设备（无检查版本）
  void uncheckedSetDevice(Device d) const noexcept override {
    (void)d;
    // no-op
  }

  // 不执行任何操作，返回默认流（Vulkan 不支持流）
  Stream getStream(Device d) const noexcept override {
    (void)d;
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, -1));
  }

  // 不执行任何操作，交换流（Vulkan 不支持流）
  Stream exchangeStream(Stream s) const noexcept override {
    (void)s;
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, -1));
  }

  // 返回设备数量为 1
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // 不支持事件记录，抛出错误
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    (void)event;
    (void)stream;
    (void)device_index;
    (void)flag;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.");
  }

  // 不支持事件阻塞，抛出错误
  void block(void* event, const Stream& stream) const override {
    (void)event;
    (void)stream;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }

  // 不支持事件查询，抛出错误
  bool queryEvent(void* event) const override {
    (void)event;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }

  // 不执行任何操作，销毁事件（Vulkan 不支持事件）
  void destroyEvent(void* event, const DeviceIndex device_index) const noexcept override {
    (void)event;
    (void)device_index;
    // no-op
  }
};

} // namespace

// 在 detail 命名空间中注册 VulkanGuardImpl 结构为 Vulkan 设备的守护实现
C10_REGISTER_GUARD_IMPL(Vulkan, VulkanGuardImpl);

} // namespace detail
} // namespace at
```