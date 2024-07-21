# `.\pytorch\c10\core\impl\DeviceGuardImplInterface.cpp`

```
// 引入头文件：包含设备保护实现接口的定义
#include <c10/core/impl/DeviceGuardImplInterface.h>

// 定义命名空间 c10::impl，用于实现与设备保护相关的功能
namespace c10::impl {

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
// 定义原子指针数组，存储设备保护实现接口的注册信息
std::atomic<const DeviceGuardImplInterface*>
    device_guard_impl_registry[static_cast<size_t>(
        DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

// 设备保护实现注册器的构造函数，将特定设备类型的实现注册到全局注册表中
DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
    DeviceType type,                              // 设备类型
    const DeviceGuardImplInterface* impl) {       // 设备保护实现接口指针
  // 将设备保护实现接口指针存储到对应设备类型的原子指针数组中
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

} // namespace c10::impl
```