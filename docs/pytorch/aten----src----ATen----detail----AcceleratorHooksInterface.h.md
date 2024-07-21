# `.\pytorch\aten\src\ATen\detail\AcceleratorHooksInterface.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <c10/core/Device.h>
// 包含 c10 库的 Device 头文件

#include <c10/core/Stream.h>
// 包含 c10 库的 Stream 头文件

namespace at {

// 声明命名空间 at

// AcceleratorHooksInterface 是由所有加速器提供的共享接口，
// 允许通用代码使用。该接口基于钩子（hook）设计，因为它对应于所有从 CPU 代码中通用调用的函数。

struct TORCH_API AcceleratorHooksInterface {
  // 加速器钩子接口结构体声明

  // 虚析构函数，仅用于禁止 -Werror=non-virtual-dtor 警告
  virtual ~AcceleratorHooksInterface() = default;

  // 检查设备 device_index 是否完全初始化
  virtual bool hasPrimaryContext(DeviceIndex device_index) const = 0;

  // 获取设备数量
  virtual DeviceIndex deviceCount() const {
    return 0;
  }

  // 设置当前设备，如果不支持则报错
  virtual void setCurrentDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support setCurrentDevice()");
  }

  // 获取当前设备，如果不支持则报错
  virtual DeviceIndex getCurrentDevice() const {
    TORCH_CHECK(false, "Backend doesn't support getCurrentDevice()");
    return -1;
  }

  // 切换设备，如果不支持则报错
  virtual DeviceIndex exchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support exchangeDevice()");
    return -1;
  }

  // 可能切换设备，如果不支持则报错
  virtual DeviceIndex maybeExchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support maybeExchangeDevice()");
    return -1;
  }
};

} // namespace at
```