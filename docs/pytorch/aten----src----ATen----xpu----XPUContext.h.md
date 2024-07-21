# `.\pytorch\aten\src\ATen\xpu\XPUContext.h`

```py
#pragma once


// 如果编译时包含了 XPU，XPU 就可用
#include <ATen/Context.h>
// 引入 XPU 的函数和工具
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

// 定义在 at::xpu 命名空间下
namespace at::xpu {

// 内联函数，检查是否有可用的 XPU 设备
inline bool is_available() {
  // 返回值为真，如果有一个以上的 XPU 设备
  return c10::xpu::device_count() > 0;
}

// 获取当前设备的属性
TORCH_XPU_API DeviceProp* getCurrentDeviceProperties();

// 获取特定设备的属性
TORCH_XPU_API DeviceProp* getDeviceProperties(DeviceIndex device);

// 从设备获取全局索引
TORCH_XPU_API int32_t getGlobalIdxFromDevice(DeviceIndex device);

} // namespace at::xpu
```