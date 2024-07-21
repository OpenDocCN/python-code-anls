# `.\pytorch\aten\src\ATen\xpu\XPUDevice.h`

```py
#pragma once

# 声明代码只被编译一次，避免重复包含头文件


#include <ATen/Context.h>
#include <c10/xpu/XPUFunctions.h>

# 包含 ATen 和 c10 XPU 相关的头文件，用于设备和上下文管理


namespace at::xpu {

# 进入 at::xpu 命名空间


inline Device getDeviceFromPtr(void* ptr) {

# 定义内联函数 getDeviceFromPtr，根据指针获取设备信息


  auto device = c10::xpu::get_device_idx_from_pointer(ptr);

# 调用 c10::xpu::get_device_idx_from_pointer 函数获取指针对应的设备索引


  return {c10::DeviceType::XPU, device};

# 返回一个包含设备类型为 XPU 和设备索引的 Device 对象


}

# 结束函数定义


} // namespace at::xpu

# 结束 at::xpu 命名空间
```