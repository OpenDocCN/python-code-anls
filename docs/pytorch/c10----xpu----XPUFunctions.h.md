# `.\pytorch\c10\xpu\XPUFunctions.h`

```py
#pragma once

#include <c10/core/Device.h>
#include <c10/xpu/XPUDeviceProp.h>
#include <c10/xpu/XPUMacros.h>

// The naming convention used here matches the naming convention of torch.xpu

namespace c10::xpu {

// 返回当前系统中检测到的设备数量，可能会打印警告信息（仅打印一次）。
C10_XPU_API DeviceIndex device_count();

// 如果未检测到任何设备，则抛出错误。
C10_XPU_API DeviceIndex device_count_ensure_non_zero();

// 返回当前活动设备的索引。
C10_XPU_API DeviceIndex current_device();

// 设置当前活动设备。
C10_XPU_API void set_device(DeviceIndex device);

// 交换当前活动设备并返回之前的设备索引。
C10_XPU_API DeviceIndex exchange_device(DeviceIndex device);

// 可能交换设备，并返回之前的设备索引。
C10_XPU_API DeviceIndex maybe_exchange_device(DeviceIndex to_device);

// 根据设备索引返回原始 SYCL 设备对象的引用。
C10_XPU_API sycl::device& get_raw_device(DeviceIndex device);

// 返回当前设备的 SYCL 上下文对象的引用。
C10_XPU_API sycl::context& get_device_context();

// 获取特定设备的设备属性信息。
C10_XPU_API void get_device_properties(
    DeviceProp* device_prop,
    DeviceIndex device);

// 根据指针获取其所在设备的索引。
C10_XPU_API DeviceIndex get_device_idx_from_pointer(void* ptr);

} // namespace c10::xpu
```