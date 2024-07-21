# `.\pytorch\torch\csrc\utils\device_lazy_init.h`

```
#pragma once

#include <c10/core/TensorOptions.h>

// device_lazy_init() is always compiled, even for CPU-only builds.

namespace torch::utils {

/**
 * This mechanism of lazy initialization is designed for each device backend.
 * Currently, CUDA and XPU follow this design. This function `device_lazy_init`
 * MUST be called before you attempt to access any Type(CUDA or XPU) object
 * from ATen, in any way. It guarantees that the device runtime status is lazily
 * initialized when the first runtime API is requested.
 *
 * Here are some common ways that a device object may be retrieved:
 *   - You call getNonVariableType or getNonVariableTypeOpt
 *   - You call toBackend() on a Type
 *
 * It's important to do this correctly, because if you forget to add it you'll
 * get an oblique error message seems like "Cannot initialize CUDA without
 * ATen_cuda library" or "Cannot initialize XPU without ATen_xpu library" if you
 * try to use CUDA or XPU functionality from a CPU-only build, which is not good
 * UX.
 */

// 声明函数 device_lazy_init，用于设备的延迟初始化
void device_lazy_init(at::DeviceType device_type);

// 设置是否需要设备初始化的标志
void set_requires_device_init(at::DeviceType device_type, bool value);

// 定义 maybe_initialize_device 函数，用于根据设备类型延迟初始化设备
inline void maybe_initialize_device(at::Device& device) {
  // Add more devices here to enable lazy initialization.
  // 如果设备是 CUDA、XPU 或者是私有使用之一，则调用 device_lazy_init 进行延迟初始化
  if (device.is_cuda() || device.is_xpu() || device.is_privateuseone()) {
    device_lazy_init(device.type());
  }
}

// 定义 maybe_initialize_device 函数，重载版本，处理可选的设备对象
inline void maybe_initialize_device(std::optional<at::Device>& device) {
  // 如果设备对象没有值，则直接返回
  if (!device.has_value()) {
    return;
  }
  // 否则调用 maybe_initialize_device(device.value()) 进行设备的延迟初始化
  maybe_initialize_device(device.value());
}

// 定义 maybe_initialize_device 函数，根据给定的 TensorOptions 初始化设备
inline void maybe_initialize_device(const at::TensorOptions& options) {
  // 获取 TensorOptions 中的设备
  auto device = options.device();
  // 调用 maybe_initialize_device(device) 进行设备的延迟初始化
  maybe_initialize_device(device);
}

// 判断指定设备类型是否已经初始化的函数
bool is_device_initialized(at::DeviceType device_type);

} // namespace torch::utils
```