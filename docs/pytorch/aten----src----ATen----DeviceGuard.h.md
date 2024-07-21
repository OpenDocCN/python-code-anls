# `.\pytorch\aten\src\ATen\DeviceGuard.h`

```
#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h> // TensorList whyyyyy

namespace at {

// Are you here because you're wondering why DeviceGuard(tensor) no
// longer works?  For code organization reasons, we have temporarily(?)
// removed this constructor from DeviceGuard.  The new way to
// spell it is:
//
//    OptionalDeviceGuard guard(device_of(tensor));

/// 返回给定张量的设备，如果张量已定义。
inline std::optional<Device> device_of(const Tensor& t) {
  // 检查张量是否已定义
  if (t.defined()) {
    // 如果已定义，返回其设备作为可选值
    return c10::make_optional(t.device());
  } else {
    // 如果未定义，返回空的可选值
    return c10::nullopt;
  }
}

/// 返回给定可选张量的设备，如果张量存在且已定义。
inline std::optional<Device> device_of(const std::optional<Tensor>& t) {
  // 检查可选张量是否有值
  return t.has_value() ? device_of(t.value()) : c10::nullopt;
}

/// 返回张量列表的设备，如果列表非空且第一个张量已定义。
/// （此函数隐式假设列表中的所有张量具有相同的设备。）
inline std::optional<Device> device_of(ITensorListRef t) {
  // 检查张量列表是否非空
  if (!t.empty()) {
    // 如果非空，返回第一个张量的设备
    return device_of(t.front());
  } else {
    // 如果空，返回空的可选值
    return c10::nullopt;
  }
}

} // namespace at


这段代码是 C++ 中的头文件，定义了一些函数用于获取张量或张量列表的设备信息，并使用了 `std::optional` 类型来处理可能不存在的情况。
```