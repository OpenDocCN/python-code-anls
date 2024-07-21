# `.\pytorch\aten\src\ATen\DeviceAccelerator.h`

```
#pragma once

// 引入C10库中的DeviceType定义
#include <c10/core/DeviceType.h>
// 引入C10库中的宏定义
#include <c10/macros/Macros.h>

// 引入ATen库中的MTIAHooksInterface.h文件，该文件定义了MTIA钩子接口
#include <ATen/detail/MTIAHooksInterface.h>
// 引入optional标准库，用于返回可能为空的加速器类型
#include <optional>

// 本文件定义了PyTorch顶层加速器的概念。
// 根据这里的定义，如果一个设备是加速器，则满足以下条件：
// - 与所有其他加速器互斥
// - 通过流/事件系统执行异步计算
// - 提供一组通用API，如AcceleratorHooksInterface所定义

// 截至目前，加速器设备包括（无特定顺序）：
// CUDA, MTIA, PrivateUse1
// 一旦所有必要的API被支持和测试，我们希望添加的包括：
// HIP, MPS, XPU

namespace at {

// 确保只有一个加速器可用（如果可能，在编译时进行检查），并返回它。
// 当checked为true时，返回的optional始终有值。
TORCH_API std::optional<c10::DeviceType> getAccelerator(bool checked = false);

} // namespace at
```