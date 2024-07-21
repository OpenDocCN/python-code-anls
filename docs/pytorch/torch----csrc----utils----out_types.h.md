# `.\pytorch\torch\csrc\utils\out_types.h`

```py
#pragma once
// 预处理指令：确保本头文件仅被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

namespace torch::utils {
// 命名空间 torch::utils 的开始

TORCH_API void check_out_type_matches(
    const at::Tensor& result,
    std::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool device_is_none);
    // 声明函数 check_out_type_matches，用于检查输出张量的类型匹配情况，
    // 参数包括输出张量 result、可选的标量类型 scalarType 及其是否为 None 的标志 scalarType_is_none、
    // 可选的布局 layout、可选的设备 device 及其是否为 None 的标志 device_is_none

}
// 命名空间 torch::utils 的结束
```