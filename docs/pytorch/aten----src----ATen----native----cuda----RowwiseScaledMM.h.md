# `.\pytorch\aten\src\ATen\native\cuda\RowwiseScaledMM.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/TensorBase.h>
// 包含 ATen 库的张量基类头文件

#include <c10/util/Optional.h>
// 包含 C10 库的可选类型头文件

namespace at::cuda::detail {
// 进入 at::cuda::detail 命名空间

TORCH_API void f8f8bf16_rowwise(
    at::Tensor XQ, // 输入张量 XQ，数据类型为 FP8
    at::Tensor WQ, // 输入张量 WQ，数据类型为 FP8
    at::Tensor x_scale, // 输入张量 x_scale，数据类型为 FP32
    at::Tensor w_scale, // 输入张量 w_scale，数据类型为 FP32
    c10::optional<at::Tensor> bias, // 可选的输入张量 bias，数据类型为 BF16
    bool use_fast_accum, // 布尔型标志，指示是否使用快速累加
    at::Tensor& out); // 输出张量 out，函数修改其内容

}  // 结束 at::cuda::detail 命名空间
```