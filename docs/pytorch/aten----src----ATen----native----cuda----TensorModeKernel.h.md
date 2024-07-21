# `.\pytorch\aten\src\ATen\native\cuda\TensorModeKernel.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <cstdint>
// 包含标准整数类型头文件，提供固定宽度整数类型的定义

namespace at {
// 命名空间 'at' 开始

class TensorBase;
// 声明名为 TensorBase 的类

}

namespace at {
namespace native {
// 命名空间 'at::native' 开始

void launch_fused_mode_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t slice_size, int64_t slices);
// 函数声明：启动融合模式的内核，接受张量值、索引、自身张量以及切片大小和数量作为参数

void launch_apply_mode_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t ndim);
// 函数声明：启动应用模式的内核，接受张量值、索引、自身张量以及维度和张量维度作为参数

}}  // namespace at::native
// 命名空间 'at::native' 结束
```