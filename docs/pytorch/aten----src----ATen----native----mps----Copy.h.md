# `.\pytorch\aten\src\ATen\native\mps\Copy.h`

```
//  Copyright © 2022 Apple Inc.
//  上面的声明指示此代码受版权保护，版权归Apple Inc.所有

#pragma once
// 一次性包含防止头文件被多次包含，确保该头文件只被编译一次
#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 类的头文件

namespace at {
namespace native {
namespace mps {

// 声明 mps_copy_ 函数，用于在非阻塞模式下从源张量复制数据到目标张量，并返回目标张量的引用
at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

// 声明 copy_blit_mps 函数，用于在内存之间直接复制数据，无返回值
void copy_blit_mps(void* dst, const void* src, size_t size);

} // namespace mps
} // namespace native
} // namespace at
// 命名空间声明结束
```