# `.\pytorch\aten\src\ATen\native\utils\Factory.h`

```py
#pragma once
// 使用 #pragma once 指令确保头文件只被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 类定义

namespace at {
namespace native {
namespace mobile {

// 在需要时分配填充的连续内存，以适应特定的内存格式
Tensor allocate_padded_contiguous_if_needed(
    const Tensor& input,
    c10::MemoryFormat memory_format);

// TODO: 在 at::native::empty() 被修改以接受自定义内存分配器后，移除此函数

// 创建一个带尾部填充的空张量
at::Tensor empty_with_tail_padding(
    IntArrayRef size,
    const caffe2::TypeMeta dtype,
    c10::MemoryFormat memory_format,
    std::optional<DimnameList> maybe_names);

} // namespace mobile
} // namespace native
} // namespace at
```