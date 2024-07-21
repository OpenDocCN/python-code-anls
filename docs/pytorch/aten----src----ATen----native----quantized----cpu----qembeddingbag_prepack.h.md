# `.\pytorch\aten\src\ATen\native\quantized\cpu\qembeddingbag_prepack.h`

```
#pragma once
#include <ATen/core/Tensor.h>

namespace at { namespace native {

// 定义一个函数签名，声明 qembeddingbag_byte_prepack_out 函数，该函数接受一个输出张量和一个权重张量作为参数，并返回输出张量的引用
Tensor& qembeddingbag_byte_prepack_out(Tensor& output, const Tensor& weight);

// 定义一个函数签名，声明 qembeddingbag_byte_prepack 函数，该函数接受一个权重张量作为参数，并返回一个张量
Tensor qembeddingbag_byte_prepack(const Tensor& weight);

// 定义一个函数签名，声明 qembeddingbag_byte_prepack_meta 函数，该函数接受一个权重张量作为参数，并返回一个张量
Tensor qembeddingbag_byte_prepack_meta(const Tensor& weight);

} // namespace native
} // namespace at
```