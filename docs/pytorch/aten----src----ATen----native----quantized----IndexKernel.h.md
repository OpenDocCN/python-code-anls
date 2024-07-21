# `.\pytorch\aten\src\ATen\native\quantized\IndexKernel.h`

```
#pragma once
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

// 定义函数指针类型，用于 masked_fill 操作的量化版本
using masked_fill_kernel_quantized_fn = void(*)(TensorIterator& iter, const Scalar& value, double scale, int zero_point);

// 定义函数指针类型，用于 index_put 操作的量化版本
using index_put_kernel_quantized_fn = void(*)(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point);

// 声明 masked_fill_kernel_quantized_stub 函数，用于分发 masked_fill 的量化版本
DECLARE_DISPATCH(masked_fill_kernel_quantized_fn, masked_fill_kernel_quantized_stub);

// 声明 index_put_kernel_quantized_stub 函数，用于分发 index_put 的量化版本
DECLARE_DISPATCH(index_put_kernel_quantized_fn, index_put_kernel_quantized_stub);

} // native
} // at
```