# `.\pytorch\aten\src\ATen\native\cpu\PixelShuffleKernel.h`

```py
#pragma once
#include <ATen/native/DispatchStub.h>
// 引入 ATen 库的 DispatchStub 头文件，用于声明和定义调度相关的功能

namespace at {
class TensorBase;
}
// 命名空间 at 下定义 TensorBase 类的前向声明

namespace at { namespace native {
// 命名空间 at::native 开始

using pixel_shuffle_fn = void(*)(TensorBase&, const TensorBase&, int64_t);
// 定义 pixel_shuffle_fn 类型别名，表示指向函数的指针，该函数接受 TensorBase 类型的两个引用参数和一个 int64_t 参数，返回 void
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_shuffle_kernel);
// 声明 pixel_shuffle_kernel 函数的调度分发，函数原型为 pixel_shuffle_fn 类型
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_unshuffle_kernel);
// 声明 pixel_unshuffle_kernel 函数的调度分发，函数原型同样为 pixel_shuffle_fn 类型

}} // at::native
// 命名空间 at::native 结束
```