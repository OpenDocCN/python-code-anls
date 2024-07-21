# `.\pytorch\aten\src\ATen\native\cpu\int_mm_kernel.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 核心 Tensor 头文件

#include <ATen/native/DispatchStub.h>
// 包含 ATen 本地调度存根头文件

namespace at::native {
// 进入 ATen 的 native 命名空间

using weight_to_int4pack_fn = void(*)(const Tensor&, const Tensor&, int, int);
// 定义函数指针类型 weight_to_int4pack_fn，用于表示 weight_to_int4pack_stub 函数签名

using int4pack_mm_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, int, const Tensor&, int, int);
// 定义函数指针类型 int4pack_mm_fn，用于表示 int4pack_mm_stub 函数签名

using int8pack_mm_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
// 定义函数指针类型 int8pack_mm_fn，用于表示 int8pack_mm_stub 函数签名

DECLARE_DISPATCH(weight_to_int4pack_fn, weight_to_int4pack_stub);
// 声明 weight_to_int4pack_stub 函数的分发声明，指向 weight_to_int4pack_fn 类型的函数

DECLARE_DISPATCH(int4pack_mm_fn, int4pack_mm_stub);
// 声明 int4pack_mm_stub 函数的分发声明，指向 int4pack_mm_fn 类型的函数

DECLARE_DISPATCH(int8pack_mm_fn, int8pack_mm_stub);
// 声明 int8pack_mm_stub 函数的分发声明，指向 int8pack_mm_fn 类型的函数

} // namespace at::native
// 结束 ATen 的 native 命名空间
```