# `.\pytorch\aten\src\ATen\native\cpu\MaxUnpoolKernel.h`

```
#pragma once
// 预处理指令：#pragma once 确保头文件只被编译一次，避免重复包含

#include <ATen/native/DispatchStub.h>
// 包含头文件 DispatchStub.h，该文件可能定义了分发函数的相关内容

namespace at {
class Tensor;
// 声明命名空间 at，并声明类 Tensor

namespace native {

using max_unpooling_fn = void(*)(Tensor&, const Tensor&, const Tensor&);
// 定义别名 max_unpooling_fn 为一个指向函数的指针类型，函数接受三个 Tensor 参数并返回 void

DECLARE_DISPATCH(max_unpooling_fn, max_unpool2d_kernel);
// 声明宏 DECLARE_DISPATCH，将 max_unpool2d_kernel 声明为一个分发函数，用于二维最大解池操作

DECLARE_DISPATCH(max_unpooling_fn, max_unpool3d_kernel);
// 声明宏 DECLARE_DISPATCH，将 max_unpool3d_kernel 声明为一个分发函数，用于三维最大解池操作

}} // at::native
// 命名空间 native 结束，命名空间 at 结束
```