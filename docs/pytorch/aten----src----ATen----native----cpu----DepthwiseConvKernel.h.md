# `.\pytorch\aten\src\ATen\native\cpu\DepthwiseConvKernel.h`

```
#pragma once
// 预处理命令，确保本文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub.h 文件，用于分发函数的声明
#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef.h 文件，用于处理数组引用

/*
  Depthwise 3x3 Winograd convolution operator
*/
// 深度可分离 3x3 Winograd 卷积操作符的声明

namespace at {
// 命名空间 at，用于包含 ATen 库中的各种类型和函数

class Tensor;
// 声明 Tensor 类，表示张量类型

namespace native {
// 命名空间 native，用于包含 ATen 库中的原生实现

using convolution_depthwise3x3_winograd_fn =
    Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t);
// convolution_depthwise3x3_winograd_fn 是一个函数指针类型，接受多个参数并返回 Tensor 类型的对象

DECLARE_DISPATCH(convolution_depthwise3x3_winograd_fn, convolution_depthwise3x3_winograd_stub);
// 使用 DECLARE_DISPATCH 宏声明 convolution_depthwise3x3_winograd_stub 函数的分发实现

}  // namespace native
}  // namespace at
// 命名空间结束
```