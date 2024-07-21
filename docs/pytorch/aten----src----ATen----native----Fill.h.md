# `.\pytorch\aten\src\ATen\native\Fill.h`

```py
// Functions that fill Tensors with constants. Implementations are in Fill.cpp.
// 填充张量的常量函数。具体实现在 Fill.cpp 中。

#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的 DispatchStub 头文件，用于分发机制的声明

namespace c10 {
class Scalar;
// 声明 c10 命名空间中的 Scalar 类
}

namespace at {
class Tensor;
struct TensorIterator;
// 声明 at 命名空间中的 Tensor 和 TensorIterator 结构体

namespace native {

DECLARE_DISPATCH(void(*)(TensorIterator&, const c10::Scalar&), fill_stub);
// 声明 fill_stub 函数的分发机制，该函数接受 TensorIterator 和 Scalar 作为参数

Tensor& fill_out(Tensor& self, const Scalar& value);
// 声明 fill_out 函数，用于填充张量 self，填充值为 value

}} // namespace at::native
// 结束 at 命名空间中 native 子命名空间的定义
```