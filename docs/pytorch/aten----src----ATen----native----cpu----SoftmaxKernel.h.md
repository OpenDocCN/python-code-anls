# `.\pytorch\aten\src\ATen\native\cpu\SoftmaxKernel.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub.h 头文件，用于声明调度函数相关内容
#include <cstdint>
// 包含标准整数类型头文件，以便使用 int64_t 类型

namespace at {
// 命名空间 at，包含 ATen 库的相关内容
class Tensor;
// 前置声明 Tensor 类，表示这是一个类的声明

namespace native {
// 命名空间 native，包含 ATen 库的本地实现部分

using forward_fn = void (*)(const Tensor&, const Tensor&);
// 定义函数指针类型 forward_fn，指向接受两个 Tensor 参数且返回 void 的函数
using backward_fn = void (*)(const Tensor &, const Tensor &, const Tensor&);
// 定义函数指针类型 backward_fn，指向接受三个 Tensor 参数且返回 void 的函数

DECLARE_DISPATCH(forward_fn, softmax_lastdim_kernel);
// 声明一个由 forward_fn 类型指针作为参数的宏，用于 softmax_lastdim_kernel 函数
DECLARE_DISPATCH(forward_fn, log_softmax_lastdim_kernel);
// 声明一个由 forward_fn 类型指针作为参数的宏，用于 log_softmax_lastdim_kernel 函数
DECLARE_DISPATCH(backward_fn, softmax_backward_lastdim_kernel);
// 声明一个由 backward_fn 类型指针作为参数的宏，用于 softmax_backward_lastdim_kernel 函数
DECLARE_DISPATCH(backward_fn, log_softmax_backward_lastdim_kernel);
// 声明一个由 backward_fn 类型指针作为参数的宏，用于 log_softmax_backward_lastdim_kernel 函数

using forward_fn_with_dim = void (*)(const Tensor &, const Tensor &, const int64_t);
// 定义函数指针类型 forward_fn_with_dim，指向接受两个 Tensor 参数和一个 int64_t 参数且返回 void 的函数
using backward_fn_with_dim = void (*)(const Tensor&, const Tensor&, const Tensor&, const int64_t);
// 定义函数指针类型 backward_fn_with_dim，指向接受三个 Tensor 参数和一个 int64_t 参数且返回 void 的函数

DECLARE_DISPATCH(forward_fn_with_dim, softmax_kernel);
// 声明一个由 forward_fn_with_dim 类型指针作为参数的宏，用于 softmax_kernel 函数
DECLARE_DISPATCH(forward_fn_with_dim, log_softmax_kernel);
// 声明一个由 forward_fn_with_dim 类型指针作为参数的宏，用于 log_softmax_kernel 函数
DECLARE_DISPATCH(backward_fn_with_dim, softmax_backward_kernel);
// 声明一个由 backward_fn_with_dim 类型指针作为参数的宏，用于 softmax_backward_kernel 函数
DECLARE_DISPATCH(backward_fn_with_dim, log_softmax_backward_kernel);
// 声明一个由 backward_fn_with_dim 类型指针作为参数的宏，用于 log_softmax_backward_kernel 函数

}
}
```