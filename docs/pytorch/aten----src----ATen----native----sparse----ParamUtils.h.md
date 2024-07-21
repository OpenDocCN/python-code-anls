# `.\pytorch\aten\src\ATen\native\sparse\ParamUtils.h`

```py
#pragma once
// 使用 pragma once 指令，确保头文件只被包含一次，防止多重包含问题

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 类头文件

#include <ATen/TensorUtils.h>
// 包含 ATen 库的 TensorUtils 头文件

#include <tuple>
// 包含 C++ 标准库的 tuple 头文件，用于支持元组类型

namespace at::native {
// 进入命名空间 at::native

TORCH_API std::tuple<Tensor, Tensor, int64_t> softmax_sparse_input_preprocessing(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float,
    CheckedFrom function_name);
// 声明函数 softmax_sparse_input_preprocessing，接受一个 Tensor 对象作为输入，
// 一个 int64_t 类型的维度参数 dim_，一个 bool 类型的 half_to_float 参数，
// 和一个 CheckedFrom 类型的 function_name 参数，并返回一个包含 Tensor、Tensor 和 int64_t 的元组

TORCH_API std::tuple<Tensor, Tensor, Tensor, int64_t> softmax_backward_sparse_input_preprocessing(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_,
    CheckedFrom function_name);
// 声明函数 softmax_backward_sparse_input_preprocessing，接受四个参数，
// 分别是 grad_ 和 output_，都是 Tensor 类型的参数，
// dim_ 是 int64_t 类型的维度参数，
// input_ 是 Tensor 类型的输入参数，
// function_name 是 CheckedFrom 类型的参数，并返回一个包含 Tensor、Tensor、Tensor 和 int64_t 的元组

} // namespace at::native
// 结束命名空间 at::native
```