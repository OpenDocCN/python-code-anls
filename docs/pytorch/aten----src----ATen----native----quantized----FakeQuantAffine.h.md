# `.\pytorch\aten\src\ATen\native\quantized\FakeQuantAffine.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
// 引入必要的头文件

namespace at {

struct TensorIterator;
// 声明 TensorIterator 结构体，用于迭代张量操作

namespace native {

using fake_quant_tensor_cachemask_fn = void (*)(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);
// 定义函数指针类型 fake_quant_tensor_cachemask_fn，表示一个函数，接受多个张量和数值作为参数，并返回 void

using fake_quant_tensor_cachemask_tensor_qparams_fn = void (*)(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& sc,
    const Tensor& z_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max);
// 定义函数指针类型 fake_quant_tensor_cachemask_tensor_qparams_fn，表示一个函数，接受多个张量和数值作为参数，并返回 void

using fake_quant_learnable_grad_tensor_fn = void (*)(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);
// 定义函数指针类型 fake_quant_learnable_grad_tensor_fn，表示一个函数，接受 TensorIterator 和多个数值作为参数，并返回 void

DECLARE_DISPATCH(fake_quant_tensor_cachemask_fn, fake_quant_tensor_cachemask_stub);
DECLARE_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_fn, fake_quant_tensor_cachemask_tensor_qparams_stub);
DECLARE_DISPATCH(fake_quant_learnable_grad_tensor_fn, fake_quant_grad_learnable_tensor_stub);
// 声明分发函数的具体实现，对应于上述的函数指针类型

using fake_quant_per_channel_fn = void (*)(
    TensorIterator &iter,
    int64_t quant_min,
    int64_t quant_max);
// 定义函数指针类型 fake_quant_per_channel_fn，表示一个函数，接受 TensorIterator 和两个数值作为参数，并返回 void

using fake_quant_per_channel_cachemask_fn = void (*)(
    TensorIterator &iter,
    TensorIterator &iter_mask,
    int64_t quant_min,
    int64_t quant_max);
// 定义函数指针类型 fake_quant_per_channel_cachemask_fn，表示一个函数，接受两个 TensorIterator 和两个数值作为参数，并返回 void

DECLARE_DISPATCH(fake_quant_per_channel_cachemask_fn, fake_quant_per_channel_cachemask_stub);
// 声明分发函数的具体实现，对应于 fake_quant_per_channel_cachemask_fn 的函数指针类型

using fake_quant_learnable_per_channel_fn = void (*)(
    TensorIterator &iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);
// 定义函数指针类型 fake_quant_learnable_per_channel_fn，表示一个函数，接受 TensorIterator 和三个数值作为参数，并返回 void

DECLARE_DISPATCH(fake_quant_learnable_per_channel_fn, fake_quant_grad_learnable_channel_stub);
// 声明分发函数的具体实现，对应于 fake_quant_learnable_per_channel_fn 的函数指针类型

} // namespace native
} // namespace at
// 结束 namespace 块
```