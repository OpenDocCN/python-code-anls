# `.\pytorch\aten\src\ATen\native\group_norm.h`

```py
#pragma once
#include <ATen/native/DispatchStub.h>
#include <cstdint>

namespace at {
class Tensor;

namespace native {

// 前向传播函数指针类型定义，计算 GroupNorm 的前向传播
using forward_fn = void (*)(
    const Tensor& /* X */,         // 输入张量 X
    const Tensor& /* gamma */,     // 缩放参数 gamma
    const Tensor& /* beta */,      // 偏移参数 beta
    int64_t /* N */,               // 批大小 N
    int64_t /* C */,               // 通道数 C
    int64_t /* HxW */,             // 高度乘以宽度 HxW
    int64_t /* group */,           // 组数 group
    double /* eps */,              // 修正值 eps
    Tensor& /* Y */,               // 输出张量 Y
    Tensor& /* mean */,            // 批均值 mean
    Tensor& /* rstd */);           // 归一化标准差 rstd

// 反向传播函数指针类型定义，计算 GroupNorm 的反向传播
using backward_fn = void (*)(
    const Tensor& /* dY */,        // 上游梯度张量 dY
    const Tensor& /* X */,         // 输入张量 X
    const Tensor& /* mean */,      // 批均值 mean
    const Tensor& /* rstd */,      // 归一化标准差 rstd
    const Tensor& /* gamma */,     // 缩放参数 gamma
    int64_t /* N */,               // 批大小 N
    int64_t /* C */,               // 通道数 C
    int64_t /* HxW */,             // 高度乘以宽度 HxW
    int64_t /* group */,           // 组数 group
    Tensor& /* dX */,              // 下游梯度张量 dX
    Tensor& /* dgamma */,          // gamma 的梯度 dgamma
    Tensor& /* dbeta */);          // beta 的梯度 dbeta

// 声明前向传播函数的调度分发器
DECLARE_DISPATCH(forward_fn, GroupNormKernel);
// 声明反向传播函数的调度分发器
DECLARE_DISPATCH(backward_fn, GroupNormBackwardKernel);

} // namespace native
} // namespace at
```