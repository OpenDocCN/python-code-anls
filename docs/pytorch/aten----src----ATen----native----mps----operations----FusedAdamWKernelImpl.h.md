# `.\pytorch\aten\src\ATen\native\mps\operations\FusedAdamWKernelImpl.h`

```py
#pragma once
#include <ATen/core/Tensor.h>  // 包含 ATen 库中的 Tensor 类定义

namespace at::native {
namespace mps {

void _fused_adamw_mps_impl_(
    at::TensorList params,             // 参数列表，包含多个 Tensor
    at::TensorList grads,              // 梯度列表，包含多个 Tensor
    at::TensorList exp_avgs,           // 指数移动平均值列表，包含多个 Tensor
    at::TensorList exp_avg_sqs,        // 指数移动平方平均值列表，包含多个 Tensor
    at::TensorList state_steps,        // 状态步数列表，包含多个 Tensor
    const double lr,                   // 学习率
    const double beta1,                // 梯度的一阶矩估计的指数衰减率
    const double beta2,                // 梯度的二阶矩估计的指数衰减率
    const double weight_decay,         // 权重衰减（L2 正则化）参数
    const double eps,                  // 用于数值稳定性的小常数
    const bool maximize,               // 是否最大化优化目标（用于优化器的方向）
    const c10::optional<at::Tensor>& grad_scale,  // 梯度缩放的可选张量
    const c10::optional<at::Tensor>& found_inf    // 检测到的无穷梯度的可选张量
);
} //namespace mps
}// namespace at::native
```