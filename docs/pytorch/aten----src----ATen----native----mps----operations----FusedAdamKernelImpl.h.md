# `.\pytorch\aten\src\ATen\native\mps\operations\FusedAdamKernelImpl.h`

```py
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

namespace at::native {
namespace mps {

void _fused_adam_mps_impl_(
    at::TensorList params,
    // 参数列表，包含模型参数的张量列表
    at::TensorList grads,
    // 梯度列表，包含对应模型参数的梯度张量列表
    at::TensorList exp_avgs,
    // 指数加权平均列表，用于存储梯度的指数加权平均值的张量列表
    at::TensorList exp_avg_sqs,
    // 梯度平方的指数加权平均列表，用于存储梯度平方的指数加权平均值的张量列表
    at::TensorList state_steps,
    // 状态步数列表，用于存储状态步数的张量列表
    const double lr,
    // 学习率
    const double beta1,
    // 一阶矩估计的指数衰减率
    const double beta2,
    // 二阶矩估计的指数衰减率
    const double weight_decay,
    // 权重衰减
    const double eps,
    // 用于数值稳定性的小常数
    const bool maximize,
    // 是否最大化优化目标
    const c10::optional<at::Tensor>& grad_scale,
    // 梯度缩放因子的可选张量
    const c10::optional<at::Tensor>& found_inf
    // 检测到的梯度是否包含无穷大值的可选张量
);

} // namespace mps
} // namespace at::native


这段代码是一个 C++ 头文件，定义了 `_fused_adam_mps_impl_` 函数，用于执行融合 Adam 优化器的实现。
```