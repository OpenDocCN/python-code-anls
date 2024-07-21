# `.\pytorch\aten\src\ATen\native\mps\operations\FusedAdamWAmsgradKernelImpl.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次，防止重复包含

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类头文件

namespace at::native {
namespace mps {

void _fused_adamw_amsgrad_mps_impl_(
    at::TensorList params, // 参数列表，包含多个张量
    at::TensorList grads, // 梯度列表，对应参数列表中每个张量的梯度
    at::TensorList exp_avgs, // 指数平均值列表，用于 Adam 优化器中的平均梯度
    at::TensorList exp_avg_sqs, // 指数平均平方列表，用于 Adam 优化器中的平方梯度平均值
    at::TensorList max_exp_avg_sqs, // 最大指数平均平方列表，用于 AMSGrad 优化器中的最大平方梯度平均值
    at::TensorList state_steps, // 状态步长列表，用于记录优化器步数的状态
    const double lr, // 学习率
    const double beta1, // Adam 优化器的 beta1 参数
    const double beta2, // Adam 优化器的 beta2 参数
    const double weight_decay, // 权重衰减参数
    const double eps, // 防止除以零的小常数
    const bool maximize, // 是否最大化优化目标
    const c10::optional<at::Tensor>& grad_scale, // 梯度缩放因子的可选张量
    const c10::optional<at::Tensor>& found_inf // 是否发现梯度中的无穷值的可选张量
);
// _fused_adamw_amsgrad_mps_impl_ 函数声明，用于实现融合的 AdamW 和 AMSGrad 优化器
// 该函数接受多个参数和梯度张量，执行优化步骤，更新参数和状态

} // namespace mps
} // namespace at::native
```