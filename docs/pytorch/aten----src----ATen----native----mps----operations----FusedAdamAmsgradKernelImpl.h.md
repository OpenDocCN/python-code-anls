# `.\pytorch\aten\src\ATen\native\mps\operations\FusedAdamAmsgradKernelImpl.h`

```
#pragma once
// 在编译时保证头文件只包含一次，防止多次引用同一个头文件

#include <ATen/core/Tensor.h>
// 引入 ATen 库的 Tensor 类定义，用于处理张量操作

namespace at::native {
namespace mps {

void _fused_adam_amsgrad_mps_impl_(
    at::TensorList params,
    // 参数列表，包含模型参数的张量列表
    at::TensorList grads,
    // 梯度列表，包含对应模型参数的梯度张量列表
    at::TensorList exp_avgs,
    // 指数加权平均列表，用于保存梯度的指数加权平均
    at::TensorList exp_avg_sqs,
    // 梯度平方的指数加权平均列表
    at::TensorList max_exp_avg_sqs,
    // 最大梯度平方的指数加权平均列表
    at::TensorList state_steps,
    // 状态步骤列表，用于保存状态更新步骤信息
    const double lr,
    // 学习率
    const double beta1,
    // Adam 方法中的一阶矩估计的指数衰减率
    const double beta2,
    // Adam 方法中的二阶矩估计的指数衰减率
    const double weight_decay,
    // 权重衰减项
    const double eps,
    // 防止除零的小常数
    const bool maximize,
    // 是否最大化优化目标
    const c10::optional<at::Tensor>& grad_scale,
    // 梯度缩放系数的可选张量
    const c10::optional<at::Tensor>& found_inf
    // 是否发现无限值的可选张量
);

} // namespace mps
} // namespace at::native
```