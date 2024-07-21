# `.\pytorch\torch\nn\utils\_expanded_weights\linear_expanded_weights.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型注解
from typing import List, Optional

# 导入 PyTorch 库
import torch
import torch.nn.functional as F

# 从本地文件导入自定义函数和工具
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    is_batch_first,
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)

# 使用装饰器将自定义的梯度函数应用到 F.linear 函数上
@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    # 前向传播函数，计算输出并设置上下文信息
    @staticmethod
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        # 如果输入的张量维度不足 2，则抛出运行时错误
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(
                "Input does not have a batch dimension. Expanded Weights expected input "
                f"of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}"
            )
        # 提取扩展参数中的关键字参数（偏置）
        expanded_kwargs = {
            "bias": expanded_args_and_kwargs[2]
            if len(expanded_args_and_kwargs) == 3
            else None
        }
        # 提取扩展参数中的位置参数
        expanded_args = expanded_args_and_kwargs[:2]
        # 检测是否按批次处理
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        # 调用辅助函数计算线性变换的输出
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        # 将位置参数和关键字参数存储在上下文中
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    # 反向传播函数，计算梯度并返回结果
    @staticmethod
    def backward(ctx, grad_output):
        # 提取存储在上下文中的位置参数和关键字参数
        input, weight = ctx.args
        bias = ctx.kwargs["bias"]
        # 初始化结果列表，用于存储梯度信息
        results: List[Optional[torch.Tensor]] = []
        results.append(None)  # for kwarg_names
        results.append(None)  # for op reference

        # 如果输入需要梯度计算，则计算权重的梯度
        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)  # weight and bias don't compute batched gradients

        # 如果不是按批次处理，则调整梯度输出和输入的维度顺序
        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)

        # 如果存在 grad_sample 字段，则直接设置权重和偏置的梯度样本
        set_grad_sample_if_exists(
            weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("n...k->nk", grad_output)
        )
        return tuple(results)
```