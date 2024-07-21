# `.\pytorch\torch\nn\utils\_expanded_weights\embedding_expanded_weights.py`

```py
# mypy: allow-untyped-defs
# 引入类型声明所需的设置，允许未经类型注释的函数定义

from typing import List, Optional
# 引入类型提示的必要模块和类

import torch
import torch.nn.functional as F
# 引入PyTorch相关模块和函数

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
)
# 从本地模块导入特定函数和类


@implements_per_sample_grads(F.embedding)
# 使用装饰器声明实现了每个样本梯度的功能
class EmbeddingPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        # 前向传播函数的静态方法
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        # 使用标准参数函数处理扩展后的参数和关键字参数

        if len(expanded_args[0].shape) == 1:
            # 如果扩展参数的第一个张量是1维的，抛出运行时错误
            raise RuntimeError(
                f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}"
            )

        output = forward_helper(F.embedding, expanded_args, expanded_kwargs)
        # 调用辅助函数执行F.embedding的前向传播计算
        ctx.input, ctx.weight = expanded_args
        # 将输入和权重保存在上下文中
        ctx.padding_idx, ctx.scale_grad_by_freq = (
            expanded_kwargs["padding_idx"],
            expanded_kwargs["scale_grad_by_freq"],
        )
        ctx.sparse = expanded_kwargs["sparse"]
        # 将填充索引、按频率缩放梯度和稀疏标志保存在上下文中
        return output
        # 返回前向传播的输出结果

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播函数的静态方法
        input, weight = ctx.input, ctx.weight
        # 从上下文中获取输入和权重
        padding_idx, scale_grad_by_freq, sparse = (
            ctx.padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
        )
        # 从上下文中获取填充索引、按频率缩放梯度和稀疏标志

        def weight_per_sample_grad(weight):
            # 计算每个样本的权重梯度
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
            # 构建索引张量，用于梯度的散布相加
            grad_sample = torch.zeros(
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            # 创建全零张量用于梯度累积
            return grad_sample.scatter_add_(
                1, index, grad_output.reshape(batch_size, -1, embedding_dim)
            )
            # 使用散布相加操作将扩展后的梯度加到权重的每个样本上

        results: List[Optional[torch.Tensor]] = []
        # 初始化结果列表，每个元素可以是None或者torch.Tensor
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        # 添加两个空项作为关键字参数名称和操作引用的占位符

        if input.requires_grad:
            # 如果输入需要计算梯度
            bw_fn = torch.ops.aten.embedding_backward
            # 使用PyTorch的C++扩展运算符
            results.append(
                bw_fn(
                    grad_output,
                    input,
                    weight.shape[0],
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )
            )
            # 执行反向传播的嵌入操作
        else:
            results.append(None)
            # 如果输入不需要计算梯度，添加空项

        # weight doesn't compute batched gradients; no other arguments are differentiable (2 not saved from forward)
        # 权重不计算批量梯度；其他参数不可微分（前向传播未保存2个参数）

        results = results + [None] * 6
        # 添加6个空项作为未保存的其他参数

        # set grad_sample field for weight with per sample gradients
        # 设置具有每个样本梯度的权重的grad_sample字段
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        # 调用函数设置权重的每个样本梯度

        return tuple(results)
        # 返回结果元组
```