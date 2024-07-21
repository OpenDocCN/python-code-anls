# `.\pytorch\torch\nn\utils\_expanded_weights\layer_norm_expanded_weights.py`

```
    # 设置类型检查时允许未类型化的定义
    # 导入所需的类型声明
    from typing import List, Optional

    # 导入 PyTorch 库
    import torch
    import torch.nn.functional as F

    # 从当前包中导入实现了 per_sample_grads 的 ExpandedWeight 类和 implements_per_sample_grads 函数装饰器
    from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
    # 从当前包中导入辅助函数和工具函数
    from .expanded_weights_utils import (
        forward_helper,
        set_grad_sample_if_exists,
        standard_kwargs,
        sum_over_all_but_batch_and_last_n,
        unpack_expanded_weight_or_tensor,
    )


    @implements_per_sample_grads(F.layer_norm)
    # 定义 LayerNormPerSampleGrad 类，继承自 torch.autograd.Function
    class LayerNormPerSampleGrad(torch.autograd.Function):
        
        @staticmethod
        # 实现前向传播方法
        def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
            # 使用 standard_kwargs 函数处理扩展参数和关键字参数
            expanded_args, expanded_kwargs = standard_kwargs(
                kwarg_names, expanded_args_and_kwargs
            )
            # 获取输入张量
            input = expanded_args[0]
            # 获取规范化的形状参数
            normalized_shape = expanded_args[1]
            
            # 如果输入张量的维度小于或等于规范化形状的维度，抛出运行时错误
            if len(input.shape) <= len(normalized_shape):
                raise RuntimeError(
                    "Expanded Weights: Layer norm should not normalize over batch dimension for per sample gradient"
                    f"computations but got that normalized shape, {normalized_shape}, matched input shape."
                )
            
            # 调用 forward_helper 函数执行实际的 layer_norm 操作
            # 返回输出张量、均值和标准差的计算结果
            output, mean, rstd = forward_helper(
                torch.native_layer_norm, expanded_args, expanded_kwargs
            )
            
            # 将扩展参数保存到上下文中
            ctx.args = expanded_args

            # 如果输入张量需要梯度计算或者权重是 ExpandedWeight 类型，保存权重到上下文中
            if input.requires_grad or isinstance(expanded_kwargs["weight"], ExpandedWeight):
                ctx.weight = expanded_kwargs["weight"]
            # 如果输入张量需要梯度计算或者偏置是 ExpandedWeight 类型，保存偏置到上下文中
            if input.requires_grad or isinstance(expanded_kwargs["bias"], ExpandedWeight):
                ctx.bias = expanded_kwargs["bias"]
            # 保存 epsilon 参数、均值和标准差到上下文中
            ctx.eps = expanded_kwargs["eps"]
            ctx.mean, ctx.rstd = mean, rstd
            
            # 返回前向传播的输出张量
            return output

        @staticmethod
    # 定义反向传播函数，用于计算梯度
    def backward(ctx, grad_output):
        # 定义计算权重每个样本梯度的函数
        def weight_per_sample_grad(weight):
            return sum_over_all_but_batch_and_last_n(
                F.layer_norm(input, normalized_shape, eps=ctx.eps) * grad_output,
                weight.dim(),
            )

        # 从上下文中获取输入和规范化形状
        input, normalized_shape = ctx.args
        # 从上下文中获取均值和倒数标准差
        mean, rstd = ctx.mean, ctx.rstd

        # 存储计算结果的列表，初始两个元素为None用于占位
        results: List[Optional[torch.Tensor]] = []
        results.append(None)  # 用于关键字参数名称
        results.append(None)  # 用于操作引用

        # 如果输入需要梯度计算
        if input.requires_grad:
            # 解包扩展的权重或张量
            weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
            bias_ = unpack_expanded_weight_or_tensor(ctx.bias)
            # 调用原生的 Layer Norm 反向传播函数计算梯度
            results.append(
                torch.ops.aten.native_layer_norm_backward(
                    grad_output,
                    input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight_,
                    bias_,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # 权重和偏置不计算批量梯度；其他参数不可微分，用None填充
        results = results + [None] * 4

        # 如果上下文中存在权重，则设置权重样本梯度字段
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(ctx.weight, weight_per_sample_grad)
        # 如果上下文中存在偏置，则设置偏置样本梯度字段
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                ctx.bias,
                lambda bias: sum_over_all_but_batch_and_last_n(grad_output, bias.dim()),
            )
        
        # 返回计算结果的元组
        return tuple(results)
```