# `.\pytorch\torch\nn\utils\_expanded_weights\group_norm_expanded_weights.py`

```py
    # 引入运算符模块，用于处理操作符相关的功能
    import operator
    # 从 functools 模块中引入 reduce 函数，用于进行可迭代对象的累积运算
    from functools import reduce
    # 从 typing 模块中引入 List 和 Optional 类型，用于类型提示
    from typing import List, Optional

    # 引入 PyTorch 库
    import torch
    # 从 torch.nn.functional 模块中引入 F，用于调用 PyTorch 中的函数
    import torch.nn.functional as F

    # 从当前目录的 expanded_weights_impl 模块中引入 ExpandedWeight 和 implements_per_sample_grads 函数
    from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
    # 从当前目录的 expanded_weights_utils 模块中引入 forward_helper, set_grad_sample_if_exists, standard_kwargs, unpack_expanded_weight_or_tensor 函数
    from .expanded_weights_utils import (
        forward_helper,
        set_grad_sample_if_exists,
        standard_kwargs,
        unpack_expanded_weight_or_tensor,
    )


    # 使用装饰器 implements_per_sample_grads 将 GroupNormPerSampleGrad 类与 F.group_norm 绑定
    @implements_per_sample_grads(F.group_norm)
    # 定义 GroupNormPerSampleGrad 类，继承自 torch.autograd.Function
    class GroupNormPerSampleGrad(torch.autograd.Function):
        # 定义静态方法 forward，用于执行前向传播计算
        @staticmethod
        def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
            # 调用 standard_kwargs 函数，处理参数和关键字参数的扩展
            expanded_args, expanded_kwargs = standard_kwargs(
                kwarg_names, expanded_args_and_kwargs
            )
            # 解包扩展后的参数
            input, num_groups = expanded_args
            # 获取输入张量的维度信息
            N = input.shape[0]
            C = input.shape[1]
            HxW = reduce(operator.mul, input.shape[2:], 1)
            # 解包扩展后的关键字参数
            weight, bias, eps = (
                expanded_kwargs["weight"],
                expanded_kwargs["bias"],
                expanded_kwargs["eps"],
            )
            # 调用 forward_helper 函数执行组规范化的前向传播计算
            output, mean, rstd = forward_helper(
                torch.native_group_norm,
                (input, weight, bias, N, C, HxW, num_groups, eps),
                {},
            )
            # 在上下文对象 ctx 中存储计算所需的变量，以便在反向传播时使用
            ctx.input, ctx.num_groups = input, num_groups
            ctx.weight, ctx.eps = weight, eps
            ctx.mean, ctx.rstd = mean, rstd
            # 如果 bias 是 ExpandedWeight 类型，则也存储在上下文中
            if isinstance(bias, ExpandedWeight):
                ctx.bias = bias
            # 如果输入张量需要梯度计算且 weight 是 ExpandedWeight 类型，则存储在上下文中
            if input.requires_grad and isinstance(weight, ExpandedWeight):
                ctx.weight = weight
            # 返回前向传播计算得到的输出张量
            return output

        @staticmethod
    def backward(ctx, grad_output):
        # 从上下文对象 ctx 中获取需要的变量
        input, num_groups = ctx.input, ctx.num_groups
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps
        mean, rstd = ctx.mean, ctx.rstd

        # 初始化一个结果列表，用于存放计算后的梯度
        results: List[Optional[torch.Tensor]] = []
        results.append(None)  # 用于关键字参数名称
        results.append(None)  # 用于操作的引用

        # 如果输入需要梯度计算
        if input.requires_grad:
            # 解压并确保权重是连续的张量
            weight_c = unpack_expanded_weight_or_tensor(
                weight, lambda t: t.contiguous()
            )
            # 确保输入是连续的张量
            input_c = input.contiguous()
            # 确保梯度输出是连续的张量（如果存在梯度输出）
            grad_output_c = (
                grad_output.contiguous() if grad_output is not None else None
            )
            # 获取输入张量的维度信息
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            # 调用 ATen 库中的 native_group_norm_backward 函数计算梯度
            bw_fn = torch.ops.aten.native_group_norm_backward
            results.append(
                bw_fn(
                    grad_output_c,
                    input_c,
                    mean,
                    rstd,
                    weight_c,
                    N,
                    C,
                    HxW,
                    num_groups,
                    (True, False, False),
                )[0]
            )
        else:
            # 如果输入不需要梯度计算，则添加 None 到结果列表
            results.append(None)

        # 权重和偏置不计算批量梯度；其它参数不可导
        results = results + [None] * 4

        # 如果上下文对象 ctx 中存在权重，则设置权重的样本梯度
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(
                weight,
                lambda _: torch.einsum(
                    "ni...->ni", F.group_norm(input, num_groups, eps=eps) * grad_output
                ),
            )
        # 如果上下文对象 ctx 中存在偏置，则设置偏置的样本梯度
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                bias, lambda _: torch.einsum("ni...->ni", grad_output)
            )
        
        # 返回结果列表作为元组
        return tuple(results)
```