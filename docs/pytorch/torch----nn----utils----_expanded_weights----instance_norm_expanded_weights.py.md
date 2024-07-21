# `.\pytorch\torch\nn\utils\_expanded_weights\instance_norm_expanded_weights.py`

```
    # 导入必要的模块和函数，允许未类型化的定义
    # 这里使用了 functools 中的 partial 函数
    # 以及 typing 中的 List 和 Optional 类型
    from functools import partial
    from typing import List, Optional

    # 导入 PyTorch 库
    import torch
    import torch.nn.functional as F

    # 导入自定义模块中的特定函数和类
    from .expanded_weights_impl import implements_per_sample_grads
    from .expanded_weights_utils import (
        forward_helper,
        set_grad_sample_if_exists,
        standard_kwargs,
        unpack_expanded_weight_or_tensor,
    )


    # 为 F.instance_norm 类添加 per-sample 梯度支持的装饰器
    @implements_per_sample_grads(F.instance_norm)
    class InstanceNormPerSampleGrad(torch.autograd.Function):
        # 实现前向传播方法
        @staticmethod
        def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
            # 定义 instance_norm 函数的部分应用，启用 cuDNN 加速
            instance_norm = partial(torch.instance_norm, cudnn_enabled=True)
            # 调用标准化的关键字参数处理函数
            expanded_args, expanded_kwargs = standard_kwargs(
                kwarg_names, expanded_args_and_kwargs
            )
            # 调用辅助函数进行前向传播计算
            output = forward_helper(instance_norm, expanded_args, expanded_kwargs)
            # 保存上下文中的输入数据
            ctx.input = expanded_args[0]
            # 保存运行时均值和方差
            ctx.running_mean, ctx.running_var = (
                expanded_kwargs["running_mean"],
                expanded_kwargs["running_var"],
            )
            # 保存权重、偏置和 epsilon 值
            ctx.weight, ctx.bias, ctx.eps = (
                expanded_kwargs["weight"],
                expanded_kwargs["bias"],
                expanded_kwargs["eps"],
            )
            # 返回前向传播的输出
            return output

        # 实现静态方法
        @staticmethod
    # 定义反向传播函数，接收上下文对象 ctx 和梯度 grad_output
    def backward(ctx, grad_output):
        # 从 ctx 中获取输入 input，运行均值 running_mean，运行方差 running_var
        # 获取权重 weight，偏置 bias，以及 epsilon 值 eps
        input, running_mean, running_var = ctx.input, ctx.running_mean, ctx.running_var
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps

        # 初始化结果列表，用于存储不同的结果项
        results: List[Optional[torch.Tensor]] = []
        results.append(None)  # 用于关键字参数名称
        results.append(None)  # 用于操作的引用

        # 如果输入需要计算梯度
        if input.requires_grad:
            # 获取输入的批量大小 b 和通道数 c
            b = input.shape[0]
            c = input.shape[1]
            # 创建新的形状为 (1, b * c, *input.shape[2:]) 的张量
            new_shape = (1, b * c, *input.shape[2:])

            # 根据输入的权重 weight 展开成扩展权重或张量
            weight_ = unpack_expanded_weight_or_tensor(
                weight, lambda orig_weight: orig_weight.repeat(b)
            )
            # 如果运行均值不为空，则重复 b 次
            running_mean_ = running_mean.repeat(b) if running_mean is not None else None
            # 如果运行方差不为空，则重复 b 次
            running_var_ = running_var.repeat(b) if running_var is not None else None
            # 将输入重塑为连续的张量，并视图为新的形状
            input_reshaped = input.contiguous().view(new_shape)
            # 将梯度输出也重塑为连续的张量，并视图为新的形状
            grad_output_reshaped = grad_output.contiguous().view(new_shape)
            # 计算输入张量在指定维度上的均值，不保持维度
            mean = torch.mean(
                input_reshaped, (0,) + tuple(range(2, input.dim())), False
            )
            # 计算输入张量在指定维度上的方差，不保持维度，使用有偏估计
            var = torch.var(
                input_reshaped,
                (0,) + tuple(range(2, input.dim())),
                keepdim=False,
                unbiased=False,
            )
            # 计算标准差的倒数，加上 epsilon 以防除零错误
            rstd = 1 / torch.sqrt(var + eps)

            # 使用本地批量归一化函数进行反向传播计算，支持所有输入类型
            res = torch.ops.aten.native_batch_norm_backward(
                grad_output_reshaped,
                input_reshaped,
                weight_,
                running_mean_,
                running_var_,
                mean,
                rstd,
                True,  # 训练模式
                eps,
                (True, False, False),  # 参数保存状态
            )
            # 将结果添加到结果列表中，并重塑为原始输入的形状
            results.append(res[0].reshape(input.shape))
        else:
            # 如果输入不需要计算梯度，添加空结果
            results.append(None)

        # 权重和偏置不计算批量梯度；其它参数不可微分（两个参数在前向传播中未保存）
        results = results + [None] * 7

        # 如果存在权重和偏置的梯度样本，设置梯度样本字段
        set_grad_sample_if_exists(
            weight,
            lambda _: torch.einsum(
                "ni...->ni", F.instance_norm(input, eps=eps) * grad_output
            ),
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("ni...->ni", grad_output)
        )
        # 返回结果元组
        return tuple(results)
```