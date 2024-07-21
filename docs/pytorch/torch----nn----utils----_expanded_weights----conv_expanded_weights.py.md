# `.\pytorch\torch\nn\utils\_expanded_weights\conv_expanded_weights.py`

```
# 导入 PyTorch 库和函数
import torch
import torch.nn.functional as F

# 导入自定义的辅助函数和类
from .conv_utils import (
    conv_args_and_kwargs,
    conv_backward,
    conv_input_for_string_padding,
    conv_picker,
)
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper

# 使用装饰器为 F.conv1d, F.conv2d, F.conv3d 实现每个样本梯度
@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    # 前向传播函数，处理输入并返回输出
    @staticmethod
    def forward(ctx, kwarg_names, conv_fn, *expanded_args_and_kwargs):
        # 解析扩展参数和关键字参数
        expanded_args, expanded_kwargs = conv_args_and_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        # 获取原始输入
        orig_input = expanded_args[0]
        # 检查是否采用了相同的填充模式
        was_same_padding = expanded_kwargs["padding"] == "same"

        # 如果填充是字符串类型，则使用 F.pad 进行必要的填充操作
        if isinstance(expanded_kwargs["padding"], str):
            kernel_size = expanded_args[1].shape[2:]
            padding, dilation = expanded_kwargs["padding"], expanded_kwargs["dilation"]
            input = conv_input_for_string_padding(
                conv_fn, padding, expanded_args[0], dilation, kernel_size
            )
            expanded_args = (input, expanded_args[1])
            # 已经完成填充，不再需要额外填充
            expanded_kwargs["padding"] = 0

        # 调用辅助函数进行前向计算
        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args
        batched_dim_size = conv_picker(conv_fn, 3, 4, 5)
        # 检查输入维度是否与预期的批量维度大小匹配
        if input.dim() != batched_dim_size:
            raise RuntimeError(
                f"Expanded Weights only support convolution with batched input, got {conv_fn} with an"
                f"unbatched input of dim {input.dim()}, expected input of dim {batched_dim_size}"
            )

        # 保存上下文中的信息以便反向传播使用
        ctx.conv_fn = conv_fn
        ctx.batch_size = orig_input.shape[0]
        ctx.input_required_grad = orig_input.requires_grad
        ctx.orig_input_shape = orig_input.shape
        ctx.was_same_padding = was_same_padding
        ctx.stride, ctx.padding = expanded_kwargs["stride"], expanded_kwargs["padding"]
        ctx.dilation, ctx.groups = (
            expanded_kwargs["dilation"],
            expanded_kwargs["groups"],
        )

        # 如果权重是 ExpandedWeight 类型，则保存输入
        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs["bias"]

        # 返回前向计算结果
        return output

    # 后向传播函数，根据前向传播的上下文信息计算梯度
    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(ctx.conv_fn, ctx, grad_output)
```