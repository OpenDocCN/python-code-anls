# `.\pytorch\torch\nn\utils\_expanded_weights\conv_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型声明
from typing import List, Optional

# 导入依赖库
import numpy as np

# 导入 PyTorch 库
import torch
import torch.nn.functional as F

# 导入本地工具模块
from .expanded_weights_utils import (
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)

# 设置阈值常量
THRESHOLD = 32


# 根据函数选择适当的卷积选项
def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt):
    if func == F.conv1d:
        return conv1dOpt
    if func == F.conv2d:
        return conv2dOpt
    else:
        assert func == F.conv3d
        return conv3dOpt


# 解析扩展参数和关键字参数
def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs):
    args = expanded_args_and_kwargs[: len(expanded_args_and_kwargs) - len(kwarg_names)]
    kwargs = expanded_args_and_kwargs[
        len(expanded_args_and_kwargs) - len(kwarg_names) :
    ]
    kwargs = dict(zip(kwarg_names, kwargs))

    # 调用卷积参数规范化函数
    return conv_normalizer(*args, **kwargs)


# 规范化卷积参数
def conv_normalizer(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    return (input, weight), {
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }


# 根据字符串类型填充方式处理卷积输入
def conv_input_for_string_padding(func, padding_style, input, dilation, kernel_size):
    if padding_style == "valid":
        return input
    else:
        # 计算字符串类型填充方式的整数填充
        padding = int_padding_for_string_padding(
            func, padding_style, dilation, kernel_size
        )
        return F.pad(input, padding)


# 计算字符串类型填充方式的整数填充
def int_padding_for_string_padding(func, padding_style, dilation, kernel_size):
    def get_dilation(i):
        return dilation[i] if isinstance(dilation, tuple) else dilation

    if padding_style == "same":
        padding: List[int] = []
        # F.pad 需要与 conv 期望的填充顺序相反的填充
        for i in range(conv_picker(func, 0, 1, 2), -1, -1):
            padding += conv_padding_for_same(get_dilation(i), kernel_size[i])
        return padding
    elif padding_style == "valid":
        return conv_picker(func, 2, 4, 6) * (0,)
    else:
        # 抛出异常，指出填充类型错误
        raise RuntimeError(
            f"got padding type of {padding_style}, only accept 'same' or 'valid'"
        )


# 计算 'same' 填充方式的卷积填充
def conv_padding_for_same(dilation, kernel_size):
    total_pad = dilation * (kernel_size - 1)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return left_pad, right_pad


# 定义卷积反向传播函数
def conv_backward(func, ctx, grad_output):
    def weight_grad_sample(weight):
        # 如果批量大小小于阈值并且分组数为1，则调用卷积组权重梯度采样函数
        if batch_size < THRESHOLD and groups == 1:
            return conv_group_weight_grad_sample(
                ctx.input,
                grad_output,
                weight_shape,
                stride,
                padding,
                dilation,
                batch_size,
                func,
            )
        else:
            # 否则调用展开卷积权重梯度采样函数
            return conv_unfold_weight_grad_sample(
                ctx.input,
                grad_output,
                weight_shape,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                func,
            )

    def expand(param):
        # 如果参数是整数，调用卷积选择器函数，生成一个卷积参数的元组
        if isinstance(param, int):
            return conv_picker(func, (param,), (param, param), (param, param, param))
        else:
            return param

    def calc_total_padding(func, was_same, padding, dilation, kernel_size):
        # 如果是相同填充，计算所有填充并返回总填充
        if was_same:
            all_padding = int_padding_for_string_padding(
                func, "same", dilation, kernel_size
            )
            # F.pad需要与卷积期望的填充相反的顺序
            total_padding = tuple(
                all_padding[i] + all_padding[i - 1]
                for i in range(len(all_padding) - 1, -1, -2)
            )
            return total_padding
        else:
            # 否则计算并返回总填充
            return tuple(2 * pad for pad in padding)

    weight_shape = ctx.weight.shape
    stride, padding, dilation, groups = (
        expand(ctx.stride),
        expand(ctx.padding),
        expand(ctx.dilation),
        ctx.groups,
    )

    kernel_size = []
    # 根据卷积选择器函数和权重形状获取卷积核大小
    for i in range(2, conv_picker(func, 3, 4, 5)):
        kernel_size.append(weight_shape[i])

    batch_size = ctx.batch_size
    results: List[Optional[torch.Tensor]] = []
    results.append(None)  # 用于关键字参数的占位符
    results.append(None)  # 用于操作参考的占位符

    # 如果使用"same"填充，可能会得到不均匀的填充，因此需要分离"padding"属性和总填充
    total_padding = calc_total_padding(
        func, ctx.was_same_padding, padding, dilation, kernel_size
    )
    # 检查上下文对象是否需要输入梯度
    if ctx.input_required_grad:
        # 初始化输出填充列表
        output_padding = []
        # 使用 conv_picker 函数获取输入维度
        input_dims = conv_picker(func, 1, 2, 3)
        # 遍历输入维度
        for i in range(input_dims):
            # 获取原始输入形状中的特定维度
            input_dim = ctx.orig_input_shape[2 + i]
            # 计算输出填充值并添加到列表中
            output_padding.append(
                (
                    total_padding[i]
                    + input_dim
                    - (kernel_size[i] * dilation[i] - dilation[i] + 1)
                )
                % stride[i]
            )
        
        # 解包扩展后的权重或张量
        weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
        # 根据函数类型选择转置卷积函数
        transpose_func = conv_picker(
            func, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d
        )
        # 执行转置卷积操作
        out = transpose_func(
            grad_output,
            weight_,
            None,
            stride,
            padding,
            tuple(output_padding),
            groups,
            dilation,
        )

        # 如果原始填充方式相同，则调整输出
        if ctx.was_same_padding:
            for i in range(len(total_padding)):
                out = torch.narrow(
                    out, 2 + i, total_padding[i] // 2, ctx.orig_input_shape[2 + i]
                )

        # 将输出结果添加到结果列表中
        results.append(out)
    else:
        # 如果不需要输入梯度，则将 None 添加到结果列表中
        results.append(None)
    
    # 由于权重和偏置不计算批量梯度，因此将 None 添加到结果列表中的相应位置
    results = results + [None] * 6

    # 如果存在权重的梯度样本，则设置权重的梯度样本字段
    set_grad_sample_if_exists(ctx.weight, weight_grad_sample)
    # 如果存在偏置的梯度样本，则设置偏置的梯度样本字段
    set_grad_sample_if_exists(
        ctx.bias, lambda _: grad_output.reshape(*grad_output.shape[:2], -1).sum(dim=2)
    )
    # 返回结果元组
    return tuple(results)
# 定义函数，用于计算卷积操作中权重梯度的样本
def conv_unfold_weight_grad_sample(
    input,
    grad_output,
    weight_shape,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    func,
):
    # 获取输入的批次大小
    n = input.shape[0]
    # 获取输入数据的通道数
    in_channels = input.shape[1]

    # 根据不同的函数选择器，选择对输入数据进行展开操作的函数
    unfold_func = conv_picker(
        func,
        lambda: F.unfold(
            input.unsqueeze(-2),
            kernel_size=(1, kernel_size[0]),
            dilation=(1, dilation[0]),
            padding=(0, padding[0]),
            stride=(1, stride[0]),
        ),
        lambda: F.unfold(
            input, kernel_size, dilation=dilation, padding=padding, stride=stride
        ),
        lambda: unfold3d(input, kernel_size, padding, stride, dilation),
    )

    # 调用选择的展开函数处理输入数据
    input = unfold_func()
    # 调整梯度输出的形状以匹配处理后的输入数据
    grad_output = grad_output.reshape(n, -1, input.shape[-1])

    # 计算权重梯度样本
    weight_grad_sample = torch.einsum("noq,npq->nop", grad_output, input)
    # 重新排列张量并提取对角线元素
    weight_grad_sample = weight_grad_sample.view(
        n,
        groups,
        -1,
        groups,
        int(in_channels / groups),
        np.prod(kernel_size),
    )
    # 执行张量运算，保留所需的维度
    weight_grad_sample = torch.einsum(
        "ngrg...->ngr...", weight_grad_sample
    ).contiguous()
    # 调整形状以匹配给定的权重形状
    shape = [n] + list(weight_shape)
    weight_grad_sample = weight_grad_sample.view(shape)
    # 返回计算得到的权重梯度样本
    return weight_grad_sample


# 定义函数，用于计算卷积操作中分组权重梯度的样本
def conv_group_weight_grad_sample(
    input,
    grad_output,
    weight_shape,
    stride,
    padding,
    dilation,
    batch_size,
    func,
):
    # 获取输入数据的通道数和梯度输出的通道数
    I = input.shape[1]
    O = grad_output.shape[1]

    # 调整输入数据的形状
    input_ = input.transpose(0, 1)
    # 调整梯度输出的形状
    grad_output_ = grad_output.view(
        grad_output.shape[0] * grad_output.shape[1], 1, *grad_output.shape[2:]
    )

    # 调用给定函数处理输入数据和梯度输出，计算分组权重梯度样本
    weight_grad_sample = func(
        input_,
        grad_output_,
        None,
        stride=dilation,
        padding=padding,
        dilation=stride,
        groups=batch_size,
    )
    # 根据函数选择器获取输入数据的维度
    input_dims = conv_picker(func, 3, 4, 5)
    # 遍历需要保留的维度，对权重梯度样本进行裁剪
    for i in range(2, input_dims):
        weight_grad_sample = weight_grad_sample.narrow(i, 0, weight_shape[i])
    # 调整形状以匹配给定的权重形状和输入数据的维度
    weight_grad_sample = weight_grad_sample.view(
        I, batch_size, O, *weight_grad_sample.shape[2:]
    )
    # 转置权重梯度样本以调整维度
    weight_grad_sample = weight_grad_sample.movedim(0, 2)
    # 返回计算得到的权重梯度样本
    return weight_grad_sample


# 定义函数，用于从5维输入张量中提取滑动窗口块
def unfold3d(
    tensor,
    kernel_size,
    padding,
    stride,
    dilation,
):
    r"""
    从批量输入张量中提取滑动本地块。

    :class:`torch.nn.Unfold` 仅支持4D输入（类似图像的批量张量）。
    该方法为5D输入实现相同操作。
    Args:
        tensor: 形状为 ``(B, C, D, H, W)`` 的输入张量。
        kernel_size: 滑动块的大小
        padding: 在输入的空间维度两侧添加的隐式零填充
        stride: 输入空间维度的滑动块的步幅
        dilation: 核点之间的间距。
    """
    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )
    # 检查输入张量的维度是否为5，如果不是则抛出值错误异常

    if dilation != (1, 1, 1):
        raise NotImplementedError(f"dilation={dilation} not supported.")
    # 如果膨胀（dilation）参数不是默认的(1, 1, 1)，则抛出未实现错误异常

    batch_size, channels, _, _, _ = tensor.shape
    # 从张量形状中获取批量大小和通道数，其余维度暂时不需要

    tensor = F.pad(
        tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )
    # 使用填充操作对张量进行填充，分别在第2维、第3维和第4维上进行填充
    # 输出形状: (B, C, D+2*padding[2], H+2*padding[1], W+2*padding[0])

    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])
    # 使用unfold方法在第2、3、4维上展开张量，每个维度上分别使用给定的内核大小和步长
    # 输出形状: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # 其中D_out、H_out、W_out的定义请参考torch.nn.Unfold类的说明

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # 对张量的维度进行排列，调整顺序以便后续操作
    # 输出形状: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
        1, 2
    )
    # 对张量进行重新形状化，将其变为二维张量，以便进一步处理
    # 输出形状: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2])

    return tensor
    # 返回处理后的张量作为结果
```