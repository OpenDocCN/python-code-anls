# `Bert-VITS2\transforms.py`

```py
# 导入 torch 库
import torch
# 从 torch.nn 库中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 导入 numpy 库并重命名为 np
import numpy as np

# 定义默认的最小区间宽度
DEFAULT_MIN_BIN_WIDTH = 1e-3
# 定义默认的最小区间高度
DEFAULT_MIN_BIN_HEIGHT = 1e-3
# 定义默认的最小导数值
DEFAULT_MIN_DERIVATIVE = 1e-3

# 定义分段有理二次转换函数
def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # 如果 tails 为 None，则使用 rational_quadratic_spline 函数，否则使用 unconstrained_rational_quadratic_spline 函数
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    # 调用 spline_fn 函数进行转换
    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    # 返回转换后的结果和对数绝对值的行列式
    return outputs, logabsdet

# 定义 searchsorted 函数
def searchsorted(bin_locations, inputs, eps=1e-6):
    # 在 bin_locations 的最后一个维度上加上 eps
    bin_locations[..., -1] += eps
    # 返回 inputs 在 bin_locations 上的搜索结果
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

# 定义 unconstrained_rational_quadratic_spline 函数
def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # 创建一个布尔掩码，标记 inputs 是否在区间内
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    # 创建一个布尔掩码，标记 inputs 是否在区间外
    outside_interval_mask = ~inside_interval_mask

    # 创建与 inputs 相同形状的零张量
    outputs = torch.zeros_like(inputs)
    # 创建与 inputs 相同形状的零张量
    logabsdet = torch.zeros_like(inputs)
    # 如果尾部是线性的
    if tails == "linear":
        # 在unnormalized_derivatives两侧填充1个0
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        # 计算常数值
        constant = np.log(np.exp(1 - min_derivative) - 1)
        # 设置unnormalized_derivatives的第一个和最后一个值为常数值
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        # 将inputs中在区间外的值赋给outputs
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        # 将区间外的logabsdet值设为0
        logabsdet[outside_interval_mask] = 0
    else:
        # 如果尾部不是线性的，抛出运行时错误
        raise RuntimeError("{} tails are not implemented.".format(tails))

    # 调用rational_quadratic_spline函数计算区间内的outputs和logabsdet
    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # 返回计算得到的outputs和logabsdet
    return outputs, logabsdet
def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # 检查输入是否在指定的区间内
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    # 获取未归一化宽度数组的最后一个维度的长度
    num_bins = unnormalized_widths.shape[-1]

    # 检查最小箱子宽度是否过大
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    # 检查最小箱子高度是否过大
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # 对未归一化宽度进行 softmax 归一化
    widths = F.softmax(unnormalized_widths, dim=-1)
    # 根据最小箱子宽度和箱子数量调整宽度
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    # 计算累积宽度
    cumwidths = torch.cumsum(widths, dim=-1)
    # 在累积宽度数组的开头填充一个 0
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    # 将累积宽度映射到指定区间
    cumwidths = (right - left) * cumwidths + left
    # 设置累积宽度数组的第一个和最后一个元素为指定区间的边界
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    # 计算每个箱子的宽度
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # 对未归一化导数进行处理
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # 对未归一化高度进行 softmax 归一化
    heights = F.softmax(unnormalized_heights, dim=-1)
    # 根据最小箱子高度和箱子数量调整高度
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    # 计算累积高度
    cumheights = torch.cumsum(heights, dim=-1)
    # 在累积高度数组的开头填充一个 0
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    # 将累积高度映射到指定区间
    cumheights = (top - bottom) * cumheights + bottom
    # 设置累积高度数组的第一个和最后一个元素为指定区间的边界
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    # 计算每个箱子的高度
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # 如果是反向操作，则根据累积高度进行搜索
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    # 否则，根据累积宽度进行搜索
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    # 获取输入所在箱子的累积宽度和宽度
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    # 获取输入所在箱子的累积高度
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    # 计算高度和宽度的比值
    delta = heights / widths
    # 从 delta 中根据 bin_idx 索引获取对应的值
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    # 从 derivatives 中根据 bin_idx 索引获取对应的值
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    # 从 derivatives 中根据 bin_idx 索引获取除了第一个元素外的所有元素
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    # 从 heights 中根据 bin_idx 索引获取对应的值
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    # 如果需要进行反向操作
    if inverse:
        # 计算 a、b、c 三个值
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        # 计算判别式的值
        discriminant = b.pow(2) - 4 * a * c
        # 断言判别式的值大于等于 0
        assert (discriminant >= 0).all()

        # 计算根的值
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # 计算输出值
        outputs = root * input_bin_widths + input_cumwidths

        # 计算 theta_one_minus_theta、denominator、derivative_numerator、logabsdet 四个值
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        # 返回 outputs 和 -logabsdet 两个值
        return outputs, -logabsdet
    # 如果不是特殊情况，计算输入值相对于输入区间宽度的比例
    theta = (inputs - input_cumwidths) / input_bin_widths
    # 计算 theta * (1 - theta) 的值
    theta_one_minus_theta = theta * (1 - theta)

    # 计算输出的分子部分
    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    # 计算输出的分母部分
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    # 计算最终输出值
    outputs = input_cumheights + numerator / denominator

    # 计算导数的分子部分
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    # 计算对数绝对值的行列式
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    # 返回计算结果
    return outputs, logabsdet
```