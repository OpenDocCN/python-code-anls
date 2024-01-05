# `d:/src/tocomm/Bert-VITS2\transforms.py`

```
import torch  # 导入torch库
from torch.nn import functional as F  # 导入torch.nn库中的functional模块，并将其重命名为F

import numpy as np  # 导入numpy库，并将其重命名为np


DEFAULT_MIN_BIN_WIDTH = 1e-3  # 定义一个名为DEFAULT_MIN_BIN_WIDTH的常量，值为0.001
DEFAULT_MIN_BIN_HEIGHT = 1e-3  # 定义一个名为DEFAULT_MIN_BIN_HEIGHT的常量，值为0.001
DEFAULT_MIN_DERIVATIVE = 1e-3  # 定义一个名为DEFAULT_MIN_DERIVATIVE的常量，值为0.001


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,  # 最小的箱子高度，默认值为DEFAULT_MIN_BIN_HEIGHT
    min_derivative=DEFAULT_MIN_DERIVATIVE,  # 最小的导数，默认值为DEFAULT_MIN_DERIVATIVE
):
    if tails is None:  # 如果tails参数为None
        spline_fn = rational_quadratic_spline  # 使用rational_quadratic_spline函数
        spline_kwargs = {}  # spline_kwargs为空字典
    else:  # 如果tails参数不为None
        spline_fn = unconstrained_rational_quadratic_spline  # 使用unconstrained_rational_quadratic_spline函数
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}  # spline_kwargs为包含tails和tail_bound键值对的字典

    outputs, logabsdet = spline_fn(  # 调用spline_fn函数，返回outputs和logabsdet
        inputs=inputs,  # 输入参数为inputs
        unnormalized_widths=unnormalized_widths,  # 输入参数为unnormalized_widths
        unnormalized_heights=unnormalized_heights,  # 输入参数为unnormalized_heights
        unnormalized_derivatives=unnormalized_derivatives,  # 输入参数为unnormalized_derivatives
        inverse=inverse,  # 输入参数为inverse
        min_bin_width=min_bin_width,  # 输入参数为min_bin_width
        min_bin_height=min_bin_height,  # 输入参数为min_bin_height
        min_derivative=min_derivative,  # 输入参数为min_derivative
        **spline_kwargs  # 使用spline_kwargs中的键值对作为额外的参数
    )
    return outputs, logabsdet
```
这段代码是一个函数的结尾，返回两个变量outputs和logabsdet。

```
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
```
这段代码定义了一个名为searchsorted的函数，它接受三个参数：bin_locations、inputs和eps（默认值为1e-6）。函数的作用是在bin_locations中搜索inputs，并返回每个input在bin_locations中的位置。

```
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
```
这段代码定义了一个名为unconstrained_rational_quadratic_spline的函数，它接受多个参数：inputs、unnormalized_widths、unnormalized_heights、unnormalized_derivatives、inverse、tails、tail_bound、min_bin_width、min_bin_height和min_derivative。函数的作用是根据这些参数计算出一个无约束的有理二次样条。
# 创建一个布尔掩码，用于判断输入是否在指定区间内
inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
# 创建一个布尔掩码，用于判断输入是否在指定区间外
outside_interval_mask = ~inside_interval_mask

# 创建一个与输入相同形状的全零张量
outputs = torch.zeros_like(inputs)
# 创建一个与输入相同形状的全零张量
logabsdet = torch.zeros_like(inputs)

# 如果尾部处理方式为"linear"
if tails == "linear":
    # 在未归一化的导数张量两侧填充一个元素
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    # 计算常数值
    constant = np.log(np.exp(1 - min_derivative) - 1)
    # 将未归一化的导数张量的第一个元素和最后一个元素设置为常数值
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # 将输入中在指定区间外的元素赋值给输出
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    # 将输入中在指定区间外的元素的对数绝对值行列式设置为0
    logabsdet[outside_interval_mask] = 0
# 如果尾部处理方式不是"linear"，抛出运行时错误
else:
    raise RuntimeError("{} tails are not implemented.".format(tails))

# 将输入中在指定区间内的元素赋值给输出
outputs[inside_interval_mask],
# 调用 rational_quadratic_spline 函数，传入参数并接收返回值
# 将返回值的第一个元素赋值给 outputs，第二个元素赋值给 logabsdet
outputs, logabsdet = rational_quadratic_spline(
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

# 返回 outputs 和 logabsdet
return outputs, logabsdet


def rational_quadratic_spline(
```

这段代码调用了名为 `rational_quadratic_spline` 的函数，并传入了一系列参数。函数返回两个值，分别赋值给 `outputs` 和 `logabsdet`。最后将 `outputs` 和 `logabsdet` 作为结果返回。
    inputs,  # 输入数据
    unnormalized_widths,  # 未归一化的宽度数组
    unnormalized_heights,  # 未归一化的高度数组
    unnormalized_derivatives,  # 未归一化的导数数组
    inverse=False,  # 是否为逆变换
    left=0.0,  # 输入数据的最小值
    right=1.0,  # 输入数据的最大值
    bottom=0.0,  # 输出数据的最小值
    top=1.0,  # 输出数据的最大值
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,  # 最小的直方图条宽度
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,  # 最小的直方图条高度
    min_derivative=DEFAULT_MIN_DERIVATIVE,  # 最小的导数值
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")  # 检查输入数据是否在指定的范围内

    num_bins = unnormalized_widths.shape[-1]  # 获取未归一化宽度数组的最后一个维度的大小，即直方图条的数量

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")  # 检查最小的直方图条宽度是否过大，超过了直方图条的数量所能容纳的范围
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")
```
如果最小柱高乘以柱的数量大于1.0，则抛出一个值错误异常，提示最小柱高对于柱的数量来说太大。

```
    widths = F.softmax(unnormalized_widths, dim=-1)
```
使用softmax函数对未归一化的宽度进行归一化。

```
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
```
计算每个柱的宽度，根据最小柱宽和柱的数量来调整宽度。

```
    cumwidths = torch.cumsum(widths, dim=-1)
```
计算每个柱的累积宽度。

```
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
```
在累积宽度的前面填充一个0，用于计算柱的左边界。

```
    cumwidths = (right - left) * cumwidths + left
```
根据给定的左右边界，对累积宽度进行线性变换，得到每个柱的实际宽度。

```
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
```
将第一个柱的左边界设置为给定的左边界，将最后一个柱的右边界设置为给定的右边界。

```
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
```
计算每个柱的宽度，通过相邻柱的边界差值得到。

```
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
```
计算每个柱的导数，通过对未归一化的导数进行softplus函数处理。

```
    heights = F.softmax(unnormalized_heights, dim=-1)
```
使用softmax函数对未归一化的高度进行归一化。

```
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
```
计算每个柱的高度，根据最小柱高和柱的数量来调整高度。

```
    cumheights = torch.cumsum(heights, dim=-1)
```
计算每个柱的累积高度。

```
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
```
在累积高度的前面填充一个0，用于计算柱的底边界。

```
    cumheights = (top - bottom) * cumheights + bottom
```
根据给定的上下边界，对累积高度进行线性变换，得到每个柱的实际高度。

```
    cumheights[..., 0] = bottom
```
将第一个柱的底边界设置为给定的下边界。

注：代码中的`left`、`right`、`bottom`和`top`是未给出的变量，需要根据上下文来确定它们的含义。
cumheights[..., -1] = top
```
将`top`的值赋给`cumheights`数组的最后一个元素。

```
heights = cumheights[..., 1:] - cumheights[..., :-1]
```
计算`cumheights`数组中每个元素与其前一个元素之间的差值，得到`heights`数组。

```
if inverse:
    bin_idx = searchsorted(cumheights, inputs)[..., None]
else:
    bin_idx = searchsorted(cumwidths, inputs)[..., None]
```
根据`inverse`的值选择不同的数组进行搜索，找到`inputs`在数组中的插入位置，并将结果存储在`bin_idx`中。

```
input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
```
根据`bin_idx`从`cumwidths`和`widths`数组中提取对应位置的值，并分别存储在`input_cumwidths`和`input_bin_widths`中。

```
input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
delta = heights / widths
input_delta = delta.gather(-1, bin_idx)[..., 0]
```
根据`bin_idx`从`cumheights`、`heights`和`widths`数组中提取对应位置的值，并分别存储在`input_cumheights`和`input_delta`中。同时，计算`heights`和`widths`的比值，得到`delta`数组。

```
input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
```
根据`bin_idx`从`derivatives`数组中提取对应位置的值，并分别存储在`input_derivatives`和`input_derivatives_plus_one`中。同时，从`derivatives`数组中提取除第一个元素外的所有元素，并根据`bin_idx`提取对应位置的值，存储在`input_derivatives_plus_one`中。

```
input_heights = heights.gather(-1, bin_idx)[..., 0]
```
根据`bin_idx`从`heights`数组中提取对应位置的值，并存储在`input_heights`中。
    if inverse:
        # 计算二次方程的系数 a, b, c
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        # 计算判别式
        discriminant = b.pow(2) - 4 * a * c
        # 断言判别式大于等于0
        assert (discriminant >= 0).all()

        # 计算根
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # 计算输出值
        outputs = root * input_bin_widths + input_cumwidths

        # 计算 theta * (1 - theta)
        theta_one_minus_theta = root * (1 - root)
        # 计算分母
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
```
这段代码计算导数的分子部分。它使用了输入的delta值的平方乘以一个由多个项相加的表达式。这个表达式包括输入的导数加一的值乘以根的平方，加上两倍输入的delta值乘以theta和(1-theta)的乘积，再加上输入的导数乘以(1-root)的平方。

```
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
```
这段代码计算对数绝对值行列式(logabsdet)。它使用torch.log函数计算导数的分子的对数，然后减去2乘以分母的对数。

```
        return outputs, -logabsdet
```
如果条件为真，返回outputs和-logabsdet。

```
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
```
如果条件为假，执行这段代码。它计算theta和theta_one_minus_theta的值。theta是输入减去输入的累积宽度再除以输入的bin宽度，而theta_one_minus_theta是theta乘以(1-theta)。

```
        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator
```
这段代码计算输出值。它使用输入的高度乘以一个由多个项相加的表达式作为分子，这个表达式包括输入的delta值乘以theta的平方加上输入的导数乘以theta_one_minus_theta。分母是输入的delta值加上一个由多个项相加的表达式，这个表达式包括输入的导数加上输入的导数加一再减去两倍的输入的delta值，再乘以theta_one_minus_theta。最后，输出值是输入的累积高度加上分子除以分母的结果。
# 计算导数的分子，使用了输入的增量、导数、以及一些中间变量
derivative_numerator = input_delta.pow(2) * (
    input_derivatives_plus_one * theta.pow(2)
    + 2 * input_delta * theta_one_minus_theta
    + input_derivatives * (1 - theta).pow(2)
)
# 计算对数绝对值行列式，使用了导数的分子和分母
logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

# 返回计算结果
return outputs, logabsdet
```