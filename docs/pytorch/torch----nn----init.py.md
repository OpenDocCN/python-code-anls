# `.\pytorch\torch\nn\init.py`

```
# mypy: allow-untyped-defs
"""This file contains utilities for initializing neural network parameters."""
import math  # 导入数学库
import warnings  # 导入警告模块
from typing import Optional as _Optional  # 导入类型提示模块，使用别名_Optional

import torch  # 导入PyTorch库
from torch import Tensor  # 从PyTorch中导入Tensor类


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.

# 使用_no_grad_*函数作为这些函数中使用`with torch.no_grad()`部分的包装器是必要的。
# JIT不支持上下文管理器，因此这些函数需要实现为内置函数。
# 使用这些包装器可以使这些内置函数保持简短和可重用。

def _no_grad_uniform_(tensor, a, b, generator=None):
    """Apply uniform distribution to `tensor` with range [a, b] in no_grad mode."""
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)


def _no_grad_normal_(tensor, mean, std, generator=None):
    """Apply normal distribution to `tensor` with mean and std in no_grad mode."""
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=None):
    """
    Apply truncated normal distribution to `tensor` with mean and std within range [a, b]
    in no_grad mode.
    """
    # 根据https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf中的方法实现
    def norm_cdf(x):
        # 计算标准正态分布的累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        # 如果均值超出了[a, b]范围2倍标准差之外，发出警告
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # 通过使用截断均匀分布生成值，然后使用正态分布的逆累积分布函数
        # 获取上下界的累积分布函数值
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 用[l, u]范围内的值均匀填充张量，然后转换到[2l-1, 2u-1]范围
        tensor.uniform_(2 * l - 1, 2 * u - 1, generator=generator)

        # 使用正态分布的逆累积分布函数转换为截断标准正态分布
        tensor.erfinv_()

        # 转换为正确的均值和标准差
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # 确保值在正确的范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def _no_grad_fill_(tensor, val):
    """Fill `tensor` with `val` in no_grad mode."""
    with torch.no_grad():
        return tensor.fill_(val)


def _no_grad_zero_(tensor):
    """Fill `tensor` with zeros in no_grad mode."""
    with torch.no_grad():
        return tensor.zero_()


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    """
    """
    Args:
        nonlinearity: 非线性函数的名称（`nn.functional` 中的名称）
        param: 非线性函数的可选参数

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # 使用负斜率为 0.2 的 leaky_relu

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    # 支持线性函数的列表
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    # 如果 nonlinearity 是线性函数或者 sigmoid 函数，则返回权重初始化的增益 1
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    # 如果 nonlinearity 是 tanh 函数，则返回权重初始化的增益 5.0 / 3
    elif nonlinearity == "tanh":
        return 5.0 / 3
    # 如果 nonlinearity 是 relu 函数，则返回权重初始化的增益 sqrt(2.0)
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    # 如果 nonlinearity 是 leaky_relu 函数
    elif nonlinearity == "leaky_relu":
        # 如果 param 为 None，则使用默认的负斜率 0.01
        if param is None:
            negative_slope = 0.01
        # 如果 param 是数值类型（int 或 float），则使用指定的负斜率
        elif (not isinstance(param, bool)
              and isinstance(param, (int, float))):
            # True/False 是 int 类型的实例，所以要先排除 bool 类型
            negative_slope = param
        else:
            # 抛出异常，说明传入的 param 参数不是有效的数值类型
            raise ValueError(f"negative_slope {param} not a valid number")
        # 返回根据负斜率计算的权重初始化的增益
        return math.sqrt(2.0 / (1 + negative_slope**2))
    # 如果 nonlinearity 是 selu 函数，则返回经验值 3.0 / 4
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # 根据经验值设置（https://github.com/pytorch/pytorch/pull/50664）
    else:
        # 抛出异常，说明不支持的 nonlinearity 函数类型
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")
# 使用均匀分布从区间 [a, b] 中随机填充输入的张量 tensor。
# 如果 tensor 支持 Torch 函数的多态性，调用 Torch 函数处理。
def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            uniform_, (tensor,), tensor=tensor, a=a, b=b, generator=generator
        )
    # 调用 _no_grad_uniform_ 函数填充 tensor，实现从均匀分布 U(a, b) 中随机抽取值的功能。
    return _no_grad_uniform_(tensor, a, b, generator)


# 使用正态分布从均值 mean，标准差 std^2 中随机填充输入的张量 tensor。
# 如果 tensor 支持 Torch 函数的多态性，调用 Torch 函数处理。
def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            normal_, (tensor,), tensor=tensor, mean=mean, std=std, generator=generator
        )
    # 调用 _no_grad_normal_ 函数填充 tensor，实现从正态分布 N(mean, std^2) 中随机抽取值的功能。
    return _no_grad_normal_(tensor, mean, std, generator)


# 使用截断正态分布从区间 [a, b] 内部的正态分布 N(mean, std^2) 中随机填充输入的张量 tensor。
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    # 调用 _no_grad_trunc_normal_ 函数填充 tensor，实现从截断正态分布中随机抽取值的功能。
    return _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=generator)


# 使用常数值 val 填充输入的张量 tensor。
def constant_(tensor: Tensor, val: float) -> Tensor:
    Args:
        tensor: an n-dimensional `torch.Tensor` to be filled with a constant value.
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    # 检查是否存在 torch 函数的变长参数版本，如果有则调用处理 torch 函数的方法
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            constant_, (tensor,), tensor=tensor, val=val
        )
    # 否则调用自定义的填充函数 _no_grad_fill_ 填充张量
    return _no_grad_fill_(tensor, val)
def ones_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    return _no_grad_zero_(tensor)


def eye_(tensor):
    r"""Fill the 2-dimensional input `Tensor` with the identity matrix.

    Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    # Use torch.no_grad() to ensure operations do not record gradients
    with torch.no_grad():
        # Fill the tensor with the identity matrix
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad)
    return tensor


def dirac_(tensor, groups=1):
    r"""Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)

    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    dimensions = tensor.ndimension()
    # Check if tensor dimension is valid for Dirac initialization
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.size()

    # Ensure first dimension is divisible by groups
    if sizes[0] % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    # Use torch.no_grad() to ensure operations do not record gradients
    with torch.no_grad():
        # Initialize tensor elements according to Dirac delta function
        tensor.zero_()

        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[
                        g * out_chans_per_grp + d,
                        d,
                        tensor.size(2) // 2,
                        tensor.size(3) // 2,
                    ] = 1
                else:  # Volumetric convolution
                    tensor[
                        g * out_chans_per_grp + d,
                        d,
                        tensor.size(2) // 2,
                        tensor.size(3) // 2,
                        tensor.size(4) // 2,
                    ] = 1
   `
    return tensor
# 计算张量的输入和输出通道数量
def _calculate_fan_in_and_fan_out(tensor):
    # 获取张量的维度数
    dimensions = tensor.dim()
    # 如果张量维度小于2，抛出数值错误异常
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    # 获取输入通道数（即第二个维度大小）
    num_input_fmaps = tensor.size(1)
    # 获取输出通道数（即第一个维度大小）
    num_output_fmaps = tensor.size(0)
    # 初始感受野大小为1
    receptive_field_size = 1
    # 如果张量维度大于2，计算感受野大小
    if tensor.dim() > 2:
        # math.prod 不一定总是可用，手动累积计算乘积
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    # 计算输入通道数
    fan_in = num_input_fmaps * receptive_field_size
    # 计算输出通道数
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    # 调用 _calculate_fan_in_and_fan_out 函数计算输入和输出通道数
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # 根据公式计算标准差
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # 计算均匀分布的上下界
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    # 调用 _no_grad_uniform_ 函数生成均匀分布的值，并返回填充后的张量
    return _no_grad_uniform_(tensor, -a, a, generator)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    # 调用 _calculate_fan_in_and_fan_out 函数计算输入和输出通道数
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # 根据公式计算标准差
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    # 调用 _no_grad_normal_ 函数生成正态分布的值，并返回填充后的张量
    return _no_grad_normal_(tensor, 0.0, std, generator)


def _calculate_correct_fan(tensor, mode):
    # 将 mode 转换为小写
    mode = mode.lower()
    # 定义有效的模式列表，只能是 "fan_in" 或 "fan_out"
    valid_modes = ["fan_in", "fan_out"]
    # 如果给定的模式不在有效模式列表中，则抛出 ValueError 异常
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    
    # 根据给定的张量计算输入和输出的扇入和扇出
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # 如果模式是 "fan_in"，则返回扇入；否则返回扇出
    return fan_in if mode == "fan_in" else fan_out
# 使用 Kaiming 均匀分布填充输入的张量 `Tensor` 中的值。
# 方法描述在 `Delving deep into rectifiers: Surpassing
# human-level performance on ImageNet classification` - He, K. et al. (2015) 中。
# 结果张量的值将从 :math:`\mathcal{U}(-\text{bound}, \text{bound})` 中抽样，
# 其中
# 
# .. math::
#     \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
# 
# 也称为 He 初始化。
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: _Optional[torch.Generator] = None,
):
    # 如果 `tensor` 使用了 Torch 函数重载机制，则调用相应的处理函数
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )

    # 如果 `tensor` 的形状包含 0，发出警告并返回 `tensor`
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    # 计算正确的 fan 数量
    fan = _calculate_correct_fan(tensor, mode)
    # 计算增益
    gain = calculate_gain(nonlinearity, a)
    # 计算标准差
    std = gain / math.sqrt(fan)
    # 计算均匀分布的边界，从标准差中计算
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    # 使用 Torch 的 `uniform_` 方法填充张量，范围为 `(-bound, bound)`
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


# 使用 Kaiming 正态分布填充输入的张量 `Tensor` 中的值。
# 方法描述在 `Delving deep into rectifiers: Surpassing
# human-level performance on ImageNet classification` - He, K. et al. (2015) 中。
# 结果张量的值将从 :math:`\mathcal{N}(0, \text{std}^2)` 中抽样，
# 其中
# 
# .. math::
#     \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
# 
# 也称为 He 初始化。
def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: _Optional[torch.Generator] = None,
):
    Args:
        tensor: an n-dimensional `torch.Tensor`  # 输入参数tensor是一个n维的PyTorch张量
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)  # 斜率a是用于此层后的整流器（仅用于'leaky_relu'）
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.  # 模式是'fan_in'（默认）或'fan_out'，影响权重方差在前向和反向传播中的变化
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).  # 非线性函数的名称，建议仅在'relu'或'leaky_relu'中使用
        generator: the torch Generator to sample from (default: None)  # 用于采样的PyTorch生成器（默认为None）

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if 0 in tensor.shape:  # 检查张量的形状是否包含0，如果是则发出警告并返回张量本身
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)  # 计算正确的fan值，根据指定的mode参数
    gain = calculate_gain(nonlinearity, a)  # 计算增益，基于非线性函数和斜率a的值
    std = gain / math.sqrt(fan)  # 计算标准差，根据增益和fan值计算得出
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)  # 用正态分布填充张量，均值为0，标准差为std，可选的生成器为generator
def orthogonal_(
    tensor,
    gain=1,
    generator: _Optional[torch.Generator] = None,
):
    r"""Fill the input `Tensor` with a (semi) orthogonal matrix.

    Described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    # 检查输入张量的维度是否至少为2
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    # 如果张量没有元素，则返回不做任何操作
    if tensor.numel() == 0:
        # no-op
        return tensor
    # 获取张量的行数
    rows = tensor.size(0)
    # 计算列数，由于张量已经被展平，直接用总元素数除以行数得到列数
    cols = tensor.numel() // rows
    # 用标准正态分布填充一个新的张量，与输入张量形状相同
    flattened = tensor.new(rows, cols).normal_(0, 1, generator=generator)

    # 如果行数小于列数，对张量进行转置
    if rows < cols:
        flattened.t_()

    # 计算 QR 分解
    q, r = torch.linalg.qr(flattened)
    # 根据文献推荐，对 Q 进行正交化处理
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    # 如果行数小于列数，再次对 Q 进行转置
    if rows < cols:
        q.t_()

    # 用 Q 更新输入张量，然后乘以增益因子
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def sparse_(
    tensor,
    sparsity,
    std=0.01,
    generator: _Optional[torch.Generator] = None,
):
    r"""Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    # 检查输入张量是否是二维的
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    # 获取张量的行数和列数
    rows, cols = tensor.shape
    # 计算每列要置零的元素个数
    num_zeros = int(math.ceil(sparsity * rows))

    # 用标准正态分布填充张量，用于非零元素的生成
    with torch.no_grad():
        tensor.normal_(0, std, generator=generator)
        # 对每一列的元素进行操作，随机选取一部分行索引置零
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


# for backward compatibility
def _make_deprecate(meth):
    new_name = meth.__name__
    old_name = new_name[:-1]
    def deprecated_init(*args, **kwargs):
        # 发出警告，提示用户该方法已弃用，建议使用新方法
        warnings.warn(
            f"`nn.init.{old_name}` is now deprecated in favor of `nn.init.{new_name}`.",
            FutureWarning,
            stacklevel=2,
        )
        # 调用原始的初始化方法（已弃用的方法）
        return meth(*args, **kwargs)

    # 设置函数文档字符串，说明该方法的用法和弃用警告
    deprecated_init.__doc__ = rf"""
    {old_name}(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.

    See :func:`~torch.nn.init.{new_name}` for details."""

    # 将函数名称设置为原始方法的名称
    deprecated_init.__name__ = old_name

    # 返回被标记为弃用的初始化方法
    return deprecated_init
# 将函数 uniform_ 进行过时标记并返回一个新的函数
uniform = _make_deprecate(uniform_)

# 将函数 normal_ 进行过时标记并返回一个新的函数
normal = _make_deprecate(normal_)

# 将函数 constant_ 进行过时标记并返回一个新的函数
constant = _make_deprecate(constant_)

# 将函数 eye_ 进行过时标记并返回一个新的函数
eye = _make_deprecate(eye_)

# 将函数 dirac_ 进行过时标记并返回一个新的函数
dirac = _make_deprecate(dirac_)

# 将函数 xavier_uniform_ 进行过时标记并返回一个新的函数
xavier_uniform = _make_deprecate(xavier_uniform_)

# 将函数 xavier_normal_ 进行过时标记并返回一个新的函数
xavier_normal = _make_deprecate(xavier_normal_)

# 将函数 kaiming_uniform_ 进行过时标记并返回一个新的函数
kaiming_uniform = _make_deprecate(kaiming_uniform_)

# 将函数 kaiming_normal_ 进行过时标记并返回一个新的函数
kaiming_normal = _make_deprecate(kaiming_normal_)

# 将函数 orthogonal_ 进行过时标记并返回一个新的函数
orthogonal = _make_deprecate(orthogonal_)

# 将函数 sparse_ 进行过时标记并返回一个新的函数
sparse = _make_deprecate(sparse_)
```