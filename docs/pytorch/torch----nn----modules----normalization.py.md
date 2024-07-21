# `.\pytorch\torch\nn\modules\normalization.py`

```py
# mypy: allow-untyped-defs
# 引入 numbers 模块，用于类型检查
import numbers
# 从 typing 模块中引入 List, Optional, Tuple, Union 类型
from typing import List, Optional, Tuple, Union

# 引入 torch 模块
import torch
# 从 torch 模块中引入 Size, Tensor 类型
from torch import Size, Tensor
# 从 torch.nn 模块中引入 functional as F, init
from torch.nn import functional as F, init
# 从 torch.nn.parameter 模块中引入 Parameter 类
from torch.nn.parameter import Parameter

# 从 ._functions 模块中引入 CrossMapLRN2d as _cross_map_lrn2d
from ._functions import CrossMapLRN2d as _cross_map_lrn2d
# 从 .module 模块中引入 Module 类
from .module import Module

# 定义模块可导出的公共成员列表
__all__ = ["LocalResponseNorm", "CrossMapLRN2d", "LayerNorm", "GroupNorm", "RMSNorm"]


# 定义 LocalResponseNorm 类，继承自 Module 类
class LocalResponseNorm(Module):
    r"""Applies local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    # 常量定义，标记对象具有的属性
    __constants__ = ["size", "alpha", "beta", "k"]
    size: int  # 滤波器尺寸
    alpha: float  # 乘法因子 alpha
    beta: float   # 指数 beta
    k: float      # 加法因子 k

    # 初始化函数，设定 LRN 的参数
    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        super().__init__()  # 调用父类 Module 的初始化函数
        self.size = size    # 初始化滤波器尺寸
        self.alpha = alpha  # 初始化 alpha 参数
        self.beta = beta    # 初始化 beta 参数
        self.k = k          # 初始化 k 参数

    # 前向传播函数，应用局部响应归一化操作
    def forward(self, input: Tensor) -> Tensor:
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    # 返回模块的额外描述信息
    def extra_repr(self):
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


# 定义 CrossMapLRN2d 类，继承自 Module 类
class CrossMapLRN2d(Module):
    size: int   # 滤波器尺寸
    alpha: float  # 乘法因子 alpha
    beta: float   # 指数 beta
    k: float      # 加法因子 k

    # 初始化函数，设定 CrossMapLRN2d 的参数
    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1
    ) -> None:
        super().__init__()  # 调用父类 Module 的初始化函数
        self.size = size    # 初始化滤波器尺寸
        self.alpha = alpha  # 初始化 alpha 参数
        self.beta = beta    # 初始化 beta 参数
        self.k = k          # 初始化 k 参数

    # 前向传播函数，应用跨通道局部归一化操作
    def forward(self, input: Tensor) -> Tensor:
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)

    # 返回模块的额外描述信息
    def extra_repr(self) -> str:
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


# 定义 Shape 类型的别名，可以是 int、List[int] 或 Size 对象
_shape_t = Union[int, List[int], Size]


# 定义 LayerNorm 类，继承自 Module 类
class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    # 定义 LayerNorm 类，用于对输入进行层归一化操作
    class LayerNorm(nn.Module):
        # 定义常量列表，包括归一化形状、epsilon值、是否逐元素仿射
        __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
        normalized_shape: Tuple[int, ...]  # 归一化的形状，由元组表示
        eps: float  # epsilon 值，用于数值稳定性
        elementwise_affine: bool  # 是否进行逐元素仿射操作
    
        def __init__(
            self,
            normalized_shape: _shape_t,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            bias: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)  # 如果是整数，转换为元组形式
            self.normalized_shape = tuple(normalized_shape)  # 设置归一化形状
            self.eps = eps  # 设置 epsilon 值
            self.elementwise_affine = elementwise_affine  # 设置是否逐元素仿射
    
            # 如果进行逐元素仿射操作
            if self.elementwise_affine:
                # 初始化权重参数，形状由 normalized_shape 决定
                self.weight = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
                # 如果启用偏置参数
                if bias:
                    # 初始化偏置参数，形状同样由 normalized_shape 决定
                    self.bias = Parameter(
                        torch.empty(self.normalized_shape, **factory_kwargs)
                    )
                else:
                    # 否则注册偏置参数为 None
                    self.register_parameter("bias", None)
            else:
                # 如果不进行逐元素仿射操作，注册权重和偏置参数为 None
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
    
            # 调用重置参数方法
            self.reset_parameters()
    
        def reset_parameters(self) -> None:
            # 如果进行逐元素仿射操作
            if self.elementwise_affine:
                # 初始化权重参数为全 1
                init.ones_(self.weight)
                # 如果偏置参数不为 None，则初始化为全 0
                if self.bias is not None:
                    init.zeros_(self.bias)
    
        def forward(self, input: Tensor) -> Tensor:
            # 调用 PyTorch 的 F.layer_norm 函数进行层归一化操作，传入输入、归一化形状、权重、偏置和 epsilon 值
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
    
        def extra_repr(self) -> str:
            # 返回描述该层归一化模块的额外信息，包括归一化形状、epsilon 值和是否逐元素仿射
            return (
                "{normalized_shape}, eps={eps}, "
                "elementwise_affine={elementwise_affine}".format(**self.__dict__)
            )
class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    # 初始化函数，用于初始化 GroupNorm 层
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ):
        # 调用父类的初始化方法
        super(GroupNorm, self).__init__()
        # 将输入参数赋值给对象的属性
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        # 如果指定了 device 和 dtype，则赋给相应的属性
        self.device = device
        self.dtype = dtype
    ) -> None:
        # 定义构造函数，初始化 BatchNorm2d 对象
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # 检查 num_channels 是否可以被 num_groups 整除，否则抛出数值错误异常
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        # 如果需要仿射变换，则创建权重和偏置参数
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            # 否则注册空的权重和偏置参数
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # 调用重置参数的函数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 如果需要仿射变换，初始化权重为全1，偏置为全0
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # 调用 F 库中的 group_norm 函数进行前向传播
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        # 返回对象的额外描述信息，包括 num_groups、num_channels、eps 和 affine 参数
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
# 定义一个 RMSNorm 类，继承自 Module 类
class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y = \frac{x}{\sqrt{\mathrm{RMS}[x] + \epsilon}} * \gamma

    The root mean squared norm is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the rms norm is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: :func:`torch.finfo(x.dtype).eps`
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    # 初始化方法，设置各种参数
    def __init__(
        self,
        normalized_shape: _shape_t,  # 归一化的形状，可以是整数、列表或 torch.Size
        eps: Optional[float] = None,  # 数值稳定性参数，默认为 torch.finfo(x.dtype).eps
        elementwise_affine: bool = True,  # 是否使用可学习的元素级仿射参数，默认为 True
        device=None,  # 设备参数，默认为 None
        dtype=None,  # 数据类型参数，默认为 None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()  # 调用父类的初始化方法
        if isinstance(normalized_shape, numbers.Integral):
            # 如果 normalized_shape 是整数，则转换为单元素元组
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # 设置归一化的形状
        self.eps = eps  # 设置数值稳定性参数
        self.elementwise_affine = elementwise_affine  # 设置是否使用元素级仿射参数
        if self.elementwise_affine:
            # 如果使用元素级仿射参数，则创建权重参数，并初始化为全为 1 的张量
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            # 如果不使用元素级仿射参数，则将权重参数注册为 None
            self.register_parameter("weight", None)
        self.reset_parameters()  # 调用重置参数的方法

    # 重置参数的方法
    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            # 如果使用元素级仿射参数，则将权重参数初始化为全为 1
            init.ones_(self.weight)
    # 定义前向传播方法，接受一个张量 x，并返回经过模块处理后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass.
        """
        # 使用 torch.nn.functional.rms_norm 函数进行前向传播计算
        # rms_norm 函数对输入张量 x 进行 root mean square normalization 的操作，
        # 使用模块内部的 normalized_shape, weight, eps 进行计算
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    # 定义额外表示方法，返回关于模块的额外信息的字符串表示
    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        # 使用字符串格式化方法构建描述模块额外信息的字符串
        # 字符串中包含 normalized_shape, eps, elementwise_affine 等模块属性的值
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
# TODO: ContrastiveNorm2d
# 对比归一化层（Contrastive Normalization），可能是一个自定义的神经网络层或者操作。

# TODO: DivisiveNorm2d
# 除法归一化层（Divisive Normalization），可能是一个自定义的神经网络层或者操作。

# TODO: SubtractiveNorm2d
# 减法归一化层（Subtractive Normalization），可能是一个自定义的神经网络层或者操作。
```