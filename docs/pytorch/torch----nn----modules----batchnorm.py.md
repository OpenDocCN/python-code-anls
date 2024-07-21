# `.\pytorch\torch\nn\modules\batchnorm.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型声明和模块
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import functional as F, init  # 导入 torch 的函数模块和初始化模块
from torch.nn.parameter import Parameter, UninitializedBuffer, UninitializedParameter  # 导入参数相关的类

from ._functions import SyncBatchNorm as sync_batch_norm  # 从自定义模块导入同步批标准化函数别名
from .lazy import LazyModuleMixin  # 导入 LazyModuleMixin 模块
from .module import Module  # 导入 Module 类

__all__ = [  # 暴露给外部的模块列表
    "BatchNorm1d",
    "LazyBatchNorm1d",
    "BatchNorm2d",
    "LazyBatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm3d",
    "SyncBatchNorm",
]

class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm."""

    _version = 2  # 模型版本号
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]  # 常量列表

    num_features: int  # 特征数量
    eps: float  # 避免除零时使用的小数值
    momentum: Optional[float]  # 动量参数，可选的浮点数
    affine: bool  # 是否进行仿射变换
    track_running_stats: bool  # 是否跟踪运行时统计信息

    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670
    # 注意：这里有意没有定义 weight 和 bias 属性，参见 GitHub 上的说明链接

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}  # 工厂参数字典
        super().__init__()  # 调用父类初始化方法
        self.num_features = num_features  # 设置特征数量
        self.eps = eps  # 设置 eps 参数
        self.momentum = momentum  # 设置 momentum 参数
        self.affine = affine  # 设置 affine 参数
        self.track_running_stats = track_running_stats  # 设置 track_running_stats 参数
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))  # 如果进行仿射变换，创建权重参数
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))  # 创建偏置参数
        else:
            self.register_parameter("weight", None)  # 如果不进行仿射变换，注册权重参数为 None
            self.register_parameter("bias", None)  # 注册偏置参数为 None
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )  # 如果跟踪运行时统计信息，注册运行时均值缓冲区
            self.register_buffer(
                "running_var", torch.ones(num_features, **factory_kwargs)
            )  # 注册运行时方差缓冲区
            self.running_mean: Optional[Tensor]  # 运行时均值的类型注释
            self.running_var: Optional[Tensor]  # 运行时方差的类型注释
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )  # 注册追踪批次数量的缓冲区
            self.num_batches_tracked: Optional[Tensor]  # 批次追踪数的类型注释
        else:
            self.register_buffer("running_mean", None)  # 如果不跟踪运行时统计信息，运行时均值设置为 None
            self.register_buffer("running_var", None)  # 运行时方差设置为 None
            self.register_buffer("num_batches_tracked", None)  # 批次追踪数设置为 None
        self.reset_parameters()  # 初始化参数
    # 重置运行时统计信息，如果开启了跟踪运行时统计信息
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # 运行时均值/方差/批次数等信息在运行时根据需要进行注册
            self.running_mean.zero_()  # 清零运行时均值，类型忽略检查
            self.running_var.fill_(1)  # 将运行时方差设为1，类型忽略检查
            self.num_batches_tracked.zero_()  # 将已跟踪的批次数清零，类型忽略联合属性和操作

    # 重置参数
    def reset_parameters(self) -> None:
        self.reset_running_stats()  # 调用重置运行时统计信息的方法
        if self.affine:
            init.ones_(self.weight)  # 将权重初始化为全1
            init.zeros_(self.bias)  # 将偏置初始化为全0

    # 检查输入维度的私有方法，未实现
    def _check_input_dim(self, input):
        raise NotImplementedError

    # 返回额外信息的字符串表示
    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    # 从状态字典加载模型参数的私有方法
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        # 如果版本号为空或小于2，并且开启了跟踪运行时统计信息
        if (version is None or version < 2) and self.track_running_stats:
            # 在版本2时添加了 num_batches_tracked 缓冲区
            # 应该有一个默认值为0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            # 如果状态字典中不存在 num_batches_tracked_key
            if num_batches_tracked_key not in state_dict:
                # 将 num_batches_tracked 设置为0，类型为长整型
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor(0, dtype=torch.long)
                )

        # 调用父类的加载状态字典方法
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类 _NormBase 的初始化方法，传入参数进行初始化
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        # 检查输入张量的维度是否符合预期
        self._check_input_dim(input)

        # 设置指数加权平均系数为 self.momentum，用于在导出到 ONNX 时更新图中的节点
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # 在训练模式下且跟踪运行统计信息时，更新批次追踪数
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # 使用累积移动平均
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # 使用指数移动平均
                    exponential_average_factor = self.momentum

        r"""
        决定是否使用小批量统计信息进行归一化，而不是使用缓冲区中的统计信息。
        在训练模式下使用小批量统计信息，在评估模式下缓冲区为 None 时也使用它们。
        """
        if self.training:
            bn_training = True
        else:
            # 如果 running_mean 和 running_var 都为 None，则使用小批量统计信息进行归一化
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        只有在需要跟踪缓冲区并且处于训练模式时才会更新缓冲区。
        因此，只有在需要更新时（即在训练模式下跟踪它们时），或者在评估模式下不使用缓冲区统计信息时，才需要传递它们。
        """
        return F.batch_norm(
            input,
            # 如果不需要跟踪缓冲区，则确保它们不会被更新
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class _LazyNormBase(LazyModuleMixin, _NormBase):
    weight: UninitializedParameter  # type: ignore[assignment]
    bias: UninitializedParameter  # type: ignore[assignment]
    def __init__(
        self,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ) -> None:
        # 定义一个包含设备和数据类型参数的字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类初始化方法，传入一些硬编码的参数来避免创建即将被覆写的张量
        super().__init__(
            0,  # 批次统计和仿射变换被设为 False
            eps,
            momentum,
            False,
            False,
            **factory_kwargs,  # 传递设备和数据类型参数
        )
        # 初始化实例属性
        self.affine = affine
        self.track_running_stats = track_running_stats
        # 如果启用了仿射变换，初始化权重和偏置参数
        if self.affine:
            self.weight = UninitializedParameter(**factory_kwargs)
            self.bias = UninitializedParameter(**factory_kwargs)
        # 如果启用了追踪运行统计信息，初始化运行均值、方差和追踪批次计数
        if self.track_running_stats:
            self.running_mean = UninitializedBuffer(**factory_kwargs)
            self.running_var = UninitializedBuffer(**factory_kwargs)
            self.num_batches_tracked = torch.tensor(
                0,
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},  # 排除数据类型参数
            )

    def reset_parameters(self) -> None:
        # 如果没有未初始化的参数并且特征数不为零，则调用父类的重置参数方法
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        # 如果存在未初始化的参数
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]  # 记录输入特征的数量
            # 如果启用了仿射变换，确保权重和偏置是未初始化的参数类型，并创建它们
            if self.affine:
                assert isinstance(self.weight, UninitializedParameter)
                assert isinstance(self.bias, UninitializedParameter)
                self.weight.materialize((self.num_features,))
                self.bias.materialize((self.num_features,))
            # 如果启用了追踪运行统计信息，确保运行均值和方差是未初始化的缓冲区类型，并创建它们
            if self.track_running_stats:
                self.running_mean.materialize((self.num_features,))
                self.running_var.materialize((self.num_features,))
            # 调用重置参数方法
            self.reset_parameters()
# 定义 BatchNorm1d 类，继承自 _BatchNorm 类
class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.
    At train time in the forward pass, the standard-deviation is calculated via the biased estimator,
    equivalent to ``torch.var(input, unbiased=False)``. However, the value stored in the
    moving average of the standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.
    Args:
        num_features: 输入的特征数或通道数 :math:`C`
        eps: 为了数值稳定性添加到分母的值。默认为 1e-5
        momentum: 用于计算 running_mean 和 running_var 的值。可以设为 ``None`` 表示累积移动平均（即简单平均）。默认为 0.1
        affine: 布尔值，设置为 ``True`` 表示该模块具有可学习的仿射参数。默认为 ``True``
        track_running_stats: 布尔值，设置为 ``True`` 表示该模块跟踪运行时的均值和方差；设置为 ``False`` 表示不跟踪这些统计数据，并初始化统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 为 ``None``。当这些缓冲区为 ``None`` 时，该模块始终使用批次统计数据，无论是在训练模式还是评估模式下。默认为 ``True``

    Shape:
        - Input: :math:`(N, C)` 或 :math:`(N, C, L)`，其中 :math:`N` 是批量大小，:math:`C` 是特征数或通道数，:math:`L` 是序列长度
        - Output: :math:`(N, C)` 或 :math:`(N, C, L)`（与输入相同的形状）

    Examples::

        >>> # 带有可学习参数
        >>> m = nn.BatchNorm1d(100)
        >>> # 不带可学习参数
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为2D或3D，否则引发错误
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")
class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm1d` module with lazy initialization.

    Lazy initialization based on the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm1d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为2D或3D，否则引发异常
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input.

    4D is a mini-batch of 2D inputs
    with additional channel dimension. Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    # 定义一个私有方法 `_check_input_dim`，用于检查输入张量的维度是否为四维
    def _check_input_dim(self, input):
        # 如果输入张量的维度不是四维，则抛出 ValueError 异常
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm2d` module with lazy initialization.

    Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm2d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        # 检查输入的维度是否为4维，如果不是则抛出异常
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input.

    5D is a mini-batch of 3D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    """
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    # 检查输入张量的维度是否为5维
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm3d` module with lazy initialization.

    Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm3d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        # 检查输入的维度是否为5维，如果不是则抛出值错误异常
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class SyncBatchNorm(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input.

    The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.



    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.



    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.



    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.



    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world



    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)



    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.
    Examples::

        >>> # xdoctest: +SKIP
        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)  # 创建一个具有可学习参数的 SyncBatchNorm 层，设置 num_features 为 100
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))  # 创建一个包含 0 到 7 的整数列表，用于表示进程组的 rank id
        >>> r1, r2 = ranks[:4], ranks[4:]  # 将 ranks 列表分成两部分，r1 包含前四个元素，r2 包含后四个元素
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]  # 创建两个分布式进程组，分别对应 r1 和 r2
        >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]  # 根据当前进程的 rank 决定使用哪个进程组
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)  # 创建一个 BatchNorm3d 层，设置 num_features 为 100，不含可学习参数，使用指定的进程组
        >>> input = torch.randn(20, 100, 35, 45, 10)  # 创建一个随机输入张量
        >>> output = m(input)  # 对输入张量进行 BatchNorm3d 操作，并获得输出张量

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)  # 将给定的 BatchNorm 网络转换为 SyncBatchNorm 网络，使用指定的进程组
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)  # 创建一个分布式数据并行 SyncBatchNorm 网络，每个进程仅支持单 GPU 操作

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None,
    ) -> None:
        """
        初始化 SyncBatchNorm 层的参数和状态。

        Args:
            num_features (int): 输入特征的数量。
            eps (float, optional): 数值稳定性控制参数，默认为 1e-5。
            momentum (float, optional): 动量参数，用于计算运行时统计，默认为 0.1。
            affine (bool, optional): 是否应用仿射变换，默认为 True。
            track_running_stats (bool, optional): 是否跟踪运行时统计信息，默认为 True。
            process_group (Optional[Any], optional): 分布式训练时使用的进程组，默认为 None。
            device: 设备参数，指定在哪个设备上创建层，默认为 None。
            dtype: 数据类型参数，默认为 None。

        """
        factory_kwargs = {"device": device, "dtype": dtype}  # 根据传入的 device 和 dtype 创建工厂参数字典
        super().__init__(  # 调用父类的初始化方法，传入同步批归一化层的相关参数
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.process_group = process_group  # 设置 SyncBatchNorm 层的进程组属性

    def _check_input_dim(self, input):
        """
        检查输入张量的维度是否符合要求。

        Args:
            input (Tensor): 输入张量。

        Raises:
            ValueError: 如果输入张量维度小于 2，则抛出异常。
        """
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {input.dim()}D input)")

    def _check_non_zero_input_channels(self, input):
        """
        检查输入通道数是否为非零。

        Args:
            input (Tensor): 输入张量。

        Raises:
            ValueError: 如果输入通道数为零，则抛出异常。
        """
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Converts all :attr:`BatchNorm*D` layers in the model to :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> # xdoctest: +SKIP("distributed")
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        # 将输入模块保存为输出模块，初始设置为相同
        module_output = module
        # 如果模块是 BatchNorm*D 类型的实例
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            # 创建一个 SyncBatchNorm 对象
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,   # 使用 BatchNorm*D 的特征数
                module.eps,            # 使用 BatchNorm*D 的 eps 参数
                module.momentum,       # 使用 BatchNorm*D 的 momentum 参数
                module.affine,         # 使用 BatchNorm*D 的 affine 参数
                module.track_running_stats,  # 使用 BatchNorm*D 的 track_running_stats 参数
                process_group,        # 使用给定的进程组参数
            )
            # 如果 BatchNorm*D 是可仿射的（即具有可学习的 weight 和 bias）
            if module.affine:
                # 使用 torch.no_grad() 来确保在计算图中不跟踪梯度
                with torch.no_grad():
                    module_output.weight = module.weight  # 复制 BatchNorm*D 的权重
                    module_output.bias = module.bias      # 复制 BatchNorm*D 的偏置
            module_output.running_mean = module.running_mean  # 复制 BatchNorm*D 的 running_mean
            module_output.running_var = module.running_var    # 复制 BatchNorm*D 的 running_var
            module_output.num_batches_tracked = module.num_batches_tracked  # 复制 BatchNorm*D 的 num_batches_tracked
            module_output.training = module.training  # 复制 BatchNorm*D 的 training 状态
            # 如果模块有 qconfig 属性，则复制该属性到 SyncBatchNorm 对象
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        # 遍历模块的每一个子模块，递归调用 convert_sync_batchnorm 方法
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        # 删除输入的模块对象，确保内存管理
        del module
        # 返回最终处理后的模块对象
        return module_output
```