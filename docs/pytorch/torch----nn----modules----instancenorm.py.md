# `.\pytorch\torch\nn\modules\instancenorm.py`

```py
# 引入警告模块，用于管理警告信息的显示
import warnings

# 引入 torch.nn.functional 中的 F 模块，提供了各种神经网络函数，包括实例归一化函数
import torch.nn.functional as F

# 引入 Tensor 类型，用于类型提示
from torch import Tensor

# 从当前目录中的 batchnorm 模块中导入 _LazyNormBase 和 _NormBase 类
from .batchnorm import _LazyNormBase, _NormBase

# 定义该模块对外公开的类名列表
__all__ = [
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
]

# _InstanceNorm 类，继承自 _NormBase 类
class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # 创建 factory_kwargs 字典，包含设备和数据类型信息
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类 _NormBase 的初始化方法，传入参数并继承其属性和方法
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    # 检查输入的维度是否符合预期
    def _check_input_dim(self, input):
        # 抛出未实现异常，提示子类需要重写该方法
        raise NotImplementedError

    # 获取无批次维度的方法
    def _get_no_batch_dim(self):
        # 抛出未实现异常，提示子类需要重写该方法
        raise NotImplementedError

    # 处理无批次输入的方法，对输入进行实例归一化并调整维度
    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    # 应用实例归一化的方法，调用 F.instance_norm 函数
    def _apply_instance_norm(self, input):
        return F.instance_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum if self.momentum is not None else 0.0,
            self.eps,
        )

    # 从状态字典中加载的方法，用于模型状态恢复
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
        # 检查是否存在版本信息，若不存在且 track_running_stats=False，则移除 running_mean 和 running_var
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            # 遍历 running_mean 和 running_var
            for name in ("running_mean", "running_var"):
                key = prefix + name
                # 如果 state_dict 中存在对应的键名，则添加到 running_stats_keys 列表中
                if key in state_dict:
                    running_stats_keys.append(key)
            # 如果有多余的 running stats buffer，则添加错误消息
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(f'"{k}"' for k in running_stats_keys),
                        klass=self.__class__.__name__,
                    )
                )
                # 从 state_dict 中移除多余的 running stats buffer
                for key in running_stats_keys:
                    state_dict.pop(key)

        # 调用父类方法，加载状态字典
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor) -> Tensor:
        # 检查输入张量的维度
        self._check_input_dim(input)

        # 计算特征维度
        feature_dim = input.dim() - self._get_no_batch_dim()
        # 检查输入张量特征维度是否与 num_features 匹配
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                # 抛出异常，期望输入的特征维度与 num_features 匹配
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.size(feature_dim)}."
                )
            else:
                # 发出警告，输入的特征维度与 num_features 不匹配，affine=False 时忽略 num_features
                warnings.warn(
                    f"input's size at dim={feature_dim} does not match num_features. "
                    "You can silence this warning by not passing in num_features, "
                    "which is not used because affine=False"
                )

        # 如果输入张量的维度为去除批处理维度后的维度，调用 _handle_no_batch_input 方法处理
        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        # 应用实例归一化
        return self._apply_instance_norm(input)
class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 2D (unbatched) or 3D (batched) input as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm1d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm1d` is applied
        on each channel of channeled data like multidimensional time series, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm1d` usually don't apply affine
        transform.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
"""
    """
    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = torch.randn(20, 100, 40)
        >>> output = m(input)
    """



    # 返回不带批次维度的维度数，即 2
    def _get_no_batch_dim(self):
        return 2

    # 检查输入张量的维度是否为 2D 或 3D，否则引发 ValueError 异常
    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")
# 定义一个继承自 _InstanceNorm 的类 InstanceNorm2d，用于应用二维实例归一化操作
class InstanceNorm2d(_InstanceNorm):
    # 文档字符串，描述了 Instance Normalization 的作用和输入输出形状
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    """

    # 开始注释 `InstanceNorm2d` 类的代码
    # 返回 InstanceNorm2d 类的文档字符串，描述了该类的功能、参数、输入输出形状以及示例用法

    def _get_no_batch_dim(self):
        # 返回表示没有批量维度的整数，这里固定返回 3
        return 3

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为 3 或 4，如果不是则抛出 ValueError 异常
        if input.dim() not in (3, 4):
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")
class InstanceNorm3d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size C (where C is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    updated via exponential moving average with the given `momentum` factor.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)
    """

    def _get_no_batch_dim(self):
        return 3
    def _get_no_batch_dim(self):
        # 返回值表示没有批次维度的情况下，维度索引为4
        return 4

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为4维或5维，否则抛出数值错误异常
        if input.dim() not in (4, 5):
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")
# 定义一个继承自 _LazyNormBase 和 _InstanceNorm 的类 LazyInstanceNorm3d，用于实现延迟初始化的 InstanceNorm3d 模块
class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm3d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm3d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)
    """

    # 设置类属性 cls_to_become 为 InstanceNorm3d 类型，用于指定该类将转变为哪个类的实例
    cls_to_become = InstanceNorm3d  # type: ignore[assignment]

    # 返回表示没有批次维度的数值，即输入张量的维度数，此处返回固定值 4
    def _get_no_batch_dim(self):
        return 4

    # 检查输入张量的维度是否为 4D 或 5D，如果不是则抛出 ValueError 异常
    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")
```