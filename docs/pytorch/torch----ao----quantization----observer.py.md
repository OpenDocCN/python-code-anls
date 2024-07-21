# `.\pytorch\torch\ao\quantization\observer.py`

```py
# mypy: allow-untyped-defs
"""
This module implements observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类元类和抽象方法装饰器
from collections import OrderedDict  # 导入有序字典模块
from functools import partial  # 导入偏函数模块
from typing import Any, List, Tuple, Optional, Dict  # 导入类型注解模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.ao.quantization.utils import (
    check_min_max_valid, calculate_qmin_qmax, is_per_tensor, is_per_channel, validate_qmin_qmax
)  # 从PyTorch量化工具中导入相关函数

__all__ = [
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_quant_observer",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "per_channel_weight_observer_range_neg_127_to_127",
    "weight_observer_range_neg_127_to_127",
    "FixedQParamsObserver",
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
]

class _PartialWrapper:
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # 对每个参数调用callable_args中的函数并将其作为partial参数，然后使用keywords运行
        # 如果关键字中存在arg_name，则跳过以便可以进行覆盖
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, arg_name: self.callable_args[arg_name]()}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args
    """
    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> # 将 _with_args 方法作为 classmethod 添加到 Foo 类中
        >>> Foo.with_args = classmethod(_with_args)
        >>> # 创建一个 foo_builder 实例，先传入参数 a=3 和 b=4
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> # 使用 foo_builder() 创建 foo_instance1
        >>> foo_instance1 = foo_builder()
        >>> # 再次使用 foo_builder() 创建 foo_instance2
        >>> foo_instance2 = foo_builder()
        >>> # 检查两个实例的内存地址是否相同
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    # 使用 partial 函数将 cls_or_self 和 kwargs 组合成一个新的函数对象
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    # 返回创建的 _PartialWrapper 实例
    return r
def _with_callable_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories args that need to be
    called at construction time.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances and those arguments should only
    be calculated at construction time. Can be used in conjunction with _with_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_callable_args = classmethod(_with_callable_args)
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_callable_args(cur_time=get_time_func).with_args(name="dan")
        >>> foo_instance1 = foo_builder()
        >>> # wait 50
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1.creation_time) == id(foo_instance2.creation_time)
        False
    """
    # 创建 _PartialWrapper 对象，将部分函数应用到 cls_or_self 上
    r = _PartialWrapper(partial(cls_or_self))
    # 返回允许在构造时调用的包装器，附带额外的关键字参数
    return r.with_callable_args(**kwargs)


ABC: Any = ABCMeta("ABC", (object,), {})  # 兼容 Python 2 和 3:


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        is_dynamic: indicator for whether the observer is a placeholder for dynamic quantization
        or static quantization
    """

    def __init__(self, dtype, is_dynamic=False):
        # 调用父类构造函数
        super().__init__()
        # 设置观察器的数据类型
        self.dtype = dtype
        # 设置是否为动态量化的标志
        self.is_dynamic = is_dynamic

    @abstractmethod
    def forward(self, x):
        # 抽象方法：处理输入张量 x 的逻辑
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        # 抽象方法：根据收集的统计信息计算量化参数
        pass

    # 设置类方法 with_args 为 _with_args 的类方法
    with_args = classmethod(_with_args)
    # 设置类方法 with_callable_args 为 _with_callable_args 的类方法


class UniformQuantizationObserverBase(ObserverBase):
    r"""Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`
    """
    """
    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Note: the version is shared by all observer types
    #
    # Version 1/None
    #   self
    #
    # Version 2 (base class only, does not include child class buffers)
    #   self
    #   |--- eps : Tensor
    #
    # Version 3
    #   for HistogramObserver only, changed the shape of uninitialized
    #   min_val and max_val buffers from torch.Size([0]) to torch.Size([])
    #   for PerChannelObservers, changed the name of the buffers from min_vals
    #   to min_val and from max_vals to max_val.
    _version = 3  # 设置观察器对象的版本号为3

    eps: torch.Tensor  # 声明一个名为eps的torch.Tensor类型的成员变量

    def __init__(
        self,
        dtype=torch.quint8,  # 设置默认数据类型为torch.quint8
        qscheme=torch.per_tensor_affine,  # 设置默认量化方案为torch.per_tensor_affine
        reduce_range=False,  # 控制是否缩减量化范围，默认为False
        quant_min=None,  # 量化的最小值，默认为None
        quant_max=None,  # 量化的最大值，默认为None
        factory_kwargs=None,  # 工厂参数字典，默认为None
        eps=torch.finfo(torch.float32).eps,  # 默认的eps值为torch.float32的最小增量
        is_dynamic=False,  # 控制是否为动态量化，默认为False
        **kwargs,  # 其他关键字参数
    ) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)  # 处理工厂参数字典
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)  # 调用父类初始化函数
        self.qscheme = qscheme  # 设置观察器对象的量化方案
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )  # 如果reduce_range为True，发出警告

        self.reduce_range = reduce_range  # 设置是否缩减量化范围的标志位
        self.register_buffer(
            "eps", torch.tensor([eps], **factory_kwargs)
        )  # 注册eps值为torch.Tensor对象，并存储在缓冲区中
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"  # 断言确保量化方案在允许的范围内

        _ALLOWED_DTYPES = (
            torch.qint8,
            torch.quint8,
            torch.quint4x2,
            torch.qint32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        )
        assert self.dtype in _ALLOWED_DTYPES, f"Default Observer only works for {_ALLOWED_DTYPES} data type"  # 断言确保数据类型在允许的范围内

        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)  # 检查是否设置了自定义的量化范围
        if self.has_customized_qrange:
            validate_qmin_qmax(quant_min, quant_max)  # 如果设置了自定义量化范围，则验证其有效性
        self.quant_min, self.quant_max = \
            calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)  # 计算量化的最小值和最大值
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
        # 从本地元数据中获取模型版本号
        version = local_metadata.get("version", None)

        # 如果版本号为空或者为1，则在state_dict中添加eps张量
        if version is None or version == 1:
            # 创建一个张量eps，其值为浮点数类型的最小值
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        # 调用父类的_load_from_state_dict方法，传递参数来加载模型状态
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        r"""Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        # 确保用户指定的量化范围包含0
        assert (
            quant_min <= 0 <= quant_max
        ), "Used-specified quantization range must include 0."
        # 确保qmin严格小于qmax，用于用户指定的量化范围
        assert (
            quant_min < quant_max
        ), "qmin must be strictly less than qmax for user-specified quantization range."

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ):
        # 这是一个抽象方法，用于计算量化参数，具体实现需要在子类中完成
        raise NotImplementedError("Cannot calculate quantization parameters in the base observer class.")

    @torch.jit.export
    def reset_min_max_vals(self):
        # 这是一个抽象方法，用于重置最小值和最大值，在给定的观察器中不能实现
        raise NotImplementedError("Cannot reset min/max values in the given observer.")
# 原先这个类被称为 `_ObserverBase`，为了向后兼容保留了旧名称。
# TODO(v1.13 之后): 删除这段代码
_ObserverBase = UniformQuantizationObserverBase

class MinMaxObserver(UniformQuantizationObserverBase):
    r"""基于运行时最小值和最大值计算量化参数的观察模块。

    这个观察器使用张量的最小值和最大值统计信息来计算量化参数。该模块记录传入张量的运行时最小值和最大值，并使用这些统计信息来计算量化参数。

    Args:
        dtype: `quantize` 节点所需的 dtype 参数，用于实现参考模型规范。
        qscheme: 要使用的量化方案
        reduce_range: 将量化数据类型的范围减少 1 位
        quant_min: 最小量化值。如果未指定，则将遵循 8 位设置。
        quant_max: 最大量化值。如果未指定，则将遵循 8 位设置。
        eps: float32 的 epsilon 值，默认为 `torch.finfo(torch.float32).eps`。

    给定运行时最小值和最大值 :math:`x_\text{min}` 和 :math:`x_\text{max}`，
    规定 :math:`s` 和 :math:`z` 的计算方式如下：

    运行时最小值/最大值 :math:`x_\text{min/max}` 计算如下：

    .. math::

        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}

    其中 :math:`X` 是观察到的张量。

    然后，规定 :math:`s` 和 :math:`z` 的计算方式如下：

    .. math::

        \begin{aligned}
            \text{如果是对称的:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{如果 dtype 是 qint8} \\
                128 & \text{否则}
            \end{cases}\\
            \text{否则:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    其中 :math:`Q_\text{min}` 和 :math:`Q_\text{max}` 是量化数据类型的最小值和最大值。

    .. warning:: :attr:`dtype` 只能使用 ``torch.qint8`` 或 ``torch.quint8``。

    .. note:: 如果运行时最小值等于运行时最大值，则将 scale 和 zero_point 设置为 1.0 和 0。
    """
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "MinMaxObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # TODO: MinMaxObserver本身不支持动态量化，但如果被MovingAverageObserver继承，并且averaging_constant为1，
        # 则支持动态量化，这里可能需要更好的错误检查。

        # 对于x86量化内核，需要确保vpmaddubsw指令不会溢出。我们允许reduce_range参数将量化范围缩小到(0,127)或(-64,63)。
        # 更多细节请参见aten/src/ATen/native/quantized/cpu/qconv.cpp
        # 对于非x86后端来说，这不是一个最佳选择，因为它会损失激活的一点精度。
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        # 注册缓冲区，用于存储最小值和最大值，默认分别为正无穷和负无穷
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        r"""记录``x``的运行时最小值和最大值。"""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # 避免保留自动求导记录
        x = x.to(self.min_val.dtype)
        # 计算x的最小值和最大值
        min_val_cur, max_val_cur = torch.aminmax(x)
        # 更新最小值和最大值
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""计算量化参数。"""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # 将最小值张量复制为正无穷大的张量
        self.min_val.copy_(torch.tensor(float("inf")))
        # 将最大值张量复制为负无穷大的张量
        self.max_val.copy_(torch.tensor(float("-inf")))
# 定义一个名为 MovingAverageMinMaxObserver 的类，继承自 MinMaxObserver 类
class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
    
    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    # 初始化函数，设置类的各种属性和参数
    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ):
    ) -> None:
        # 如果量化方案不是 torch.per_tensor_symmetric 或 torch.per_tensor_affine，则抛出 NotImplementedError 异常
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                f"MovingAverageMinMaxObserver's qscheme only support \
                torch.per_tensor_symmetric and torch.per_tensor_affine. \
                but got: {qscheme}"
            )
        
        # 设置移动平均常数
        self.averaging_constant = averaging_constant
        
        # 如果是动态量化且移动平均常数不为1，则抛出 NotImplementedError 异常
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )
        
        # 调用父类构造函数，传入参数
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )

    def forward(self, x_orig):
        # 如果输入张量 x_orig 的元素个数为0，则直接返回 x_orig
        if x_orig.numel() == 0:
            return x_orig
        
        # 对输入张量 x_orig 进行 detach 操作，避免保留 autograd tape
        x = x_orig.detach()
        
        # 将 x 转换为与 min_val 相同的数据类型
        x = x.to(self.min_val.dtype)
        
        # 将当前的 min_val 和 max_val 赋给局部变量 min_val 和 max_val
        min_val = self.min_val
        max_val = self.max_val
        
        # 如果当前的 min_val 和 max_val 都是无穷大和无穷小，则计算 x 的全局最小值和最大值
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            # 否则，计算当前 x 的最小值和最大值
            min_val_cur, max_val_cur = torch.aminmax(x)
            
            # 根据移动平均常数更新 min_val 和 max_val
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        
        # 将更新后的 min_val 复制给 self.min_val
        self.min_val.copy_(min_val)
        
        # 将更新后的 max_val 复制给 self.max_val
        self.max_val.copy_(max_val)
        
        # 返回原始输入张量 x_orig
        return x_orig
# 定义一个类，用于计算基于每个通道的运行最小和最大值的量化参数的观察器模块
class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis 通道轴的索引
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec. quantize节点所需的数据类型参数，用于实现参考模型规范
        qscheme: Quantization scheme to be used 要使用的量化方案
        reduce_range: Reduces the range of the quantized data type by 1 bit
                      减少量化数据类型范围1位
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
                   最小量化值。如果未指定，则将遵循8位设置。
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
                   最大量化值。如果未指定，则将遵循8位设置。
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
             float32的epsilon值，默认为`torch.finfo(torch.float32).eps`。

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    # 用于存储每个通道的运行最小值的张量
    min_val: torch.Tensor
    # 用于存储每个通道的运行最大值的张量
    max_val: torch.Tensor

    # 初始化函数
    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ):
    ) -> None:
        # 如果量化方案不是按通道的，抛出未实现错误
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        # 如果是动态量化，抛出未实现错误
        if is_dynamic:
            raise NotImplementedError(
                "PerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        # 调用父类的初始化方法，设置量化器的属性
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        # 使用工厂参数初始化工厂关键字
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        # 设置通道轴
        self.ch_axis = ch_axis
        # 注册缓冲区，存储最小值和最大值的张量
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        # 如果是对称量化且需要减少范围，且数据类型是quint8，抛出未实现错误
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        # 调用内部的_forward方法处理前向传播
        return self._forward(x_orig)

    def _forward(self, x_orig):
        # 如果输入张量为空，则直接返回
        if x_orig.numel() == 0:
            return x_orig
        # 分离输入张量，避免保留自动求导信息
        x = x_orig.detach()
        # 获取当前存储的最小值和最大值
        min_val = self.min_val
        max_val = self.max_val
        # 获取输入张量的维度
        x_dim = x.size()

        # 创建新的轴列表，用于重新排列张量
        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        
        # 需要保持最小值和最大值的数据类型一致，因为缓冲区的更新是原地进行的
        y = y.to(self.min_val.dtype)
        
        # 将张量展平，从第一个维度开始
        y = torch.flatten(y, start_dim=1)
        
        # 如果最小值或最大值的缓冲区为空，计算y的最小值和最大值
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            # 否则，计算当前y的最小值和最大值，并更新最小值和最大值
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        
        # 调整最小值和最大值的缓冲区大小，并复制计算得到的最小值和最大值
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        
        # 返回原始输入张量
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        # 调用内部方法计算量化参数
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        # 返回额外的描述信息，包括最小值和最大值
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        # 从本地元数据中获取版本号，若存在且小于3，则使用"min_vals"和"max_vals"作为状态变量名
        version = local_metadata.get("version", None)
        if version is not None and version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            # 否则使用"min_val"和"max_val"作为状态变量名
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        # 遍历状态变量名列表
        for name in local_state:
            # 构建状态字典中的键
            key = prefix + name
            # 若键存在于状态字典中
            if key in state_dict:
                # 获取对应的值
                val = state_dict[key]
                # 自定义处理方式，允许加载大小为N的min_val或max_val到大小为0的未初始化缓冲区中
                # 这里进行缓冲区的重新调整大小，并复制值到父类的默认状态字典加载代码中
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    # 若遇到未预期的状态变量名，则发出警告
                    warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
                # 对于TorchScript模块，由于不调用模块.py中定义的_load_from_state_dict函数，
                # 在此需要更新属性
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        # 若遇到未预期的状态变量名，则发出警告
                        warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
            elif strict:
                # 若启用严格模式并且键不存在于状态字典中，则将其添加到缺失键列表中
                missing_keys.append(key)

        # 若非TorchScript模式，则调用父类的_load_from_state_dict函数加载状态字典
        if not torch.jit.is_scripting():
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        # 调用内部_load_from_state_dict函数，用于加载状态字典（适用于TorchScript）
        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """重置最小值和最大值。"""
        # 这里曾经使用torch.ones，但由于JIT编译器可以通过常见子表达式消除进行优化，
        # 在这种情况下，min_val和max_val指向同一个张量。
        # 因此改为使用torch.rand创建一个空张量作为最小值和最大值的新值。
        self.min_val = torch.rand(0, )
        self.max_val = torch.rand(0, )
# 定义一个继承自PerChannelMinMaxObserver的观察器类，用于计算基于运行时每通道最小值和最大值的量化参数
class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max. 用于计算运行时最小值和最大值的平均常数
        ch_axis: Channel axis 通道轴
        dtype: Quantized data type 量化后的数据类型
        qscheme: Quantization scheme to be used 量化方案
        reduce_range: Reduces the range of the quantized data type by 1 bit 减少量化数据类型的范围 1 位
        quant_min: Minimum quantization value 最小量化值
        quant_max: Maximum quantization value 最大量化值
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`. 浮点数32位的epsilon值，默认为torch.float32的eps值

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    # 初始化方法，设置观察器的参数
    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        # 如果量化方案不是每通道的，则抛出未实现的错误
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        # 如果是动态量化，则抛出未实现的错误
        if is_dynamic:
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        # 调用父类的初始化方法，传入参数
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )
        # 设置平均常数
        self.averaging_constant = averaging_constant
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            # 如果输入张量为空，直接返回空张量
            return x_orig
        x = x_orig.detach()  # 避免保留自动求导的记录
        x = x.to(self.min_val.dtype)  # 将张量转换为与 self.min_val 相同的数据类型
        min_val = self.min_val  # 获取当前对象的最小值
        max_val = self.max_val  # 获取当前对象的最大值
        x_dim = x.size()  # 获取输入张量的维度信息

        # 创建一个新的轴顺序列表，用于重新排列张量的轴
        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0  # 将通道轴移动到第一个位置
        new_axis_list[0] = self.ch_axis  # 将第一个位置的轴移动到通道轴位置
        y = x.permute(new_axis_list)  # 根据新的轴顺序重新排列输入张量
        y = torch.flatten(y, start_dim=1)  # 将张量展平，从第一个维度开始

        if min_val.numel() == 0 or max_val.numel() == 0:
            # 如果最小值或最大值为空，使用 torch.aminmax 计算 y 的最小值和最大值
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            # 否则，计算当前 y 的最小值和最大值，并进行平均值调整
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)

        # 调整 self.min_val 和 self.max_val 的形状，并复制计算得到的最小值和最大值
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        # 返回原始输入张量 x_orig
        return x_orig
# 定义一个类 HistogramObserver，继承自 UniformQuantizationObserverBase
# 该类用于记录张量值的运行直方图以及最小/最大值，并且能够计算 scale 和 zero_point。
class HistogramObserver(UniformQuantizationObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
               直方图使用的 bin 的数量
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations
                       直方图的上采样因子，用于在观察中插值具有不同范围的直方图
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
               用于实现参考模型规范的 quantize 节点的 dtype 参数
        qscheme: Quantization scheme to be used
                 要使用的量化方案
        reduce_range: Reduces the range of the quantized data type by 1 bit
                      将量化数据类型的范围减少 1 位
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
             float32 的 epsilon 值，默认为 `torch.finfo(torch.float32).eps`。

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
       创建传入输入的直方图。
       The histogram is computed continuously, and the ranges per bin change
       with every new tensor observed.
       直方图持续计算，并且每次观察到新张量时，每个 bin 的范围都会更改。
    2. Search the distribution in the histogram for optimal min/max values.
       在直方图中搜索最优的 min/max 值的分布。
       The search for the min/max values ensures the minimization of the
       quantization error with respect to the floating point model.
       搜索 min/max 值确保在浮点模型的量化误差最小化。

    3. Compute the scale and zero point the same way as in the
       :class:`~torch.ao.quantization.MinMaxObserver`
       以与 torch.ao.quantization.MinMaxObserver 类相同的方式计算 scale 和 zero point。
    """
    # 定义类成员变量
    histogram: torch.Tensor  # 用于记录张量值的运行直方图
    min_val: torch.Tensor    # 记录最小值的张量
    max_val: torch.Tensor    # 记录最大值的张量

    # 初始化函数
    def __init__(
        self,
        bins: int = 2048,                   # 直方图的 bin 数量，默认为 2048
        upsample_rate: int = 128,           # 直方图的上采样率，默认为 128
        dtype: torch.dtype = torch.quint8,  # 用于 quantize 节点的数据类型，默认为 torch.quint8
        qscheme=torch.per_tensor_affine,    # 要使用的量化方案，默认为 torch.per_tensor_affine
        reduce_range=False,                 # 是否减少量化数据类型的范围，默认为 False
        quant_min=None,                     # 量化的最小值，默认为 None
        quant_max=None,                     # 量化的最大值，默认为 None
        factory_kwargs=None,                # 工厂参数，默认为 None
        eps=torch.finfo(torch.float32).eps, # float32 的 epsilon 值，默认为 torch.float32 的 eps
        is_dynamic=False,                   # 是否为动态量化，默认为 False
        **kwargs,                           # 其他关键字参数
    ):
    ) -> None:
        # 如果量化方案不是 torch.per_tensor_symmetric 或 torch.per_tensor_affine，则抛出未实现错误
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # 如果是动态量化，则抛出未实现错误
        if is_dynamic:
            raise NotImplementedError(
                "HistogramObserver doesn't support dynamic quantization"
            )
        # 调用父类构造函数，初始化观察器
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )
        # 将 factory_kwargs 转换为 torch.nn.factory_kwargs 的形式
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        # 设置直方图的分 bin 数量
        self.bins = bins
        # 注册缓冲区：直方图，初始为全零
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        # 注册缓冲区：最小值，初始为正无穷
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        # 注册缓冲区：最大值，初始为负无穷
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        # 目标 nbins 设为 dtype 的位数的 2 次方
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        # 设置上采样率
        self.upsample_rate = upsample_rate

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        计算在 delta_begin 和 delta_end 之间均匀分布值的范数。
        目前仅支持 L2 范数计算。

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        # 计算范数，使用三次方的差除以 3
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        # 返回密度乘以范数
        return density * norm
    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        # 计算每个 bin 的宽度
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        # 计算目标 bin 的宽度
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        # 创建一个张量，表示源 bin 的索引
        src_bin = torch.arange(self.bins, device=self.histogram.device)

        # 计算每个源 bin 开始和结束位置到第一个目标 bin 开始和结束位置的距离
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # 确定每个源 bin 开始和结束位置所属的目标 bin
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )

        # 计算密度
        density = self.histogram / bin_width

        # 初始化 norm 张量
        norm = torch.zeros(self.bins, device=self.histogram.device)

        # 计算第一部分的 norm
        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        # 计算第二部分的 norm
        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        # 计算第三部分的 norm
        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        # 返回 norm 的总和
        return norm.sum().item()
    # 非线性参数搜索函数，返回一个包含新的最小值和最大值的元组
    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        # 确保直方图的大小与指定的分箱数相符
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        # 计算每个分箱的宽度
        bin_width = (self.max_val - self.min_val) / self.bins

        # 计算直方图数据的总和
        total = torch.sum(self.histogram).item()
        # 计算直方图数据的累积和
        cSum = torch.cumsum(self.histogram, dim=0)

        # 设置步长和初始边界
        stepsize = 1e-5  # 粒度
        alpha = 0.0  # 下界
        beta = 1.0  # 上界
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        # 开始非线性搜索过程
        while alpha < beta:
            # 计算下一个步长
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # 在量化边界之间找到左右分箱
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # 决定下一步的移动
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # 移动开始分箱
                next_start_bin = l
                alpha = next_alpha
            else:
                # 移动结束分箱
                next_end_bin = r
                beta = next_beta

            # 如果没有移动分箱，则继续循环
            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # 使用下一个开始和结束分箱计算量化误差
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            # 如果误差增大，则结束循环
            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        # 根据最终的开始和结束分箱计算新的最小值和最大值
        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # 确保下采样率与上采样率及直方图分箱数的比例关系为：
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # 这使我们可以使用分辨率s的公共网格来对齐输入直方图
        # start_idx将min_val映射到直方图分箱的索引位置

        # 计算直方图分箱的宽度，确保避免分母接近FP32的最小正子正常数导致的下溢
        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = int(
            torch.ceil(
                ((combined_max - combined_min) / (self.max_val - self.min_val)) * upsample_rate
            ).item()
        )
        # 计算e，用于调整combined_max和combined_min以匹配目标范围
        e = downsample_rate / upsample_rate * (self.max_val - self.min_val) - (combined_max - combined_min)
        # 计算start_idx，表示原始直方图中开始插入新数据的位置
        start_idx = int(
            torch.round((self.min_val - combined_min) / (self.max_val - self.min_val) * self.bins * upsample_rate).item()
        )
        combined_max = combined_max + e
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        new_hist: torch.Tensor,
        upsample_rate: int,
        downsample_rate: int,
        start_idx: int,
        Nbins: int,
    ) -> torch.Tensor:
        # 首先对新数据的直方图进行上采样，上采样因子为L
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # 将上采样后的直方图插入到初始化为零的输出直方图中
        # 插入位置由start_idx确定，因为输出直方图可以覆盖更广的范围
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # 计算积分直方图，需要双精度以确保没有溢出
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1 :: downsample_rate]
        # 最后执行插值操作
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        # 将插值后的直方图加到原始直方图中并返回
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist
    # 重置直方图的状态，设置最小和最大值
    def reset_histogram(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        # 将 self.min_val 重新调整为与给定 min_val 相同的形状，并复制其值
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        # 将 self.max_val 重新调整为与给定 max_val 相同的形状，并复制其值
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        # 确保 min_val 和 max_val 是标量
        assert (
            min_val.numel() == 1 and max_val.numel() == 1
        ), "histogram min/max values must be scalar."
        # 计算 x 的直方图，并将结果存储在 self.histogram 中
        torch.histc(
            x, self.bins, min=min_val, max=max_val, out=self.histogram  # type: ignore[arg-type]
        )

    @torch.jit.export
    # 计算量化参数
    def calculate_qparams(self):
        # 检查 min_val 和 max_val 是否未初始化
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            # 若未初始化，发出警告并返回默认的量化参数
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0], device=self.min_val.device.type)
        # 确保直方图的 bins 数量与 self.histogram 的长度相等
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        # 使用非线性参数搜索方法寻找新的最小值和最大值
        new_min, new_max = self._non_linear_param_search()

        # 计算并返回量化参数
        return self._calculate_qparams(new_min, new_max)

    # 将对象状态保存到 state_dict 中
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # 保存 min_val 和 max_val 到 state_dict
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    # 从 state_dict 加载对象状态
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
        # 获取本地元数据中的版本信息
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # 如果 min_val 和 max_val 未初始化，更新其形状以适应 v2 和 v3 之间的差异
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        # 加载本地状态（min_val 和 max_val）到当前对象中
        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        # 调用父类方法加载剩余的状态
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    # 定义一个方法 `extra_repr`，用于返回对象的字符串表示形式，包括最小值和最大值
    def extra_repr(self):
        # 使用 f-string 格式化字符串，将最小值和最大值嵌入返回的字符串中
        return f"min_val={self.min_val}, max_val={self.max_val}"
class FixedQParamsObserver(ObserverBase):
    r"""
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """

    scale: torch.Tensor  # 用于存储固定的量化比例因子的张量
    zero_point: torch.Tensor  # 用于存储固定的零点的张量

    def __init__(
        self,
        scale,
        zero_point,
        dtype=torch.quint8,  # 默认量化的数据类型为 torch.quint8
        qscheme=torch.per_tensor_affine,  # 默认量化方案为 torch.per_tensor_affine
        quant_min=0,  # 默认的量化最小值为 0
        quant_max=255,  # 默认的量化最大值为 255
        is_dynamic=False,  # 是否是动态量化，默认为 False
        **kwargs,
    ):
        if is_dynamic:
            raise NotImplementedError(
                "FixedQParamsObserver doesn't support dynamic quantization"
            )
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)  # 调用父类的构造函数
        self.quant_min = quant_min  # 设置量化的最小值
        self.quant_max = quant_max  # 设置量化的最大值
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float))  # 注册固定的量化比例因子为缓冲区
        self.register_buffer('zero_point', torch.tensor([zero_point], dtype=torch.int))  # 注册固定的零点为缓冲区
        self.dtype = dtype  # 设置数据类型
        self.qscheme = qscheme  # 设置量化方案

    def forward(self, X):
        return X  # 简单地返回输入 X

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point  # 返回当前的量化参数比例因子和零点值


class PlaceholderObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """

    def __init__(
        self, dtype=torch.float32, custom_op_name="", compute_dtype=None,
        quant_min=None, quant_max=None, qscheme=None, eps=None,
        is_dynamic=False,
    ):
        # 这个观察器不执行任何操作，只是将其配置传递给量化模块的 `from_float()` 方法

        super().__init__(dtype=dtype, is_dynamic=is_dynamic, qscheme=qscheme)  # 调用父类的构造函数，传递 dtype, is_dynamic, qscheme

        # 下面是一些参数的初始化和设置
        self.quant_min = quant_min  # 设置量化的最小值
        self.quant_max = quant_max  # 设置量化的最大值
        self.custom_op_name = custom_op_name  # 设置自定义操作名称
        self.compute_dtype = compute_dtype  # 设置计算数据类型（已弃用，推荐使用 is_dynamic=True）
        self.is_dynamic = is_dynamic  # 是否是动态量化
    ) -> None:
        # 调用父类的构造函数，设置数据类型和是否动态的标志
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        # 如果量化方案未指定，则默认为每张量的仿射量化
        if qscheme is None:
            qscheme = torch.per_tensor_affine
        # 如果 eps 未指定，则使用 float32 类型的机器精度
        if eps is None:
            eps = torch.finfo(torch.float32).eps

        # 输入目标运算符的数据类型，例如动态量化操作的数据类型将是 float32
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.custom_op = custom_op_name
        # 用于动态量化计算类型的配置
        if compute_dtype:
            is_dynamic = True
            # 警告提示，建议使用 `is_dynamic` 替代 `compute_dtype`，后者将在 PyTorch 的未来版本中弃用
            warnings.warn(
                "Please use `is_dynamic` instead of `compute_dtype`. \
                    `compute_dtype` will be deprecated in a future release \
                    of PyTorch."
            )

    def forward(self, x):
        # 前向传播函数，直接返回输入张量 x
        return x

    @torch.jit.export
    def extra_repr(self):
        # 返回对象的额外字符串表示形式，包括数据类型和是否动态量化的信息
        return f"dtype={self.dtype}, is_dynamic={self.is_dynamic}"

    @torch.jit.export
    def calculate_qparams(self):
        # 抛出异常，表示 PlaceholderObserver 类型不应调用 calculate_qparams 函数
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for PlaceholderObserver"
        )
class RecordingObserver(ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """
    # 定义类变量 __annotations__，指定 tensor_val 属性为包含可选的 torch.Tensor 的列表
    __annotations__ = {"tensor_val": List[Optional[torch.Tensor]]}

    def __init__(self, dtype=torch.quint8):
        # 调用父类的构造函数，初始化 dtype，并设置 is_dynamic 为 False
        super().__init__(dtype=dtype, is_dynamic=False)  # type: ignore[call-arg]
        # 初始化 tensor_val 属性为空列表
        self.tensor_val = []

    def forward(self, x):
        # 将输入 x 的克隆添加到 tensor_val 列表中
        self.tensor_val.append(x.clone())
        # 返回输入 x 本身
        return x

    @torch.jit.export
    def calculate_qparams(self):
        # 抛出异常，指示不应调用此函数
        raise Exception("calculate_qparams should not be called for RecordingObserver")  # noqa: TRY002

    @torch.jit.export
    def get_tensor_value(self):
        # 返回 tensor_val 属性，记录的所有张量值
        return self.tensor_val


class NoopObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """

    def __init__(self, dtype=torch.float16, custom_op_name="") -> None:
        # 调用父类的构造函数，初始化 dtype 和 is_dynamic 属性
        super().__init__(dtype=dtype, is_dynamic=False)
        # 初始化自定义属性 dtype 和 custom_op
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        # 返回输入 x 本身，不进行任何操作
        return x

    @torch.jit.export
    def calculate_qparams(self):
        # 抛出异常，指示不应调用此函数
        raise Exception("calculate_qparams should not be called for NoopObserver")  # noqa: TRY002

class ReuseInputObserver(ObserverBase):
    r""" This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```py
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """
    def __init__(self):
        # 调用父类的构造函数，初始化 dtype 和 is_dynamic 属性
        super().__init__(torch.quint8, is_dynamic=False)

    def forward(self, x):
        # 返回输入 x 本身，不进行任何操作
        return x

    @torch.jit.export
    def calculate_qparams(self):
        # 抛出异常，指示不应调用此函数
        raise Exception("calculate_qparams should not be called for ReuseInputObserver")  # noqa: TRY002

def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    # 检查给定的模块是否是 torch.jit.RecursiveScriptModule 类型的实例
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # 获取模块的限定名，形式类似 '__torch__.torch.ao.quantization.observer.___torch_mangle_2.MinMaxObserver'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        # 移除限定名中的 '.___torch_mangle_\d+' 形式的字符串，保留观察器类型的名称
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        # 检查观察器类型名称是否在处理后的模块名称中
        return obs_type_name in name
    # 如果模块不是 torch.jit.RecursiveScriptModule 类型的实例，则返回 False
    return False
def _is_activation_post_process(module):
    """
    Check if the module is an activation post-process module for quantization.
    It checks if the module is an instance of ObserverBase or FakeQuantizeBase
    from torch.ao.quantization, or if it matches a specific observer script module.
    """
    return (
        isinstance(module, (torch.ao.quantization.ObserverBase,
                            torch.ao.quantization.FakeQuantizeBase)) or _is_observer_script_module(module, "quantization.observer")
    )


def _is_per_channel_script_obs_instance(module):
    """
    Check if the module is an instance of per-channel observer script modules.
    Specifically checks for PerChannelMinMaxObserver and MovingAveragePerChannelMinMaxObserver
    under the quantization.observer namespace.
    """
    if isinstance(module, torch.jit.RecursiveScriptModule):
        return _is_observer_script_module(
            module, "quantization.observer.PerChannelMinMaxObserver"
        ) or _is_observer_script_module(
            module, "quantization.observer.MovingAveragePerChannelMinMaxObserver"
        )
    return False


def get_observer_state_dict(mod):
    """
    Returns the state dict corresponding to the observer stats of the given module.
    It traverses the model's state_dict and extracts observer-related stats.
    """
    od = OrderedDict()
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        for k, v in mod.state_dict().items():
            if "observer" in k:
                od[k] = v
    else:
        # path for GraphModule and nn.Module (eager mode)
        for k, v in mod.state_dict().items():
            if "activation_post_process" in k:
                od[k] = v
    od._metadata = mod.state_dict()._metadata  # type: ignore[attr-defined]
    return od


def load_observer_state_dict(mod, obs_dict):
    """
    Given an input model and a state_dict containing observer stats,
    loads the stats back into the model.
    """
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    for name, module in mod.named_modules():
        prefix = name + "."
        if _is_activation_post_process(module):
            if _is_per_channel_script_obs_instance(module):
                # For per-channel observers, custom load_from_state_dict is called to resize tensors.
                # This is particularly needed when the module is scripted.
                module._load_from_state_dict_script(
                    obs_dict, prefix, {}, True, missing_keys, unexpected_keys, []
                )
            else:
                module._load_from_state_dict(
                    obs_dict, prefix, {}, False, missing_keys, unexpected_keys, []
                )
    for k in missing_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(f"Missing keys for observer {k} in state_dict")  # noqa: TRY002
    for k in unexpected_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(f"Unexpected keys for observer {k} in state_dict")  # noqa: TRY002


# Restrict activations to be in the range (0,127)
default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
"""
Default observer for static quantization, usually used for debugging.
"""

default_placeholder_observer = PlaceholderObserver
"""
Placeholder observer used for static quantization.
"""
# 默认的调试观察器，通常用于将量化设置为 torch.float16。
default_debug_observer = RecordingObserver

# 默认的仅用于调试的观察器。
default_weight_observer = MinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
)

# 默认的权重观察器，使用 torch.qint8 数据类型，采用 torch.per_tensor_symmetric 量化方案。
# 观察范围为 [-127, +127] 的对称量化，不包括 -128。
weight_observer_range_neg_127_to_127 = MinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
    quant_min=-127, quant_max=127, eps=2 ** -12)

# 默认的直方图观察器，通常用于 PTQ（Post Training Quantization）。
default_histogram_observer = HistogramObserver.with_args(quant_min=0, quant_max=127)

# 默认的逐通道权重观察器，通常用于支持逐通道权重量化的后端，如 `fbgemm`。
default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)

# 逐通道对称权重观察器，使用 torch.qint8 数据类型，采用 torch.per_channel_symmetric 量化方案。
# 观察范围为 [-127, +127] 的对称量化，不包括 -128。
per_channel_weight_observer_range_neg_127_to_127 = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
    quant_min=-127, quant_max=127, eps=2 ** -12)

# 默认的动态量化观察器，用于动态量化。
default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=torch.quint8, quant_min=0, quant_max=255, is_dynamic=True,
)

# 默认的浮点数量化参数观察器，使用 torch.quint8 数据类型，采用 torch.per_channel_affine_float_qparams 量化方案。
default_float_qparams_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)

# 默认的浮点数量化参数观察器，使用 torch.quint4x2 数据类型，采用 torch.per_channel_affine_float_qparams 量化方案。
default_float_qparams_observer_4bit = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint4x2, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)

# TODO: 在将来的 PR 中删除这些默认值，并强制激活函数明确指定其输出范围。
# 固定量化参数操作的默认观察器，量化范围为 [-1, 1]。
default_fixed_qparams_range_neg1to1_observer = FixedQParamsObserver.with_args(
    scale=2.0 / 256.0, zero_point=128, dtype=torch.quint8, quant_min=0, quant_max=255)

# 固定量化参数操作的默认观察器，量化范围为 [0, 1]。
default_fixed_qparams_range_0to1_observer = FixedQParamsObserver.with_args(
    scale=1.0 / 256.0, zero_point=0, dtype=torch.quint8, quant_min=0, quant_max=255)

# 以下两个变量保留了一段时间以保持向后兼容性，在几个发布版本后移除。
# 默认的对称固定量化参数观察器，等同于 default_fixed_qparams_range_neg1to1_observer。
default_symmetric_fixed_qparams_observer = default_fixed_qparams_range_neg1to1_observer

# 默认的仿射固定量化参数观察器，等同于 default_fixed_qparams_range_0to1_observer。
default_affine_fixed_qparams_observer = default_fixed_qparams_range_0to1_observer

# 默认的重复使用输入观察器，用于像 reshape 这样的操作符，重用操作符输入的观察器。
default_reuse_input_observer = ReuseInputObserver
```