# `.\pytorch\torch\nn\modules\linear.py`

```py
# mypy: allow-untyped-defs
# 引入数学库
import math
# 引入类型提示
from typing import Any

# 引入 PyTorch 库
import torch
# 引入 Tensor 类型
from torch import Tensor
# 引入 torch.nn.functional 别名 F，引入初始化模块
from torch.nn import functional as F, init
# 引入参数和未初始化参数类
from torch.nn.parameter import Parameter, UninitializedParameter

# 引入 LazyModuleMixin 和 Module 类
from .lazy import LazyModuleMixin
from .module import Module


# 导出的模块名列表
__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]


# 定义 Identity 类，继承自 Module 类
class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    # 构造方法
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    # 前向传播方法
    def forward(self, input: Tensor) -> Tensor:
        return input


# 定义 Linear 类，继承自 Module 类
class Linear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # 常量列表，用于序列化
    __constants__ = ["in_features", "out_features"]
    # 输入特征数和输出特征数
    in_features: int
    out_features: int
    # 权重张量
    weight: Tensor

    # 构造方法
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # 允许未知类型定义
        **kwargs: Any
    ) -> None:
        super().__init__()
    ) -> None:
        # 定义构造函数，初始化线性层对象
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的构造函数
        super().__init__()
        # 设置输入特征数和输出特征数
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重参数，形状为(out_features, in_features)
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        # 如果需要偏置，则初始化偏置参数
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            # 否则注册偏置参数为None
            self.register_parameter("bias", None)
        # 调用重置参数的方法
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Kaiming 均匀初始化权重参数
        # 其中 a=sqrt(5) 等同于 uniform(-1/sqrt(in_features), 1/sqrt(in_features)) 初始化
        # 更多详情参见 https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 如果存在偏置参数，则初始化偏置参数
        if self.bias is not None:
            # 计算权重参数的 fan_in
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            # 计算均匀分布的边界
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # 使用均匀分布初始化偏置参数
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播方法，使用 F.linear 计算线性层输出
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        # 返回对象的额外描述信息，包括输入特征数、输出特征数和是否有偏置
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
# 该类存在的唯一目的是避免在使用不正确量化的注意力层脚本时触发一个不明确的错误。
# 详细信息请参见此问题：https://github.com/pytorch/pytorch/issues/58969
# TODO: 在量化API使用错误时快速失败，然后移除此类并用普通的 Linear 类替换其使用

class NonDynamicallyQuantizableLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # 调用父类的构造方法初始化线性层，设置输入特征数、输出特征数、是否包含偏置、设备和数据类型
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )


class Bilinear(Module):
    r"""对输入数据应用双线性变换： :math:`y = x_1^T A x_2 + b`。

    Args:
        in1_features: 第一个输入样本的大小
        in2_features: 第二个输入样本的大小
        out_features: 每个输出样本的大小
        bias: 如果设置为 False，则该层将不会学习一个可加的偏置。
            默认值：``True``

    Shape:
        - Input1: :math:`(*, H_{in1})`，其中 :math:`H_{in1}=\text{in1\_features}`，并且
          :math:`*` 表示任意数量的附加维度，包括没有。除了最后一个维度外，输入的所有维度应该是相同的。
        - Input2: :math:`(*, H_{in2})`，其中 :math:`H_{in2}=\text{in2\_features}`。
        - Output: :math:`(*, H_{out})`，其中 :math:`H_{out}=\text{out\_features}`，
          并且除了最后一个维度外，所有维度与输入的形状相同。

    Attributes:
        weight: 模块的可学习权重，形状为
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`。
            这些值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 初始化，其中
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   模块的可学习偏置，形状为 :math:`(\text{out\_features})`。
                如果 :attr:`bias` 是 ``True``，这些值从
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 初始化，其中
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """

    __constants__ = ["in1_features", "in2_features", "out_features"]
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # 定义初始化函数，设定初始参数，包括设备和数据类型
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类初始化方法
        super().__init__()
        # 设置输入特征、输出特征、输出特征数目
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        # 创建权重参数，并初始化为指定形状的张量
        self.weight = Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )

        # 如果启用偏置，则创建偏置参数并初始化为指定形状的张量，否则将偏置参数注册为 None
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        # 调用重置参数方法，初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 根据权重的第二个维度的大小计算均匀分布的边界
        bound = 1 / math.sqrt(self.weight.size(1))
        # 使用均匀分布初始化权重参数
        init.uniform_(self.weight, -bound, bound)
        # 如果存在偏置参数，则使用均匀分布初始化偏置参数
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # 执行双线性操作，计算输入特征对应的输出张量
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        # 返回描述该层结构的字符串，包括输入特征数、输出特征数、是否有偏置
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
# 定义了一个继承自 LazyModuleMixin 和 Linear 的类 LazyLinear
class LazyLinear(LazyModuleMixin, Linear):
    r"""A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.
    """

    # 类属性，指示该类在初始化完成后将变为 Linear 类型
    cls_to_become = Linear  # type: ignore[assignment]
    # 权重参数，使用 UninitializedParameter 类型
    weight: UninitializedParameter
    # 偏置参数，使用 UninitializedParameter 类型
    bias: UninitializedParameter  # type: ignore[assignment]

    # 初始化方法
    def __init__(
        self, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        # 初始化参数工厂的关键字参数
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法，创建一个临时 Linear 对象，bias 被硬编码为 False 避免创建即将被覆盖的张量
        super().__init__(0, 0, False)
        # 初始化权重参数为 UninitializedParameter 对象
        self.weight = UninitializedParameter(**factory_kwargs)
        # 设置输出特征数
        self.out_features = out_features
        # 如果 bias 为 True，则初始化偏置参数为 UninitializedParameter 对象
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    # 重置参数的方法
    def reset_parameters(self) -> None:
        # 如果没有未初始化的参数并且输入特征数不为 0，则调用父类的重置参数方法
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    # 初始化参数的方法，根据输入的形状进行初始化
    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        # 如果存在未初始化的参数
        if self.has_uninitialized_params():
            with torch.no_grad():
                # 推断输入特征数
                self.in_features = input.shape[-1]
                # 材料化权重参数为指定形状的张量
                self.weight.materialize((self.out_features, self.in_features))
                # 如果存在偏置参数，则材料化偏置参数为指定形状的张量
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                # 重置参数
                self.reset_parameters()


# TODO: PartialLinear - maybe in sparse?
```