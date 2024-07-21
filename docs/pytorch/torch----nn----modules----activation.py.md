# `.\pytorch\torch\nn\modules\activation.py`

```py
# mypy: allow-untyped-defs
# 导入警告模块，用于处理警告信息
import warnings
# 导入类型提示模块
from typing import Optional, Tuple

# 导入PyTorch相关模块
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter

# 导入自定义模块
from .linear import NonDynamicallyQuantizableLinear
from .module import Module

# 定义公开的模块列表
__all__ = [
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]

# 阈值函数模块，继承自Module类
class Threshold(Module):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["threshold", "value", "inplace"]

    threshold: float
    value: float
    inplace: bool

    # 初始化函数，设置阈值、替换值和是否原地操作的标志
    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super().__init__()
        self.threshold = threshold  # 设置阈值
        self.value = value  # 设置替换值
        self.inplace = inplace  # 设置是否原地操作的标志
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    # 前向传播函数，对输入Tensor进行阈值处理并返回处理后的Tensor
    def forward(self, input: Tensor) -> Tensor:
        return F.threshold(input, self.threshold, self.value, self.inplace)

    # 返回模块的额外描述信息，包括阈值、替换值和是否原地操作的信息
    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"threshold={self.threshold}, value={self.value}{inplace_str}"


# ReLU激活函数模块，继承自Module类
class ReLU(Module):
    r"""Applies the rectified linear unit function element-wise.

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """

    __constants__ = ["inplace"]
    inplace: bool


    # inplace 是一个布尔型参数，用于指示是否原地修改数据



    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace


    # 初始化函数，设置是否原地修改数据的参数
    def __init__(self, inplace: bool = False):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 inplace 参数赋值给对象的 inplace 属性
        self.inplace = inplace



    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)


    # 前向传播函数，应用 ReLU 激活函数
    def forward(self, input: Tensor) -> Tensor:
        # 调用 PyTorch 的 F.relu 函数，根据 inplace 参数决定是否原地修改数据
        return F.relu(input, inplace=self.inplace)



    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


    # 返回额外的表示信息
    def extra_repr(self) -> str:
        # 根据 inplace 参数生成对应的字符串描述
        inplace_str = "inplace=True" if self.inplace else ""
        # 返回生成的描述信息
        return inplace_str
# 定义一个名为 RReLU 的类，继承自 Module 类
class RReLU(Module):
    # 文档字符串，描述了 RReLU 函数的功能和特性
    r"""Applies the randomized leaky rectified linear unit function, element-wise.

    Method described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})` during training while during
    evaluation :math:`a` is fixed with :math:`a = \frac{\text{lower} + \text{upper}}{2}`.

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/RReLU.png

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    # 定义常量列表，指定需要在序列化时保存的属性名称
    __constants__ = ["lower", "upper", "inplace"]

    # 类属性，定义 RReLU 类的三个属性：lower, upper, inplace
    lower: float
    upper: float
    inplace: bool

    # 构造方法，初始化 RReLU 实例
    def __init__(
        self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False
    ):
        # 调用父类的构造方法
        super().__init__()
        # 设置 RReLU 实例的 lower 属性
        self.lower = lower
        # 设置 RReLU 实例的 upper 属性
        self.upper = upper
        # 设置 RReLU 实例的 inplace 属性
        self.inplace = inplace

    # 前向传播方法，接收输入张量 input，返回经 RReLU 激活后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 rrelu 函数，传递当前实例的 lower, upper, inplace 和当前是否为训练模式的信息
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    # 生成额外描述信息的方法，返回一个字符串，描述了 RReLU 实例的 lower 和 upper 属性及是否原地操作的信息
    def extra_repr(self):
        # 根据 inplace 属性决定是否包含 "inplace=True" 字符串
        inplace_str = ", inplace=True" if self.inplace else ""
        # 返回描述信息字符串
        return f"lower={self.lower}, upper={self.upper}{inplace_str}"


# 定义一个名为 Hardtanh 的类，继承自 Module 类
class Hardtanh(Module):
    # 文档字符串，描述了 HardTanh 函数的功能和特性
    r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 定义常量列表，指定需要在序列化时保存的属性名称
    __constants__ = ["min_val", "max_val", "inplace"]

    # 类属性，定义 Hardtanh 类的三个属性：min_val, max_val, inplace
    min_val: float
    max_val: float
    inplace: bool
    # 定义类的属性：最小值，最大值，是否原地操作的标志
    min_val: float
    max_val: float
    inplace: bool

    # 初始化方法，设置对象的初始属性
    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        
        # 如果使用了旧版参数名min_value，则发出警告，并将其赋值给min_val
        if min_value is not None:
            warnings.warn(
                "keyword argument `min_value` is deprecated and rename to `min_val`",
                FutureWarning,
                stacklevel=2,
            )
            min_val = min_value
        
        # 如果使用了旧版参数名max_value，则发出警告，并将其赋值给max_val
        if max_value is not None:
            warnings.warn(
                "keyword argument `max_value` is deprecated and rename to `max_val`",
                FutureWarning,
                stacklevel=2,
            )
            max_val = max_value

        # 设置对象的属性值
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        
        # 断言：最大值必须大于最小值，否则触发异常
        assert self.max_val > self.min_val

    # 前向传播方法，应用hardtanh函数处理输入数据
    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    # 提供额外的对象表示信息，以字符串形式返回对象的属性
    def extra_repr(self) -> str:
        # 如果原地操作标志为True，则额外表示中包含"inplace=True"
        inplace_str = ", inplace=True" if self.inplace else ""
        # 返回描述对象的属性值的字符串
        return f"min_val={self.min_val}, max_val={self.max_val}{inplace_str}"
class ReLU6(Hardtanh):
    r"""Applies the ReLU6 function element-wise.

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        # 调用父类 Hardtanh 的构造函数，设定 ReLU6 的上下界为 0 和 6
        super().__init__(0.0, 6.0, inplace)

    def extra_repr(self) -> str:
        # 根据 inplace 参数返回描述字符串
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Sigmoid(Module):
    r"""Applies the Sigmoid function element-wise.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # 使用 PyTorch 提供的 sigmoid 函数计算 Sigmoid 激活后的输出
        return torch.sigmoid(input)


class Hardsigmoid(Module):
    r"""Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardsigmoid.png

    Examples::

        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # 调用父类 Module 的构造函数，初始化 inplace 参数
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # 使用 PyTorch 提供的 hardsigmoid 函数计算 Hardsigmoid 激活后的输出
        return F.hardsigmoid(input, self.inplace)


class Tanh(Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # 使用 PyTorch 提供的 tanh 函数计算 Tanh 激活后的输出
        return torch.tanh(input)


class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """



    __constants__ = ["inplace"]
    inplace: bool



    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace



    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)



    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
# 定义一个自定义的 Mish 激活函数，继承自 nn.Module
class Mish(Module):
    r"""Applies the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Mish.png

    Examples::

        >>> m = nn.Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 定义常量列表，这里仅包含一个 inplace 标志位
    __constants__ = ["inplace"]
    # 是否原地操作的标志位
    inplace: bool

    # 初始化方法
    def __init__(self, inplace: bool = False):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 inplace 属性
        self.inplace = inplace

    # 前向传播方法，接收一个输入张量并返回处理后的输出张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.mish 函数，应用 Mish 激活函数
        return F.mish(input, inplace=self.inplace)

    # 返回该模块的额外描述信息的方法
    def extra_repr(self) -> str:
        # 根据 inplace 属性的值生成相应的描述字符串
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


# 定义一个自定义的 Hardswish 激活函数，继承自 nn.Module
class Hardswish(Module):
    r"""Applies the Hardswish function, element-wise.

    Method described in the paper: `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardswish.png

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 定义常量列表，这里仅包含一个 inplace 标志位
    __constants__ = ["inplace"]

    # 是否原地操作的标志位
    inplace: bool

    # 初始化方法
    def __init__(self, inplace: bool = False) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 设置 inplace 属性
        self.inplace = inplace

    # 前向传播方法，接收一个输入张量并返回处理后的输出张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.hardswish 函数，应用 Hardswish 激活函数
        return F.hardswish(input, self.inplace)


# 定义一个自定义的 ELU 激活函数，继承自 nn.Module
class ELU(Module):
    r"""Applies the Exponential Linear Unit (ELU) function, element-wise.

    Method described in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
    Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

    ELU is defined as:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ELU.png
    """

    # 初始化方法略去，因为示例中未提供该部分的代码
    """
    Examples::
    
        >>> m = nn.ELU()  # 创建一个ELU激活函数的实例对象m
        >>> input = torch.randn(2)  # 生成一个形状为(2,)的随机张量input
        >>> output = m(input)  # 使用ELU激活函数对input进行处理得到output
    """
    
    __constants__ = ["alpha", "inplace"]
    alpha: float  # ELU激活函数的参数alpha，控制ELU函数的斜率，初始化为1.0
    inplace: bool  # 表示是否进行原地操作的标志位，初始化为False
    
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()  # 调用父类的构造函数进行初始化
        self.alpha = alpha  # 初始化对象的alpha属性
        self.inplace = inplace  # 初始化对象的inplace属性
    
    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)  # 调用torch.nn.functional.elu函数进行ELU激活操作
    
    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""  # 根据inplace属性确定是否进行原地操作的字符串表示
        return f"alpha={self.alpha}{inplace_str}"  # 返回描述对象状态的字符串，包括alpha参数和是否原地操作的信息
class CELU(Module):
    r"""Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/CELU.png

    Examples::

        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """

    # 定义 CELU 激活函数类，继承自 Module 类
    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        # 初始化方法，设置 alpha 和 inplace 参数
        super().__init__()
        self.alpha = alpha  # 设置 alpha 参数
        self.inplace = inplace  # 设置 inplace 参数

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播方法，调用 F.celu 函数进行计算
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        # 返回额外的表示信息，用于描述初始化参数
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{inplace_str}"


class SELU(Module):
    r"""Applies the SELU function element-wise.

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    .. warning::
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SELU.png

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    # 定义 SELU 激活函数类，继承自 Module 类
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # 初始化方法，设置 inplace 参数
        super().__init__()
        self.inplace = inplace  # 设置 inplace 参数

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播方法，调用 F.selu 函数进行计算
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        # 返回额外的表示信息，用于描述初始化参数
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
    # 定义一个门控线性单元（GLU）模块，用于输入张量的半分部分与经过 sigmoid 函数处理后的张量的按位乘积
    class GLU(nn.Module):
        # 定义常量参数列表，包含 GLU 模块的维度
        __constants__ = ["dim"]
        dim: int
    
        # GLU 模块的初始化函数，设置 GLU 操作的维度，默认为最后一个维度
        def __init__(self, dim: int = -1) -> None:
            super().__init__()
            self.dim = dim
    
        # 前向传播函数，对输入的张量进行 GLU 操作，使用 torch.nn.functional 中的 glu 函数
        def forward(self, input: Tensor) -> Tensor:
            return F.glu(input, self.dim)
    
        # 返回 GLU 模块的额外表示，描述 GLU 操作的维度信息
        def extra_repr(self) -> str:
            return f"dim={self.dim}"
# 定义一个 GELU 激活函数的类，继承自 Module 类
class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["approximate"]
    approximate: str

    # 初始化函数，设定激活函数的近似算法类型
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    # 前向传播函数，调用 PyTorch 的 F.gelu 函数进行计算
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    # 返回描述当前激活函数配置的字符串
    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


# 定义一个 Hardshrink 激活函数的类，继承自 Module 类
class Hardshrink(Module):
    r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["lambd"]
    lambd: float

    # 初始化函数，设定 Hardshrink 激活函数的阈值参数
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    # 前向传播函数，调用 PyTorch 的 F.hardshrink 函数进行计算
    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    # 返回描述当前激活函数配置的字符串
    def extra_repr(self) -> str:
        return f"{self.lambd}"


# 定义一个 LeakyReLU 激活函数的类，继承自 Module 类
class LeakyReLU(Module):
    r"""Applies the LeakyReLU function element-wise.

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``
    """
    # 初始化函数，设定 LeakyReLU 激活函数的负斜率和是否原地操作的参数
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    # 前向传播函数，调用 PyTorch 的 F.leaky_relu 函数进行计算
    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
    Shape:
        - Input: :math:`(*)` where `*` means any number of additional dimensions
        - Output: :math:`(*)`, same shape as the input

    .. image:: ../scripts/activation_images/LeakyReLU.png

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 定义 LeakyReLU 类，继承自 nn.Module
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool  # 是否进行原地操作的标志
    negative_slope: float  # 负斜率参数

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.negative_slope = negative_slope  # 初始化负斜率参数
        self.inplace = inplace  # 初始化原地操作标志

    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 leaky_relu 函数进行前向传播
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""  # 根据 inplace 属性生成对应的字符串
        return f"negative_slope={self.negative_slope}{inplace_str}"  # 返回描述实例的字符串
# 定义一个名为 LogSigmoid 的神经网络模块，继承自 Module 类
class LogSigmoid(Module):
    # LogSigmoid 函数的描述和数学公式
    r"""Applies the Logsigmoid function element-wise.

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/LogSigmoid.png

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 前向传播函数，接受一个张量输入 input，返回 LogSigmoid(input)
    def forward(self, input: Tensor) -> Tensor:
        return F.logsigmoid(input)


# 定义一个名为 Softplus 的神经网络模块，继承自 Module 类
class Softplus(Module):
    # Softplus 函数的描述和数学公式
    r"""Applies the Softplus function element-wise.

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softplus.png

    Examples::

        >>> m = nn.Softplus()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 类常量定义
    __constants__ = ["beta", "threshold"]
    # beta 和 threshold 的类型和默认值
    beta: float
    threshold: float

    # 初始化函数，接受 beta 和 threshold 两个参数
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta  # 初始化 beta
        self.threshold = threshold  # 初始化 threshold

    # 前向传播函数，接受一个张量输入 input，返回 Softplus(input)
    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)

    # 返回一个描述对象状态的字符串，包括 beta 和 threshold 的值
    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


# 定义一个名为 Softshrink 的神经网络模块，继承自 Module 类
class Softshrink(Module):
    # Softshrink 函数的描述和数学公式
    r"""Applies the soft shrinkage function element-wise.

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # 类常量定义
    __constants__ = ["lambd"]
    # lambd 的类型和默认值
    lambd: float

    # 初始化函数，接受 lambd 一个参数
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd  # 初始化 lambd

    # 前向传播函数，接受一个张量输入 input，返回 Softshrink(input)
    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)
    # 定义一个方法 `extra_repr`，返回对象属性 `lambd` 的字符串表示
    def extra_repr(self) -> str:
        # 将对象的属性 `lambd` 转换为字符串并返回
        return str(self.lambd)
# 检查输入张量的设备类型是否在预定义的设备类型列表中
def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    # 如果输入张量不为 None
    if x is not None:
        # 返回输入张量的设备类型是否在 ['cpu', 'cuda', torch.utils.backend_registration._privateuse1_backend_name] 中
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    # 如果输入张量为 None，则返回 True
    return True


# 检查输入张量是否需要梯度
def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    # 如果输入张量不为 None
    if x is not None:
        # 返回输入张量是否需要梯度
        return x.requires_grad
    # 如果输入张量为 None，则返回 False
    return False


# 检查当前是否正在进行 FX 追踪
def _is_make_fx_tracing():
    # 如果当前不是在进行脚本化（scripting）
    if not torch.jit.is_scripting():
        # 获取当前的调度模式栈
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        # 返回栈中是否有 ProxyTorchDispatchMode 类型的对象
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        # 如果当前正在脚本化，则返回 False
        return False


# 多头注意力机制模块
class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information from different representation subspaces.
    
    Method described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiHeadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.
    Args:
        embed_dim: 模型的总维度。
        num_heads: 并行注意力头的数量。注意，embed_dim 将被分割到 num_heads 个部分（每个头的维度为 embed_dim // num_heads）。
        dropout: attn_output_weights 上的 dropout 概率。默认为 0.0（不使用 dropout）。
        bias: 如果指定，则在输入/输出投影层中添加偏置。默认为 True。
        add_bias_kv: 如果指定，则在维度为 0 的键和值序列中添加偏置。默认为 False。
        add_zero_attn: 如果指定，则在维度为 1 的键和值序列中添加一批新的零。默认为 False。
        kdim: 键的特征总数。默认为 None（使用 kdim=embed_dim）。
        vdim: 值的特征总数。默认为 None（使用 vdim=embed_dim）。
        batch_first: 如果为 True，则输入和输出张量的形状为 (batch, seq, feature)。默认为 False（seq, batch, feature）。

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    # 常量定义
    __constants__ = ["batch_first"]
    
    # 可选的键和值的偏置张量
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    # 初始化函数
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        # 如果 embed_dim 或 num_heads 小于等于 0，则抛出数值错误异常
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        # 初始化工厂参数字典，用于创建参数时指定设备和数据类型
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法
        super().__init__()
        # 设置实例变量
        self.embed_dim = embed_dim
        # 如果 kdim 为 None，则使用 embed_dim 作为默认值
        self.kdim = kdim if kdim is not None else embed_dim
        # 如果 vdim 为 None，则使用 embed_dim 作为默认值
        self.vdim = vdim if vdim is not None else embed_dim
        # 检查是否 qkv 共享同一个 embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置 dropout 概率
        self.dropout = dropout
        # 是否以 batch_first 模式处理输入
        self.batch_first = batch_first
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 断言条件：embed_dim 必须能够被 num_heads 整除
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # 根据 _qkv_same_embed_dim 初始化参数权重
        if not self._qkv_same_embed_dim:
            # 初始化 q_proj_weight 参数
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            # 初始化 k_proj_weight 参数
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            # 初始化 v_proj_weight 参数
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            # 不注册 in_proj_weight 参数
            self.register_parameter("in_proj_weight", None)
        else:
            # 初始化 in_proj_weight 参数
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            # 不注册 q_proj_weight, k_proj_weight, v_proj_weight 参数
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        # 根据 bias 初始化 in_proj_bias 参数
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            # 不注册 in_proj_bias 参数
            self.register_parameter("in_proj_bias", None)
        # 初始化 out_proj 线性层
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        # 根据 add_bias_kv 初始化 bias_k 和 bias_v 参数
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        # 是否添加零注意力
        self.add_zero_attn = add_zero_attn

        # 重置所有参数的初始值
        self._reset_parameters()

    # 重置参数的初始化方法
    def _reset_parameters(self):
        # 根据 _qkv_same_embed_dim 来选择初始化方法
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        # 如果存在 in_proj_bias，则将其值设为 0
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        # 如果存在 bias_k，则使用 xavier_normal_ 方法初始化其值
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        # 如果存在 bias_v，则使用 xavier_normal_ 方法初始化其值
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)
    def __setstate__(self, state):
        # 加载由 v1.1.0 生成的旧 MultiheadAttention 检查点的支持
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        # 调用父类的 __setstate__ 方法来设置对象状态
        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # 此方法定义了 MultiheadAttention 模块的前向传播逻辑
        ...

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            # 如果存在 key_padding_mask，则设置 mask_type 为 1，并使用 key_padding_mask 作为 merged_mask
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # 如果存在 attn_mask，则进入此分支
            batch_size, seq_len, _ = query.shape
            # 设置 mask_type 为 2
            mask_type = 2

            # 将 attn_mask 扩展为 4D 张量
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                # 如果同时存在 key_padding_mask，则将其扩展后与 attn_mask 合并
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # 如果没有 attn_mask 和 key_padding_mask，则返回 None, None
        return merged_mask, mask_type
# 定义一个 PReLU 激活函数的类，继承自 nn.Module
class PReLU(Module):
    r"""Applies the element-wise PReLU function.

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \ge 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.

    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. image:: ../scripts/activation_images/PReLU.png

    Examples::

        >>> m = nn.PReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        # 创建一个形状为 (num_parameters,) 的可学习参数 weight
        self.weight = Parameter(torch.empty(num_parameters, **factory_kwargs))
        # 调用 reset_parameters 方法初始化 weight
        self.reset_parameters()

    def reset_parameters(self):
        # 使用常数值 init 初始化 weight
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.prelu 函数实现 PReLU 激活函数的前向传播
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        # 返回描述该类实例的额外字符串，指明 num_parameters 属性的值
        return f"num_parameters={self.num_parameters}"


# 定义一个 Softsign 激活函数的类，继承自 nn.Module
class Softsign(Module):
    r"""Applies the element-wise Softsign function.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softsign.png

    Examples::

        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.softsign 函数实现 Softsign 激活函数的前向传播
        return F.softsign(input)


# 定义一个 Tanhshrink 激活函数的类，继承自 nn.Module
class Tanhshrink(Module):
    r"""Applies the element-wise Tanhshrink function.

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)
    """
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
          输入: :math:`(*)`，其中 :math:`*` 表示任意数量的维度。
        - Output: :math:`(*)`, same shape as the input.
          输出: :math:`(*)`，与输入相同的形状。

    .. image:: ../scripts/activation_images/Tanhshrink.png
       图像:: ../scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        在正向传播过程中，应用 tanhshrink 函数到输入张量上，并返回结果张量。
        
        参数:
            input (Tensor): 输入张量，可以是任意形状的张量。

        返回:
            Tensor: 经过 tanhshrink 函数处理后的张量，形状与输入相同。

        """
        return F.tanhshrink(input)
# 定义 Softmin 类，继承自 Module 类
class Softmin(Module):
    r"""Applies the Softmin function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    # 定义常量 __constants__，指定了维度 dim 为可选的整数
    __constants__ = ["dim"]
    dim: Optional[int]

    # Softmin 类的初始化方法
    def __init__(self, dim: Optional[int] = None) -> None:
        # 调用父类 Module 的初始化方法
        super().__init__()
        # 设置 Softmin 对象的维度 dim
        self.dim = dim

    # 用于反序列化 Softmin 对象状态的方法
    def __setstate__(self, state):
        # 调用父类 Module 的反序列化方法
        super().__setstate__(state)
        # 如果对象没有维度属性，则将其设置为 None
        if not hasattr(self, "dim"):
            self.dim = None

    # 定义前向传播方法
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 模块中的 softmin 函数进行计算，指定堆栈级别为 5
        return F.softmin(input, self.dim, _stacklevel=5)

    # 返回描述 Softmin 对象额外信息的字符串
    def extra_repr(self):
        return f"dim={self.dim}"


# 定义 Softmax 类，继承自 Module 类
class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecified
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)

    """

    # 定义常量 __constants__，指定了维度 dim 为可选的整数
    __constants__ = ["dim"]
    dim: Optional[int]

    # Softmax 类的初始化方法
    def __init__(self, dim: Optional[int] = None) -> None:
        # 调用父类 Module 的初始化方法
        super().__init__()
        # 设置 Softmax 对象的维度 dim
        self.dim = dim

    # 用于反序列化 Softmax 对象状态的方法
    def __setstate__(self, state):
        # 调用父类 Module 的反序列化方法
        super().__setstate__(state)
        # 如果对象没有维度属性，则将其设置为 None
        if not hasattr(self, "dim"):
            self.dim = None

    # 定义前向传播方法
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 模块中的 softmax 函数进行计算，指定堆栈级别为 5
        return F.softmax(input, self.dim, _stacklevel=5)

    # 返回描述 Softmax 对象额外信息的字符串
    def extra_repr() -> str:
        return f"dim={self.dim}"
class Softmax2d(Module):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # 检查输入张量的维度是否为3D或4D，如果不是则抛出异常
        if input.dim() not in (3, 4):
            raise ValueError(
                f"Softmax2d: expected input to be 3D or 4D, got {input.dim()}D instead"
            )
        # 对输入张量在第三个维度上应用 softmax 操作，并返回结果张量
        return F.softmax(input, -3, _stacklevel=5)


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        # 调用父类构造函数初始化 LogSoftmax 模块
        super().__init__()
        # 设置 LogSoftmax 操作的维度参数
        self.dim = dim

    def __setstate__(self, state):
        # 调用父类方法设置 LogSoftmax 模块的状态
        super().__setstate__(state)
        # 如果 LogSoftmax 模块中不存在维度参数，则设置为 None
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        # 对输入张量在指定维度上应用 log_softmax 操作，并返回结果张量
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        # 返回 LogSoftmax 模块的额外信息，包括维度参数的字符串表示
        return f"dim={self.dim}"
```