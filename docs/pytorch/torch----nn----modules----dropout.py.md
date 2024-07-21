# `.\pytorch\torch\nn\modules\dropout.py`

```py
# 导入 torch.nn.functional 中的 F 模块，用于神经网络中的函数操作
# 导入 Tensor 类型，用于声明函数参数和返回值的张量类型
import torch.nn.functional as F
from torch import Tensor

# 从当前目录下的 module 模块中导入 Module 类
from .module import Module

# 定义 __all__ 列表，指定可以通过 `from module import *` 导入的公共接口
__all__ = [
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]

# 定义 _DropoutNd 类，继承自 Module 类
class _DropoutNd(Module):
    # 定义常量列表 __constants__，包含了类的常量 'p' 和 'inplace'
    __constants__ = ["p", "inplace"]
    p: float  # 概率 p，表示随机将输入张量的某些元素置零的概率
    inplace: bool  # 是否原地操作标志位，表示是否在原张量上进行操作

    # 初始化函数，接受参数 p（概率）和 inplace（是否原地操作）
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()  # 调用父类 Module 的初始化函数
        # 如果概率 p 小于 0 或大于 1，则抛出 ValueError 异常
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p  # 将参数 p 赋值给实例变量 self.p
        self.inplace = inplace  # 将参数 inplace 赋值给实例变量 self.inplace

    # extra_repr 方法，返回描述模块的额外字符串信息
    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"  # 返回格式化的描述字符串

# Dropout 类，继承自 _DropoutNd 类
class Dropout(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`.
    ...
    """

    # forward 方法，定义前向传播逻辑，接收输入张量 input，返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 F.dropout 函数，实现随机将输入张量元素置零
        return F.dropout(input, self.p, self.training, self.inplace)

# Dropout1d 类，继承自 _DropoutNd 类
class Dropout1d(_DropoutNd):
    r"""Randomly zero out entire channels.
    ...
    """

    # forward 方法，定义前向传播逻辑，接收输入张量 input，返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 F.dropout 函数，实现随机将输入张量元素置零
        return F.dropout(input, self.p, self.training, self.inplace)
    """
    In this case, :func:`nn.Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`.
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout1d(p=0.2)
        >>> input = torch.randn(20, 16, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Applies dropout to the input tensor in the 1-dimensional space.

        Args:
            input (Tensor): input tensor to which dropout will be applied.

        Returns:
            Tensor: output tensor after applying dropout.

        Notes:
            This method applies the dropout operation defined by :func:`F.dropout1d`.
            It uses the dropout probability `self.p` and respects the training mode
            indicated by `self.training`.

        """
        return F.dropout1d(input, self.p, self.training, self.inplace)
class Dropout2d(_DropoutNd):
    r"""Randomly zero out entire channels.

    A channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`.

    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    .. warning ::
        Due to historical reasons, this class will perform 1D channel-wise dropout
        for 3D inputs (as done by :class:`nn.Dropout1d`). Thus, it currently does NOT
        support inputs without a batch dimension of shape :math:`(C, H, W)`. This
        behavior will change in a future release to interpret 3D inputs as no-batch-dim
        inputs. To maintain the old behavior, switch to :class:`nn.Dropout1d`.

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, L)`.
        - Output: :math:`(N, C, H, W)` or :math:`(N, C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        # 调用 nn.functional 中的 dropout2d 函数，对输入进行二维随机丢弃
        return F.dropout2d(input, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):
    r"""Randomly zero out entire channels.

    A channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`.

    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv3d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    ```
    """
    三维 Dropout 操作类。

    Dropout 是一种在神经网络训练中随机失活部分神经元的技术，有助于防止过拟合。

    Args:
        p (float, optional): 失活概率，即某神经元被随机失活的概率。
        inplace (bool, optional): 如果设置为 ``True``，将在原地执行操作。

    Shape:
        - 输入: :math:`(N, C, D, H, W)` 或者 :math:`(C, D, H, W)`。
        - 输出: 与输入相同的形状 :math:`(N, C, D, H, W)` 或者 :math:`(C, D, H, W)`。

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    参考文献: Efficient Object Localization Using Convolutional Networks
    https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        前向传播函数。

        Args:
            input (Tensor): 输入张量。

        Returns:
            Tensor: 经过 Dropout3d 处理后的张量。
        """
        return F.dropout3d(input, self.p, self.training, self.inplace)
# 定义 AlphaDropout 类，继承自 _DropoutNd 类
class AlphaDropout(_DropoutNd):
    r"""Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        p (float): probability of an element to be dropped. Default: 0.5
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.AlphaDropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    # 重写 forward 方法，对输入应用 Alpha Dropout
    def forward(self, input: Tensor) -> Tensor:
        # 调用 functional 模块中的 alpha_dropout 函数，实现 Alpha Dropout 功能
        return F.alpha_dropout(input, self.p, self.training)


# 定义 FeatureAlphaDropout 类，继承自 _DropoutNd 类
class FeatureAlphaDropout(_DropoutNd):
    r"""Randomly masks out entire channels.

    A channel is a feature map,
    e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]` of the input tensor). Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function. More details
    can be found in the paper `Self-Normalizing Neural Networks`_ .

    Each element will be masked independently for each sample on every forward
    call with probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    Usually the input comes from :class:`nn.AlphaDropout` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.AlphaDropout` will help promote independence between
    feature maps and should be used instead.
    """
    Args:
        p (float, optional): probability of an element to be zeroed. Default: 0.5
            控制每个元素被置零的概率，默认为0.5
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place
            如果设置为 ``True``，则进行原地操作

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
            输入张量的形状可以是 :math:`(N, C, D, H, W)` 或者 :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).
            输出张量的形状与输入相同，可以是 :math:`(N, C, D, H, W)` 或者 :math:`(C, D, H, W)`

    Examples::

        >>> m = nn.FeatureAlphaDropout(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        对输入张量进行前向传播，应用特征级Alpha Dropout操作。

        Args:
            input (Tensor): 输入张量

        Returns:
            Tensor: 输出张量，经过特征级Alpha Dropout处理后的结果
        """
        return F.feature_alpha_dropout(input, self.p, self.training)
```