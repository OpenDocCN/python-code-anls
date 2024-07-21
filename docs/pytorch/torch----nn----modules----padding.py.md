# `.\pytorch\torch\nn\modules\padding.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类型定义
from typing import Sequence, Tuple

import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t

# 从自定义模块中导入基类和辅助函数
from .module import Module
from .utils import _ntuple, _pair, _quadruple


# TODO: grad_output size asserts in THNN

# 导出的模块列表，用于模块的公开接口
__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]


# CircularPadNd 类，继承自 Module 基类
class _CircularPadNd(Module):
    # 定义常量字段 padding
    __constants__ = ["padding"]
    padding: Sequence[int]

    # 抽象方法，用于检查输入维度
    def _check_input_dim(self, input):
        raise NotImplementedError

    # 前向传播方法，对输入进行循环填充并返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        # 使用 torch.nn.functional.pad 函数进行循环填充
        return F.pad(input, self.padding, "circular")

    # 返回对象的额外表示信息，包括填充参数
    def extra_repr(self) -> str:
        return f"{self.padding}"


# CircularPad1d 类，实现了一维循环填充功能
class CircularPad1d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.CircularPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 3., 0., 1., 2., 3., 0., 1.],
                 [6., 7., 4., 5., 6., 7., 4., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad1d((3, 1))
        >>> m(input)
        tensor([[[1., 2., 3., 0., 1., 2., 3., 0.],
                 [5., 6., 7., 4., 5., 6., 7., 4.]]])
    """

    # 声明填充参数为一个二元组
    padding: Tuple[int, int]

    # 初始化方法，接受填充参数，调用基类的初始化方法
    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        # 调用辅助函数 _pair 处理填充参数，确保是二元组形式
        self.padding = _pair(padding)

    # 重载基类方法，检查输入张量的维度是否为 2 或 3
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


# CircularPad2d 类，实现了二维循环填充功能
class CircularPad2d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.
    
    (后续代码省略)
    """
        Tensor values at the beginning of the dimension are used to pad the end,
        and values at the end are used to pad the beginning. If negative padding is
        applied then the ends of the tensor get removed.
    
        For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.
    
        Args:
            padding (int, tuple): the size of the padding. If is `int`, uses the same
                padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
                :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
    
        Shape:
            - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
            - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
    
              :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
    
              :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
    
        Examples::
    
            >>> m = nn.CircularPad2d(2)
            >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
            >>> input
            tensor([[[[0., 1., 2.],
                      [3., 4., 5.],
                      [6., 7., 8.]]]])
            >>> m(input)
            tensor([[[[4., 5., 3., 4., 5., 3., 4.],
                      [7., 8., 6., 7., 8., 6., 7.],
                      [1., 2., 0., 1., 2., 0., 1.],
                      [4., 5., 3., 4., 5., 3., 4.],
                      [7., 8., 6., 7., 8., 6., 7.],
                      [1., 2., 0., 1., 2., 0., 1.],
                      [4., 5., 3., 4., 5., 3., 4.]]]])
            >>> # using different paddings for different sides
            >>> m = nn.CircularPad2d((1, 1, 2, 0))
            >>> m(input)
            tensor([[[[5., 3., 4., 5., 3.],
                      [8., 6., 7., 8., 6.],
                      [2., 0., 1., 2., 0.],
                      [5., 3., 4., 5., 3.],
                      [8., 6., 7., 8., 6.]]]])
    
        """
    
        padding: Tuple[int, int, int, int]
    
        def __init__(self, padding: _size_4_t) -> None:
            super().__init__()
            # Initialize the CircularPad2d layer with padding values as a tuple
            self.padding = _quadruple(padding)
    
        def _check_input_dim(self, input):
            # Check if the input tensor is 3D or 4D, raise error otherwise
            if input.dim() != 3 and input.dim() != 4:
                raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")
# 定义一个继承自 _CircularPadNd 的类，用于对输入张量进行圆形填充
class CircularPad3d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.CircularPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    """

    padding: Tuple[int, int, int, int, int, int]  # 定义 padding 参数类型为包含六个整数的元组

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()  # 调用父类 _CircularPadNd 的构造方法
        self.padding = _ntuple(6)(padding)  # 将输入的 padding 转换为长度为 6 的元组

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为 4D 或 5D，如果不是则抛出 ValueError 异常
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")
    """
    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
                   3.5000],
                 [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
                   3.5000]]])
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
                 [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])
    """

    padding: Tuple[int, int]  # 用于指定填充的左右两边大小的元组

    def __init__(self, padding: _size_2_t, value: float):
        super().__init__(value)  # 调用父类的初始化方法，传入填充值
        self.padding = _pair(padding)  # 将传入的填充参数转换成长度为2的元组
class ConstantPad2d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad2d(2, 3.5)
        >>> input = torch.randn(1, 2, 2)
        >>> input
        tensor([[[ 1.6585,  0.4320],
                 [-0.8701, -0.4649]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
                 [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    """

    __constants__ = ["padding", "value"]
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t, value: float) -> None:
        # 调用父类的构造函数，传入填充的常量值
        super().__init__(value)
        # 将传入的填充参数转换成四元组形式
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
    padding: Tuple[int, int, int, int, int, int]


    # 定义类属性 `padding`，表示六维的填充参数，每维度分别对应不同的填充量
    padding: Tuple[int, int, int, int, int, int]



    def __init__(self, padding: _size_6_t, value: float) -> None:


    # 构造函数，初始化 ConstantPad3d 类的实例
    def __init__(self, padding: _size_6_t, value: float) -> None:



        super().__init__(value)


        # 调用父类的构造函数，传入 value 参数，初始化父类
        super().__init__(value)



        self.padding = _ntuple(6)(padding)


        # 将传入的 padding 参数转换为六元组，赋值给实例的 padding 属性
        self.padding = _ntuple(6)(padding)
class _ReflectionPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        # 使用 torch.nn.functional.pad() 函数对输入张量进行反射填充
        return F.pad(input, self.padding, "reflect")

    def extra_repr(self) -> str:
        # 返回当前填充参数的字符串表示形式
        return f"{self.padding}"


class ReflectionPad1d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::
        >>> m = nn.ReflectionPad1d(2)
        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
                 [6., 5., 4., 5., 6., 7., 6., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad1d((3, 1))
        >>> m(input)
        tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
                 [7., 6., 5., 4., 5., 6., 7., 6.]]])
    """

    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        # 调用父类构造函数，初始化填充参数
        super().__init__()
        # 将输入的填充参数转换为长度为2的元组
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
    ```
    """
    定义一个 ReflectionPad2d 类，用于对输入进行反射填充的操作

    Parameters:
    - padding: Tuple[int, int, int, int]，表示填充每个维度的大小

    """
    
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        """
        ReflectionPad2d 类的初始化方法

        Parameters:
        - padding: _size_4_t，输入参数的类型为四元组

        """
        super().__init__()
        # 将输入的填充参数转换为四元组，确保格式统一
        self.padding = _quadruple(padding)
class ReflectionPad3d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad3d(1)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
        >>> m(input)
        tensor([[[[[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]],
                  [[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]]]]])
    """

    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        # 将输入的 padding 参数转换为一个长度为 6 的元组，表示六个方向的填充大小
        self.padding = _ntuple(6)(padding)


class _ReplicationPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        # 使用 torch.nn.functional.pad() 函数进行输入张量的边界复制填充
        return F.pad(input, self.padding, "replicate")

    def extra_repr(self) -> str:
        # 返回描述实例状态的字符串，包括填充的大小
        return f"{self.padding}"


class ReplicationPad1d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
    """
    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReplicationPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
                 [4., 4., 4., 5., 6., 7., 7., 7.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad1d((3, 1))
        >>> m(input)
        tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
                 [4., 4., 4., 4., 5., 6., 7., 7.]]])
    """

    padding: Tuple[int, int]

    # 初始化函数，接受一个名为 padding 的元组参数
    def __init__(self, padding: _size_2_t) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 padding 转换为一对整数，赋值给对象的 padding 属性
        self.padding = _pair(padding)
class ReplicationPad2d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReplicationPad2d(2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [3., 3., 3., 4., 5., 5., 5.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [3., 3., 4., 5., 5.],
                  [6., 6., 7., 8., 8.]]]])

    """

    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        # Initialize the padding attribute with the provided tuple of integers
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    """
    # 定义 ReplicationPad3d 类，用于在三维输入张量周围进行复制填充
    padding: Tuple[int, int, int, int, int, int]

    # 初始化方法，接受一个六元组 padding 参数，并调用 _ntuple(6) 将其转换为长度为 6 的元组
    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        # 将传入的 padding 参数转换为长度为 6 的元组并赋值给实例变量 self.padding
        self.padding = _ntuple(6)(padding)
class ZeroPad1d(ConstantPad1d):
    r"""Pads the input tensor boundaries with zero.
    用零填充输入张量的边界。

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.
    对于`N`维填充，请使用:func:`torch.nn.functional.pad()`。

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
        填充（int，tuple）：填充的大小。如果是`int`，则在两个边界中使用相同的填充。如果是2-`tuple`，则使用(:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
        形状:
        - 输入: :math:`(C, W_{in})` 或 :math:`(N, C, W_{in})`。
        - 输出: :math:`(C, W_{out})` 或 :math:`(N, C, W_{out})`，其中
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::
    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000, -1.0491, -0.7152, -0.0749,  0.8530,  0.0000,
                   0.0000],
                 [ 0.0000,  0.0000, -1.3287,  1.8966,  0.1466, -0.2771,  0.0000,
                   0.0000]]])
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000,  0.0000],
                 [ 0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000,  0.0000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad1d((3, 1))
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000],
                 [ 0.0000,  0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000]]])
    """

    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__(padding, 0.0)
        使用给定的填充大小和值初始化父类ConstantPad1d
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        return f"{self.padding}"
        返回填充值的字符串表示形式


class ZeroPad2d(ConstantPad2d):
    r"""Pads the input tensor boundaries with zero.
    用零填充输入张量的边界。

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.
    对于`N`维填充，请使用:func:`torch.nn.functional.pad()`。

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
        填充（int，tuple）：填充的大小。如果是`int`，则在所有边界中使用相同的填充。如果是4-`tuple`，则使用(:math:`\text{padding\_left}`, :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
        形状:
        - 输入: :math:`(N, C, H_{in}, W_{in})` 或 :math:`(C, H_{in}, W_{in})`.
        - 输出: :math:`(N, C, H_{out}, W_{out})` 或 :math:`(C, H_{out}, W_{out})`，其中
          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`
    # 定义一个名为 ZeroPad2d 的 nn.Module 类，用于二维零填充操作
    class ZeroPad2d(nn.Module):
    
        # 初始化方法，接受一个长度为4的元组作为填充参数
        def __init__(self, padding: _size_4_t) -> None:
            # 调用父类的初始化方法
            super().__init__(padding, 0.0)
    
        # 返回一个描述实例状态的字符串，包括填充参数
        def extra_repr(self) -> str:
            return f"{self.padding}"
class ZeroPad3d(ConstantPad3d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ZeroPad3d(3)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
        >>> output = m(input)
    """

    padding: Tuple[int, int, int, int, int, int]

    # 初始化函数，继承自 ConstantPad3d，用给定的 padding 初始化对象
    def __init__(self, padding: _size_6_t) -> None:
        super().__init__(padding, 0.0)

    # 返回对象的额外描述信息，包括当前的 padding 大小
    def extra_repr(self) -> str:
        return f"{self.padding}"
```