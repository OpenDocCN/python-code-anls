# `.\pytorch\torch\nn\modules\pixelshuffle.py`

```
# 导入 torch.nn.functional 中的 F 模块
import torch.nn.functional as F
# 导入 Tensor 类型
from torch import Tensor
# 从当前目录下的 module 模块中导入 Module 类
from .module import Module

# 声明 __all__ 列表，指定了可以导出的模块名
__all__ = ["PixelShuffle", "PixelUnshuffle"]

# 定义 PixelShuffle 类，继承自 Module 类
class PixelShuffle(Module):
    r"""Rearrange elements in a tensor according to an upscaling factor.

    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et al. (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \div \text{upscale\_factor}^2

    .. math::
        H_{out} = H_{in} \times \text{upscale\_factor}

    .. math::
        W_{out} = W_{in} \times \text{upscale\_factor}

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    # 声明常量 __constants__，指定了 upscale_factor 作为常量
    __constants__ = ["upscale_factor"]
    # 声明 upscale_factor 为整数类型
    upscale_factor: int

    # 构造函数，初始化 upscale_factor
    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    # 前向传播函数，调用 F.pixel_shuffle 函数进行像素重排操作
    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_shuffle(input, self.upscale_factor)

    # 返回额外的字符串表示，描述了 upscale_factor
    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"


# 定义 PixelUnshuffle 类，继承自 Module 类
class PixelUnshuffle(Module):
    r"""Reverse the PixelShuffle operation.

    Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
    in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`, where r is a downscale factor.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et al. (2016) for more details.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \times \text{downscale\_factor}^2

    .. math::
        H_{out} = H_{in} \div \text{downscale\_factor}

    .. math::
        W_{out} = W_{in} \div \text{downscale\_factor}
    Examples::

        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = torch.randn(1, 1, 12, 12)
        >>> output = pixel_unshuffle(input)
        >>> print(output.size())
        torch.Size([1, 9, 4, 4])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    # 定义一个类 nn.PixelUnshuffle，用于执行像素解拼操作，可以通过指定的 downscale_factor 控制解拼时的尺度缩小倍数
    __constants__ = ["downscale_factor"]
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 初始化类的 downscale_factor 属性，表示解拼时的尺度缩小倍数
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 pixel_unshuffle 函数，对输入的张量 input 进行像素解拼操作
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        # 返回一个描述类额外信息的字符串，包括 downscale_factor 的值
        return f"downscale_factor={self.downscale_factor}"
```