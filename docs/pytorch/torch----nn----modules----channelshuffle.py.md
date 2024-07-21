# `.\pytorch\torch\nn\modules\channelshuffle.py`

```
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["ChannelShuffle"]


# 定义一个继承自 Module 的 ChannelShuffle 类
class ChannelShuffle(Module):
    r"""Divides and rearranges the channels in a tensor.

    This operation divides the channels in a tensor of shape :math:`(*, C , H, W)`
    into g groups as :math:`(*, \frac{C}{g}, g, H, W)` and shuffles them,
    while retaining the original tensor shape in the final output.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[13., 14.],
                  [15., 16.]]]])
        >>> output = channel_shuffle(input)
        >>> output
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[13., 14.],
                  [15., 16.]]]])
    """

    # 定义保存常量 "groups" 的元组
    __constants__ = ["groups"]
    groups: int

    # 初始化函数，接收一个 groups 参数
    def __init__(self, groups: int) -> None:
        # 调用父类 Module 的初始化函数
        super().__init__()
        # 设置对象的 groups 属性为传入的参数值
        self.groups = groups

    # 前向传播函数，接收一个输入张量 input，返回经过 channel shuffle 后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 模块的 channel_shuffle 函数
        return F.channel_shuffle(input, self.groups)

    # 返回对象的额外描述信息，描述对象的属性
    def extra_repr(self) -> str:
        return f"groups={self.groups}"
```