# `.\pytorch\torch\nn\modules\upsampling.py`

```
# 允许在类型检查时定义未加类型注解的函数
# 从 typing 模块导入 Optional 类型注解
from typing import Optional

# 从 torch.nn.functional 模块中导入 F 模块
import torch.nn.functional as F
# 从 torch 模块中导入 Tensor 类型注解
from torch import Tensor
# 从 torch.nn.common_types 模块中导入一些特定类型注解
from torch.nn.common_types import _ratio_2_t, _ratio_any_t, _size_2_t, _size_any_t

# 从当前目录下的 module.py 文件中导入 Module 类
from .module import Module

# 定义本模块中可导出的类列表
__all__ = ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"]

# 定义一个名为 Upsample 的类，继承自 Module 类
class Upsample(Module):
    # 文档字符串：对多通道 1D（时间）、2D（空间）或 3D（立体）数据进行上采样
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    # 输入数据假设为 `minibatch x channels x [optional depth] x [optional height] x width` 的形式
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    因此，对于空间输入，我们期望一个 4D 张量；对于立体输入，我们期望一个 5D 张量。

    # 可用于上采样的算法有最近邻和线性、双线性、三线性以及三线性插值，分别适用于 3D、4D 和 5D 输入张量。
    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor, respectively.

    # 可以提供 :attr:`scale_factor` 或目标输出 :attr:`size` 来计算输出大小（不能同时给出，因为这是模棱两可的）
    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    # 参数：
    Args:
        # 输出空间尺寸
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        # 空间大小的乘法因子。如果是元组，它必须与输入大小匹配。
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        # 上采样算法：'nearest'、'linear'、'bilinear'、'bicubic' 和 'trilinear'
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        # 是否对齐角像素
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        # 重新计算用于插值计算的 scale_factor
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.

    # 形状：
    Shape:
        - 输入：:math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - 输出：:math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          或 :math:`(N, C, D_{out}, H_{out}, W_{out})`, 其中

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor
    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor
    # 计算输出宽度 W_out，使用向下取整函数 floor，根据输入宽度 W_in 和缩放因子 scale_factor 计算得出

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, `bicubic`, and `trilinear`) don't proportionally
        align the output and input pixels, and thus the output values can depend
        on the input size. This was the default behavior for these modes up to
        version 0.3.1. Since then, the default behavior is
        ``align_corners = False``. See below for concrete examples on how this
        affects the outputs.
    # 警告：当 ``align_corners = True`` 时，线性插值模式（`linear`、`bilinear`、`bicubic` 和 `trilinear`）不会按比例对齐输出和输入像素，因此输出值可能依赖于输入大小。这是这些模式的默认行为，直到版本 0.3.1。此后，默认行为变为 ``align_corners = False``。请参见下面的具体示例，了解这如何影响输出结果。

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.
    # 注意：如果您想要进行下采样/一般性调整大小操作，应使用 :func:`~nn.functional.interpolate` 函数。
    # 常量定义，列出了类的常量名称
    __constants__ = [
        "size",  # 输入的尺寸
        "scale_factor",  # 缩放因子
        "mode",  # 插值模式，如 nearest 或 bilinear
        "align_corners",  # 是否对齐角点的标志
        "name",  # 名称
        "recompute_scale_factor",  # 是否重新计算缩放因子的标志
    ]
    name: str  # 声明一个名为 name 的类型为 str 的成员变量
    size: Optional[_size_any_t]  # 声明一个名为 size 的可选类型成员变量，类型为 _size_any_t
    scale_factor: Optional[_ratio_any_t]  # 可选的尺度因子，用于指定缩放比例
    mode: str  # 插值模式，指定如何进行插值（默认为最近邻插值）
    align_corners: Optional[bool]  # 可选的布尔值，指定是否对齐角点
    recompute_scale_factor: Optional[bool]  # 可选的布尔值，指定是否重新计算尺度因子

    def __init__(
        self,
        size: Optional[_size_any_t] = None,  # 可选的目标尺寸或输出尺寸
        scale_factor: Optional[_ratio_any_t] = None,  # 可选的尺度因子，用于指定缩放比例
        mode: str = "nearest",  # 插值模式，默认为最近邻插值
        align_corners: Optional[bool] = None,  # 可选的布尔值，指定是否对齐角点
        recompute_scale_factor: Optional[bool] = None,  # 可选的布尔值，指定是否重新计算尺度因子
    ) -> None:
        super().__init__()  # 调用父类的构造函数
        self.name = type(self).__name__  # 设置实例的名称为类名
        self.size = size  # 初始化目标尺寸或输出尺寸
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)  # 如果尺度因子是元组，则转换为浮点数元组
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None  # 否则转换为浮点数或者设为None
        self.mode = mode  # 初始化插值模式
        self.align_corners = align_corners  # 初始化对齐角点参数
        self.recompute_scale_factor = recompute_scale_factor  # 初始化重新计算尺度因子参数

    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )  # 调用PyTorch的插值函数进行前向传播

    def __setstate__(self, state):
        if "recompute_scale_factor" not in state:
            state["recompute_scale_factor"] = True  # 如果状态中没有尺度因子的重新计算参数，则设置为True

        super().__setstate__(state)  # 调用父类的状态设置方法

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = "scale_factor=" + repr(self.scale_factor)  # 如果存在尺度因子，则添加到信息字符串中
        else:
            info = "size=" + repr(self.size)  # 否则添加目标尺寸到信息字符串中
        info += ", mode=" + repr(self.mode)  # 添加插值模式到信息字符串中
        return info  # 返回描述实例的字符串
class UpsamplingBilinear2d(Upsample):
    r"""Applies a 2D bilinear upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`. It is
        equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor
    """

    def __init__(
        self,
        size: Optional[_size_2_t] = None,
        scale_factor: Optional[_ratio_2_t] = None,
    ) -> None:
        # 调用父类构造函数，初始化上采样层
        super().__init__(size, scale_factor, mode="bilinear")
    # 定义一个新的类 nn.UpsamplingBilinear2d，继承自父类 nn.Module
    def __init__(
        self,
        # 可选参数：指定输出的尺寸大小
        size: Optional[_size_2_t] = None,
        # 可选参数：指定尺寸的缩放比例
        scale_factor: Optional[_ratio_2_t] = None,
    ) -> None:
        # 调用父类的初始化方法，传入 size、scale_factor、mode="bilinear" 和 align_corners=True
        super().__init__(size, scale_factor, mode="bilinear", align_corners=True)
```