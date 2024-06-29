# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\axes_rgb.py`

```py
from types import MethodType  # 导入 MethodType 类型，用于动态绑定方法

import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from .axes_divider import make_axes_locatable, Size  # 从当前目录导入 make_axes_locatable 和 Size
from .mpl_axes import Axes, SimpleAxisArtist  # 从当前目录导入 Axes 和 SimpleAxisArtist 类


def make_rgb_axes(ax, pad=0.01, axes_class=None, **kwargs):
    """
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        要在其中创建 RGB 轴的 Axes 实例。
    pad : float, optional
        用于填充的 Axes 高度的分数。
    axes_class : `matplotlib.axes.Axes` or None, optional
        用于 R、G 和 B 轴的 Axes 类。如果为 None，则使用与 *ax* 相同的类。
    **kwargs
        传递给 *axes_class* 的初始化参数，用于 R、G 和 B 轴。
    """

    divider = make_axes_locatable(ax)  # 创建一个 Axes 分割器对象

    pad_size = pad * Size.AxesY(ax)  # 计算填充的大小，作为 Axes 高度的一部分

    xsize = ((1-2*pad)/3) * Size.AxesX(ax)  # 计算每个子图的宽度
    ysize = ((1-2*pad)/3) * Size.AxesY(ax)  # 计算每个子图的高度

    divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])  # 设置水平分割参数
    divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])  # 设置垂直分割参数

    ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))  # 设置主 Axes 的位置定位器

    ax_rgb = []
    if axes_class is None:
        axes_class = type(ax)

    for ny in [4, 2, 0]:
        ax1 = axes_class(ax.get_figure(), ax.get_position(original=True),
                         sharex=ax, sharey=ax, **kwargs)  # 创建 R、G、B 轴实例
        locator = divider.new_locator(nx=2, ny=ny)  # 创建新的定位器
        ax1.set_axes_locator(locator)  # 设置每个轴的位置定位器
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)  # 设置刻度标签不可见
        try:
            for axis in ax1.axis.values():
                axis.major_ticklabels.set_visible(False)  # 设置主刻度标签不可见
        except AttributeError:
            pass

        ax_rgb.append(ax1)  # 将创建的轴实例添加到列表中

    fig = ax.get_figure()
    for ax1 in ax_rgb:
        fig.add_axes(ax1)  # 将创建的轴实例添加到主图的图形对象中

    return ax_rgb  # 返回包含 R、G、B 轴实例的列表


class RGBAxes:
    """
    4-panel `~.Axes.imshow` (RGB, R, G, B).

    Layout::

        ┌───────────────┬─────┐
        │               │  R  │
        │               ├─────┤
        │      RGB      │  G  │
        │               ├─────┤
        │               │  B  │
        └───────────────┴─────┘

    子类可以重写 ``_defaultAxesClass`` 属性。
    默认情况下，RGBAxes 使用 `.mpl_axes.Axes`。

    Attributes
    ----------
    RGB : ``_defaultAxesClass``
        用于三通道 `~.Axes.imshow` 的 Axes 对象。
    R : ``_defaultAxesClass``
        用于红色通道 `~.Axes.imshow` 的 Axes 对象。
    G : ``_defaultAxesClass``
        用于绿色通道 `~.Axes.imshow` 的 Axes 对象。
    B : ``_defaultAxesClass``
        用于蓝色通道 `~.Axes.imshow` 的 Axes 对象。
    """

    _defaultAxesClass = Axes  # 默认使用 Axes 类作为轴对象
    # 初始化函数，用于创建一个包含 RGB 和各个单通道图像的对象
    def __init__(self, *args, pad=0, **kwargs):
        """
        Parameters
        ----------
        pad : float, default: 0
            Axes 高度的一部分，作为填充。
        axes_class : `~matplotlib.axes.Axes`
            要使用的 Axes 类。如果未提供，则使用 `_defaultAxesClass`。
        *args
            传递给 RGB Axes 的 *axes_class* 初始化参数
        **kwargs
            传递给 RGB、R、G 和 B Axes 的 *axes_class* 初始化参数
        """
        # 从 kwargs 中弹出 'axes_class' 参数，如果不存在则使用默认的 self._defaultAxesClass
        axes_class = kwargs.pop("axes_class", self._defaultAxesClass)
        # 创建 RGB 图像的 Axes 对象，并赋值给 self.RGB
        self.RGB = ax = axes_class(*args, **kwargs)
        # 将 Axes 对象添加到其所属的 Figure 中
        ax.get_figure().add_axes(ax)
        # 使用 make_rgb_axes 函数创建并返回 RGB、R、G、B 四个 Axes 对象
        self.R, self.G, self.B = make_rgb_axes(
            ax, pad=pad, axes_class=axes_class, **kwargs)
        # 设置四个 Axes 对象的线条颜色和刻度标记颜色为白色
        for ax1 in [self.RGB, self.R, self.G, self.B]:
            if isinstance(ax1.axis, MethodType):
                # 如果 Axes 对象的 axis 是 MethodType 类型，则创建新的 AxisDict 对象
                ad = Axes.AxisDict(self)
                ad.update(
                    bottom=SimpleAxisArtist(ax1.xaxis, 1, ax1.spines["bottom"]),
                    top=SimpleAxisArtist(ax1.xaxis, 2, ax1.spines["top"]),
                    left=SimpleAxisArtist(ax1.yaxis, 1, ax1.spines["left"]),
                    right=SimpleAxisArtist(ax1.yaxis, 2, ax1.spines["right"]))
            else:
                # 否则直接使用现有的 axis 对象
                ad = ax1.axis
            # 设置 axis 对象的所有线条颜色为白色
            ad[:].line.set_color("w")
            # 设置 axis 对象的主刻度标记边缘颜色为白色
            ad[:].major_ticks.set_markeredgecolor("w")

    # 显示 RGB 图像及其各个单通道图像的函数
    def imshow_rgb(self, r, g, b, **kwargs):
        """
        Create the four images {rgb, r, g, b}.

        Parameters
        ----------
        r, g, b : array-like
            红色、绿色和蓝色通道的数组。
        **kwargs
            传递给四幅图像的 `~.Axes.imshow` 调用的参数。

        Returns
        -------
        rgb : `~matplotlib.image.AxesImage`
        r : `~matplotlib.image.AxesImage`
        g : `~matplotlib.image.AxesImage`
        b : `~matplotlib.image.AxesImage`
        """
        # 检查输入的 r、g、b 三个数组的形状是否一致，如果不一致则引发 ValueError 异常
        if not (r.shape == g.shape == b.shape):
            raise ValueError(
                f'Input shapes ({r.shape}, {g.shape}, {b.shape}) do not match')
        # 将 r、g、b 三个数组合并成一个 RGB 图像数组
        RGB = np.dstack([r, g, b])
        # 创建与 RGB 形状相同的零数组，分别用于单通道图像 R、G、B
        R = np.zeros_like(RGB)
        R[:, :, 0] = r  # 将 r 数组的值赋给 R 的红色通道
        G = np.zeros_like(RGB)
        G[:, :, 1] = g  # 将 g 数组的值赋给 G 的绿色通道
        B = np.zeros_like(RGB)
        B[:, :, 2] = b  # 将 b 数组的值赋给 B 的蓝色通道
        # 在 RGB、R、G、B 四个 Axes 对象上分别显示相应的图像，并返回四个 `~matplotlib.image.AxesImage` 对象
        im_rgb = self.RGB.imshow(RGB, **kwargs)
        im_r = self.R.imshow(R, **kwargs)
        im_g = self.G.imshow(G, **kwargs)
        im_b = self.B.imshow(B, **kwargs)
        return im_rgb, im_r, im_g, im_b
```