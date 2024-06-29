# `D:\src\scipysrc\matplotlib\galleries\examples\misc\demo_ribbon_box.py`

```py
"""
==========
Ribbon Box
==========

"""

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np

# 从matplotlib中导入必要的模块和类
from matplotlib import cbook
from matplotlib import colors as mcolors
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox, BboxTransformTo, TransformedBbox

# 定义一个RibbonBox类
class RibbonBox:

    # 加载原始图像
    original_image = plt.imread(
        cbook.get_sample_data("Minduka_Present_Blue_Pack.png"))
    
    # 定义切割位置
    cut_location = 70
    
    # 提取颜色和亮度通道
    b_and_h = original_image[:, :, 2:3]
    color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
    alpha = original_image[:, :, 3:4]
    nx = original_image.shape[1]

    # 初始化方法，根据颜色创建图像
    def __init__(self, color):
        rgb = mcolors.to_rgb(color)
        self.im = np.dstack(
            [self.b_and_h - self.color * (1 - np.array(rgb)), self.alpha])

    # 获取拉伸后的图像
    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)
        return np.vstack(
            [self.im[:self.cut_location],
             np.broadcast_to(
                 self.im[self.cut_location], (ny2 - ny, nx, nch)),
             self.im[self.cut_location:]])


# 定义一个RibbonBoxImage类，继承自AxesImage类
class RibbonBoxImage(AxesImage):
    zorder = 1

    # 初始化方法，接受轴、边界框、颜色等参数
    def __init__(self, ax, bbox, color, *, extent=(0, 1, 0, 1), **kwargs):
        super().__init__(ax, extent=extent, **kwargs)
        self._bbox = bbox
        self._ribbonbox = RibbonBox(color)
        self.set_transform(BboxTransformTo(bbox))

    # 绘制方法，根据边界框的高宽比例进行图像拉伸
    def draw(self, renderer):
        stretch_factor = self._bbox.height / self._bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self.get_array() is None or self.get_array().shape[0] != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)

        super().draw(renderer)


# 主函数，用于生成主图表
def main():
    # 创建图和轴
    fig, ax = plt.subplots()

    # 年份和箱子高度数据
    years = np.arange(2004, 2009)
    heights = [7900, 8100, 7900, 6900, 2800]
    
    # 箱子的颜色
    box_colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.7, 0.5, 0.8),
        (0.3, 0.8, 0.7),
    ]

    # 遍历年份、高度和颜色，创建RibbonBoxImage实例并添加到轴上
    for year, h, bc in zip(years, heights, box_colors):
        bbox0 = Bbox.from_extents(year - 0.4, 0., year + 0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        ax.add_artist(RibbonBoxImage(ax, bbox, bc, interpolation="bicubic"))
        ax.annotate(str(h), (year, h), va="bottom", ha="center")

    # 设置轴的x和y轴限制
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 10000)

    # 创建背景渐变图像，并显示在轴上
    background_gradient = np.zeros((2, 2, 4))
    background_gradient[:, :, :3] = [1, 1, 0]
    background_gradient[:, :, 3] = [[0.1, 0.3], [0.3, 0.5]]  # alpha通道
    ax.imshow(background_gradient, interpolation="bicubic", zorder=0.1,
              extent=(0, 1, 0, 1), transform=ax.transAxes)

    # 显示图表
    plt.show()


# 调用主函数
main()
```