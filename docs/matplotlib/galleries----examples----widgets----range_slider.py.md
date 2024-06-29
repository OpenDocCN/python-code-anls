# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\range_slider.py`

```py
"""
======================================
Thresholding an Image with RangeSlider
======================================

Using the RangeSlider widget to control the thresholding of an image.

The RangeSlider widget can be used similarly to the `.widgets.Slider`
widget. The major difference is that RangeSlider's ``val`` attribute
is a tuple of floats ``(lower val, upper val)`` rather than a single float.

See :doc:`/gallery/widgets/slider_demo` for an example of using
a ``Slider`` to control a single float.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于生成数据

from matplotlib.widgets import RangeSlider  # 导入 RangeSlider 组件

# generate a fake image
np.random.seed(19680801)  # 设定随机数种子以便重现随机生成的数据
N = 128  # 图像尺寸
img = np.random.randn(N, N)  # 生成一个随机的 N x N 大小的二维数组作为图像数据

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建一个包含两个子图的图像窗口
fig.subplots_adjust(bottom=0.25)  # 调整子图布局，确保底部空间充足

im = axs[0].imshow(img)  # 在第一个子图中显示图像
axs[1].hist(img.flatten(), bins='auto')  # 在第二个子图中绘制图像像素强度的直方图
axs[1].set_title('Histogram of pixel intensities')  # 设置第二个子图的标题

# Create the RangeSlider
slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])  # 在图像窗口中添加一个坐标轴用于放置 RangeSlider
slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max())  # 创建 RangeSlider 实例，用于设置阈值范围

# Create the Vertical lines on the histogram
lower_limit_line = axs[1].axvline(slider.val[0], color='k')  # 在直方图中添加垂直线，表示下限
upper_limit_line = axs[1].axvline(slider.val[1], color='k')  # 在直方图中添加垂直线，表示上限


def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    im.norm.vmin = val[0]  # 更新图像的 colormap 下限
    im.norm.vmax = val[1]  # 更新图像的 colormap 上限

    # Update the position of the vertical lines
    lower_limit_line.set_xdata([val[0], val[0]])  # 更新下限线的位置
    upper_limit_line.set_xdata([val[1], val[1]])  # 更新上限线的位置

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()  # 重新绘制图像窗口，确保更新生效


slider.on_changed(update)  # 将 update 函数注册为 RangeSlider 值变化时的回调函数
plt.show()  # 显示图像窗口
```