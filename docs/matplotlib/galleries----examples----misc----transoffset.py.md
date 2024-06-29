# `D:\src\scipysrc\matplotlib\galleries\examples\misc\transoffset.py`

```py
"""
======================
transforms.offset_copy
======================

This illustrates the use of `.transforms.offset_copy` to
make a transform that positions a drawing element such as
a text string at a specified offset in screen coordinates
(dots or inches) relative to a location given in any
coordinates.

Every Artist (Text, Line2D, etc.) has a transform that can be
set when the Artist is created, such as by the corresponding
pyplot function.  By default, this is usually the Axes.transData
transform, going from data units to screen pixels.  We can
use the `.offset_copy` function to make a modified copy of
this transform, where the modification consists of an
offset.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
import numpy as np  # 导入numpy模块

import matplotlib.transforms as mtransforms  # 导入matplotlib.transforms模块

xs = np.arange(7)  # 创建一个包含0到6的数组
ys = xs**2  # 计算xs数组中每个元素的平方，创建ys数组

fig = plt.figure(figsize=(5, 10))  # 创建一个大小为5x10的图形窗口
ax = plt.subplot(2, 1, 1)  # 创建一个2行1列的子图，并选择第1个子图作为当前子图

# 如果我们希望每个文本实例具有相同的偏移量，
# 我们只需要创建一个变换。为了获取传递给offset_copy的变换参数，
# 我们首先需要创建Axes；上面的subplot函数是这样做的一种方式。
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=0.10, units='inches')

for x, y in zip(xs, ys):
    plt.plot(x, y, 'ro')  # 在图中绘制红色圆点
    plt.text(x, y, '%d, %d' % (int(x), int(y)), transform=trans_offset)  # 在指定位置绘制文本

# offset_copy也适用于极坐标图。
ax = plt.subplot(2, 1, 2, projection='polar')  # 创建一个极坐标子图，并选择第2个子图作为当前子图

trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       y=6, units='dots')

for x, y in zip(xs, ys):
    plt.polar(x, y, 'ro')  # 在极坐标图中绘制红色圆点
    plt.text(x, y, '%d, %d' % (int(x), int(y)),
             transform=trans_offset,
             horizontalalignment='center',
             verticalalignment='bottom')  # 在指定位置绘制文本，指定水平和垂直对齐方式

plt.show()  # 显示图形
```