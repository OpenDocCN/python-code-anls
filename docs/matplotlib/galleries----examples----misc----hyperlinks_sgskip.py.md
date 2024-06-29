# `D:\src\scipysrc\matplotlib\galleries\examples\misc\hyperlinks_sgskip.py`

```
"""
==========
Hyperlinks
==========

This example demonstrates how to set hyperlinks on various kinds of elements.

This currently only works with the SVG backend.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 数学计算库

import matplotlib.cm as cm  # 导入 matplotlib 的色彩映射模块

# %%

fig = plt.figure()  # 创建一个新的图形对象
s = plt.scatter([1, 2, 3], [4, 5, 6])  # 绘制散点图，并将对象赋给 s
s.set_urls(['https://www.bbc.com/news', 'https://www.google.com/', None])  # 为散点图设置点击链接
fig.savefig('scatter.svg')  # 将图形保存为 scatter.svg

# %%

fig = plt.figure()  # 创建一个新的图形对象
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)  # 创建一个均匀间隔的数组
X, Y = np.meshgrid(x, y)  # 生成网格坐标
Z1 = np.exp(-X**2 - Y**2)  # 计算第一个高斯分布
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  # 计算第二个高斯分布
Z = (Z1 - Z2) * 2  # 计算两个高斯分布的差值

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,  # 绘制图像并设置插值、颜色映射等属性
                origin='lower', extent=(-3, 3, -3, 3))

im.set_url('https://www.google.com/')  # 为图像设置点击链接
fig.savefig('image.svg')  # 将图形保存为 image.svg
```