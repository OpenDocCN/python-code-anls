# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\zoom_window.py`

```
"""
===========
Zoom Window
===========

This example shows how to connect events in one window, for example, a mouse
press, to another figure window.

If you click on a point in the first window, the z and y limits of the second
will be adjusted so that the center of the zoom in the second window will be
the (x, y) coordinates of the clicked point.

Note the diameter of the circles in the scatter are defined in points**2, so
their size is independent of the zoom.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# 创建源图和放大图的子图对象
figsrc, axsrc = plt.subplots(figsize=(3.7, 3.7))
figzoom, axzoom = plt.subplots(figsize=(3.7, 3.7))

# 设置源图的初始 x 和 y 范围，并关闭自动缩放
axsrc.set(xlim=(0, 1), ylim=(0, 1), autoscale_on=False,
          title='Click to zoom')

# 设置放大图的初始 x 和 y 范围，并关闭自动缩放
axzoom.set(xlim=(0.45, 0.55), ylim=(0.4, 0.6), autoscale_on=False,
           title='Zoom window')

# 生成随机数据
x, y, s, c = np.random.rand(4, 200)
s *= 200

# 在源图和放大图上绘制散点图
axsrc.scatter(x, y, s, c)
axzoom.scatter(x, y, s, c)

# 定义鼠标按下事件处理函数
def on_press(event):
    # 如果不是鼠标左键按下，则返回
    if event.button != 1:
        return
    # 获取鼠标点击的坐标
    x, y = event.xdata, event.ydata
    # 设置放大图的新 x 和 y 范围
    axzoom.set_xlim(x - 0.1, x + 0.1)
    axzoom.set_ylim(y - 0.1, y + 0.1)
    # 重新绘制放大图
    figzoom.canvas.draw()

# 将鼠标按下事件连接到源图的绘图画布
figsrc.canvas.mpl_connect('button_press_event', on_press)

# 显示图形
plt.show()
```