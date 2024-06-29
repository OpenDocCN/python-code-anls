# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\image_slices_viewer.py`

```py
"""
============
Scroll event
============

In this example a scroll wheel event is used to scroll through 2D slices of
3D data.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库并简称为 plt
import numpy as np  # 导入 numpy 库并简称为 np


class IndexTracker:
    def __init__(self, ax, X):
        self.index = 0  # 初始化索引为0
        self.X = X  # 将输入的数据存储为实例变量
        self.ax = ax  # 将输入的坐标轴对象存储为实例变量
        self.im = ax.imshow(self.X[:, :, self.index])  # 在坐标轴上显示X的第一个切片，并存储图像对象
        self.update()  # 更新显示内容

    def on_scroll(self, event):
        # 打印滚轮事件的按钮和步长
        print(event.button, event.step)
        # 根据滚轮向上或向下滚动来增加或减少索引
        increment = 1 if event.button == 'up' else -1
        max_index = self.X.shape[-1] - 1  # 计算索引的最大值
        self.index = np.clip(self.index + increment, 0, max_index)  # 确保索引在合法范围内
        self.update()  # 更新显示内容

    def update(self):
        # 更新图像数据
        self.im.set_data(self.X[:, :, self.index])
        # 设置坐标轴标题，显示当前索引
        self.ax.set_title(
            f'Use scroll wheel to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()


x, y, z = np.ogrid[-10:10:100j, -10:10:100j, 1:10:20j]  # 生成三维网格数据
X = np.sin(x * y * z) / (x * y * z)  # 计算三维数据的sin值并进行归一化处理

fig, ax = plt.subplots()  # 创建图形和坐标轴对象

# 创建 IndexTracker 实例，并通过变量 tracker 使其在整个图形的生命周期内存在
tracker = IndexTracker(ax, X)

# 将图形的滚轮事件连接到 IndexTracker 实例的 on_scroll 方法上
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()  # 显示图形
```