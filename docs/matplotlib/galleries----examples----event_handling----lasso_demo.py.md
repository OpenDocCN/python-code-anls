# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\lasso_demo.py`

```
"""
==========
Lasso Demo
==========

Use a lasso to select a set of points and get the indices of the selected points.
A callback is used to change the color of the selected points.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块
import numpy as np  # 导入 NumPy 库，并使用别名 np

from matplotlib import colors as mcolors  # 导入 matplotlib 中的 colors 模块，并使用别名 mcolors
from matplotlib import path  # 导入 matplotlib 中的 path 模块
from matplotlib.collections import RegularPolyCollection  # 导入 matplotlib 中的 RegularPolyCollection 类
from matplotlib.widgets import Lasso  # 导入 matplotlib 中的 Lasso 类


class LassoManager:
    def __init__(self, ax, data):
        # The information of whether a point has been selected or not is stored in the
        # collection's array (0 = out, 1 = in), which then gets colormapped to blue
        # (out) and red (in).
        self.collection = RegularPolyCollection(
            6, sizes=(100,), offset_transform=ax.transData,
            offsets=data, array=np.zeros(len(data)),
            clim=(0, 1), cmap=mcolors.ListedColormap(["tab:blue", "tab:red"]))
        ax.add_collection(self.collection)  # 将 RegularPolyCollection 对象添加到坐标轴 ax 上
        canvas = ax.figure.canvas  # 获取坐标轴所在的画布
        canvas.mpl_connect('button_press_event', self.on_press)  # 绑定鼠标按下事件到 self.on_press 方法
        canvas.mpl_connect('button_release_event', self.on_release)  # 绑定鼠标释放事件到 self.on_release 方法

    def callback(self, verts):
        data = self.collection.get_offsets()  # 获取当前集合中的偏移量数据
        self.collection.set_array(path.Path(verts).contains_points(data))  # 根据 verts 绘制的路径来更新集合的数组
        canvas = self.collection.figure.canvas  # 获取集合所在的画布
        canvas.draw_idle()  # 在事件循环的空闲时绘制画布
        del self.lasso  # 删除当前的 Lasso 对象的引用

    def on_press(self, event):
        canvas = self.collection.figure.canvas  # 获取集合所在的画布
        if event.inaxes is not self.collection.axes or canvas.widgetlock.locked():
            return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)  # 创建一个新的 Lasso 对象
        canvas.widgetlock(self.lasso)  # 获取一个对绘图小部件的锁定

    def on_release(self, event):
        canvas = self.collection.figure.canvas  # 获取集合所在的画布
        if hasattr(self, 'lasso') and canvas.widgetlock.isowner(self.lasso):
            canvas.widgetlock.release(self.lasso)  # 释放对绘图小部件的锁定


if __name__ == '__main__':
    np.random.seed(19680801)  # 设置随机数种子以确保可重复性
    ax = plt.figure().add_subplot(  # 创建一个新的图形，并在其上添加一个子图
        xlim=(0, 1), ylim=(0, 1), title='Lasso points using left mouse button')  # 设置子图的 x 和 y 轴范围，并设置标题
    manager = LassoManager(ax, np.random.rand(100, 2))  # 创建 LassoManager 的实例对象
    plt.show()  # 显示绘制的图形
```