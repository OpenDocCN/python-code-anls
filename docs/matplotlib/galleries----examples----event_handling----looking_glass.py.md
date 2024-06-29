# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\looking_glass.py`

```py
"""
=============
Looking Glass
=============

Example using mouse events to simulate a looking glass for inspecting data.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy库用于生成随机数据

import matplotlib.patches as patches  # 导入matplotlib中的patches模块，用于创建图形对象

# 设置随机数种子以便结果可重现
np.random.seed(19680801)

# 生成随机数据
x, y = np.random.rand(2, 200)

# 创建图形和坐标系
fig, ax = plt.subplots()

# 创建一个黄色半透明圆形对象并添加到坐标系中
circ = patches.Circle((0.5, 0.5), 0.25, alpha=0.8, fc='yellow')
ax.add_patch(circ)

# 绘制随机散点图，并设置透明度
ax.plot(x, y, alpha=0.2)

# 绘制原始的随机线条，并设置透明度，并且设置剪裁路径为圆形
line, = ax.plot(x, y, alpha=1.0, clip_path=circ)

# 设置图表标题
ax.set_title("Left click and drag to move looking glass")


# 定义事件处理类
class EventHandler:
    def __init__(self):
        # 连接图形的鼠标按下、释放和移动事件到相应的处理函数
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        # 初始化圆心的位置
        self.x0, self.y0 = circ.center
        self.pressevent = None

    # 鼠标按下事件处理函数
    def on_press(self, event):
        # 如果不在坐标系内部按下鼠标，则不处理
        if event.inaxes != ax:
            return
        # 如果不在圆形内部按下鼠标，则不处理
        if not circ.contains(event)[0]:
            return
        # 记录按下的事件
        self.pressevent = event

    # 鼠标释放事件处理函数
    def on_release(self, event):
        # 清空按下事件的记录
        self.pressevent = None
        # 更新圆形的圆心位置
        self.x0, self.y0 = circ.center

    # 鼠标移动事件处理函数
    def on_move(self, event):
        # 如果没有按下鼠标或者移动不在按下的坐标系内，则不处理
        if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
            return
        # 计算移动的距离
        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        # 更新圆形的圆心位置
        circ.center = self.x0 + dx, self.y0 + dy
        # 更新剪裁路径为新的圆形位置
        line.set_clip_path(circ)
        # 重新绘制图形
        fig.canvas.draw()

# 创建事件处理对象
handler = EventHandler()

# 显示绘制的图形
plt.show()
```