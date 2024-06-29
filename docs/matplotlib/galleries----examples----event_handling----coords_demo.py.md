# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\coords_demo.py`

```
"""
===========================
Mouse move and click events
===========================

An example of how to interact with the plotting canvas by connecting to move
and click events.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.backend_bases import MouseButton  # 从 matplotlib.backend_bases 模块中导入 MouseButton 类

# 生成示例数据
t = np.arange(0.0, 1.0, 0.01)  # 创建一个从 0 到 1（不包含）的数组，步长为 0.01
s = np.sin(2 * np.pi * t)  # 计算这些点对应的正弦值

# 创建图形和坐标系
fig, ax = plt.subplots()  # 创建一个图形对象和一个坐标系对象
ax.plot(t, s)  # 在坐标系中绘制正弦波形

# 定义鼠标移动事件的回调函数
def on_move(event):
    if event.inaxes:  # 如果事件发生在坐标系内部
        print(f'data coords {event.xdata} {event.ydata},',  # 打印数据坐标
              f'pixel coords {event.x} {event.y}')  # 打印像素坐标

# 定义鼠标点击事件的回调函数
def on_click(event):
    if event.button is MouseButton.LEFT:  # 如果是左键点击
        print('disconnecting callback')  # 打印断开回调的消息
        plt.disconnect(binding_id)  # 断开事件绑定的回调函数

# 将鼠标移动事件和点击事件连接到 matplotlib 的交互事件
binding_id = plt.connect('motion_notify_event', on_move)  # 将 on_move 函数绑定到鼠标移动事件
plt.connect('button_press_event', on_click)  # 将 on_click 函数绑定到鼠标点击事件

plt.show()  # 显示绘制的图形
"""
```