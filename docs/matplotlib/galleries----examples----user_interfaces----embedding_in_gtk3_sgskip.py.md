# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_gtk3_sgskip.py`

```
"""
=================
Embedding in GTK3
=================

Demonstrate adding a FigureCanvasGTK3Agg widget to a Gtk.ScrolledWindow using
GTK3 accessed via pygobject.
"""

import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import numpy as np

from matplotlib.backends.backend_gtk3agg import \
    FigureCanvasGTK3Agg as FigureCanvas  # 导入用于在GTK3中嵌入Matplotlib图形的类
from matplotlib.figure import Figure  # 导入Matplotlib图形类

win = Gtk.Window()  # 创建一个GTK窗口对象
win.connect("delete-event", Gtk.main_quit)  # 连接窗口关闭事件到Gtk主循环的退出函数
win.set_default_size(400, 300)  # 设置窗口的默认尺寸为400x300像素
win.set_title("Embedding in GTK3")  # 设置窗口标题为"Embedding in GTK3"

fig = Figure(figsize=(5, 4), dpi=100)  # 创建一个Matplotlib图形对象，尺寸为5x4英寸，分辨率为100dpi
ax = fig.add_subplot()  # 添加一个子图到图形对象fig

t = np.arange(0.0, 3.0, 0.01)  # 创建一个从0到3的数组，步长为0.01
s = np.sin(2*np.pi*t)  # 计算sin函数值
ax.plot(t, s)  # 在子图ax上绘制t和s的曲线

sw = Gtk.ScrolledWindow()  # 创建一个滚动窗口对象sw
win.add(sw)  # 将滚动窗口sw添加到GTK窗口win中

# 设置滚动窗口边框宽度为10像素
sw.set_border_width(10)

canvas = FigureCanvas(fig)  # 创建一个FigureCanvas对象canvas，用于在GTK中显示Matplotlib图形
canvas.set_size_request(800, 600)  # 设置canvas的显示尺寸为800x600像素
sw.add(canvas)  # 将canvas添加到滚动窗口sw中

win.show_all()  # 显示窗口win及其所有子组件
Gtk.main()  # 进入GTK主循环，等待事件处理
```