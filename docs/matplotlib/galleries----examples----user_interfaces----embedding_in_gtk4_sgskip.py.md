# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_gtk4_sgskip.py`

```
"""
=================
Embedding in GTK4
=================

Demonstrate adding a FigureCanvasGTK4Agg widget to a Gtk.ScrolledWindow using
GTK4 accessed via pygobject.
"""

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

import numpy as np

from matplotlib.backends.backend_gtk4agg import \
    FigureCanvasGTK4Agg as FigureCanvas  # 导入 matplotlib 中 GTK4 的绘图组件
from matplotlib.figure import Figure  # 导入 matplotlib 的 Figure 类


def on_activate(app):
    win = Gtk.ApplicationWindow(application=app)  # 创建 GTK 应用窗口对象
    win.set_default_size(400, 300)  # 设置窗口默认尺寸
    win.set_title("Embedding in GTK4")  # 设置窗口标题

    fig = Figure(figsize=(5, 4), dpi=100)  # 创建 matplotlib 的 Figure 对象
    ax = fig.add_subplot()  # 添加一个子图到 Figure 上
    t = np.arange(0.0, 3.0, 0.01)  # 创建一个 numpy 数组
    s = np.sin(2*np.pi*t)  # 计算正弦函数值
    ax.plot(t, s)  # 在子图上绘制正弦函数图像

    # 创建一个带有边距的滚动窗口对象
    sw = Gtk.ScrolledWindow(margin_top=10, margin_bottom=10,
                            margin_start=10, margin_end=10)
    win.set_child(sw)  # 将滚动窗口设置为窗口的子组件

    canvas = FigureCanvas(fig)  # 创建一个 matplotlib 的绘图区域组件
    canvas.set_size_request(800, 600)  # 设置绘图区域的大小
    sw.set_child(canvas)  # 将绘图区域组件设置为滚动窗口的子组件

    win.show()  # 显示窗口


app = Gtk.Application(application_id='org.matplotlib.examples.EmbeddingInGTK4')  # 创建 GTK 应用对象
app.connect('activate', on_activate)  # 连接 activate 信号到 on_activate 函数
app.run(None)  # 运行 GTK 应用
```