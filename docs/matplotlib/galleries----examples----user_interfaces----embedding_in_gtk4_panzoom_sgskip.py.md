# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_gtk4_panzoom_sgskip.py`

```py
"""
===========================================
Embedding in GTK4 with a navigation toolbar
===========================================

Demonstrate NavigationToolbar with GTK4 accessed via pygobject.
"""

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

import numpy as np

from matplotlib.backends.backend_gtk4 import \
    NavigationToolbar2GTK4 as NavigationToolbar
from matplotlib.backends.backend_gtk4agg import \
    FigureCanvasGTK4Agg as FigureCanvas
from matplotlib.figure import Figure


def on_activate(app):
    # 创建一个 GTK 应用窗口
    win = Gtk.ApplicationWindow(application=app)
    # 设置窗口默认尺寸
    win.set_default_size(400, 300)
    # 设置窗口标题
    win.set_title("Embedding in GTK4")

    # 创建一个 Matplotlib 图形对象
    fig = Figure(figsize=(5, 4), dpi=100)
    # 在图形对象上添加一个子图
    ax = fig.add_subplot(1, 1, 1)
    # 生成一组数据
    t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2*np.pi*t)
    # 在子图上绘制数据
    ax.plot(t, s)

    # 创建一个垂直布局的 GTK 容器
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    # 将 vbox 设置为窗口的主要子元素
    win.set_child(vbox)

    # 创建一个 Matplotlib 的绘图区域并添加到 vbox 中
    canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    canvas.set_hexpand(True)
    canvas.set_vexpand(True)
    vbox.append(canvas)

    # 创建一个 Matplotlib 导航工具栏并添加到 vbox 中
    toolbar = NavigationToolbar(canvas)
    vbox.append(toolbar)

    # 显示窗口
    win.show()


# 创建一个 GTK 应用实例
app = Gtk.Application(
    application_id='org.matplotlib.examples.EmbeddingInGTK4PanZoom')
# 连接应用的 'activate' 信号到 on_activate 函数
app.connect('activate', on_activate)
# 运行 GTK 应用，不传递参数
app.run(None)
```