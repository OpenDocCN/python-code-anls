# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_gtk3_panzoom_sgskip.py`

```
"""
===========================================
Embedding in GTK3 with a navigation toolbar
===========================================

Demonstrate NavigationToolbar with GTK3 accessed via pygobject.
"""

import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import numpy as np

from matplotlib.backends.backend_gtk3 import \
    NavigationToolbar2GTK3 as NavigationToolbar  # 导入 GTK3 的 Matplotlib 导航工具栏
from matplotlib.backends.backend_gtk3agg import \
    FigureCanvasGTK3Agg as FigureCanvas  # 导入 GTK3 的 Matplotlib 图形画布
from matplotlib.figure import Figure  # 导入 Matplotlib 图形对象

win = Gtk.Window()  # 创建 GTK 窗口对象
win.connect("delete-event", Gtk.main_quit)  # 连接窗口关闭事件到 Gtk.main_quit 函数
win.set_default_size(400, 300)  # 设置窗口默认大小
win.set_title("Embedding in GTK3")  # 设置窗口标题

fig = Figure(figsize=(5, 4), dpi=100)  # 创建 Matplotlib 图形对象
ax = fig.add_subplot(1, 1, 1)  # 添加子图到图形对象
t = np.arange(0.0, 3.0, 0.01)
s = np.sin(2*np.pi*t)
ax.plot(t, s)  # 在子图上绘制 sin 曲线

vbox = Gtk.VBox()  # 创建垂直布局容器对象
win.add(vbox)  # 将垂直布局容器添加到窗口中

# Add canvas to vbox
canvas = FigureCanvas(fig)  # 创建 Matplotlib 图形画布对象，实际上是 GTK 的 DrawingArea
vbox.pack_start(canvas, True, True, 0)  # 将图形画布添加到垂直布局容器中

# Create toolbar
toolbar = NavigationToolbar(canvas)  # 创建 Matplotlib 导航工具栏对象
vbox.pack_start(toolbar, False, False, 0)  # 将导航工具栏添加到垂直布局容器中

win.show_all()  # 显示窗口及其所有子组件
Gtk.main()  # 进入 GTK 主循环
```