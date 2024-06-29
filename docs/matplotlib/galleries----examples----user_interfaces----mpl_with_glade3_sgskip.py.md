# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\mpl_with_glade3_sgskip.py`

```py
"""
=======================
Matplotlib with Glade 3
=======================
"""

from pathlib import Path  # 导入 Path 类，用于处理文件路径

import gi  # 导入 gi 模块，用于 GTK+ 库的集成

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk  # 导入 Gtk 模块，用于创建 GTK+ 应用程序界面

import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib.backends.backend_gtk3agg import \
    FigureCanvasGTK3Agg as FigureCanvas  # 导入 Matplotlib GTK3Agg 后端的 Canvas 组件
from matplotlib.figure import Figure  # 导入 Matplotlib 的 Figure 组件


class Window1Signals:
    def on_window1_destroy(self, widget):
        Gtk.main_quit()  # 当窗口关闭时退出 Gtk 主循环


def main():
    builder = Gtk.Builder()  # 创建 Gtk.Builder 对象，用于从 Glade 文件构建界面
    builder.add_objects_from_file(
        str(Path(__file__).parent / "mpl_with_glade3.glade"),  # 加载 Glade 文件中的窗口对象
        ("window1", ""))
    builder.connect_signals(Window1Signals())  # 将信号与处理函数关联
    window = builder.get_object("window1")  # 获取 Glade 文件中的 window1 窗口对象
    sw = builder.get_object("scrolledwindow1")  # 获取 Glade 文件中的 scrolledwindow1 对象

    # Start of Matplotlib specific code
    figure = Figure(figsize=(8, 6), dpi=71)  # 创建一个 Figure 对象，设置大小和分辨率
    axis = figure.add_subplot()  # 添加一个子图到 Figure 上
    t = np.arange(0.0, 3.0, 0.01)  # 生成一个从0到3的数组，步长为0.01
    s = np.sin(2*np.pi*t)  # 计算正弦函数值
    axis.plot(t, s)  # 绘制正弦函数图像到子图上

    axis.set_xlabel('time [s]')  # 设置 x 轴标签
    axis.set_ylabel('voltage [V]')  # 设置 y 轴标签

    canvas = FigureCanvas(figure)  # 创建一个 FigureCanvas 对象，用于在 GTK 界面中显示 Matplotlib 图像
    canvas.set_size_request(800, 600)  # 设置 Canvas 组件的尺寸
    sw.add(canvas)  # 将 Canvas 添加到 scrolledwindow1 中显示
    # End of Matplotlib specific code

    window.show_all()  # 显示窗口及其所有子控件
    Gtk.main()  # 启动 Gtk 主循环，进入事件处理状态


if __name__ == "__main__":
    main()  # 当作为主程序运行时调用 main 函数
```