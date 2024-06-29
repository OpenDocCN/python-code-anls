# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_qt_sgskip.py`

```
"""
===============
Embedding in Qt
===============

Simple Qt application embedding Matplotlib canvases.  This program will work
equally well using any Qt binding (PyQt6, PySide6, PyQt5, PySide2).  The
binding can be selected by setting the :envvar:`QT_API` environment variable to
the binding name, or by first importing it.
"""

# 导入所需的模块
import sys  # 导入系统模块
import time  # 导入时间模块

import numpy as np  # 导入NumPy库

# 导入Matplotlib相关模块
from matplotlib.backends.backend_qtagg import FigureCanvas  # 导入Matplotlib的画布类
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar  # 导入Matplotlib的QT导航工具栏
from matplotlib.backends.qt_compat import QtWidgets  # 导入QtWidgets模块
from matplotlib.figure import Figure  # 导入Matplotlib的Figure类


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 创建主窗口部件
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)  # 将主窗口部件设置为中心部件
        layout = QtWidgets.QVBoxLayout(self._main)  # 创建垂直布局管理器

        # 创建静态画布
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # 将静态画布的导航工具栏添加到布局
        layout.addWidget(NavigationToolbar(static_canvas, self))
        # 将静态画布添加到布局
        layout.addWidget(static_canvas)

        # 创建动态画布
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # 将动态画布的导航工具栏添加到布局
        layout.addWidget(NavigationToolbar(dynamic_canvas, self))
        # 将动态画布添加到布局
        layout.addWidget(dynamic_canvas)

        # 在静态画布上创建子图并绘制正切函数图像
        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        # 在动态画布上创建子图并绘制正弦函数图像
        self._dynamic_ax = dynamic_canvas.figure.subplots()
        t = np.linspace(0, 10, 101)
        # 创建Line2D对象
        self._line, = self._dynamic_ax.plot(t, np.sin(t + time.time()))
        # 创建定时器，并设定定时器回调函数为更新画布函数
        self._timer = dynamic_canvas.new_timer(50)
        self._timer.add_callback(self._update_canvas)
        self._timer.start()

    def _update_canvas(self):
        t = np.linspace(0, 10, 101)
        # 根据时间偏移更新正弦函数图像的数据
        self._line.set_data(t, np.sin(t + time.time()))
        # 重新绘制画布
        self._line.figure.canvas.draw()


if __name__ == "__main__":
    # 检查是否已经有运行中的QApplication实例（例如，在IDE中运行时）
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)  # 如果没有实例，则创建QApplication实例

    # 创建并显示主窗口应用程序
    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()  # 启动Qt事件循环
```