# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qtagg.py`

```py
"""
Render to qt from agg.
"""

import ctypes  # 导入 ctypes 库，用于处理 C 数据类型和调用 DLL 函数

from matplotlib.transforms import Bbox  # 导入 Bbox 类，用于表示图形的边界框

from .qt_compat import QT_API, QtCore, QtGui  # 导入 QT_API、QtCore 和 QtGui，用于跨不同版本的 PyQt/PySide 兼容性
from .backend_agg import FigureCanvasAgg  # 导入 FigureCanvasAgg 类，用于绘制到 Agg canvas
from .backend_qt import _BackendQT, FigureCanvasQT  # 导入 _BackendQT 和 FigureCanvasQT 类，用于 Qt 后端支持
from .backend_qt import (  # 导入 FigureManagerQT 和 NavigationToolbar2QT 类
    FigureManagerQT, NavigationToolbar2QT)


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):
    """
    A Qt widget for rendering Matplotlib plots onto Agg canvas.
    Inherits from both FigureCanvasAgg and FigureCanvasQT.
    """

    def paintEvent(self, event):
        """
        Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        self._draw_idle()  # 如果有绘图请求，执行绘图操作

        # 如果画布没有渲染器，等待 FigureCanvasAgg.draw(self) 被调用
        if not hasattr(self, 'renderer'):
            return

        painter = QtGui.QPainter(self)  # 创建 QPainter 对象，用于在 widget 上绘制
        try:
            rect = event.rect()  # 获取事件的矩形区域
            # 使用屏幕 DPI 比例来缩放矩形尺寸，以获取正确的 Figure 坐标值（而不是 QT5 的坐标）
            width = rect.width() * self.device_pixel_ratio
            height = rect.height() * self.device_pixel_ratio
            left, top = self.mouseEventCoords(rect.topLeft())  # 获取鼠标事件的坐标
            bottom = top - height  # 根据图像高度调整 "top" 边界以匹配坐标系
            right = left + width  # 根据图像宽度调整右边界
            bbox = Bbox([[left, bottom], [right, top]])  # 创建图像边界框
            buf = memoryview(self.copy_from_bbox(bbox))  # 从边界框复制数据到缓冲区

            # 根据不同的 QT_API 版本处理缓冲区指针
            if QT_API == "PyQt6":
                from PyQt6 import sip
                ptr = int(sip.voidptr(buf))
            else:
                ptr = buf

            painter.eraseRect(rect)  # 清除 widget 画布
            # 创建 QImage 对象，用于在 widget 上显示图像
            qimage = QtGui.QImage(ptr, buf.shape[1], buf.shape[0],
                                  QtGui.QImage.Format.Format_RGBA8888)
            qimage.setDevicePixelRatio(self.device_pixel_ratio)  # 设置设备像素比例
            origin = QtCore.QPoint(rect.left(), rect.top())  # 设置图像的原点坐标
            painter.drawImage(origin, qimage)  # 在指定坐标处绘制图像

            # 解决 PySide2 下 QImage 内存泄漏的问题
            if QT_API == "PySide2" and QtCore.__version_info__ < (5, 12):
                ctypes.c_long.from_address(id(buf)).value = 1

            self._draw_rect_callback(painter)  # 执行绘制矩形的回调函数
        finally:
            painter.end()  # 结束绘图操作
    # 调用父类的 print_figure 方法，传递所有位置参数和关键字参数
    super().print_figure(*args, **kwargs)
    # 在某些情况下，Qt 可能在关闭文件保存对话框后自行触发绘制事件。当这种情况发生时，
    # 我们需要确保内部画布被重新绘制。然而，如果用户正在使用自动选择的 Qt 后端，但使用
    # 不同的后端（如 pgf）保存，我们不希望触发 Qt 的完整绘制，因此只需设置下一次绘制的标志。
    self._draw_pending = True
# 将 _BackendQTAgg 类导出为 QT 后端的一部分
@_BackendQT.export
# 定义 _BackendQTAgg 类，继承自 _BackendQT 类
class _BackendQTAgg(_BackendQT):
    # 设置 FigureCanvas 类为 FigureCanvasQTAgg
    FigureCanvas = FigureCanvasQTAgg
```