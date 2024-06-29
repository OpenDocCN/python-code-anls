# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qtcairo.py`

```py
import ctypes  # 导入 ctypes 库，用于处理 C 数据类型和调用 DLLs

from .backend_cairo import cairo, FigureCanvasCairo  # 导入 cairo 和 FigureCanvasCairo 类
from .backend_qt import _BackendQT, FigureCanvasQT  # 导入 _BackendQT 和 FigureCanvasQT 类
from .qt_compat import QT_API, QtCore, QtGui  # 导入 QT_API, QtCore 和 QtGui 模块


class FigureCanvasQTCairo(FigureCanvasCairo, FigureCanvasQT):
    def draw(self):
        if hasattr(self._renderer.gc, "ctx"):  # 检查是否有上下文对象 gc.ctx
            self._renderer.dpi = self.figure.dpi  # 设置渲染器的 DPI 为图形对象的 DPI
            self.figure.draw(self._renderer)  # 调用图形对象的 draw 方法进行绘制
        super().draw()  # 调用父类 FigureCanvasQT 的 draw 方法

    def paintEvent(self, event):
        width = int(self.device_pixel_ratio * self.width())  # 计算绘制宽度，考虑设备像素比
        height = int(self.device_pixel_ratio * self.height())  # 计算绘制高度，考虑设备像素比
        if (width, height) != self._renderer.get_canvas_width_height():
            # 如果宽高与当前画布大小不一致，则创建新的 Cairo 图像表面
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            self._renderer.set_context(cairo.Context(surface))  # 设置渲染器的上下文为新创建的 Cairo 上下文
            self._renderer.dpi = self.figure.dpi  # 设置渲染器的 DPI 为图形对象的 DPI
            self.figure.draw(self._renderer)  # 调用图形对象的 draw 方法进行绘制
        buf = self._renderer.gc.ctx.get_target().get_data()  # 获取渲染器的上下文目标数据
        if QT_API == "PyQt6":
            from PyQt6 import sip
            ptr = int(sip.voidptr(buf))  # 使用 PyQt6 的 sip 模块获取缓冲区指针
        else:
            ptr = buf  # 否则直接使用 buf
        qimage = QtGui.QImage(
            ptr, width, height,
            QtGui.QImage.Format.Format_ARGB32_Premultiplied)  # 创建 QImage 对象，使用渲染器的上下文数据
        # 修复 PySide2 下的 QImage 内存泄漏问题，根据 QT_API 和 PySide2 版本号
        if QT_API == "PySide2" and QtCore.__version_info__ < (5, 12):
            ctypes.c_long.from_address(id(buf)).value = 1
        qimage.setDevicePixelRatio(self.device_pixel_ratio)  # 设置 QImage 的设备像素比
        painter = QtGui.QPainter(self)  # 创建 QPainter 对象，用于在窗口中绘制
        painter.eraseRect(event.rect())  # 擦除指定区域的内容
        painter.drawImage(0, 0, qimage)  # 在指定位置绘制 QImage
        self._draw_rect_callback(painter)  # 调用回调函数绘制额外的矩形
        painter.end()  # 结束绘制


@_BackendQT.export
class _BackendQTCairo(_BackendQT):
    FigureCanvas = FigureCanvasQTCairo  # 将 FigureCanvasQTCairo 类设置为 _BackendQTCairo 的 FigureCanvas
```