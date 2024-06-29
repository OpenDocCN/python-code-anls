# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk4cairo.py`

```py
from contextlib import nullcontext  # 导入 nullcontext 上下文管理器，用于创建一个空的上下文

from .backend_cairo import FigureCanvasCairo  # 导入 FigureCanvasCairo 类，作为 Cairo 后端的画布
from .backend_gtk4 import GLib, Gtk, FigureCanvasGTK4, _BackendGTK4  # 导入 GTK4 相关模块和类

class FigureCanvasGTK4Cairo(FigureCanvasCairo, FigureCanvasGTK4):
    _context_is_scaled = True  # 设置画布上下文已缩放标志为 True

    def on_draw_event(self, widget, ctx):
        if self._idle_draw_id:  # 如果存在空闲绘制的标识符
            GLib.source_remove(self._idle_draw_id)  # 移除空闲绘制的标识符
            self._idle_draw_id = 0  # 重置空闲绘制的标识符为 0
            self.draw()  # 执行绘制操作

        with (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):  # 如果存在工具栏，则使用其提供的等待光标上下文管理器
            self._renderer.set_context(ctx)  # 设置渲染器的上下文
            scale = self.device_pixel_ratio  # 获取设备像素比例
            # 将物理绘制缩放到逻辑尺寸
            ctx.scale(1 / scale, 1 / scale)  # 对上下文进行缩放操作
            allocation = self.get_allocation()  # 获取分配给窗口部件的大小和位置
            Gtk.render_background(
                self.get_style_context(), ctx,
                allocation.x, allocation.y,
                allocation.width, allocation.height)  # 使用 GTK 渲染背景
            self._renderer.dpi = self.figure.dpi  # 设置渲染器的 DPI 属性为图形的 DPI
            self.figure.draw(self._renderer)  # 绘制图形

@_BackendGTK4.export
class _BackendGTK4Cairo(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Cairo  # 将 FigureCanvas 指定为 FigureCanvasGTK4Cairo 类
```