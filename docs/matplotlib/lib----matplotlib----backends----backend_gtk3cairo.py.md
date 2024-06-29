# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk3cairo.py`

```py
from contextlib import nullcontext  # 导入空上下文管理器，用于条件性地创建上下文

from .backend_cairo import FigureCanvasCairo  # 导入Cairo后端的画布类
from .backend_gtk3 import GLib, Gtk, FigureCanvasGTK3, _BackendGTK3  # 导入GTK3相关类


class FigureCanvasGTK3Cairo(FigureCanvasCairo, FigureCanvasGTK3):
    def on_draw_event(self, widget, ctx):
        if self._idle_draw_id:
            GLib.source_remove(self._idle_draw_id)  # 移除之前的闲置绘制任务
            self._idle_draw_id = 0
            self.draw()  # 执行绘制操作

        with (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):  # 若工具栏存在，则使用等待光标上下文管理器
            self._renderer.set_context(ctx)  # 设置渲染器的上下文
            scale = self.device_pixel_ratio  # 获取设备像素比例
            # 缩放物理绘图到逻辑大小
            ctx.scale(1 / scale, 1 / scale)
            allocation = self.get_allocation()  # 获取分配的空间
            Gtk.render_background(
                self.get_style_context(), ctx,  # 在给定上下文中渲染背景
                allocation.x, allocation.y,
                allocation.width, allocation.height)
            self._renderer.dpi = self.figure.dpi  # 设置渲染器的 DPI
            self.figure.draw(self._renderer)  # 在画布上绘制图形


@_BackendGTK3.export
class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Cairo  # 设置后端类为GTK3与Cairo结合的画布
```