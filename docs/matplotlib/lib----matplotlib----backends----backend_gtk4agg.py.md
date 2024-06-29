# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk4agg.py`

```py
# 导入NumPy库，用于处理数组和向量化操作
import numpy as np

# 导入matplotlib的cbook模块，用于一般的实用函数和类
from .. import cbook

# 导入本地的backend_agg和backend_gtk4模块
from . import backend_agg, backend_gtk4

# 从backend_gtk4模块中导入GLib、Gtk和_BackendGTK4类
from .backend_gtk4 import GLib, Gtk, _BackendGTK4

# 导入cairo库，前提是已经通过_backend_gtk检查了cairo的存在
import cairo


class FigureCanvasGTK4Agg(backend_agg.FigureCanvasAgg,
                          backend_gtk4.FigureCanvasGTK4):
    # 定义on_draw_event方法，处理绘图事件
    def on_draw_event(self, widget, ctx):
        # 如果存在空闲绘图标识_idle_draw_id，则移除该标识并重置为0，然后执行绘图操作
        if self._idle_draw_id:
            GLib.source_remove(self._idle_draw_id)
            self._idle_draw_id = 0
            self.draw()

        # 获取设备像素比例
        scale = self.device_pixel_ratio
        # 获取部件的分配尺寸
        allocation = self.get_allocation()

        # 使用Gtk的方法渲染背景
        Gtk.render_background(
            self.get_style_context(), ctx,
            allocation.x, allocation.y,
            allocation.width, allocation.height)

        # 获取渲染器的RGBA缓冲区，并转换为预乘的ARGB32格式
        buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
            np.asarray(self.get_renderer().buffer_rgba()))
        
        # 获取缓冲区的高度、宽度和通道数
        height, width, _ = buf.shape
        
        # 创建cairo图像表面，使用缓冲区数据，并指定格式为ARGB32
        image = cairo.ImageSurface.create_for_data(
            buf.ravel().data, cairo.FORMAT_ARGB32, width, height)
        
        # 设置图像表面的设备比例
        image.set_device_scale(scale, scale)
        
        # 设置绘图上下文的源为图像表面，并在指定位置(0, 0)处绘制
        ctx.set_source_surface(image, 0, 0)
        ctx.paint()

        # 返回False，表示事件处理完毕
        return False


# 将_FigureCanvasGTK4Agg类导出为_BackendGTK4Agg类的FigureCanvas属性
@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
```