# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk3agg.py`

```
import numpy as np  # 导入 numpy 库，用于数值计算

from .. import cbook, transforms  # 导入当前目录上级的 cbook 和 transforms 模块
from . import backend_agg, backend_gtk3  # 导入当前目录的 backend_agg 和 backend_gtk3 模块
from .backend_gtk3 import GLib, Gtk, _BackendGTK3  # 从 backend_gtk3 模块中导入 GLib、Gtk、_BackendGTK3

import cairo  # 导入 cairo 库，此处假设 _backend_gtk 已经检查了 cairo 库的存在


class FigureCanvasGTK3Agg(backend_agg.FigureCanvasAgg,
                          backend_gtk3.FigureCanvasGTK3):
    def __init__(self, figure):
        super().__init__(figure=figure)  # 调用父类的构造方法初始化 FigureCanvasAgg 和 FigureCanvasGTK3
        self._bbox_queue = []  # 初始化 _bbox_queue 属性为空列表

    def on_draw_event(self, widget, ctx):
        if self._idle_draw_id:  # 如果存在 _idle_draw_id，表示之前注册的绘制事件
            GLib.source_remove(self._idle_draw_id)  # 移除之前注册的绘制事件的标识符
            self._idle_draw_id = 0  # 将 _idle_draw_id 重置为0
            self.draw()  # 调用 draw 方法重新绘制

        scale = self.device_pixel_ratio  # 获取设备像素比例
        allocation = self.get_allocation()  # 获取分配的区域
        w = allocation.width * scale  # 计算实际宽度
        h = allocation.height * scale  # 计算实际高度

        if not len(self._bbox_queue):  # 如果 _bbox_queue 为空
            Gtk.render_background(
                self.get_style_context(), ctx,
                allocation.x, allocation.y,
                allocation.width, allocation.height)
            bbox_queue = [transforms.Bbox([[0, 0], [w, h]])]  # 创建一个包含整个画布范围的 Bbox 对象列表
        else:
            bbox_queue = self._bbox_queue  # 否则使用已有的 _bbox_queue

        for bbox in bbox_queue:
            x = int(bbox.x0)  # 获取bbox左下角x坐标
            y = h - int(bbox.y1)  # 获取bbox左下角y坐标（考虑坐标系差异）
            width = int(bbox.x1) - int(bbox.x0)  # 计算bbox宽度
            height = int(bbox.y1) - int(bbox.y0)  # 计算bbox高度

            buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
                np.asarray(self.copy_from_bbox(bbox)))  # 从 bbox 区域复制像素数据，并进行颜色格式转换
            image = cairo.ImageSurface.create_for_data(
                buf.ravel().data, cairo.FORMAT_ARGB32, width, height)  # 使用转换后的像素数据创建 Cairo 图像表面
            image.set_device_scale(scale, scale)  # 设置图像表面的设备比例
            ctx.set_source_surface(image, x / scale, y / scale)  # 设置上下文的绘制源为图像表面
            ctx.paint()  # 执行绘制操作

        if len(self._bbox_queue):  # 如果 _bbox_queue 不为空
            self._bbox_queue = []  # 清空 _bbox_queue

        return False  # 返回 False 表示处理完毕

    def blit(self, bbox=None):
        # 如果 bbox 为 None，则将整个画布内容复制到 gtk；否则只复制 bbox 定义的区域
        if bbox is None:
            bbox = self.figure.bbox  # 使用 figure 的 bbox

        scale = self.device_pixel_ratio  # 获取设备像素比例
        allocation = self.get_allocation()  # 获取分配的区域
        x = int(bbox.x0 / scale)  # 计算复制区域左下角 x 坐标
        y = allocation.height - int(bbox.y1 / scale)  # 计算复制区域左下角 y 坐标
        width = (int(bbox.x1) - int(bbox.x0)) // scale  # 计算复制区域宽度
        height = (int(bbox.y1) - int(bbox.y0)) // scale  # 计算复制区域高度

        self._bbox_queue.append(bbox)  # 将 bbox 添加到 _bbox_queue 中
        self.queue_draw_area(x, y, width, height)  # 将指定区域添加到重新绘制队列中


@_BackendGTK3.export
class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Agg  # 将 FigureCanvasGTK3Agg 指定为 _BackendGTK3Cairo 的 FigureCanvas
```