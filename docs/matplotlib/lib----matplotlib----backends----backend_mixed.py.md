# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_mixed.py`

```
import numpy as np

from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing


class MixedModeRenderer:
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """

    def __init__(self, figure, width, height, dpi, vector_renderer,
                 raster_renderer_class=None,
                 bbox_inches_restore=None):
        """
        Parameters
        ----------
        figure : `~matplotlib.figure.Figure`
            The figure instance.
        width : scalar
            The width of the canvas in logical units
        height : scalar
            The height of the canvas in logical units
        dpi : float
            The dpi of the canvas
        vector_renderer : `~matplotlib.backend_bases.RendererBase`
            An instance of a subclass of
            `~matplotlib.backend_bases.RendererBase` that will be used for the
            vector drawing.
        raster_renderer_class : `~matplotlib.backend_bases.RendererBase`, optional
            The renderer class to use for the raster drawing. If not provided,
            this will use the Agg backend (which is currently the only viable
            option anyway.)
        """
        # 如果未指定光栅渲染器类，则使用默认的 RendererAgg
        if raster_renderer_class is None:
            raster_renderer_class = RendererAgg

        # 设置光栅渲染器类
        self._raster_renderer_class = raster_renderer_class
        # 设置画布的逻辑单位宽度和高度
        self._width = width
        self._height = height
        # 设置画布的 DPI
        self.dpi = dpi

        # 设置矢量渲染器
        self._vector_renderer = vector_renderer

        # 初始化时光栅渲染器为空
        self._raster_renderer = None

        # 保存对图形的引用，因为需要在栅格化之前和之后更改图形的 DPI
        self.figure = figure
        self._figdpi = figure.dpi

        # 用于恢复边界框英寸的设置
        self._bbox_inches_restore = bbox_inches_restore

        # 设置当前使用的渲染器为矢量渲染器
        self._renderer = vector_renderer

    def __getattr__(self, attr):
        """
        Proxy method to redirect attribute access to the underlying renderer.
        """
        # 将未重写的属性访问重定向到基础渲染器上
        return getattr(self._renderer, attr)
    def start_rasterizing(self):
        """
        进入“光栅化”模式。所有随后的绘图命令（直到调用 `stop_rasterizing` 为止）
        将使用光栅后端进行绘制。
        """
        # 将图形的 DPI 临时设为指定的 DPI
        self.figure.dpi = self.dpi
        # 当使用紧凑边界框时
        if self._bbox_inches_restore:
            # 处理图形以备光栅化，返回处理后的边界框
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore)
            self._bbox_inches_restore = r

        # 使用指定尺寸和 DPI 创建光栅化渲染器对象
        self._raster_renderer = self._raster_renderer_class(
            self._width*self.dpi, self._height*self.dpi, self.dpi)
        # 将当前渲染器设为光栅化渲染器
        self._renderer = self._raster_renderer

    def stop_rasterizing(self):
        """
        退出“光栅化”模式。自上次调用 `start_rasterizing` 以来进行的所有绘图将通过调用
        `draw_image` 被复制到矢量后端。
        """

        # 将当前渲染器设为矢量渲染器
        self._renderer = self._vector_renderer

        height = self._height * self.dpi
        # 将光栅渲染器的 RGBA 缓冲区转换为 NumPy 数组
        img = np.asarray(self._raster_renderer.buffer_rgba())
        # 获取非零切片
        slice_y, slice_x = cbook._get_nonzero_slices(img[..., 3])
        # 根据切片裁剪图像
        cropped_img = img[slice_y, slice_x]
        if cropped_img.size:
            gc = self._renderer.new_gc()
            # TODO: 如果混合模式的分辨率与图形的 DPI 不同，图像必须进行缩放（dpi -> _figdpi）。
            #       并非所有后端都支持此操作。
            # 绘制图像到当前渲染器上
            self._renderer.draw_image(
                gc,
                slice_x.start * self._figdpi / self.dpi,
                (height - slice_y.stop) * self._figdpi / self.dpi,
                cropped_img[::-1])
        # 清空光栅渲染器对象
        self._raster_renderer = None

        # 恢复图形的 DPI
        self.figure.dpi = self._figdpi

        # 当使用紧凑边界框时
        if self._bbox_inches_restore:
            # 处理图形以备光栅化，返回处理后的边界框
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore,
                                               self._figdpi)
            self._bbox_inches_restore = r
```