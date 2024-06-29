# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_wxagg.py`

```py
# 导入 wx 模块，用于与 wxPython GUI 库进行交互
import wx

# 从本地模块中导入 FigureCanvasAgg 类
from .backend_agg import FigureCanvasAgg
# 从本地模块中导入 _BackendWx 和 _FigureCanvasWxBase 类
from .backend_wx import _BackendWx, _FigureCanvasWxBase
# 从本地模块中导入 NavigationToolbar2WxAgg 类，用作导航工具栏
from .backend_wx import (  # noqa: F401 # pylint: disable=W0611
    NavigationToolbar2Wx as NavigationToolbar2WxAgg)


class FigureCanvasWxAgg(FigureCanvasAgg, _FigureCanvasWxBase):
    def draw(self, drawDC=None):
        """
        使用 agg 渲染图形。
        """
        # 调用 FigureCanvasAgg 类的 draw 方法，渲染图形
        FigureCanvasAgg.draw(self)
        # 创建位图对象，用于显示图形
        self.bitmap = self._create_bitmap()
        # 设置绘制标志为 True
        self._isDrawn = True
        # 更新 GUI 显示
        self.gui_repaint(drawDC=drawDC)

    def blit(self, bbox=None):
        """
        绘制指定的区域。
        """
        # 调用 _create_bitmap 方法创建位图对象
        bitmap = self._create_bitmap()
        # 如果未指定 bbox 区域，则直接用新位图替换旧位图
        if bbox is None:
            self.bitmap = bitmap
        else:
            # 否则，使用内存设备上下文操作进行部分位图复制
            srcDC = wx.MemoryDC(bitmap)
            destDC = wx.MemoryDC(self.bitmap)
            x = int(bbox.x0)
            y = int(self.bitmap.GetHeight() - bbox.y1)
            destDC.Blit(x, y, int(bbox.width), int(bbox.height), srcDC, x, y)
            destDC.SelectObject(wx.NullBitmap)
            srcDC.SelectObject(wx.NullBitmap)
        # 更新 GUI 显示
        self.gui_repaint()

    def _create_bitmap(self):
        """
        从渲染器的 RGBA 缓冲区创建 wx.Bitmap 对象。
        """
        # 获取渲染器的 RGBA 缓冲区
        rgba = self.get_renderer().buffer_rgba()
        h, w, _ = rgba.shape
        # 创建 wx.Bitmap 对象，并设置 DPI 缩放因子
        bitmap = wx.Bitmap.FromBufferRGBA(w, h, rgba)
        bitmap.SetScaleFactor(self.GetDPIScaleFactor())
        return bitmap


@_BackendWx.export
class _BackendWxAgg(_BackendWx):
    FigureCanvas = FigureCanvasWxAgg
```