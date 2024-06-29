# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_wxcairo.py`

```py
# 导入 wxcairo 库中的 wxcairo 模块，作为 wxcairo 的别名引入
import wx.lib.wxcairo as wxcairo

# 从当前目录下的 backend_cairo 模块中导入 cairo 和 FigureCanvasCairo 类
from .backend_cairo import cairo, FigureCanvasCairo

# 从当前目录下的 backend_wx 模块中导入 _BackendWx 和 _FigureCanvasWxBase 类
from .backend_wx import _BackendWx, _FigureCanvasWxBase

# 从当前目录下的 backend_wx 模块中导入 NavigationToolbar2WxCairo 类，并将其重命名为 NavigationToolbar2WxCairo
# 注：禁止 Flake8 Linter (F401) 报告未使用和 Pylint (W0611) 报告未使用的警告
from .backend_wx import (
    NavigationToolbar2Wx as NavigationToolbar2WxCairo
)  # noqa: F401 # pylint: disable=W0611


class FigureCanvasWxCairo(FigureCanvasCairo, _FigureCanvasWxBase):
    def draw(self, drawDC=None):
        # 获取图形对象的边界框大小，并转换为整数类型
        size = self.figure.bbox.size.astype(int)
        
        # 创建一个 ARGB32 格式的图像表面，尺寸为 size
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
        
        # 将渲染器的上下文设置为 surface 的 Cairo 上下文
        self._renderer.set_context(cairo.Context(surface))
        
        # 设置渲染器的 DPI 为图形对象的 DPI
        self._renderer.dpi = self.figure.dpi
        
        # 在渲染器上绘制图形对象
        self.figure.draw(self._renderer)
        
        # 从 surface 创建 wxcairo 的位图对象
        self.bitmap = wxcairo.BitmapFromImageSurface(surface)
        
        # 设置绘制标志为 True
        self._isDrawn = True
        
        # 调用 GUI 重新绘制方法，传入 drawDC 参数
        self.gui_repaint(drawDC=drawDC)


@_BackendWx.export
class _BackendWxCairo(_BackendWx):
    # 设置 BackendWxCairo 类的 FigureCanvas 类为 FigureCanvasWxCairo
    FigureCanvas = FigureCanvasWxCairo
```