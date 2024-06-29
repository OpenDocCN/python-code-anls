# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_tkagg.py`

```py
from . import _backend_tk
from .backend_agg import FigureCanvasAgg
from ._backend_tk import _BackendTk, FigureCanvasTk
from ._backend_tk import (  # noqa: F401 # pylint: disable=W0611
    FigureManagerTk, NavigationToolbar2Tk)

class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    # 绘制方法重载，调用父类方法后再调用 blit 方法
    def draw(self):
        super().draw()
        self.blit()

    # 更新画布方法，使用 Tkinter 的底层绘图接口
    def blit(self, bbox=None):
        _backend_tk.blit(self._tkphoto, self.renderer.buffer_rgba(),
                         (0, 1, 2, 3), bbox=bbox)

# 将 _BackendTkAgg 导出为 _BackendTk 的子类，定义了 FigureCanvas 为 FigureCanvasTkAgg 类
@_BackendTk.export
class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg
```