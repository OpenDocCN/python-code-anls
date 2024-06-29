# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qt5agg.py`

```py
"""
Render to qt from agg
"""
# 从 backends 模块中导入相关内容
from .. import backends

# 设置 _QT_FORCE_QT5_BINDING 为 True，强制使用 Qt5 绑定
backends._QT_FORCE_QT5_BINDING = True

# 从 backend_qtagg 模块中导入以下类，忽略 F401 错误和 E402 错误，禁用 W0611 警告
from .backend_qtagg import (
    _BackendQTAgg, FigureCanvasQTAgg, FigureManagerQT, NavigationToolbar2QT,
    FigureCanvasAgg, FigureCanvasQT
)

# 使用 _BackendQTAgg.export 装饰器导出 _BackendQT5Agg 类
@_BackendQTAgg.export
class _BackendQT5Agg(_BackendQTAgg):
    pass
```