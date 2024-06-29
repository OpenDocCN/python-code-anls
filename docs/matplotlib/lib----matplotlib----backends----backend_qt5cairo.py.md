# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qt5cairo.py`

```py
# 从上级目录的 backends 模块中导入 backends 对象
from .. import backends

# 设置 backends 模块中的 _QT_FORCE_QT5_BINDING 属性为 True，强制使用 Qt5 绑定
backends._QT_FORCE_QT5_BINDING = True

# 导入 backend_qtcairo 模块中的以下对象：
# _BackendQTCairo, FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT
# 使用 noqa: F401 来禁止未使用的导入的警告，
# 使用 E402 来忽略与导入顺序相关的 Pylint 警告，
# 使用 pylint: disable=W0611 来禁用未使用的导入的 Pylint 警告
from .backend_qtcairo import (
    _BackendQTCairo, FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT
)

# 使用 _BackendQTCairo.export 装饰器来导出下面定义的 _BackendQT5Cairo 类
@_BackendQTCairo.export
class _BackendQT5Cairo(_BackendQTCairo):
    pass
```