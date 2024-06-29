# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qt5.py`

```
# 导入相对路径下的 backends 模块
from .. import backends

# 设置 backends 模块中的 _QT_FORCE_QT5_BINDING 属性为 True，强制使用 QT5 绑定
backends._QT_FORCE_QT5_BINDING = True

# 导入 backend_qt 模块的特定内容
from .backend_qt import (  # noqa
    SPECIAL_KEYS,  # 特殊键定义
    # 公共 API
    cursord, _create_qApp, _BackendQT, TimerQT, MainWindow, FigureCanvasQT,
    FigureManagerQT, ToolbarQt, NavigationToolbar2QT, SubplotToolQt,
    SaveFigureQt, ConfigureSubplotsQt, RubberbandQt,
    HelpQt, ToolCopyToClipboardQT,
    # 内部重新导出的内容
    FigureCanvasBase,  FigureManagerBase, MouseButton, NavigationToolbar2,
    TimerBase, ToolContainerBase, figureoptions, Gcf
)

# 导入 backend_qt 模块，并指定其为 _backend_qt 别名
from . import backend_qt as _backend_qt  # noqa

# 使用装饰器将 _BackendQT5 类导出为 _BackendQT 的一部分
@_BackendQT.export
class _BackendQT5(_BackendQT):
    pass

# 定义 __getattr__ 函数，用于动态获取模块属性
def __getattr__(name):
    # 如果请求的属性名为 'qApp'，则返回 _backend_qt 模块中的 qApp 属性
    if name == 'qApp':
        return _backend_qt.qApp
    # 如果请求的属性名不存在，则抛出 AttributeError 异常
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```