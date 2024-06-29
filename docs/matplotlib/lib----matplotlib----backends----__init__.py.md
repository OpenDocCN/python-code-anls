# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\__init__.py`

```py
# 导入模块中的 BackendFilter 和 backend_registry，忽略 F401 类型的导入警告
from .registry import BackendFilter, backend_registry  # noqa: F401

# NOTE: plt.switch_backend() 在导入时会添加一个 "backend" 属性到这里，以支持向后兼容。
# _QT_FORCE_QT5_BINDING 变量，用于控制是否强制使用 Qt5 绑定，默认为 False。
_QT_FORCE_QT5_BINDING = False
```