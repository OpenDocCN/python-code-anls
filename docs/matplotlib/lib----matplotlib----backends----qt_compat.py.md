# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\qt_compat.py`

```
"""
Qt binding and backend selector.

The selection logic is as follows:
- if any of PyQt6, PySide6, PyQt5, or PySide2 have already been
  imported (checked in that order), use it;
- otherwise, if the QT_API environment variable (used by Enthought) is set, use
  it to determine which binding to use;
- otherwise, use whatever the rcParams indicate.
"""

# 导入必要的库和模块
import operator
import os
import platform
import sys

# 导入第三方库
from packaging.version import parse as parse_version

# 导入 Matplotlib 库
import matplotlib as mpl

# 导入 _QT_FORCE_QT5_BINDING 模块
from . import _QT_FORCE_QT5_BINDING

# 定义可能的 Qt 绑定的字符串常量
QT_API_PYQT6 = "PyQt6"
QT_API_PYSIDE6 = "PySide6"
QT_API_PYQT5 = "PyQt5"
QT_API_PYSIDE2 = "PySide2"

# 获取 QT_API 环境变量值并转换为小写
QT_API_ENV = os.environ.get("QT_API")
if QT_API_ENV is not None:
    QT_API_ENV = QT_API_ENV.lower()

# 定义一个映射表，将 QT_API_ENV 转换为请求的绑定
_ETS = {
    "pyqt6": QT_API_PYQT6,
    "pyside6": QT_API_PYSIDE6,
    "pyqt5": QT_API_PYQT5,
    "pyside2": QT_API_PYSIDE2,
}

# 首先检查是否已经导入了任何 Qt 组件
if sys.modules.get("PyQt6.QtCore"):
    QT_API = QT_API_PYQT6
elif sys.modules.get("PySide6.QtCore"):
    QT_API = QT_API_PYSIDE6
elif sys.modules.get("PyQt5.QtCore"):
    QT_API = QT_API_PYQT5
elif sys.modules.get("PySide2.QtCore"):
    QT_API = QT_API_PYSIDE2
# 否则，检查 QT_API 环境变量（来自 Enthought）是否设置
elif (mpl.rcParams._get_backend_or_none() or "").lower().startswith("qt5"):
    if QT_API_ENV in ["pyqt5", "pyside2"]:
        QT_API = _ETS[QT_API_ENV]
    else:
        _QT_FORCE_QT5_BINDING = True  # noqa
        QT_API = None
# 如果选择了非 Qt 后端但仍然到达此处（可能的情况是在不使用 pyplot 的情况下完全手动嵌入 Matplotlib 到 Qt 应用程序中）
elif QT_API_ENV is None:
    QT_API = None
elif QT_API_ENV in _ETS:
    QT_API = _ETS[QT_API_ENV]
else:
    raise RuntimeError(
        "The environment variable QT_API has the unrecognized value {!r}; "
        "valid values are {}".format(QT_API_ENV, ", ".join(_ETS)))

# 定义函数 _setup_pyqt5plus()
def _setup_pyqt5plus():
    global QtCore, QtGui, QtWidgets, __version__
    global _isdeleted, _to_int

    # 如果 QT_API 是 PyQt6
    if QT_API == QT_API_PYQT6:
        from PyQt6 import QtCore, QtGui, QtWidgets, sip
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _to_int = operator.attrgetter('value')
    # 如果 QT_API 是 PySide6
    elif QT_API == QT_API_PYSIDE6:
        from PySide6 import QtCore, QtGui, QtWidgets, __version__
        import shiboken6
        def _isdeleted(obj): return not shiboken6.isValid(obj)
        if parse_version(__version__) >= parse_version('6.4'):
            _to_int = operator.attrgetter('value')
        else:
            _to_int = int
    # 如果 QT_API 等于 QT_API_PYQT5，则执行以下代码块
    elif QT_API == QT_API_PYQT5:
        # 从 PyQt5 中导入 QtCore, QtGui, QtWidgets 模块
        from PyQt5 import QtCore, QtGui, QtWidgets
        # 导入 sip 模块
        import sip
        # 将 __version__ 设为 PyQt5 的版本字符串
        __version__ = QtCore.PYQT_VERSION_STR
        # 重定义 QtCore.Signal 为 QtCore.pyqtSignal
        QtCore.Signal = QtCore.pyqtSignal
        # 重定义 QtCore.Slot 为 QtCore.pyqtSlot
        QtCore.Slot = QtCore.pyqtSlot
        # 重定义 QtCore.Property 为 QtCore.pyqtProperty
        QtCore.Property = QtCore.pyqtProperty
        # 将 _isdeleted 设为 sip 模块的 isdeleted 函数
        _isdeleted = sip.isdeleted
        # 将 _to_int 设为 Python 内置的 int 函数
        _to_int = int
    # 如果 QT_API 等于 QT_API_PYSIDE2，则执行以下代码块
    elif QT_API == QT_API_PYSIDE2:
        # 从 PySide2 中导入 QtCore, QtGui, QtWidgets, __version__ 模块
        from PySide2 import QtCore, QtGui, QtWidgets, __version__
        try:
            # 尝试从 PySide2 导入 shiboken2 模块
            from PySide2 import shiboken2
        except ImportError:
            # 如果导入失败，则导入 shiboken2 模块
            import shiboken2
        # 定义 _isdeleted 函数，检查对象是否有效
        def _isdeleted(obj):
            return not shiboken2.isValid(obj)
        # 将 _to_int 设为 Python 内置的 int 函数
        _to_int = int
    # 如果 QT_API 不是已知的 PyQt5 或 PySide2，则引发断言错误
    else:
        raise AssertionError(f"Unexpected QT_API: {QT_API}")
# 如果 QT_API 在支持的 Qt 版本列表中，则调用 _setup_pyqt5plus() 进行设置
if QT_API in [QT_API_PYQT6, QT_API_PYQT5, QT_API_PYSIDE6, QT_API_PYSIDE2]:
    _setup_pyqt5plus()
# 如果 QT_API 为 None，则根据 _QT_FORCE_QT5_BINDING 设置候选列表 _candidates
elif QT_API is None:  # 参考上文关于 dict.__getitem__ 的说明。
    if _QT_FORCE_QT5_BINDING:
        _candidates = [
            (_setup_pyqt5plus, QT_API_PYQT5),
            (_setup_pyqt5plus, QT_API_PYSIDE2),
        ]
    else:
        _candidates = [
            (_setup_pyqt5plus, QT_API_PYQT6),
            (_setup_pyqt5plus, QT_API_PYSIDE6),
            (_setup_pyqt5plus, QT_API_PYQT5),
            (_setup_pyqt5plus, QT_API_PYSIDE2),
        ]
    # 遍历候选列表，并尝试执行设置函数 _setup()
    for _setup, QT_API in _candidates:
        try:
            _setup()
        except ImportError:
            continue
        break
    else:
        # 如果所有的候选设置均失败，则抛出 ImportError 异常
        raise ImportError(
            "Failed to import any of the following Qt binding modules: {}"
            .format(", ".join([QT_API for _, QT_API in _candidates]))
        )
else:  # 我们不应该到达这里。
    # 如果出现意料之外的 QT_API，则抛出 AssertionError 异常
    raise AssertionError(f"Unexpected QT_API: {QT_API}")

# 获取当前 Qt 库的版本信息，并将其转换为元组类型 _version_info
_version_info = tuple(QtCore.QLibraryInfo.version().segments())

# 如果当前导入的 Qt 版本低于 (5, 12)，则抛出 ImportError 异常
if _version_info < (5, 12):
    raise ImportError(
        f"The Qt version imported is "
        f"{QtCore.QLibraryInfo.version().toString()} but Matplotlib requires "
        f"Qt>=5.12")

# 修复 macOS Big Sur 上的问题，仅在特定条件下执行
# 参考链接：https://bugreports.qt.io/browse/QTBUG-87014，在 Qt 5.15.2 中修复
if (sys.platform == 'darwin' and
        parse_version(platform.mac_ver()[0]) >= parse_version("10.16") and
        _version_info < (5, 15, 2)):
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

# 后向兼容处理函数 _exec
def _exec(obj):
    # 根据 PyQt6 使用 exec，其它情况使用 exec_
    obj.exec() if hasattr(obj, "exec") else obj.exec_()
```