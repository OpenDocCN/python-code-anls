# `.\comic-translate\app\ui\dayu_widgets\switch.py`

```py
"""
MSwitch
"""
# 导入必要的模块
from __future__ import absolute_import   # 导入绝对路径
from __future__ import division         # 导入除法运算
from __future__ import print_function   # 导入打印函数

# 导入第三方模块
from PySide6 import QtCore     # 导入 PySide6 的 QtCore 模块
from PySide6 import QtWidgets  # 导入 PySide6 的 QtWidgets 模块

# 导入本地模块
from . import dayu_theme   # 导入当前包中的 dayu_theme 模块
from .mixin import cursor_mixin   # 导入当前包中的 cursor_mixin 模块


@cursor_mixin   # 使用 cursor_mixin 装饰器修饰 MSwitch 类
class MSwitch(QtWidgets.QRadioButton):
    """
    Switching Selector.

    Property:
        dayu_size: the size of switch widget. int
    """

    def __init__(self, parent=None):
        """
        初始化 MSwitch 对象。

        :param parent: 父级窗口对象，默认为 None
        """
        super(MSwitch, self).__init__(parent)   # 调用父类 QtWidgets.QRadioButton 的初始化方法
        self._dayu_size = dayu_theme.default_size   # 设置默认大小为 dayu_theme.default_size
        self.setAutoExclusive(False)   # 设置不自动互斥

    def minimumSizeHint(self):
        """
        重写 QRadioButton 的最小尺寸提示方法。不需要文本空间。

        :return: QtCore.QSize 对象，表示的是控件的最小尺寸
        """
        height = self._dayu_size * 1.2   # 根据 dayu_size 计算高度
        return QtCore.QSize(int(height), int(height / 2))   # 返回计算后的尺寸作为建议的最小尺寸

    def get_dayu_size(self):
        """
        获取当前的开关尺寸。

        :return: int，表示当前的开关尺寸
        """
        return self._dayu_size   # 返回当前的 dayu_size

    def set_dayu_size(self, value):
        """
        设置开关的尺寸。

        :param value: int，表示要设置的开关尺寸
        :return: None
        """
        self._dayu_size = value   # 设置 dayu_size 为指定值
        self.style().polish(self)   # 刷新控件样式

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)   # 定义 dayu_size 属性，与 get_dayu_size 和 set_dayu_size 关联

    def huge(self):
        """将 MSwitch 设置为巨大尺寸"""
        self.set_dayu_size(dayu_theme.huge)   # 调用 set_dayu_size 方法，设置为巨大尺寸
        return self   # 返回当前对象引用

    def large(self):
        """将 MSwitch 设置为大尺寸"""
        self.set_dayu_size(dayu_theme.large)   # 调用 set_dayu_size 方法，设置为大尺寸
        return self   # 返回当前对象引用

    def medium(self):
        """将 MSwitch 设置为中等尺寸"""
        self.set_dayu_size(dayu_theme.medium)   # 调用 set_dayu_size 方法，设置为中等尺寸
        return self   # 返回当前对象引用

    def small(self):
        """将 MSwitch 设置为小尺寸"""
        self.set_dayu_size(dayu_theme.small)   # 调用 set_dayu_size 方法，设置为小尺寸
        return self   # 返回当前对象引用

    def tiny(self):
        """将 MSwitch 设置为微小尺寸"""
        self.set_dayu_size(dayu_theme.tiny)   # 调用 set_dayu_size 方法，设置为微小尺寸
        return self   # 返回当前对象引用
```