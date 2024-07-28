# `.\comic-translate\app\ui\dayu_widgets\spin_box.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""
Custom Stylesheet for QSpinBox, QDoubleSpinBox, QDateTimeEdit, QDateEdit, QTimeEdit.
Only add size arg for their __init__.
"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme
from .mixin import cursor_mixin


@cursor_mixin
class MSpinBox(QtWidgets.QSpinBox):
    """
    MSpinBox just use stylesheet and add dayu_size. No more extend.
    Property:
        dayu_size: The height of MSpinBox
    """

    def __init__(self, parent=None):
        super(MSpinBox, self).__init__(parent=parent)
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the MSpinBox height
        :return: integer
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the MSpinBox size.
        :param value: integer
        :return: None
        """
        self._dayu_size = value
        self.style().polish(self)

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """Set MSpinBox to huge size"""
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MSpinBox to large size"""
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MSpinBox to medium size"""
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MSpinBox to small size"""
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """Set MSpinBox to tiny size"""
        self.set_dayu_size(dayu_theme.tiny)
        return self


@cursor_mixin
class MDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """
    MDoubleSpinBox just use stylesheet and add dayu_size. No more extend.
    Property:
        dayu_size: The height of MDoubleSpinBox
    """

    def __init__(self, parent=None):
        super(MDoubleSpinBox, self).__init__(parent=parent)
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the MDoubleSpinBox height
        :return: integer
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the MDoubleSpinBox size.
        :param value: integer
        :return: None
        """
        self._dayu_size = value
        self.style().polish(self)

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """Set MDoubleSpinBox to huge size"""
        self.set_dayu_size(dayu_theme.huge)
        return self
    def large(self):
        """将 MDoubleSpinBox 设置为大号大小"""
        # 调用 set_dayu_size 方法，设置 MDoubleSpinBox 的大小为 dayu_theme.large
        self.set_dayu_size(dayu_theme.large)
        # 返回当前对象实例，以支持链式调用
        return self

    def medium(self):
        """将 MDoubleSpinBox 设置为中号大小"""
        # 调用 set_dayu_size 方法，设置 MDoubleSpinBox 的大小为 dayu_theme.medium
        self.set_dayu_size(dayu_theme.medium)
        # 返回当前对象实例，以支持链式调用
        return self

    def small(self):
        """将 MDoubleSpinBox 设置为小号大小"""
        # 调用 set_dayu_size 方法，设置 MDoubleSpinBox 的大小为 dayu_theme.small
        self.set_dayu_size(dayu_theme.small)
        # 返回当前对象实例，以支持链式调用
        return self

    def tiny(self):
        """将 MDoubleSpinBox 设置为超小号大小"""
        # 调用 set_dayu_size 方法，设置 MDoubleSpinBox 的大小为 dayu_theme.tiny
        self.set_dayu_size(dayu_theme.tiny)
        # 返回当前对象实例，以支持链式调用
        return self
@cursor_mixin
class MDateTimeEdit(QtWidgets.QDateTimeEdit):
    """
    MDateTimeEdit just use stylesheet and add dayu_size. No more extend.
    Property:
        dayu_size: The height of MDateTimeEdit
    """

    def __init__(self, datetime=None, parent=None):
        # 如果未提供 datetime 参数，则调用父类的构造函数，初始化一个 QDateTimeEdit 控件
        if datetime is None:
            super(MDateTimeEdit, self).__init__(parent=parent)
        else:
            # 如果提供了 datetime 参数，则使用该参数初始化 QDateTimeEdit 控件
            super(MDateTimeEdit, self).__init__(datetime, parent=parent)
        # 设置默认的 dayu_size 属性为 dayu_theme.default_size
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the MDateTimeEdit height
        :return: integer
        """
        # 返回当前的 dayu_size 属性值
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the MDateTimeEdit size.
        :param value: integer
        :return: None
        """
        # 设置 dayu_size 属性为指定的 value 值，并更新控件的样式
        self._dayu_size = value
        self.style().polish(self)

    # 定义 dayu_size 属性，通过 QtCore.Property 将其绑定到 get_dayu_size 和 set_dayu_size 方法
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """Set MDateTimeEdit to huge size"""
        # 将 dayu_size 设置为 dayu_theme.huge，并返回当前实例
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MDateTimeEdit to large size"""
        # 将 dayu_size 设置为 dayu_theme.large，并返回当前实例
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MDateTimeEdit to  medium"""
        # 将 dayu_size 设置为 dayu_theme.medium，并返回当前实例
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MDateTimeEdit to small size"""
        # 将 dayu_size 设置为 dayu_theme.small，并返回当前实例
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """Set MDateTimeEdit to tiny size"""
        # 将 dayu_size 设置为 dayu_theme.tiny，并返回当前实例
        self.set_dayu_size(dayu_theme.tiny)
        return self


@cursor_mixin
class MDateEdit(QtWidgets.QDateEdit):
    """
    MDateEdit just use stylesheet and add dayu_size. No more extend.
    Property:
        dayu_size: The height of MDateEdit
    """

    def __init__(self, date=None, parent=None):
        # 如果未提供 date 参数，则调用父类的构造函数，初始化一个 QDateEdit 控件
        if date is None:
            super(MDateEdit, self).__init__(parent=parent)
        else:
            # 如果提供了 date 参数，则使用该参数初始化 QDateEdit 控件
            super(MDateEdit, self).__init__(date, parent=parent)
        # 设置默认的 dayu_size 属性为 dayu_theme.default_size
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the MDateEdit height
        :return: integer
        """
        # 返回当前的 dayu_size 属性值
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the MDateEdit size.
        :param value: integer
        :return: None
        """
        # 设置 dayu_size 属性为指定的 value 值，并更新控件的样式
        self._dayu_size = value
        self.style().polish(self)

    # 定义 dayu_size 属性，通过 QtCore.Property 将其绑定到 get_dayu_size 和 set_dayu_size 方法
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """Set MDateEdit to huge size"""
        # 将 dayu_size 设置为 dayu_theme.huge，并返回当前实例
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MDateEdit to large size"""
        # 将 dayu_size 设置为 dayu_theme.large，并返回当前实例
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MDateEdit to  medium"""
        # 将 dayu_size 设置为 dayu_theme.medium，并返回当前实例
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MDateEdit to small size"""
        # 将 dayu_size 设置为 dayu_theme.small，并返回当前实例
        self.set_dayu_size(dayu_theme.small)
        return self
    def tiny(self):
        """Set MDateEdit to tiny size"""
        # 调用自定义方法，设置日期编辑器为微小尺寸
        self.set_dayu_size(dayu_theme.tiny)
        # 返回设置后的对象本身
        return self
@cursor_mixin
class MTimeEdit(QtWidgets.QTimeEdit):
    """
    MTimeEdit just use stylesheet and add dayu_size. No more extend.
    Property:
        dayu_size: The height of MTimeEdit
    """

    def __init__(self, time=None, parent=None):
        # 调用父类的构造函数，初始化时间编辑器
        if time is None:
            super(MTimeEdit, self).__init__(parent=parent)
        else:
            super(MTimeEdit, self).__init__(time, parent=parent)
        # 设置默认的 dayu_size 属性为 dayu_theme.default_size
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the MTimeEdit height
        :return: integer
        """
        # 返回当前的 dayu_size 属性值
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the MTimeEdit size.
        :param value: integer
        :return: None
        """
        # 设置 dayu_size 属性为指定的值，并刷新样式
        self._dayu_size = value
        self.style().polish(self)

    # 定义 dayu_size 属性，其类型为 int，读取方法为 get_dayu_size，设置方法为 set_dayu_size
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """Set MTimeEdit to huge size"""
        # 将 MTimeEdit 设置为巨大尺寸
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MTimeEdit to large size"""
        # 将 MTimeEdit 设置为大尺寸
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MTimeEdit to medium size"""
        # 将 MTimeEdit 设置为中等尺寸
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MTimeEdit to small size"""
        # 将 MTimeEdit 设置为小尺寸
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """Set MTimeEdit to tiny size"""
        # 将 MTimeEdit 设置为微小尺寸
        self.set_dayu_size(dayu_theme.tiny)
        return self
```