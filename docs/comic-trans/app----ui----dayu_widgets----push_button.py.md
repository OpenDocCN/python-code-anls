# `.\comic-translate\app\ui\dayu_widgets\push_button.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""
MPushButton.
"""
# 导入未来模块，确保代码在Python 2和Python 3中兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme
from .mixin import cursor_mixin
from .mixin import focus_shadow_mixin

# 通过cursor_mixin和focus_shadow_mixin装饰器为MPushButton类添加功能混合特性
@cursor_mixin
@focus_shadow_mixin
class MPushButton(QtWidgets.QPushButton):
    """
    QPushButton.

    Property:
        dayu_size: The size of push button
        dayu_type: The type of push button.
    """

    # 定义MPushButton类的几种默认类型
    DefaultType = "default"
    PrimaryType = "primary"
    SuccessType = "success"
    WarningType = "warning"
    DangerType = "danger"

    def __init__(self, text="", icon=None, parent=None):
        # 根据传入的参数初始化按钮，如果没有图标则使用文本初始化按钮
        if icon is None:
            super(MPushButton, self).__init__(text=text, parent=parent)
        else:
            super(MPushButton, self).__init__(icon=icon, text=text, parent=parent)
        # 初始化按钮的dayu_type属性为DefaultType
        self._dayu_type = MPushButton.DefaultType
        # 初始化按钮的dayu_size属性为dayu_theme.default_size

        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        Get the push button height
        :return: integer
        """
        # 返回按钮的dayu_size属性，即按钮的高度
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the avatar size.
        :param value: integer
        :return: None
        """
        # 设置按钮的dayu_size属性为传入的值，并更新按钮的样式
        self._dayu_size = value
        self.style().polish(self)

    def get_dayu_type(self):
        """
        Get the push button type.
        :return: string.
        """
        # 返回按钮的dayu_type属性，即按钮的类型
        return self._dayu_type

    def set_dayu_type(self, value):
        """
        Set the push button type.
        :return: None
        """
        # 设置按钮的dayu_type属性为传入的值，确保传入的值是预定义的几种类型之一，否则抛出异常
        if value in [
            MPushButton.DefaultType,
            MPushButton.PrimaryType,
            MPushButton.SuccessType,
            MPushButton.WarningType,
            MPushButton.DangerType,
        ]:
            self._dayu_type = value
        else:
            raise ValueError(
                "Input argument 'value' should be one of " "default/primary/success/warning/danger string."
            )
        # 更新按钮的样式
        self.style().polish(self)

    # 使用QtCore.Property定义dayu_type属性，使其具有属性特性，可以通过get_dayu_type和set_dayu_type方法进行访问和设置
    dayu_type = QtCore.Property(str, get_dayu_type, set_dayu_type)
    # 使用QtCore.Property定义dayu_size属性，使其具有属性特性，可以通过get_dayu_size和set_dayu_size方法进行访问和设置
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def primary(self):
        """Set MPushButton to PrimaryType"""
        # 设置按钮类型为PrimaryType
        self.set_dayu_type(MPushButton.PrimaryType)
        return self

    def success(self):
        """Set MPushButton to SuccessType"""
        # 设置按钮类型为SuccessType
        self.set_dayu_type(MPushButton.SuccessType)
        return self

    def warning(self):
        """Set MPushButton to  WarningType"""
        # 设置按钮类型为WarningType
        self.set_dayu_type(MPushButton.WarningType)
        return self
    def danger(self):
        """设置 MPushButton 控件为危险类型"""
        self.set_dayu_type(MPushButton.DangerType)
        return self

    def huge(self):
        """设置 MPushButton 控件为巨大尺寸"""
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """设置 MPushButton 控件为大尺寸"""
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """设置 MPushButton 控件为中等尺寸"""
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """设置 MPushButton 控件为小尺寸"""
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """设置 MPushButton 控件为微小尺寸"""
        self.set_dayu_size(dayu_theme.tiny)
        return self
```