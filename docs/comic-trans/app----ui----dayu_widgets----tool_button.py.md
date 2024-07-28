# `.\comic-translate\app\ui\dayu_widgets\tool_button.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""MToolButton"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme  # 导入本地模块 dayu_theme
from .mixin import cursor_mixin  # 导入本地模块 cursor_mixin
from .qt import MIcon  # 导入本地模块 MIcon


@cursor_mixin
class MToolButton(QtWidgets.QToolButton):
    """MToolButton"""

    def __init__(self, parent=None):
        super(MToolButton, self).__init__(parent=parent)
        self._dayu_svg = None  # 初始化属性 _dayu_svg 为 None
        self.setAutoExclusive(False)  # 设置工具按钮不互斥
        self.setAutoRaise(True)  # 设置自动浮起效果

        self._polish_icon()  # 调用 _polish_icon 方法初始化图标样式
        self.toggled.connect(self._polish_icon)  # 绑定 toggled 信号到 _polish_icon 方法
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)  # 设置大小策略为最小

        self._dayu_size = dayu_theme.default_size  # 初始化 _dayu_size 属性为默认大小

    @QtCore.Slot(bool)
    def _polish_icon(self, checked=None):
        """根据 _dayu_svg 属性设置按钮图标"""
        if self._dayu_svg:
            if self.isCheckable() and self.isChecked():
                self.setIcon(MIcon(self._dayu_svg, dayu_theme.primary_color))
            else:
                self.setIcon(MIcon(self._dayu_svg))

    def enterEvent(self, event):
        """重写进入事件以突出显示图标"""
        if self._dayu_svg:
            self.setIcon(MIcon(self._dayu_svg, dayu_theme.primary_color))
        return super(MToolButton, self).enterEvent(event)

    def leaveEvent(self, event):
        """重写离开事件以恢复图标"""
        self._polish_icon()
        return super(MToolButton, self).leaveEvent(event)

    def get_dayu_size(self):
        """
        获取工具按钮的高度
        :return: 整数
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        设置工具按钮的大小。
        :param value: 整数
        :return: None
        """
        self._dayu_size = value
        self.style().polish(self)
        if self.toolButtonStyle() == QtCore.Qt.ToolButtonIconOnly:
            self.setFixedSize(QtCore.QSize(self._dayu_size, self._dayu_size))
            self.setIconSize(QtCore.QSize(self._dayu_size, self._dayu_size))

    def get_dayu_svg(self):
        """获取当前 SVG 路径"""
        return self._dayu_svg

    def set_dayu_svg(self, path):
        """设置当前 SVG 路径"""
        self._dayu_svg = path
        self._polish_icon()

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def huge(self):
        """设置 MToolButton 为巨大尺寸"""
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """设置 MToolButton 为大尺寸"""
        self.set_dayu_size(dayu_theme.large)
        return self
    def medium(self):
        """Set MToolButton to medium size"""
        # 调用外部 dayu_theme 的 medium 尺寸设置方法
        self.set_dayu_size(dayu_theme.medium)
        # 返回当前对象，支持链式调用
        return self

    def small(self):
        """Set MToolButton to small size"""
        # 调用外部 dayu_theme 的 small 尺寸设置方法
        self.set_dayu_size(dayu_theme.small)
        # 返回当前对象，支持链式调用
        return self

    def tiny(self):
        """Set MToolButton to tiny size"""
        # 调用外部 dayu_theme 的 tiny 尺寸设置方法
        self.set_dayu_size(dayu_theme.tiny)
        # 返回当前对象，支持链式调用
        return self

    def svg(self, path):
        """Set current svg path"""
        # 设置当前的 SVG 图标路径
        self.set_dayu_svg(path)
        # 返回当前对象，支持链式调用
        return self

    def icon_only(self):
        """Set tool button style to icon only"""
        # 设置工具按钮样式为仅图标
        self.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        # 设置按钮固定尺寸为 dayu_size
        self.setFixedSize(QtCore.QSize(self._dayu_size, self._dayu_size))
        # 返回当前对象，支持链式调用
        return self

    def text_only(self):
        """Set tool button style to text only"""
        # 设置工具按钮样式为仅文本
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        # 返回当前对象，支持链式调用
        return self

    def text_beside_icon(self):
        """Set tool button style to text beside icon"""
        # 设置工具按钮样式为文本旁边显示图标
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        # 返回当前对象，支持链式调用
        return self

    def text_under_icon(self):
        """Set tool button style to text under icon"""
        # 设置工具按钮样式为文本下方显示图标
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        # 返回当前对象，支持链式调用
        return self
```