# `.\comic-translate\app\ui\dayu_widgets\alert.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""
MAlert class.
"""
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
import functools

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets
import six

# Import local modules
from . import dayu_theme
from .avatar import MAvatar
from .label import MLabel
from .mixin import property_mixin
from .qt import MPixmap
from .qt import get_scale_factor
from .tool_button import MToolButton


@property_mixin
class MAlert(QtWidgets.QWidget):
    """
    Alert component for feedback.

    Property:
        dayu_type: The feedback type with different color container.
        dayu_text: The feedback string showed in container.
    """

    InfoType = "info"
    SuccessType = "success"
    WarningType = "warning"
    ErrorType = "error"

    def __init__(self, text="", parent=None, flags=QtCore.Qt.Widget):
        # 调用父类的构造函数初始化窗口部件
        super(MAlert, self).__init__(parent, flags)
        # 设置窗口部件具有样式化背景属性
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        # 创建一个头像标签，用于显示小图标
        self._icon_label = MAvatar()
        self._icon_label.set_dayu_size(dayu_theme.tiny)  # 设置头像标签的大小
        # 创建一个内容标签，用于显示反馈信息
        self._content_label = MLabel().secondary()
        # 创建一个关闭按钮，用于关闭警报窗口
        self._close_button = MToolButton().svg("close_line.svg").tiny().icon_only()
        self._close_button.clicked.connect(functools.partial(self.setVisible, False))  # 连接关闭按钮的点击事件
        scale_x, _ = get_scale_factor()
        margin = 8 * scale_x
        # 创建水平布局，并设置边距
        self._main_lay = QtWidgets.QHBoxLayout()
        self._main_lay.setContentsMargins(margin, margin, margin, margin)
        # 将头像标签、内容标签和关闭按钮添加到水平布局中
        self._main_lay.addWidget(self._icon_label)
        self._main_lay.addWidget(self._content_label)
        self._main_lay.addStretch()
        self._main_lay.addWidget(self._close_button)

        # 设置当前窗口部件的布局为水平布局
        self.setLayout(self._main_lay)

        # 初始化时显示图标和关闭按钮，设置反馈类型为信息类型，并设置反馈文本内容
        self.set_show_icon(True)
        self.set_closable(False)
        self._dayu_type = None
        self._dayu_text = None
        self.set_dayu_type(MAlert.InfoType)
        self.set_dayu_text(text)

    def set_closable(self, closable):
        """Display the close icon button or not."""
        self._close_button.setVisible(closable)  # 根据参数设置关闭按钮的可见性

    def set_show_icon(self, show_icon):
        """Display the information type icon or not."""
        self._icon_label.setVisible(show_icon)  # 根据参数设置头像标签的可见性

    def _set_dayu_text(self):
        self._content_label.setText(self._dayu_text)  # 设置内容标签显示的文本内容
        self.setVisible(bool(self._dayu_text))  # 根据文本内容决定警报窗口部件是否可见

    def set_dayu_text(self, value):
        """Set the feedback content."""
        if isinstance(value, six.string_types):
            self._dayu_text = value  # 如果输入的内容是字符串类型，则设置为反馈文本内容
        else:
            raise TypeError("Input argument 'value' should be string type, " "but get {}".format(type(value)))  # 如果输入的内容不是字符串类型，抛出类型错误异常
        self._set_dayu_text()  # 更新显示反馈文本内容
    def _set_dayu_type(self):
        # 设置图标标签的图像，根据反馈类型选择相应的 SVG 图片和颜色
        self._icon_label.set_dayu_image(
            MPixmap(
                "{}_fill.svg".format(self._dayu_type),
                vars(dayu_theme).get(self._dayu_type + "_color"),
            )
        )
        # 刷新样式
        self.style().polish(self)

    def set_dayu_type(self, value):
        """设置反馈类型。"""
        # 检查输入的反馈类型是否合法
        if value in [
            MAlert.InfoType,
            MAlert.SuccessType,
            MAlert.WarningType,
            MAlert.ErrorType,
        ]:
            self._dayu_type = value
        else:
            # 如果类型不合法，抛出异常
            raise ValueError("Input argument 'value' should be one of " "info/success/warning/error string.")
        # 更新显示的反馈类型
        self._set_dayu_type()

    def get_dayu_type(self):
        """
        获取 MAlert 的反馈类型。
        :return: str
        """
        return self._dayu_type

    def get_dayu_text(self):
        """
        获取 MAlert 的反馈消息。
        :return: six.string_types
        """
        return self._dayu_text

    # 定义属性 dayu_text，关联到 get_dayu_text 和 set_dayu_text 方法
    dayu_text = QtCore.Property(six.text_type, get_dayu_text, set_dayu_text)
    
    # 定义属性 dayu_type，关联到 get_dayu_type 和 set_dayu_type 方法
    dayu_type = QtCore.Property(str, get_dayu_type, set_dayu_type)

    def info(self):
        """设置 MAlert 的反馈类型为 InfoType"""
        self.set_dayu_type(MAlert.InfoType)
        return self

    def success(self):
        """设置 MAlert 的反馈类型为 SuccessType"""
        self.set_dayu_type(MAlert.SuccessType)
        return self

    def warning(self):
        """设置 MAlert 的反馈类型为 WarningType"""
        self.set_dayu_type(MAlert.WarningType)
        return self

    def error(self):
        """设置 MAlert 的反馈类型为 ErrorType"""
        self.set_dayu_type(MAlert.ErrorType)
        return self

    def closable(self):
        """设置 MAlert 可关闭"""
        self.set_closable(True)
        return self
```