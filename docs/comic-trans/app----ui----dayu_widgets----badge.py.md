# `.\comic-translate\app\ui\dayu_widgets\badge.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
"""
MBadge
"""
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets

# Import local modules
from . import utils


class MBadge(QtWidgets.QWidget):
    """
    Badge normally appears in proximity to notifications or user avatars with eye-catching appeal,
    typically displaying unread messages count.
    Show something at the wrapped widget top right.
    There is 3 type styles:
        dot: show a dot
        count: show a number at
        text: show a string

    Property:
        dayu_dot: bool
        dayu_text: six.string_types
        dayu_count: int
        dayu_overflow: int
    """

    def __init__(self, widget=None, parent=None):
        super(MBadge, self).__init__(parent)
        # Initialize the badge with the given widget and parent
        self._widget = widget
        self._overflow_count = 99  # Default maximum count before overflow

        self._dot = False  # Flag for dot style
        self._text = None   # Text content (not used in current implementation)
        self._count = None  # Count number to display

        # Button setup for displaying the badge
        self._badge_button = QtWidgets.QPushButton()
        self._badge_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        # Layout setup
        self._main_lay = QtWidgets.QGridLayout()
        self._main_lay.setContentsMargins(0, 0, 0, 0)
        if widget is not None:
            self._main_lay.addWidget(widget, 0, 0)  # Add widget to the layout
        self._main_lay.addWidget(self._badge_button, 0, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)  # Add badge button to the layout
        self.setLayout(self._main_lay)  # Set the layout for the widget

    def get_dayu_overflow(self):
        """
        Get the current overflow number.
        :return: int
        """
        return self._overflow_count

    def set_dayu_overflow(self, num):
        """
        Set the overflow number to control when the count exceeds a certain limit.
        :param num: int - new maximum number before overflow
        :return: None
        """
        self._overflow_count = num
        self._update_number()  # Update the displayed number if necessary

    def get_dayu_dot(self):
        """
        Check if the dot style is enabled.
        :return: bool
        """
        return self._dot

    def set_dayu_dot(self, show):
        """
        Set the dot style visibility.
        :param show: bool - whether to show the dot
        :return: None
        """
        self._dot = show
        self._badge_button.setText("")  # Clear any text on the badge button
        self._badge_button.setVisible(show)  # Set the visibility of the badge button
        self.style().polish(self)  # Update the style of the widget

    def get_dayu_count(self):
        """
        Get the current count number displayed.
        :return: int
        """
        return self._count

    def set_dayu_count(self, num):
        """
        Set the count number to display.
        :param num: int - number to display
        :return: None
        """
        self._count = num
        self._update_number()  # Update the displayed number

    def _update_number(self):
        """
        Update the displayed number on the badge button based on the current count.
        This method determines whether to show the count or a dot based on the current settings.
        :return: None
        """
        if self._dot:
            self._badge_button.setText("")
        elif self._count is not None:
            if self._count > self._overflow_count:
                self._badge_button.setText("{}+".format(self._overflow_count))
            else:
                self._badge_button.setText(str(self._count))
        else:
            self._badge_button.setText("")
    # 更新显示的数字，并设置按钮文本为格式化后的数字
    self._badge_button.setText(utils.overflow_format(self._count, self._overflow_count))
    # 根据计数值决定是否显示按钮
    self._badge_button.setVisible(self._count > 0)
    # 重置标志位为False
    self._dot = False
    # 更新部件的样式
    self.style().polish(self)

def get_dayu_text(self):
    """
    获取当前显示的文本
    :return: six.string_types
    """
    return self._text

def set_dayu_text(self, text):
    """
    设置当前样式以显示文本
    :param text: six.string_types
    :return: None
    """
    self._text = text
    # 设置按钮文本为给定的文本
    self._badge_button.setText(self._text)
    # 根据文本内容决定是否显示按钮
    self._badge_button.setVisible(bool(self._text))
    # 重置标志位为False
    self._dot = False
    # 更新部件的样式
    self.style().polish(self)

# 定义一个Qt属性 dayu_overflow，其获取器和设置器分别为get_dayu_overflow和set_dayu_overflow

# 定义一个Qt属性 dayu_dot，其获取器和设置器分别为get_dayu_dot和set_dayu_dot

# 定义一个Qt属性 dayu_count，其获取器和设置器分别为get_dayu_count和set_dayu_count

# 定义一个Qt属性 dayu_text，其获取器和设置器分别为get_dayu_text和set_dayu_text

@classmethod
def dot(cls, show=False, widget=None):
    """
    创建一个带有点样式的徽章。
    :param show: bool
    :param widget: 被包装的部件
    :return: 徽章实例
    """
    # 创建一个类实例
    inst = cls(widget=widget)
    # 设置实例的点标记
    inst.set_dayu_dot(show)
    return inst

@classmethod
def count(cls, count=0, widget=None):
    """
    创建一个带有数字样式的徽章。
    :param count: int
    :param widget: 被包装的部件
    :return: 徽章实例
    """
    # 创建一个类实例
    inst = cls(widget=widget)
    # 设置实例的计数
    inst.set_dayu_count(count)
    return inst

@classmethod
def text(cls, text="", widget=None):
    """
    创建一个带有文本样式的徽章。
    :param text: six.string_types
    :param widget: 被包装的部件
    :return: 徽章实例
    """
    # 创建一个类实例
    inst = cls(widget=widget)
    # 设置实例的文本
    inst.set_dayu_text(text)
    return inst
```