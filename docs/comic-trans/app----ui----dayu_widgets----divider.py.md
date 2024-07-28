# `.\comic-translate\app\ui\dayu_widgets\divider.py`

```py
# 指定脚本的解释器为 Python，并声明文件编码格式为 UTF-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################

"""
MDivider
"""
# 导入未来版本的特性模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets
import six

# 导入本地模块
from .label import MLabel

class MDivider(QtWidgets.QWidget):
    """
    A divider line separates different content.

    Property:
        dayu_text: six.string_types
    """

    # 对齐方式映射表，将 Qt 对齐枚举值映射到整数百分比
    _alignment_map = {
        QtCore.Qt.AlignCenter: 50,
        QtCore.Qt.AlignLeft: 20,
        QtCore.Qt.AlignRight: 80,
    }

    def __init__(
        self,
        text="",
        orientation=QtCore.Qt.Horizontal,
        alignment=QtCore.Qt.AlignCenter,
        parent=None,
    ):
        # 初始化函数，设置基本属性和布局
        super(MDivider, self).__init__(parent)
        self._orient = orientation
        self._text_label = MLabel().secondary()  # 创建一个 MLabel 实例并设置为次要样式
        self._left_frame = QtWidgets.QFrame()   # 创建左侧的 QFrame 实例
        self._right_frame = QtWidgets.QFrame()  # 创建右侧的 QFrame 实例
        self._main_lay = QtWidgets.QHBoxLayout()  # 创建水平布局管理器
        self._main_lay.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距为 0
        self._main_lay.setSpacing(0)  # 设置布局的间距为 0
        self._main_lay.addWidget(self._left_frame)  # 将左侧框架添加到布局
        self._main_lay.addWidget(self._text_label)  # 将文本标签添加到布局
        self._main_lay.addWidget(self._right_frame)  # 将右侧框架添加到布局
        self.setLayout(self._main_lay)  # 设置布局管理器为当前部件的布局

        if orientation == QtCore.Qt.Horizontal:
            # 如果是水平方向，则设置左右两侧框架为水平线样式和 Sunken 风格
            self._left_frame.setFrameShape(QtWidgets.QFrame.HLine)
            self._left_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
            self._right_frame.setFrameShape(QtWidgets.QFrame.HLine)
            self._right_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        else:
            # 如果是垂直方向，则隐藏文本标签和右侧框架，设置左侧框架为垂直线样式和 Sunken 风格，固定宽度为 5
            self._text_label.setVisible(False)
            self._right_frame.setVisible(False)
            self._left_frame.setFrameShape(QtWidgets.QFrame.VLine)
            self._left_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.setFixedWidth(5)  # 设置部件的固定宽度为 5

        # 根据对齐方式调整左右框架的伸展因子，以控制它们的比例
        self._main_lay.setStretchFactor(self._left_frame, self._alignment_map.get(alignment, 50))
        self._main_lay.setStretchFactor(self._right_frame, 100 - self._alignment_map.get(alignment, 50))
        self._text = None
        self.set_dayu_text(text)  # 调用设置文本的方法并传入初始文本值

    def set_dayu_text(self, value):
        """
        Set the divider's text.
        When text is empty, hide the text_label and right_frame to ensure the divider not has a gap.

        :param value: six.string_types
        :return: None
        """
        self._text = value
        self._text_label.setText(value)  # 设置文本标签的文本内容为传入值
        if self._orient == QtCore.Qt.Horizontal:
            self._text_label.setVisible(bool(value))  # 根据文本内容是否为空决定是否显示文本标签
            self._right_frame.setVisible(bool(value))  # 根据文本内容是否为空决定是否显示右侧框架
    def get_dayu_text(self):
        """
        获取当前的文本内容
        :return: six.string_types 返回一个字符串类型
        """
        return self._text

    dayu_text = QtCore.Property(six.string_types[0], get_dayu_text, set_dayu_text)

    @classmethod
    def left(cls, text=""):
        """创建一个文本居左的水平分隔线。"""
        return cls(text, alignment=QtCore.Qt.AlignLeft)

    @classmethod
    def right(cls, text=""):
        """创建一个文本居右的水平分隔线。"""
        return cls(text, alignment=QtCore.Qt.AlignRight)

    @classmethod
    def center(cls, text=""):
        """创建一个文本居中的水平分隔线。"""
        return cls(text, alignment=QtCore.Qt.AlignCenter)

    @classmethod
    def vertical(cls):
        """创建一个垂直分隔线。"""
        return cls(orientation=QtCore.Qt.Vertical)
```