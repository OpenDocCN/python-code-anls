# `.\comic-translate\app\ui\dayu_widgets\menu_tab_widget.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
"""A Navigation menu"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme
from .button_group import MButtonGroupBase
from .divider import MDivider
from .tool_button import MToolButton

class MBlockButton(MToolButton):
    """MBlockButton"""

    def __init__(self, parent=None):
        super(MBlockButton, self).__init__(parent)
        self.setCheckable(True)

class MBlockButtonGroup(MButtonGroupBase):
    """MBlockButtonGroup"""

    sig_checked_changed = QtCore.Signal(int)

    def __init__(self, tab, orientation=QtCore.Qt.Horizontal, parent=None):
        super(MBlockButtonGroup, self).__init__(orientation=orientation, parent=parent)
        self.set_spacing(1)
        self._menu_tab = tab
        self._button_group.setExclusive(True)  # 设置按钮组为互斥模式，即只能选中一个按钮
        self._button_group.buttonClicked.connect(self._on_button_clicked)  # 连接按钮点击事件到处理函数

    def _on_button_clicked(self, button):
        # Get the id of the clicked button and emit it
        button_id = self._button_group.id(button)  # 获取点击按钮的 id
        self.sig_checked_changed.emit(button_id)  # 发送按钮 id 到信号

    def create_button(self, data_dict):
        button = MBlockButton()
        if data_dict.get("svg"):
            button.svg(data_dict.get("svg"))  # 如果提供了 svg 数据，设置按钮的 svg 图标
        if data_dict.get("text"):
            if data_dict.get("svg") or data_dict.get("icon"):
                button.text_beside_icon()  # 如果同时有文本和图标或者 svg，则设置文本在图标旁边显示
            else:
                button.text_only()  # 否则只显示文本
        else:
            button.icon_only()  # 如果没有文本，则只显示图标
        button.set_dayu_size(self._menu_tab.get_dayu_size())  # 设置按钮的大小
        return button

    def update_size(self, size):
        for button in self._button_group.buttons():
            button.set_dayu_size(size)  # 更新所有按钮的大小

    def set_dayu_checked(self, value):
        """Set current checked button's id"""
        button = self._button_group.button(value)  # 获取指定 id 的按钮
        button.setChecked(True)  # 设置该按钮为选中状态
        self.sig_checked_changed.emit(value)  # 发送按钮 id 到信号

    def get_dayu_checked(self):
        """Get current checked button's id"""
        return self._button_group.checkedId()  # 返回当前选中按钮的 id

    dayu_checked = QtCore.Property(int, get_dayu_checked, set_dayu_checked, notify=sig_checked_changed)

class MMenuTabWidget(QtWidgets.QWidget):
    """MMenuTabWidget"""
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super(MMenuTabWidget, self).__init__(parent=parent)
        # 创建一个 MBlockButtonGroup 实例，并将其设置为工具栏按钮组件
        self.tool_button_group = MBlockButtonGroup(tab=self, orientation=orientation)

        # 根据传入的 orientation 参数选择水平或垂直布局
        if orientation == QtCore.Qt.Horizontal:
            # 创建水平布局，并设置边距
            self._bar_layout = QtWidgets.QHBoxLayout()
            self._bar_layout.setContentsMargins(10, 0, 10, 0)
        else:
            # 创建垂直布局，并设置边距
            self._bar_layout = QtWidgets.QVBoxLayout()
            self._bar_layout.setContentsMargins(0, 0, 0, 0)

        # 将工具栏按钮组件添加到布局中
        self._bar_layout.addWidget(self.tool_button_group)
        # 添加伸缩空间
        self._bar_layout.addStretch()

        # 创建一个 QWidget 作为工具栏的容器，并设置其布局为 _bar_layout
        bar_widget = QtWidgets.QWidget()
        bar_widget.setObjectName("bar_widget")
        bar_widget.setLayout(self._bar_layout)
        # 设置 bar_widget 的背景样式
        bar_widget.setAttribute(QtCore.Qt.WA_StyledBackground)

        # 创建主布局，将 bar_widget 添加到其中
        main_lay = QtWidgets.QVBoxLayout()
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        main_lay.addWidget(bar_widget)

        # 如果 orientation 是水平的，则添加一个 MDivider 分割线到主布局中
        if orientation == QtCore.Qt.Horizontal:
            main_lay.addWidget(MDivider())

        # 添加一些间距到主布局中
        main_lay.addSpacing(5)
        # 设置当前组件的布局为主布局 main_lay
        self.setLayout(main_lay)

        # 设置默认的 dayu_size 为 dayu_theme 中定义的 large 大小
        self._dayu_size = dayu_theme.large

    def tool_bar_append_widget(self, widget):
        """将一个小部件添加到工具栏的最右侧位置。"""
        self._bar_layout.addWidget(widget)

    def tool_bar_insert_widget(self, widget):
        """将一个小部件插入到工具栏的最左侧位置。"""
        self._bar_layout.insertWidget(0, widget)

    def add_menu(self, data_dict, index=None):
        """向菜单中添加一个新的选项。"""
        self.tool_button_group.add_button(data_dict, index)

    def get_dayu_size(self):
        """
        获取菜单选项的尺寸。
        :return: 整数
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        设置菜单选项的尺寸。
        :param value: 整数
        :return: 无
        """
        self._dayu_size = value
        # 更新工具栏按钮组件的尺寸
        self.tool_button_group.update_size(self._dayu_size)
        # 刷新样式
        self.style().polish(self)

    # 定义 dayu_size 的属性，可以通过属性访问和设置菜单选项的尺寸
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)
```