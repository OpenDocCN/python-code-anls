# `.\comic-translate\app\ui\dayu_widgets\line_tab_widget.py`

```py
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
"""MLineTabWidget"""

# Import future modules
# 导入未来版本兼容模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# Import local modules
# 导入本地模块
from . import dayu_theme
from .button_group import MButtonGroupBase
from .divider import MDivider
from .stacked_widget import MStackedWidget
from .tool_button import MToolButton

class MUnderlineButton(MToolButton):
    """MUnderlineButton"""

    def __init__(self, parent=None):
        super(MUnderlineButton, self).__init__(parent)
        # 设置按钮为可选中状态
        self.setCheckable(True)


class MUnderlineButtonGroup(MButtonGroupBase):
    """MUnderlineButtonGroup"""

    # 定义信号，当按钮选中状态改变时触发
    sig_checked_changed = QtCore.Signal(int)

    def __init__(self, tab, parent=None):
        super(MUnderlineButtonGroup, self).__init__(parent=parent)
        # 设置按钮组关联的标签页
        self._line_tab = tab
        # 设置按钮之间的间距为1像素
        self.set_spacing(1)
        # 设置按钮组为互斥选择模式
        self._button_group.setExclusive(True)
        # 连接按钮点击信号与自定义的选中状态改变信号
        self._button_group.buttonClicked[int].connect(self.sig_checked_changed)

    def create_button(self, data_dict):
        # 创建下划线按钮实例
        button = MUnderlineButton(parent=self)
        # 根据数据字典设置按钮的图标
        if data_dict.get("svg"):
            button.svg(data_dict.get("svg"))
        # 根据数据字典设置按钮的文本显示方式
        if data_dict.get("text"):
            if data_dict.get("svg") or data_dict.get("icon"):
                button.text_beside_icon()
            else:
                button.text_only()
        else:
            button.icon_only()
        # 设置按钮的大小
        button.set_dayu_size(self._line_tab.get_dayu_size())
        return button

    def update_size(self, size):
        # 更新按钮组内所有按钮的大小
        for button in self._button_group.buttons():
            button.set_dayu_size(size)

    def set_dayu_checked(self, value):
        """Set current checked button's id"""
        # 设置指定 id 的按钮为选中状态
        button = self._button_group.button(value)
        button.setChecked(True)
        # 发送选中状态改变信号
        self.sig_checked_changed.emit(value)

    def get_dayu_checked(self):
        """Get current checked button's id"""
        # 返回当前选中按钮的 id
        return self._button_group.checkedId()

    # 定义属性 dayu_checked，通过 get_dayu_checked 和 set_dayu_checked 方法获取和设置选中按钮的 id，并在选中状态改变时发出通知
    dayu_checked = QtCore.Property(int, get_dayu_checked, set_dayu_checked, notify=sig_checked_changed)


class MLineTabWidget(QtWidgets.QWidget):
    """MLineTabWidget"""
    def __init__(self, alignment=QtCore.Qt.AlignCenter, parent=None):
        # 调用父类的构造函数，初始化一个带有指定对齐方式的 MLineTabWidget 对象
        super(MLineTabWidget, self).__init__(parent=parent)
        # 创建一个 MUnderlineButtonGroup 对象作为工具栏按钮的分组
        self.tool_button_group = MUnderlineButtonGroup(tab=self)
        # 创建水平布局对象作为工具栏的容器
        self.bar_layout = QtWidgets.QHBoxLayout()
        # 设置水平布局的边距为0
        self.bar_layout.setContentsMargins(0, 0, 0, 0)
        # 根据 alignment 参数的不同，设置工具栏的对齐方式和内容
        if alignment == QtCore.Qt.AlignCenter:
            self.bar_layout.addStretch()
            self.bar_layout.addWidget(self.tool_button_group)
            self.bar_layout.addStretch()
        elif alignment == QtCore.Qt.AlignLeft:
            self.bar_layout.addWidget(self.tool_button_group)
            self.bar_layout.addStretch()
        elif alignment == QtCore.Qt.AlignRight:
            self.bar_layout.addStretch()
            self.bar_layout.addWidget(self.tool_button_group)
        
        # 创建 MStackedWidget 对象用于管理多个页面的堆栈
        self.stack_widget = MStackedWidget()
        # 将工具按钮组的信号连接到堆栈控件的当前页面索引变更槽函数
        self.tool_button_group.sig_checked_changed.connect(self.stack_widget.setCurrentIndex)
        
        # 创建垂直布局对象作为主布局容器
        main_lay = QtWidgets.QVBoxLayout()
        # 设置垂直布局的边距为0，间距为0
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        # 将工具栏布局添加到主布局中
        main_lay.addLayout(self.bar_layout)
        # 添加一个分割线到主布局中
        main_lay.addWidget(MDivider())
        # 添加间距到主布局中
        main_lay.addSpacing(5)
        # 将堆栈控件添加到主布局中
        main_lay.addWidget(self.stack_widget)
        # 设置主布局为 MLineTabWidget 的布局
        self.setLayout(main_lay)
        # 设置 dayu_size 初始值为 dayu_theme 的默认大小
        self._dayu_size = dayu_theme.default

    def append_widget(self, widget):
        """Add the widget to line tab's right position."""
        # 将指定的 widget 添加到工具栏布局的末尾
        self.bar_layout.addWidget(widget)

    def insert_widget(self, widget):
        """Insert the widget to line tab's left position."""
        # 将指定的 widget 插入到工具栏布局的开头
        self.bar_layout.insertWidget(0, widget)

    def add_tab(self, widget, data_dict):
        """Add a tab"""
        # 将指定的 widget 添加到堆栈控件中作为新的页面
        self.stack_widget.addWidget(widget)
        # 使用 data_dict 中的数据添加一个按钮到工具按钮组，并将其与新页面关联
        self.tool_button_group.add_button(data_dict, self.stack_widget.count() - 1)

    def get_dayu_size(self):
        """
        Get the line tab size.
        :return: integer
        """
        # 返回当前的 dayu_size 值
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the line tab size.
        :param value: integer
        :return: None
        """
        # 设置 dayu_size 的值为指定的 value
        self._dayu_size = value
        # 更新工具按钮组的大小
        self.tool_button_group.update_size(self._dayu_size)
        # 刷新当前控件的样式
        self.style().polish(self)

    # 定义一个 dayu_size 的属性，用于获取和设置 MLineTabWidget 的大小
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)
```