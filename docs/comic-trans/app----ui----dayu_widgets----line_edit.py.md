# `.\comic-translate\app\ui\dayu_widgets\line_edit.py`

```py
# 设置脚本的解释器为 Python，并且指定编码为 UTF-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################

"""MLineEdit
Get the user input is a text field
"""

# 导入将来版本所需的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import functools

# 导入第三方模块
from PySide6 import QtCore, QtGui
from PySide6 import QtWidgets
import six

# 导入本地模块
from . import dayu_theme
from .browser import MClickBrowserFileToolButton
from .browser import MClickBrowserFolderToolButton
from .browser import MClickSaveFileToolButton
from .mixin import focus_shadow_mixin
from .push_button import MPushButton
from .tool_button import MToolButton

@focus_shadow_mixin
class MLineEdit(QtWidgets.QLineEdit):
    """MLineEdit"""

    # 延迟文本改变的信号，参数是字符串类型
    sig_delay_text_changed = QtCore.Signal(six.string_types[0])

    def __init__(self, text="", parent=None):
        # 初始化方法
        super(MLineEdit, self).__init__(text, parent)
        
        # 创建水平布局
        self._main_layout = QtWidgets.QHBoxLayout()
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addStretch()

        # 前缀和后缀部件初始化为空
        self._prefix_widget = None
        self._suffix_widget = None

        # 将水平布局设置为当前窗口的布局
        self.setLayout(self._main_layout)

        # 设置属性"history"，其值为当前文本的属性
        self.setProperty("history", self.property("text"))

        # 设置文本边距
        self.setTextMargins(2, 0, 2, 0)

        # 创建延迟定时器
        self._delay_timer = QtCore.QTimer()
        self._delay_timer.setInterval(500)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._slot_delay_text_changed)

        # 连接文本改变信号到延迟启动的槽函数
        self.textChanged.connect(self._slot_begin_to_start_delay)

        # 设置dayu_size默认为dayu_theme中的默认大小
        self._dayu_size = dayu_theme.default_size

    def get_dayu_size(self):
        """
        获取dayu_size属性的值，即按钮的高度
        :return: 整数
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        设置dayu_size属性的值，即按钮的高度
        :param value: 整数
        :return: None
        """
        self._dayu_size = value

        # 如果_prefix_widget有set_dayu_size方法，则调用它设置大小
        if hasattr(self._prefix_widget, "set_dayu_size"):
            self._prefix_widget.set_dayu_size(self._dayu_size)

        # 如果_suffix_widget有set_dayu_size方法，则调用它设置大小
        if hasattr(self._suffix_widget, "set_dayu_size"):
            self._suffix_widget.set_dayu_size(self._dayu_size)

        # 更新样式
        self.style().polish(self)

    # 定义dayu_size属性，类型为整数，其getter和setter为get_dayu_size和set_dayu_size方法
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def set_delay_duration(self, millisecond):
        """设置延迟定时器的超时时长."""
        self._delay_timer.setInterval(millisecond)

    @QtCore.Slot()
    def _slot_delay_text_changed(self):
        # 延迟定时器超时时发射sig_delay_text_changed信号，带上当前文本内容
        self.sig_delay_text_changed.emit(self.text())

    @QtCore.Slot(six.text_type)
    def _slot_begin_to_start_delay(self, _):
        # 当文本改变时，如果延迟定时器正在运行，则停止它；然后重新启动定时器
        if self._delay_timer.isActive():
            self._delay_timer.stop()
        self._delay_timer.start()
    def get_prefix_widget(self):
        """获取前缀部件，用于用户编辑"""
        return self._prefix_widget

    def set_prefix_widget(self, widget):
        """设置行编辑左侧起始部件"""
        if self._prefix_widget:
            index = self._main_layout.indexOf(self._prefix_widget)
            self._main_layout.takeAt(index)
            self._prefix_widget.setVisible(False)
        # 如果部件是 MPushButton 类型：
        widget.setProperty("combine", "horizontal")
        widget.setProperty("position", "left")
        if hasattr(widget, "set_dayu_size"):
            widget.set_dayu_size(self._dayu_size)

        margin = self.textMargins()
        margin.setLeft(margin.left() + widget.width())
        self.setTextMargins(margin)

        self._main_layout.insertWidget(0, widget)
        self._prefix_widget = widget
        return widget

    def get_suffix_widget(self):
        """获取后缀部件，用于用户编辑"""
        return self._suffix_widget

    def set_suffix_widget(self, widget):
        """设置行编辑右侧结束部件"""
        if self._suffix_widget:
            index = self._main_layout.indexOf(self._suffix_widget)
            self._main_layout.takeAt(index)
            self._suffix_widget.setVisible(False)
        # 如果部件是 MPushButton 类型：
        widget.setProperty("combine", "horizontal")
        widget.setProperty("position", "right")
        if hasattr(widget, "set_dayu_size"):
            widget.set_dayu_size(self._dayu_size)

        margin = self.textMargins()
        margin.setRight(margin.right() + widget.width())
        self.setTextMargins(margin)
        self._main_layout.addWidget(widget)
        self._suffix_widget = widget
        return widget

    def setText(self, text):
        """重写 setText 方法以保存文本到历史记录"""
        self.setProperty("history", "{}\n{}".format(self.property("history"), text))
        return super(MLineEdit, self).setText(text)

    def clear(self):
        """重写 clear 方法以清空历史记录"""
        self.setProperty("history", "")
        return super(MLineEdit, self).clear()

    def search(self):
        """为 MLineEdit 添加搜索图标按钮。"""
        suffix_button = MToolButton().icon_only().svg("close_line.svg")
        suffix_button.clicked.connect(self.clear)
        self.set_suffix_widget(suffix_button)
        self.setPlaceholderText(self.tr("输入关键字进行搜索..."))
        return self
    def error(self):
        """为 MLineEdit 添加错误信息存储工具集，显示红色样式"""

        @QtCore.Slot()
        def _slot_show_detail(self):
            # 创建一个只读的 QTextEdit 对话框
            dialog = QtWidgets.QTextEdit(self)
            dialog.setReadOnly(True)
            # 获取主屏幕的几何信息
            screen = QtGui.QGuiApplication.primaryScreen()
            geo = screen.geometry()
            # 设置对话框的位置和大小为主屏幕宽高的四分之一
            dialog.setGeometry(geo.width() / 2, geo.height() / 2, geo.width() / 4, geo.height() / 4)
            # 设置对话框的标题为"Error Detail Information"
            dialog.setWindowTitle(self.tr("Error Detail Information"))
            # 设置对话框显示的文本为对象的历史属性
            dialog.setText(self.property("history"))
            # 将对话框设为模态对话框
            dialog.setWindowFlags(QtCore.Qt.Dialog)
            # 显示对话框
            dialog.show()

        # 设置对象的属性"dayu_type"为"error"
        self.setProperty("dayu_type", "error")
        # 将对象设为只读状态
        self.setReadOnly(True)
        # 创建一个仅有图标的 MToolButton，并加载指定的 SVG 图标
        _suffix_button = MToolButton().icon_only().svg("detail_line.svg")
        # 将按钮的点击事件连接到 _slot_show_detail 方法，并传递当前对象 self 作为参数
        _suffix_button.clicked.connect(functools.partial(_slot_show_detail, self))
        # 设置对象的后缀部件为 _suffix_button
        self.set_suffix_widget(_suffix_button)
        # 设置对象的占位文本为"Error information will be here..."
        self.setPlaceholderText(self.tr("Error information will be here..."))
        # 返回对象本身
        return self

    def search_engine(self, text="Search"):
        """为 MLineEdit 添加一个 MPushButton 作为后缀，用于搜索"""
        # 创建一个带有指定文本的 MPushButton
        _suffix_button = MPushButton(text=text).primary()
        # 将按钮的点击事件连接到对象的 returnPressed 方法
        _suffix_button.clicked.connect(self.returnPressed)
        # 设置按钮的固定宽度为100
        _suffix_button.setFixedWidth(100)
        # 设置对象的后缀部件为 _suffix_button
        self.set_suffix_widget(_suffix_button)
        # 设置对象的占位文本为"Enter key word to search..."
        self.setPlaceholderText(self.tr("Enter key word to search..."))
        # 返回对象本身
        return self

    def file(self, filters=None):
        """为 MLineEdit 添加一个 MClickBrowserFileToolButton，用于选择文件"""
        # 创建一个 MClickBrowserFileToolButton 实例
        _suffix_button = MClickBrowserFileToolButton()
        # 连接按钮的 sig_file_changed 信号到对象的 setText 方法
        _suffix_button.sig_file_changed.connect(self.setText)
        # 设置按钮的文件过滤器为给定的 filters，如果没有指定则为空列表
        _suffix_button.set_dayu_filters(filters or [])
        # 连接对象的 textChanged 信号到按钮的 set_dayu_path 方法
        self.textChanged.connect(_suffix_button.set_dayu_path)
        # 设置对象的后缀部件为 _suffix_button
        self.set_suffix_widget(_suffix_button)
        # 设置对象的占位文本为"Click button to browser files"
        self.setPlaceholderText(self.tr("Click button to browser files"))
        # 返回对象本身
        return self

    def save_file(self, filters=None):
        """为 MLineEdit 添加一个 MClickSaveFileToolButton，用于设置保存文件"""
        # 创建一个 MClickSaveFileToolButton 实例
        _suffix_button = MClickSaveFileToolButton()
        # 连接按钮的 sig_file_changed 信号到对象的 setText 方法
        _suffix_button.sig_file_changed.connect(self.setText)
        # 设置按钮的文件过滤器为给定的 filters，如果没有指定则为空列表
        _suffix_button.set_dayu_filters(filters or [])
        # 连接对象的 textChanged 信号到按钮的 set_dayu_path 方法
        self.textChanged.connect(_suffix_button.set_dayu_path)
        # 设置对象的后缀部件为 _suffix_button
        self.set_suffix_widget(_suffix_button)
        # 设置对象的占位文本为"Click button to set save file"
        self.setPlaceholderText(self.tr("Click button to set save file"))
        # 返回对象本身
        return self

    def folder(self):
        """为 MLineEdit 添加一个 MClickBrowserFolderToolButton，用于选择文件夹"""
        # 创建一个 MClickBrowserFolderToolButton 实例
        _suffix_button = MClickBrowserFolderToolButton()
        # 连接按钮的 sig_folder_changed 信号到对象的 setText 方法
        _suffix_button.sig_folder_changed.connect(self.setText)
        # 连接对象的 textChanged 信号到按钮的 set_dayu_path 方法
        self.textChanged.connect(_suffix_button.set_dayu_path)
        # 设置对象的后缀部件为 _suffix_button
        self.set_suffix_widget(_suffix_button)
        # 设置对象的占位文本为"Click button to browser folder"
        self.setPlaceholderText(self.tr("Click button to browser folder"))
        # 返回对象本身
        return self
    def huge(self):
        """Set MLineEdit to huge size"""
        # 调用内部方法设置 MLineEdit 控件为巨大尺寸
        self.set_dayu_size(dayu_theme.huge)
        # 返回当前对象，以支持方法链式调用
        return self

    def large(self):
        """Set MLineEdit to large size"""
        # 调用内部方法设置 MLineEdit 控件为大尺寸
        self.set_dayu_size(dayu_theme.large)
        # 返回当前对象，以支持方法链式调用
        return self

    def medium(self):
        """Set MLineEdit to medium size"""
        # 调用内部方法设置 MLineEdit 控件为中等尺寸
        self.set_dayu_size(dayu_theme.medium)
        # 返回当前对象，以支持方法链式调用
        return self

    def small(self):
        """Set MLineEdit to small size"""
        # 调用内部方法设置 MLineEdit 控件为小尺寸
        self.set_dayu_size(dayu_theme.small)
        # 返回当前对象，以支持方法链式调用
        return self

    def tiny(self):
        """Set MLineEdit to tiny size"""
        # 调用内部方法设置 MLineEdit 控件为微小尺寸
        self.set_dayu_size(dayu_theme.tiny)
        # 返回当前对象，以支持方法链式调用
        return self

    def password(self):
        """Set MLineEdit to password echo mode"""
        # 将 MLineEdit 控件设置为密码回显模式
        self.setEchoMode(QtWidgets.QLineEdit.Password)
        # 返回当前对象，以支持方法链式调用
        return self
```