# `.\comic-translate\app\ui\dayu_widgets\drawer.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.6
# Email : muyanru345@163.com
###################################################################
"""MDrawer"""
# 导入未来模块，确保代码向后兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from .divider import MDivider  # 导入自定义的 MDivider 类
from .label import MLabel  # 导入自定义的 MLabel 类
from .qt import get_scale_factor  # 导入自定义的 get_scale_factor 函数
from .tool_button import MToolButton  # 导入自定义的 MToolButton 类


class MDrawer(QtWidgets.QWidget):
    """
    A panel which slides in from the edge of the screen.
    """

    LeftPos = "left"  # 定义 MDrawer 左侧位置的常量字符串
    RightPos = "right"  # 定义 MDrawer 右侧位置的常量字符串
    TopPos = "top"  # 定义 MDrawer 顶部位置的常量字符串
    BottomPos = "bottom"  # 定义 MDrawer 底部位置的常量字符串

    sig_closed = QtCore.Signal()  # 定义一个信号 sig_closed，用于表示抽屉关闭事件
    def __init__(self, title, position="right", closable=True, parent=None):
        super(MDrawer, self).__init__(parent)
        self.setObjectName("message")
        self.setWindowFlags(QtCore.Qt.Popup)
        # 设置窗口标志为弹出窗口，用于显示在顶层
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        # 设置窗口具有样式化背景属性

        self._title_label = MLabel(parent=self).h4()
        # 创建一个标题标签，并设置其为H4格式
        self._title_label.setText(title)
        # 设置标题标签的文本内容为给定的标题文字

        self._close_button = MToolButton(parent=self).icon_only().svg("close_line.svg").small()
        # 创建一个关闭按钮，仅包含图标，并设置图标为close_line.svg，大小为small
        self._close_button.clicked.connect(self.close)
        # 将关闭按钮的点击事件连接到自身的关闭方法
        self._close_button.setVisible(closable or False)
        # 根据closable参数确定关闭按钮是否可见

        self._title_extra_lay = QtWidgets.QHBoxLayout()
        _title_lay = QtWidgets.QHBoxLayout()
        _title_lay.addWidget(self._title_label)
        _title_lay.addStretch()
        _title_lay.addLayout(self._title_extra_lay)
        _title_lay.addWidget(self._close_button)
        # 设置标题栏的布局，包括标题标签、额外布局和关闭按钮

        self._bottom_lay = QtWidgets.QHBoxLayout()
        self._bottom_lay.addStretch()
        # 创建底部布局并添加拉伸空间

        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        # 创建滚动区域，并允许其内部的widget大小可调整

        self._main_lay = QtWidgets.QVBoxLayout()
        self._main_lay.addLayout(_title_lay)
        self._main_lay.addWidget(MDivider())
        self._main_lay.addWidget(self._scroll_area)
        self._main_lay.addWidget(MDivider())
        self._main_lay.addLayout(self._bottom_lay)
        # 设置主布局，包括标题栏布局、分割线、滚动区域、分割线和底部布局
        self.setLayout(self._main_lay)
        # 将主布局应用于当前窗口

        self._position = position
        # 存储给定的位置参数

        self._close_timer = QtCore.QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.close)
        self._close_timer.timeout.connect(self.sig_closed)
        self._close_timer.setInterval(300)
        self._is_first_close = True
        # 创建一个定时器，用于在关闭窗口时执行操作，并设置首次关闭标志为True

        self._pos_ani = QtCore.QPropertyAnimation(self)
        self._pos_ani.setTargetObject(self)
        self._pos_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._pos_ani.setDuration(300)
        self._pos_ani.setPropertyName(b"pos")
        # 创建位置动画对象，用于窗口动画效果

        self._opacity_ani = QtCore.QPropertyAnimation()
        self._opacity_ani.setTargetObject(self)
        self._opacity_ani.setDuration(300)
        self._opacity_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._opacity_ani.setPropertyName(b"windowOpacity")
        self._opacity_ani.setStartValue(0.0)
        self._opacity_ani.setEndValue(1.0)
        # 创建透明度动画对象，用于窗口淡入效果

    def set_widget(self, widget):
        self._scroll_area.setWidget(widget)
        # 设置滚动区域内的widget

    def add_widget_to_bottom(self, button):
        self._bottom_lay.addWidget(button)
        # 将给定的按钮添加到底部布局中

    def add_widget_to_top(self, button):
        self._title_extra_lay.addWidget(button)
        # 将给定的按钮添加到额外标题布局中的顶部

    def _fade_out(self):
        self._pos_ani.setDirection(QtCore.QAbstractAnimation.Backward)
        self._pos_ani.start()
        # 设置位置动画方向为向后（反向），并启动位置动画
        self._opacity_ani.setDirection(QtCore.QAbstractAnimation.Backward)
        self._opacity_ani.start()
        # 设置透明度动画方向为向后（反向），并启动透明度动画
    # 启动位置动画和透明度动画
    def _fade_int(self):
        self._pos_ani.start()
        self._opacity_ani.start()

    # 设置 MDrawer 的正确位置
    def _set_proper_position(self):
        # 获取父级窗口
        parent = self.parent()
        # 获取父级窗口的几何属性
        parent_geo = parent.geometry()
        
        # 根据设定的位置，调整抽屉的位置和大小
        if self._position == MDrawer.LeftPos:
            # 计算左侧位置
            pos = parent_geo.topLeft() if parent.parent() is None else parent.mapToGlobal(parent_geo.topLeft())
            target_x = pos.x()
            target_y = pos.y()
            self.setFixedHeight(parent_geo.height())
            self._pos_ani.setStartValue(QtCore.QPoint(target_x - self.width(), target_y))
            self._pos_ani.setEndValue(QtCore.QPoint(target_x, target_y))
        
        if self._position == MDrawer.RightPos:
            # 计算右侧位置
            pos = parent_geo.topRight() if parent.parent() is None else parent.mapToGlobal(parent_geo.topRight())
            self.setFixedHeight(parent_geo.height())
            target_x = pos.x() - self.width()
            target_y = pos.y()
            self._pos_ani.setStartValue(QtCore.QPoint(target_x + self.width(), target_y))
            self._pos_ani.setEndValue(QtCore.QPoint(target_x, target_y))
        
        if self._position == MDrawer.TopPos:
            # 计算顶部位置
            pos = parent_geo.topLeft() if parent.parent() is None else parent.mapToGlobal(parent_geo.topLeft())
            self.setFixedWidth(parent_geo.width())
            target_x = pos.x()
            target_y = pos.y()
            self._pos_ani.setStartValue(QtCore.QPoint(target_x, target_y - self.height()))
            self._pos_ani.setEndValue(QtCore.QPoint(target_x, target_y))
        
        if self._position == MDrawer.BottomPos:
            # 计算底部位置
            pos = parent_geo.bottomLeft() if parent.parent() is None else parent.mapToGlobal(parent_geo.bottomLeft())
            self.setFixedWidth(parent_geo.width())
            target_x = pos.x()
            target_y = pos.y() - self.height()
            self._pos_ani.setStartValue(QtCore.QPoint(target_x, target_y + self.height()))
            self._pos_ani.setEndValue(QtCore.QPoint(target_x, target_y))

    # 设置抽屉的位置
    def set_dayu_position(self, value):
        """
        设置 MDrawer 的位置。
        可选值为 top/right/bottom/left，默认为 right。
        :param value: str
        :return: None
        """
        self._position = value
        # 获取缩放因子
        scale_x, _ = get_scale_factor()
        # 根据位置设置抽屉的固定高度或宽度
        if value in [MDrawer.BottomPos, MDrawer.TopPos]:
            self.setFixedHeight(200 * scale_x)
        else:
            self.setFixedWidth(200 * scale_x)

    # 获取抽屉当前的位置
    def get_dayu_position(self):
        """
        获取 MDrawer 的位置
        :return: str
        """
        return self._position

    # 将 dayu_position 方法作为 QtCore 属性
    dayu_position = QtCore.Property(str, get_dayu_position, set_dayu_position)

    # 将抽屉的位置设置为左侧
    def left(self):
        """将抽屉位置设置为左侧"""
        self.set_dayu_position(MDrawer.LeftPos)
        return self

    # 将抽屉的位置设置为右侧
    def right(self):
        """将抽屉位置设置为右侧"""
        self.set_dayu_position(MDrawer.RightPos)
        return self
    def top(self):
        """
        将抽屉的位置设置为顶部

        使用 MDrawer.TopPos 常量来设置抽屉的位置到顶部，
        然后返回当前对象本身。
        """
        self.set_dayu_position(MDrawer.TopPos)
        return self

    def bottom(self):
        """
        将抽屉的位置设置为底部

        使用 MDrawer.BottomPos 常量来设置抽屉的位置到底部，
        然后返回当前对象本身。
        """
        self.set_dayu_position(MDrawer.BottomPos)
        return self

    def show(self):
        """
        显示抽屉

        调用内部方法设置适当的位置和淡入效果，然后调用父类的显示方法。
        特别注释：用于中文输入（timmyliang）
        最后激活当前窗口。
        """
        self._set_proper_position()
        self._fade_int()
        super(MDrawer, self).show()
        self.activateWindow()

    def closeEvent(self, event):
        """
        处理关闭事件

        如果是第一次关闭，则启动关闭计时器、淡出效果，并忽略当前事件；
        否则，接受该事件（即允许关闭窗口）。
        """
        if self._is_first_close:
            self._is_first_close = False
            self._close_timer.start()
            self._fade_out()
            event.ignore()
        else:
            event.accept()
```