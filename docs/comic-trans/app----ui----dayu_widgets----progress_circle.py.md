# `.\comic-translate\app\ui\dayu_widgets\progress_circle.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""MProgressCircle"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme
from . import utils
from .label import MLabel

class MProgressCircle(QtWidgets.QProgressBar):
    """
    MProgressCircle: Display the current progress of an operation flow.
    When you need to display the completion percentage of an operation.

    Property:
        dayu_width: int
        dayu_color: str
    """

    def __init__(self, dashboard=False, parent=None):
        super(MProgressCircle, self).__init__(parent)
        self._main_lay = QtWidgets.QHBoxLayout()  # 创建水平布局对象_main_lay
        self._default_label = MLabel().h3()  # 创建默认的标签对象_default_label，并设置为h3样式
        self._default_label.setAlignment(QtCore.Qt.AlignCenter)  # 设置默认标签对象的文本居中对齐
        self._main_lay.addWidget(self._default_label)  # 将默认标签对象添加到水平布局中
        self.setLayout(self._main_lay)  # 将水平布局_main_lay设置为当前窗口的布局
        self._color = None  # 初始化颜色属性_color为None
        self._width = None  # 初始化宽度属性_width为None

        self._start_angle = 90 * 16  # 初始化起始角度_start_angle为90度的16进制值
        self._max_delta_angle = 360 * 16  # 初始化最大角度变化值_max_delta_angle为360度的16进制值
        self._height_factor = 1.0  # 初始化高度因子_height_factor为1.0
        self._width_factor = 1.0  # 初始化宽度因子_width_factor为1.0
        if dashboard:
            self._start_angle = 225 * 16  # 若dashboard为True，则将起始角度_start_angle设置为225度的16进制值
            self._max_delta_angle = 270 * 16  # 若dashboard为True，则将最大角度变化值_max_delta_angle设置为270度的16进制值
            self._height_factor = (2 + pow(2, 0.5)) / 4 + 0.03  # 若dashboard为True，则重新计算高度因子_height_factor

        self.set_dayu_width(dayu_theme.progress_circle_default_radius)  # 调用set_dayu_width方法，设置当前圆形控件的宽度为默认半径
        self.set_dayu_color(dayu_theme.primary_color)  # 调用set_dayu_color方法，设置当前圆形控件的前景色为主题色

    def set_widget(self, widget):
        """
        Set a custom widget to show on the circle's inner center
        and replace the default percent label
        :param widget: QWidget
        :return: None
        """
        self.setTextVisible(False)  # 不显示默认的百分比标签
        if not widget.styleSheet():
            widget.setStyleSheet("background:transparent")  # 如果自定义部件没有样式表，则设置背景为透明
        self._main_lay.addWidget(widget)  # 将自定义部件添加到水平布局_main_lay中

    def get_dayu_width(self):
        """
        Get current circle fixed width
        :return: int
        """
        return self._width  # 返回当前圆形控件的宽度值

    def set_dayu_width(self, value):
        """
        Set current circle fixed width
        :param value: int
        :return: None
        """
        self._width = value  # 设置当前圆形控件的宽度为指定值value
        # 根据宽度因子和高度因子设置控件的固定大小
        self.setFixedSize(QtCore.QSize(int(self._width * self._width_factor), int(self._width * self._height_factor)))

    def get_dayu_color(self):
        """
        Get current circle foreground color
        :return: str
        """
        return self._color  # 返回当前圆形控件的前景色值

    def set_dayu_color(self, value):
        """
        Set current circle's foreground color
        :param value: str
        :return:
        """
        self._color = value  # 设置当前圆形控件的前景色为指定值value
        self.update()  # 更新控件显示，以更新颜色

    dayu_color = QtCore.Property(str, get_dayu_color, set_dayu_color)  # 定义圆形控件的前景色属性
    # 定义一个名为 dayu_width 的 QtCore 属性，类型为 int，获取方法为 get_dayu_width，设置方法为 set_dayu_width
    dayu_width = QtCore.Property(int, get_dayu_width, set_dayu_width)

    # 重写 QProgressBar 的 paintEvent 方法
    def paintEvent(self, event):
        """Override QProgressBar's paintEvent."""
        # 如果进度条显示的文本内容不等于默认标签的文本内容，则更新默认标签的文本
        if self.text() != self._default_label.text():
            self._default_label.setText(self.text())
        # 如果文本可见性状态不等于默认标签的可见性状态，则更新默认标签的可见性
        if self.isTextVisible() != self._default_label.isVisible():
            self._default_label.setVisible(self.isTextVisible())

        # 计算当前进度的百分比
        percent = utils.get_percent(self.value(), self.minimum(), self.maximum())
        # 获取 dayu_width 属性的值作为总宽度
        total_width = self.get_dayu_width()
        # 计算画笔宽度
        pen_width = int(3 * total_width / 50.0)
        # 计算圆的半径
        radius = total_width - pen_width - 1

        # 创建 QPainter 对象，并设置抗锯齿渲染
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)

        # 绘制背景圆形
        pen_background = QtGui.QPen()
        pen_background.setWidth(pen_width)
        pen_background.setColor(QtGui.QColor(dayu_theme.background_selected_color))
        pen_background.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen_background)
        painter.drawArc(
            pen_width / 2.0 + 1,
            pen_width / 2.0 + 1,
            radius,
            radius,
            self._start_angle,
            -self._max_delta_angle,
        )

        # 绘制前景圆形，表示当前进度
        pen_foreground = QtGui.QPen()
        pen_foreground.setWidth(pen_width)
        pen_foreground.setColor(QtGui.QColor(self._color))
        pen_foreground.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen_foreground)
        painter.drawArc(
            pen_width / 2.0 + 1,
            pen_width / 2.0 + 1,
            radius,
            radius,
            self._start_angle,
            -percent * 0.01 * self._max_delta_angle,
        )
        painter.end()

    @classmethod
    def dashboard(cls, parent=None):
        """Create a dashboard style MCircle"""
        # 创建一个 dashboard 样式的 MProgressCircle 实例，带有指定的父组件
        return MProgressCircle(dashboard=True, parent=parent)
```