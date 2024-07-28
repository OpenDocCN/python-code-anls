# `.\comic-translate\app\ui\dayu_widgets\loading.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.4
# Email : muyanru345@163.com
###################################################################
"""
MLoading
"""
# 导入未来模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme
from .qt import MPixmap

class MLoading(QtWidgets.QWidget):
    """
    Show a loading animation image.
    """

    def __init__(self, size=None, color=None, parent=None):
        super(MLoading, self).__init__(parent)
        # 如果未提供尺寸，则使用默认尺寸
        size = size or dayu_theme.default_size
        # 设置小部件的固定大小
        self.setFixedSize(QtCore.QSize(size, size))
        # 创建并缩放加载动画的图片对象
        self.pix = MPixmap("loading.svg", color or dayu_theme.primary_color).scaledToWidth(
            size, QtCore.Qt.SmoothTransformation
        )
        # 初始化旋转角度
        self._rotation = 0
        # 创建属性动画对象
        self._loading_ani = QtCore.QPropertyAnimation()
        self._loading_ani.setTargetObject(self)
        # 设置动画持续时间为1秒
        self._loading_ani.setDuration(1000)
        # 设置动画作用的属性为rotation
        self._loading_ani.setPropertyName(b"rotation")
        # 设置动画起始值和结束值
        self._loading_ani.setStartValue(0)
        self._loading_ani.setEndValue(360)
        # 设置动画循环次数为无限循环
        self._loading_ani.setLoopCount(-1)
        # 启动动画
        self._loading_ani.start()

    def _set_rotation(self, value):
        # 设置_rotation属性的值，并触发更新
        self._rotation = value
        self.update()

    def _get_rotation(self):
        # 获取_rotation属性的值
        return self._rotation

    # 定义rotation属性，使其可以访问和修改_rotation属性
    rotation = QtCore.Property(int, _get_rotation, _set_rotation)

    def paintEvent(self, event):
        """override the paint event to paint the 1/4 circle image."""
        # 创建绘图对象
        painter = QtGui.QPainter(self)
        # 设置平滑变换渲染
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        # 将绘图原点移动到图片中心
        painter.translate(self.pix.width() / 2, self.pix.height() / 2)
        # 根据当前_rotation角度旋转绘图对象
        painter.rotate(self._rotation)
        # 绘制加载动画图片
        painter.drawPixmap(
            -self.pix.width() / 2,
            -self.pix.height() / 2,
            self.pix.width(),
            self.pix.height(),
            self.pix,
        )
        painter.end()
        # 调用父类的paintEvent方法
        return super(MLoading, self).paintEvent(event)

    @classmethod
    def huge(cls, color=None):
        """Create a MLoading with huge size"""
        # 创建一个巨大尺寸的MLoading对象
        return cls(dayu_theme.huge, color)

    @classmethod
    def large(cls, color=None):
        """Create a MLoading with large size"""
        # 创建一个大尺寸的MLoading对象
        return cls(dayu_theme.large, color)

    @classmethod
    def medium(cls, color=None):
        """Create a MLoading with medium size"""
        # 创建一个中等尺寸的MLoading对象
        return cls(dayu_theme.medium, color)

    @classmethod
    def small(cls, color=None):
        """Create a MLoading with small size"""
        # 创建一个小尺寸的MLoading对象
        return cls(dayu_theme.small, color)

    @classmethod
    # 定义一个类方法 tiny，用于创建一个尺寸很小的 MLoading 对象
    def tiny(cls, color=None):
        """Create a MLoading with tiny size"""
        # 使用 dayu_theme.tiny 和给定的颜色参数创建一个 MLoading 对象并返回
        return cls(dayu_theme.tiny, color)
class MLoadingWrapper(QtWidgets.QWidget):
    """
    A wrapper widget to show the loading widget or hide.
    Property:
        dayu_loading: bool. current loading state.
    """

    def __init__(self, widget, loading=True, parent=None):
        super(MLoadingWrapper, self).__init__(parent)
        self._widget = widget  # 存储传入的主要显示部件
        self._mask_widget = QtWidgets.QFrame()  # 创建一个QFrame用作遮罩层
        self._mask_widget.setObjectName("mask")  # 设置遮罩层的对象名为"mask"
        self._mask_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # 设置遮罩层的大小策略为扩展
        self._loading_widget = MLoading()  # 创建一个MLoading实例作为加载动画部件
        self._loading_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # 设置加载动画部件的大小策略为扩展

        self._main_lay = QtWidgets.QGridLayout()  # 创建一个网格布局管理器实例
        self._main_lay.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距为0
        self._main_lay.addWidget(widget, 0, 0)  # 将主要显示部件添加到布局的(0, 0)位置
        self._main_lay.addWidget(self._mask_widget, 0, 0)  # 将遮罩层添加到布局的(0, 0)位置（覆盖主要显示部件）
        self._main_lay.addWidget(self._loading_widget, 0, 0, QtCore.Qt.AlignCenter)  # 将加载动画部件添加到布局的(0, 0)位置，并居中对齐
        self.setLayout(self._main_lay)  # 将布局设置为当前窗口部件的布局
        self._loading = None  # 初始化加载状态为None
        self.set_dayu_loading(loading)  # 设置初始的加载状态

    def _set_loading(self):
        self._loading_widget.setVisible(self._loading)  # 设置加载动画部件的可见性为当前加载状态
        self._mask_widget.setVisible(self._loading)  # 设置遮罩层的可见性为当前加载状态

    def set_dayu_loading(self, loading):
        """
        Set current state to loading or not
        :param loading: bool 传入的加载状态
        :return: None
        """
        self._loading = loading  # 更新当前加载状态
        self._set_loading()  # 更新加载动画部件和遮罩层的可见性

    def get_dayu_loading(self):
        """
        Get current loading widget is loading or not.
        :return: bool 返回当前加载状态
        """
        return self._loading

    dayu_loading = QtCore.Property(bool, get_dayu_loading, set_dayu_loading)
```