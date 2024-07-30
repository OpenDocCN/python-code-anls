# `.\comic-translate\app\ui\dayu_widgets\completer.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: TimmyLiang
# Date  : 2021.12
# Email : 820472580@qq.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore      # 导入 PySide6 的核心模块
from PySide6 import QtGui       # 导入 PySide6 的图形界面模块
from PySide6 import QtWidgets   # 导入 PySide6 的窗口部件模块

# Import local modules
from . import dayu_theme        # 从当前包中导入 dayu_theme 模块
from .mixin import property_mixin  # 从当前包中导入 property_mixin 模块


@property_mixin
class MCompleter(QtWidgets.QCompleter):
    ITEM_HEIGHT = 28  # 类级别的常量定义，表示每个项目的高度为 28 像素

    def __init__(self, parent=None):
        super(MCompleter, self).__init__(parent)
        self.setProperty("animatable", True)  # 设置属性 "animatable" 为 True，表示支持动画效果

        popup = self.popup()  # 获取 QCompleter 的弹出窗口
        dayu_theme.apply(popup)  # 应用 dayu_theme 的样式到弹出窗口

        # 创建控制窗口透明度动画的 QPropertyAnimation 对象
        self._opacity_anim = QtCore.QPropertyAnimation(popup, b"windowOpacity")
        self.setProperty("anim_opacity_duration", 300)  # 设置动画持续时间为 300 毫秒
        self.setProperty("anim_opacity_curve", "OutCubic")  # 设置透明度动画的缓动曲线为 OutCubic
        self.setProperty("anim_opacity_start", 0)  # 设置透明度动画的起始值为 0
        self.setProperty("anim_opacity_end", 1)  # 设置透明度动画的结束值为 1

        # 创建控制窗口大小动画的 QPropertyAnimation 对象
        self._size_anim = QtCore.QPropertyAnimation(popup, b"size")
        self.setProperty("anim_size_duration", 300)  # 设置动画持续时间为 300 毫秒
        self.setProperty("anim_size_curve", "OutCubic")  # 设置大小动画的缓动曲线为 OutCubic

        popup.installEventFilter(self)  # 安装事件过滤器，以便监控弹出窗口的事件

    def _set_anim_opacity_duration(self, value):
        self._opacity_anim.setDuration(value)  # 设置透明度动画的持续时间

    def _set_anim_opacity_curve(self, value):
        curve = getattr(QtCore.QEasingCurve, value, None)
        assert curve, "invalid QEasingCurve"
        self._opacity_anim.setEasingCurve(curve)  # 设置透明度动画的缓动曲线

    def _set_anim_opacity_start(self, value):
        self._opacity_anim.setStartValue(value)  # 设置透明度动画的起始值

    def _set_anim_opacity_end(self, value):
        self._opacity_anim.setEndValue(value)  # 设置透明度动画的结束值

    def _set_anim_size_duration(self, value):
        self._size_anim.setDuration(value)  # 设置大小动画的持续时间

    def _set_anim_size_curve(self, value):
        curve = getattr(QtCore.QEasingCurve, value, None)
        assert curve, "invalid QEasingCurve"
        self._size_anim.setEasingCurve(curve)  # 设置大小动画的缓动曲线

    def _set_anim_size_start(self, value):
        self._size_anim.setStartValue(value)  # 设置大小动画的起始值

    def _set_anim_size_end(self, value):
        self._size_anim.setEndValue(value)  # 设置大小动画的结束值
    # 初始化尺寸动画的起始大小
    def init_size(self):
        # 弹出窗口对象
        popup = self.popup()
        
        # 获取弹出窗口的数据模型
        model = popup.model()
        
        # 获取当前控件的宽度
        width = self.widget().width()
        
        # 获取弹出窗口的推荐最大高度
        max_height = popup.sizeHint().height()
        
        # 获取数据模型中第一行第一列的项的高度作为每项的默认高度
        item_height = model.data(model.index(0, 0), QtCore.Qt.SizeHintRole)
        
        # 计算列表的总高度
        height = (item_height or self.ITEM_HEIGHT) * model.rowCount()
        
        # 限制列表高度不超过最大高度
        height = height if height < max_height else max_height
        
        # 获取动画起始大小属性值，如果不存在则使用默认的空大小
        start_size = self.property("anim_size_start")
        start_size = start_size if start_size else QtCore.QSize(0, 0)
        
        # 获取动画结束大小属性值，如果不存在则使用当前计算得到的宽度和高度
        end_size = self.property("anim_size_end")
        end_size = end_size if end_size else QtCore.QSize(width, height)
        
        # 设置尺寸动画的起始值和结束值
        self._size_anim.setStartValue(start_size)
        self._size_anim.setEndValue(end_size)

    # 启动动画效果
    def start_anim(self):
        # 初始化尺寸
        self.init_size()
        
        # 启动透明度动画
        self._opacity_anim.start()
        
        # 启动尺寸动画
        self._size_anim.start()

    # 事件过滤器，处理控件的显示事件
    def eventFilter(self, widget, event):
        # 如果事件类型是控件显示，并且控件支持动画效果
        if event.type() == QtCore.QEvent.Show and self.property("animatable"):
            # 启动动画效果
            self.start_anim()
        
        # 继续处理默认的事件过滤器逻辑
        return super(MCompleter, self).eventFilter(widget, event)
```