# `.\comic-translate\app\ui\dayu_widgets\message.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""MMessage"""
# Import future modules
# 导入未来版本的模块支持
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
from .avatar import MAvatar
from .label import MLabel
from .loading import MLoading
from .qt import MPixmap
from .tool_button import MToolButton

class MMessage(QtWidgets.QWidget):
    InfoType = "info"  # 消息类型为信息
    SuccessType = "success"  # 消息类型为成功
    WarningType = "warning"  # 消息类型为警告
    ErrorType = "error"  # 消息类型为错误
    LoadingType = "loading"  # 消息类型为加载中

    default_config = {"duration": 2, "top": 24}  # 默认配置为消息显示时长2秒，顶部距离为24像素

    sig_closed = QtCore.Signal()  # 定义一个信号 sig_closed，用于表示消息关闭事件
    def __init__(self, text, duration=None, dayu_type=None, closable=False, parent=None):
        # 调用父类的初始化方法，将父对象设置为parent
        super(MMessage, self).__init__(parent)
        # 设置当前对象的对象名为"message"
        self.setObjectName("message")
        # 设置窗口标志，使窗口无边框、对话框风格且始终置顶显示
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        # 设置窗口关闭时自动删除对象
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        # 设置窗口具有样式化背景
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        # 根据给定的dayu_type选择加载相应的图标，如果没有指定，则使用默认图标
        if dayu_type == MMessage.LoadingType:
            _icon_label = MLoading.tiny()
        else:
            _icon_label = MAvatar.tiny()
            # 获取当前类型对应的颜色主题，并设置图像为当前类型的填充图像
            current_type = dayu_type or MMessage.InfoType
            _icon_label.set_dayu_image(
                MPixmap(
                    "{}_fill.svg".format(current_type),
                    vars(dayu_theme).get(current_type + "_color"),
                )
            )

        # 设置内容标签，显示传入的文本内容
        self._content_label = MLabel(parent=self)
        self._content_label.setText(text)

        # 设置关闭按钮，如果可关闭或未设置显示持续时间，则按钮可见
        self._close_button = MToolButton(parent=self).icon_only().svg("close_line.svg").tiny()
        self._close_button.clicked.connect(self.close)  # 点击关闭按钮时连接到关闭函数
        self._close_button.setVisible(closable or duration is None)

        # 设置主布局为水平布局，依次添加图标、内容标签、拉伸空间和关闭按钮
        self._main_lay = QtWidgets.QHBoxLayout()
        self._main_lay.addWidget(_icon_label)
        self._main_lay.addWidget(self._content_label)
        self._main_lay.addStretch()
        self._main_lay.addWidget(self._close_button)
        self.setLayout(self._main_lay)

        # 如果设置了显示持续时间，则创建定时器来自动关闭消息窗口和执行淡出动画
        if duration is not None:
            _close_timer = QtCore.QTimer(self)
            _close_timer.setSingleShot(True)
            _close_timer.timeout.connect(self.close)  # 定时器超时时连接到关闭函数
            _close_timer.timeout.connect(self.sig_closed)  # 定时器超时时连接到信号sig_closed
            _close_timer.setInterval(duration * 1000)  # 设置定时器超时时间（毫秒）

            _ani_timer = QtCore.QTimer(self)
            _ani_timer.timeout.connect(self._fade_out)  # 定时器超时时连接到淡出函数
            _ani_timer.setInterval(duration * 1000 - 300)  # 设置定时器超时时间（毫秒），略早于关闭定时器

            _close_timer.start()  # 启动定时器
            _ani_timer.start()  # 启动动画定时器

        # 设置位置动画对象，对当前对象执行位置动画
        self._pos_ani = QtCore.QPropertyAnimation(self)
        self._pos_ani.setTargetObject(self)
        self._pos_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)  # 设置动画缓和曲线
        self._pos_ani.setDuration(300)  # 设置动画持续时间（毫秒）
        self._pos_ani.setPropertyName(b"pos")  # 设置动画作用属性为位置属性

        # 设置透明度动画对象，对当前对象执行透明度动画
        self._opacity_ani = QtCore.QPropertyAnimation()
        self._opacity_ani.setTargetObject(self)
        self._opacity_ani.setDuration(300)  # 设置动画持续时间（毫秒）
        self._opacity_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)  # 设置动画缓和曲线
        self._opacity_ani.setPropertyName(b"windowOpacity")  # 设置动画作用属性为窗口透明度属性
        self._opacity_ani.setStartValue(0.0)  # 设置动画起始值
        self._opacity_ani.setEndValue(1.0)  # 设置动画结束值

        # 设置适当的位置给消息窗口，执行淡入效果
        self._set_proper_position(parent)
        self._fade_int()  # 执行淡入函数

    def _fade_out(self):
        # 设置位置动画反向播放
        self._pos_ani.setDirection(QtCore.QAbstractAnimation.Backward)
        self._pos_ani.start()  # 启动位置动画
        # 设置透明度动画反向播放
        self._opacity_ani.setDirection(QtCore.QAbstractAnimation.Backward)
        self._opacity_ani.start()  # 启动透明度动画
    # 启动位置动画和不透明度动画
    def _fade_int(self):
        self._pos_ani.start()
        self._opacity_ani.start()

    # 根据父窗口设置合适的位置
    def _set_proper_position(self, parent):
        # 获取父窗口的几何信息
        parent_geo = parent.geometry()
        
        # 如果父窗口是顶级窗口类型
        if parent.isWindowType():
            # 位置为父窗口的左上角
            pos = parent_geo.topLeft()
        elif parent in QtWidgets.QApplication.topLevelWidgets():
            # 父窗口虽然是独立窗口，但还有父级，例如在Maya中开发的工具窗口
            pos = parent_geo.topLeft()
        else:
            # 将父窗口左上角的位置映射到全局坐标系
            pos = parent.mapToGlobal(parent_geo.topLeft())
        
        offset = 0
        # 遍历父窗口的子控件
        for child in parent.children():
            # 如果子控件是 MMessage 实例且可见
            if isinstance(child, MMessage) and child.isVisible():
                # 更新偏移量为子控件的 y 坐标的最大值
                offset = max(offset, child.y())
        
        # 计算基础位置
        base = pos.y() + MMessage.default_config.get("top")
        # 目标 x 坐标位于父窗口中心左偏 100 像素
        target_x = pos.x() + parent_geo.width() / 2 - 100
        # 目标 y 坐标为偏移量加上基础位置，或者基础位置本身（如果没有偏移量）
        target_y = (offset + 50) if offset else base
        
        # 设置位置动画的起始和结束值
        self._pos_ani.setStartValue(QtCore.QPoint(target_x, target_y - 40))
        self._pos_ani.setEndValue(QtCore.QPoint(target_x, target_y))

    @classmethod
    # 显示普通消息
    def info(cls, text, parent, duration=None, closable=None):
        inst = cls(
            text,
            dayu_type=MMessage.InfoType,
            duration=duration,
            closable=closable if closable is not None else duration is None,
            parent=parent,
        )
        inst.show()
        return inst

    @classmethod
    # 显示成功消息
    def success(cls, text, parent, duration=None, closable=None):
        inst = cls(
            text,
            dayu_type=MMessage.SuccessType,
            duration=duration,
            closable=closable,
            parent=parent,
        )
        inst.show()
        return inst

    @classmethod
    # 显示警告消息
    def warning(cls, text, parent, duration=None, closable=None):
        inst = cls(
            text,
            dayu_type=MMessage.WarningType,
            duration=duration,
            closable=closable,
            parent=parent,
        )
        inst.show()
        return inst

    @classmethod
    # 显示错误消息
    def error(cls, text, parent, duration=None, closable=None):
        inst = cls(
            text,
            dayu_type=MMessage.ErrorType,
            duration=duration,
            closable=closable,
            parent=parent,
        )
        inst.show()
        return inst

    @classmethod
    # 显示带有加载动画的消息
    def loading(cls, text, parent):
        inst = cls(text, dayu_type=MMessage.LoadingType, parent=parent)
        inst.show()
        return inst

    @classmethod
    # 定义一个类方法 `config`，用于配置全局 MMessage 的持续时间和顶部位置参数
    def config(cls, duration=None, top=None):
        """
        Config the global MMessage duration and top setting.
        :param duration: int (unit is second)  # 持续时间，单位为秒，可选参数
        :param top: int (unit is px)  # 顶部位置，单位为像素，可选参数
        :return: None  # 返回空值
        """
        # 如果 duration 参数不为 None，则设置默认配置字典中的 "duration" 键为该值
        if duration is not None:
            cls.default_config["duration"] = duration
        # 如果 top 参数不为 None，则设置默认配置字典中的 "top" 键为该值
        if top is not None:
            cls.default_config["top"] = top
```