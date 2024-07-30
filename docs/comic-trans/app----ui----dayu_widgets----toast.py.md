# `.\comic-translate\app\ui\dayu_widgets\toast.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""
MToast
"""
# Import future modules
# 导入未来模块，确保代码在Python 2和3中兼容
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


class MToast(QtWidgets.QWidget):
    """
    MToast
    A Phone style message.
    """

    InfoType = "info"
    SuccessType = "success"
    WarningType = "warning"
    ErrorType = "error"
    LoadingType = "loading"

    default_config = {
        "duration": 2,
    }

    sig_closed = QtCore.Signal()
    def __init__(self, text, duration=None, dayu_type=None, parent=None):
        super(MToast, self).__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Dialog
            | QtCore.Qt.WindowStaysOnTopHint
        )
        
        # 设置窗口属性：透明背景、关闭时删除、样式背景
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        _icon_lay = QtWidgets.QHBoxLayout()
        _icon_lay.addStretch()

        if dayu_type == MToast.LoadingType:
            # 如果类型为加载类型，则添加加载动画
            _icon_lay.addWidget(MLoading(size=dayu_theme.huge, color=dayu_theme.text_color_inverse))
        else:
            # 否则添加指定类型的图标
            _icon_label = MAvatar()
            _icon_label.set_dayu_size(dayu_theme.toast_icon_size)
            _icon_label.set_dayu_image(
                MPixmap(
                    "{}_line.svg".format(dayu_type or MToast.InfoType),
                    dayu_theme.text_color_inverse,
                )
            )
            _icon_lay.addWidget(_icon_label)
        _icon_lay.addStretch()

        _content_label = MLabel()
        _content_label.setText(text)
        _content_label.setAlignment(QtCore.Qt.AlignCenter)

        _main_lay = QtWidgets.QVBoxLayout()
        _main_lay.setContentsMargins(0, 0, 0, 0)
        _main_lay.addStretch()
        _main_lay.addLayout(_icon_lay)
        _main_lay.addSpacing(10)
        _main_lay.addWidget(_content_label)
        _main_lay.addStretch()
        self.setLayout(_main_lay)
        self.setFixedSize(QtCore.QSize(dayu_theme.toast_size, dayu_theme.toast_size))

        _close_timer = QtCore.QTimer(self)
        _close_timer.setSingleShot(True)
        _close_timer.timeout.connect(self.close)
        _close_timer.timeout.connect(self.sig_closed)
        _close_timer.setInterval((duration or self.default_config.get("duration")) * 1000)
        self.has_played = False

        if dayu_type != MToast.LoadingType:
            # 如果不是加载类型的提示，启动定时器关闭提示窗口
            _close_timer.start()

        self._opacity_ani = QtCore.QPropertyAnimation()
        self._opacity_ani.setTargetObject(self)
        self._opacity_ani.setDuration(300)
        self._opacity_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._opacity_ani.setPropertyName(b"windowOpacity")
        self._opacity_ani.setStartValue(0.0)
        self._opacity_ani.setEndValue(0.9)

        # 设置窗口位置居中，并启动淡入动画
        self._get_center_position(parent)
        self._fade_int()

    def closeEvent(self, event):
        if self.has_played:
            # 如果已经播放过动画，接受关闭事件
            event.accept()
        else:
            # 否则执行淡出动画，并忽略关闭事件
            self._fade_out()
            event.ignore()

    def _fade_out(self):
        # 标记已播放动画，设置动画方向为向后，并在动画结束后关闭窗口
        self.has_played = True
        self._opacity_ani.setDirection(QtCore.QAbstractAnimation.Backward)
        self._opacity_ani.finished.connect(self.close)
        self._opacity_ani.start()

    def _fade_int(self):
        # 启动淡入动画
        self._opacity_ani.start()
    def _get_center_position(self, parent):
        # 获取父窗口的几何信息
        parent_geo = parent.geometry()
        # 如果父窗口是顶级窗口（独立窗口）
        if parent.isWindowType():
            # 设置位置为父窗口的左上角
            pos = parent_geo.topLeft()
        # 如果父窗口是顶级窗口的子窗口
        elif parent in QtWidgets.QApplication.topLevelWidgets():
            # 设置位置为父窗口的左上角
            pos = parent_geo.topLeft()
        else:
            # 将父窗口左上角映射为全局坐标，并设置位置
            pos = parent.mapToGlobal(parent_geo.topLeft())
        
        # 初始化偏移量为0
        offset = 0
        # 遍历父窗口的子控件
        for child in parent.children():
            # 如果子控件是 MToast 类型且可见
            if isinstance(child, MToast) and child.isVisible():
                # 计算偏移量，取最大值
                offset = max(offset, child.y())
        
        # 计算目标 x 和 y 坐标，使当前控件位于父窗口中心
        target_x = pos.x() + parent_geo.width() / 2 - self.width() / 2
        target_y = pos.y() + parent_geo.height() / 2 - self.height() / 2
        
        # 设置当前控件的位置属性
        self.setProperty("pos", QtCore.QPoint(target_x, target_y))

    @classmethod
    def info(cls, text, parent, duration=None):
        """显示一个普通的提示消息"""
        # 创建一个 MToast 实例，显示信息类型为 InfoType
        inst = cls(text, duration=duration, dayu_type=MToast.InfoType, parent=parent)
        # 显示实例
        inst.show()
        # 返回实例对象
        return inst

    @classmethod
    def success(cls, text, parent, duration=None):
        """显示一个成功的提示消息"""
        # 创建一个 MToast 实例，显示信息类型为 SuccessType
        inst = cls(text, duration=duration, dayu_type=MToast.SuccessType, parent=parent)
        # 显示实例
        inst.show()
        # 返回实例对象
        return inst

    @classmethod
    def warning(cls, text, parent, duration=None):
        """显示一个警告的提示消息"""
        # 创建一个 MToast 实例，显示信息类型为 WarningType
        inst = cls(text, duration=duration, dayu_type=MToast.WarningType, parent=parent)
        # 显示实例
        inst.show()
        # 返回实例对象
        return inst

    @classmethod
    def error(cls, text, parent, duration=None):
        """显示一个错误的提示消息"""
        # 创建一个 MToast 实例，显示信息类型为 ErrorType
        inst = cls(text, duration=duration, dayu_type=MToast.ErrorType, parent=parent)
        # 显示实例
        inst.show()
        # 返回实例对象
        return inst

    @classmethod
    def loading(cls, text, parent):
        """显示一个带有加载动画的提示消息。
        需要手动关闭此窗口。"""
        # 创建一个 MToast 实例，显示信息类型为 LoadingType
        inst = cls(text, dayu_type=MToast.LoadingType, parent=parent)
        # 显示实例
        inst.show()
        # 返回实例对象
        return inst

    @classmethod
    def config(cls, duration):
        """
        配置全局 MToast 的默认显示时长。
        :param duration: int（单位为秒）
        :return: None
        """
        # 如果 duration 不为空
        if duration is not None:
            # 设置默认配置中的显示时长
            cls.default_config["duration"] = duration
```