# `.\comic-translate\app\ui\dayu_widgets\popup.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: timmyliang
# Date  : 2021.12
# Email : 820472580@qq.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore     # 导入 PySide6 核心模块
from PySide6 import QtGui      # 导入 PySide6 图形模块
from PySide6 import QtWidgets  # 导入 PySide6 控件模块

# Import local modules
from .mixin import hover_shadow_mixin   # 导入本地 hover_shadow_mixin 混合类
from .mixin import property_mixin       # 导入本地 property_mixin 混合类


@hover_shadow_mixin  # 使用 hover_shadow_mixin 装饰器
@property_mixin          # 使用 property_mixin 装饰器
class MPopup(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super(MPopup, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Popup)  # 设置窗口标志为 Popup 类型，用于弹出窗口
        self.mouse_pos = None
        self.setProperty("movable", True)    # 设置属性 "movable" 为 True，可移动
        self.setProperty("animatable", True)  # 设置属性 "animatable" 为 True，可动画化
        QtCore.QTimer.singleShot(0, self.post_init)  # 在 0 秒后执行 post_init 方法

        self._opacity_anim = QtCore.QPropertyAnimation(self, b"windowOpacity")  # 创建窗口透明度动画对象
        self.setProperty("anim_opacity_duration", 300)  # 设置透明度动画持续时间为 300 毫秒
        self.setProperty("anim_opacity_curve", "OutCubic")  # 设置透明度动画缓动曲线为 OutCubic
        self.setProperty("anim_opacity_start", 0)  # 设置透明度动画起始值为 0
        self.setProperty("anim_opacity_end", 1)    # 设置透明度动画结束值为 1

        self._size_anim = QtCore.QPropertyAnimation(self, b"size")  # 创建窗口大小动画对象
        self.setProperty("anim_size_duration", 300)  # 设置大小动画持续时间为 300 毫秒
        self.setProperty("anim_size_curve", "OutCubic")  # 设置大小动画缓动曲线为 OutCubic
        self.setProperty("border_radius", 15)  # 设置窗口边界半径为 15 像素

    def post_init(self):
        start_size = self.property("anim_size_start")  # 获取动画大小的起始值
        size = self.sizeHint()  # 获取窗口的推荐大小
        start_size = start_size if start_size else QtCore.QSize(0, size.height())  # 如果起始大小不存在，则使用默认值
        end_size = self.property("anim_size_end")  # 获取动画大小的结束值
        end_size = end_size if end_size else size  # 如果结束大小不存在，则使用窗口的推荐大小
        self.setProperty("anim_size_start", start_size)  # 设置动画大小的起始值
        self.setProperty("anim_size_end", end_size)      # 设置动画大小的结束值

    def update_mask(self):
        rectPath = QtGui.QPainterPath()  # 创建绘制路径对象
        end_size = self.property("anim_size_end")  # 获取动画大小的结束值
        rect = QtCore.QRectF(0, 0, end_size.width(), end_size.height())  # 创建矩形区域对象
        radius = self.property("border_radius")  # 获取边界半径属性值
        rectPath.addRoundedRect(rect, radius, radius)  # 向路径对象添加圆角矩形
        self.setMask(QtGui.QRegion(rectPath.toFillPolygon().toPolygon()))  # 设置窗口的遮罩

    def _get_curve(self, value):
        curve = getattr(QtCore.QEasingCurve, value, None)  # 获取指定名称的缓动曲线对象
        if not curve:
            raise TypeError("Invalid QEasingCurve")  # 如果曲线对象不存在，抛出类型错误异常
        return curve

    def _set_border_radius(self, value):
        QtCore.QTimer.singleShot(0, self.update_mask)  # 在 0 秒后更新窗口遮罩

    def _set_anim_opacity_duration(self, value):
        self._opacity_anim.setDuration(value)  # 设置透明度动画的持续时间

    def _set_anim_opacity_curve(self, value):
        self._opacity_anim.setEasingCurve(self._get_curve(value))  # 设置透明度动画的缓动曲线

    def _set_anim_opacity_start(self, value):
        self._opacity_anim.setStartValue(value)  # 设置透明度动画的起始值

    def _set_anim_opacity_end(self, value):
        self._opacity_anim.setEndValue(value)  # 设置透明度动画的结束值
    # 设置动画的持续时间
    def _set_anim_size_duration(self, value):
        self._size_anim.setDuration(value)

    # 设置动画的缓动曲线
    def _set_anim_size_curve(self, value):
        self._size_anim.setEasingCurve(self._get_curve(value))

    # 设置动画的起始值
    def _set_anim_size_start(self, value):
        self._size_anim.setStartValue(value)

    # 设置动画的结束值，并在动画完成后调用 update_mask 方法更新掩码
    def _set_anim_size_end(self, value):
        self._size_anim.setEndValue(value)
        QtCore.QTimer.singleShot(0, self.update_mask)

    # 启动动画效果
    def start_anim(self):
        self._size_anim.start()
        self._opacity_anim.start()

    # 处理鼠标按下事件，记录鼠标位置
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pos = event.pos()
        return super(MPopup, self).mousePressEvent(event)

    # 处理鼠标释放事件，清空鼠标位置信息
    def mouseReleaseEvent(self, event):
        self.mouse_pos = None
        return super(MPopup, self).mouseReleaseEvent(event)

    # 处理鼠标移动事件，根据鼠标位置调整窗口位置（如果窗口可移动）
    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and self.mouse_pos and self.property("movable"):
            self.move(self.mapToGlobal(event.pos() - self.mouse_pos))
        return super(MPopup, self).mouseMoveEvent(event)

    # 显示弹出窗口，在需要时播放动画效果，并将窗口移动到鼠标当前位置
    def show(self):
        if self.property("animatable"):
            self.start_anim()
        self.move(QtGui.QCursor.pos())
        super(MPopup, self).show()
        # NOTES(timmyliang): 用于处理中文输入时的激活窗口操作
        self.activateWindow()
```