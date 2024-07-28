# `.\comic-translate\app\ui\dayu_widgets\mixin.py`

```py
"""
mixin decorators to add Qt class feature.
"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets


def property_mixin(cls):
    """Mixin decorator to run function after dynamic property value changed"""

    def _new_event(self, event):
        # Check if the event type is DynamicPropertyChange
        if event.type() == QtCore.QEvent.DynamicPropertyChange:
            # Convert property name to string and decode it
            prp = event.propertyName().data().decode()
            # Check if there is a corresponding setter function
            if hasattr(self, "_set_{}".format(prp)):
                # Get the callback function and call it with the property value
                callback = getattr(self, "_set_{}".format(prp))
                callback(self.property(str(prp)))
        return super(cls, self).event(event)

    setattr(cls, "event", _new_event)
    return cls


def cursor_mixin(cls):
    """
    Cursor mixin decorator:
    - Changes the widget cursor to Qt.PointingHandCursor when mouse enters and widget is enabled.
    - Changes the widget cursor to Qt.ForbiddenCursor when widget is disabled and mouse enters.
    """

    # Store references to original event methods
    old_enter_event = cls.enterEvent
    old_leave_event = cls.leaveEvent
    old_hide_event = cls.hideEvent
    old_focus_out_event = cls.focusOutEvent

    def _revert_cursor(self):
        # Restore the cursor when mouse leaves
        if self.__dict__.get("__dayu_enter", False):
            while self.__dict__.get("__dayu_enter_count", 0) > 0:
                QtWidgets.QApplication.restoreOverrideCursor()
                self.__dict__.update({"__dayu_enter_count": self.__dict__.get("__dayu_enter_count", 0) - 1})
            self.__dict__.update({"__dayu_enter": False})

    def _new_enter_event(self, *args, **kwargs):
        # Set pointing hand cursor when mouse enters
        self.__dict__.update({"__dayu_enter": True})
        self.__dict__.update({"__dayu_enter_count": self.__dict__.get("__dayu_enter_count", 0) + 1})
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.PointingHandCursor if self.isEnabled() else QtCore.Qt.ForbiddenCursor
        )
        return old_enter_event(self, *args, **kwargs)

    def _new_leave_event(self, *args, **kwargs):
        # Revert cursor when mouse leaves
        _revert_cursor(self)
        return old_leave_event(self, *args, **kwargs)

    def _new_hide_event(self, *args, **kwargs):
        # Revert cursor when widget is hidden
        _revert_cursor(self)
        return old_hide_event(self, *args, **kwargs)

    def _new_focus_out_event(self, *args, **kwargs):
        # Revert cursor when focus is lost
        _revert_cursor(self)
        return old_focus_out_event(self, *args, **kwargs)

    # Replace original event methods with new ones
    cls.enterEvent = _new_enter_event
    cls.leaveEvent = _new_leave_event
    cls.hideEvent = _new_hide_event
    cls.focusOutEvent = _new_focus_out_event
    return cls


def focus_shadow_mixin(cls):
    """
    Focus shadow effect mixin decorator:
    - Adds shadow effect for decorated class when widget is focused.
    - Enables shadow effect when focus is gained.
    - Disables shadow effect when focus is lost.
    """
    # 备份原始的 focusInEvent 方法
    old_focus_in_event = cls.focusInEvent
    # 备份原始的 focusOutEvent 方法
    old_focus_out_event = cls.focusOutEvent

    # 定义新的 focusInEvent 方法
    def _new_focus_in_event(self, *args, **kwargs):
        # 调用原始的 focusInEvent 方法
        old_focus_in_event(self, *args, **kwargs)
        
        # 如果没有设置图形效果
        if not self.graphicsEffect():
            # 导入本地模块
            from . import dayu_theme

            # 创建阴影效果
            shadow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
            
            # 获取当前控件的 dayu_type 属性并获取对应颜色
            dayu_type = self.property("dayu_type")
            color = vars(dayu_theme).get("{}_color".format(dayu_type or "primary"))
            
            # 设置阴影效果的颜色、偏移和模糊半径
            shadow_effect.setColor(QtGui.QColor(color))
            shadow_effect.setOffset(0, 0)
            shadow_effect.setBlurRadius(5)
            shadow_effect.setEnabled(False)
            
            # 将阴影效果应用于当前控件
            self.setGraphicsEffect(shadow_effect)
        
        # 如果控件是启用状态，则启用图形效果
        if self.isEnabled():
            self.graphicsEffect().setEnabled(True)

    # 定义新的 focusOutEvent 方法
    def _new_focus_out_event(self, *args, **kwargs):
        # 调用原始的 focusOutEvent 方法
        old_focus_out_event(self, *args, **kwargs)
        
        # 如果设置了图形效果，则禁用它
        if self.graphicsEffect():
            self.graphicsEffect().setEnabled(False)

    # 将新定义的方法设置为类的新的 focusInEvent 方法
    setattr(cls, "focusInEvent", _new_focus_in_event)
    # 将新定义的方法设置为类的新的 focusOutEvent 方法
    setattr(cls, "focusOutEvent", _new_focus_out_event)
    # 返回更新后的类
    return cls
# 为类添加悬停阴影效果的装饰器
def hover_shadow_mixin(cls):
    # 保存原始的 enterEvent 和 leaveEvent 方法
    old_enter_event = cls.enterEvent
    old_leave_event = cls.leaveEvent

    # 新的 enterEvent 方法，加入悬停阴影效果逻辑
    def _new_enter_event(self, *args, **kwargs):
        # 调用原始的 enterEvent 方法
        old_enter_event(self, *args, **kwargs)
        # 如果当前没有设置图形效果
        if not self.graphicsEffect():
            # 导入本地模块
            from . import dayu_theme
            # 创建阴影效果对象
            shadow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
            # 获取并设置阴影颜色
            dayu_type = self.property("type")
            color = vars(dayu_theme).get("{}_color".format(dayu_type or "primary"))
            shadow_effect.setColor(QtGui.QColor(color))
            shadow_effect.setOffset(0, 0)
            shadow_effect.setBlurRadius(5)
            shadow_effect.setEnabled(False)
            # 将阴影效果应用到当前部件
            self.setGraphicsEffect(shadow_effect)
        # 如果部件启用，则启用图形效果
        if self.isEnabled():
            self.graphicsEffect().setEnabled(True)

    # 新的 leaveEvent 方法，禁用悬停阴影效果
    def _new_leave_event(self, *args, **kwargs):
        # 调用原始的 leaveEvent 方法
        old_leave_event(self, *args, **kwargs)
        # 如果存在图形效果，则禁用它
        if self.graphicsEffect():
            self.graphicsEffect().setEnabled(False)

    # 替换类的 enterEvent 和 leaveEvent 方法为新的实现
    setattr(cls, "enterEvent", _new_enter_event)
    setattr(cls, "leaveEvent", _new_leave_event)
    return cls


def _stackable(widget):
    """Used for stacked_animation_mixin to only add mixin for widget who can stacked."""
    # 使用 widget() 获取当前部件，使用 currentChanged 播放动画
    # 目前仅 QTabWidget 和 QStackedWidget 可以使用这个装饰器
    return issubclass(widget, QtWidgets.QWidget) and hasattr(widget, "widget") and hasattr(widget, "currentChanged")


# 用于堆叠部件的动画效果的装饰器
def stacked_animation_mixin(cls):
    # 如果部件不支持堆叠，直接返回原始部件类
    if not _stackable(cls):
        return cls
    # 保存原始的 __init__ 方法
    old_init = cls.__init__
    # 定义一个新的初始化方法，接受任意数量的位置参数和关键字参数
    def _new_init(self, *args, **kwargs):
        # 调用旧的初始化方法，传递所有的位置参数和关键字参数
        old_init(self, *args, **kwargs)
        # 初始化前一个索引为0，用于动画效果控制
        self._previous_index = 0
        
        # 创建一个用于显示动画的属性动画对象
        self._to_show_pos_ani = QtCore.QPropertyAnimation()
        self._to_show_pos_ani.setDuration(400)  # 设置动画持续时间为400毫秒
        self._to_show_pos_ani.setPropertyName(b"pos")  # 设置动画作用的属性为位置
        self._to_show_pos_ani.setEndValue(QtCore.QPoint(0, 0))  # 设置动画结束时的位置为(0, 0)
        self._to_show_pos_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)  # 设置动画的缓动曲线
        
        # 创建一个用于隐藏动画的属性动画对象，设置与上述类似
        self._to_hide_pos_ani = QtCore.QPropertyAnimation()
        self._to_hide_pos_ani.setDuration(400)
        self._to_hide_pos_ani.setPropertyName(b"pos")
        self._to_hide_pos_ani.setEndValue(QtCore.QPoint(0, 0))
        self._to_hide_pos_ani.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        
        # 创建一个透明度效果对象和对应的属性动画
        self._opacity_eff = QtWidgets.QGraphicsOpacityEffect()
        self._opacity_ani = QtCore.QPropertyAnimation()
        self._opacity_ani.setDuration(400)
        self._opacity_ani.setEasingCurve(QtCore.QEasingCurve.InCubic)
        self._opacity_ani.setPropertyName(b"opacity")
        self._opacity_ani.setStartValue(0.0)
        self._opacity_ani.setEndValue(1.0)
        self._opacity_ani.setTargetObject(self._opacity_eff)  # 设置动画作用对象为透明度效果对象
        self._opacity_ani.finished.connect(self._disable_opacity)  # 动画结束时连接禁用透明度方法
        self.currentChanged.connect(self._play_anim)  # 切换当前控件时连接播放动画方法

    # 播放动画的方法，接受一个索引作为参数
    def _play_anim(self, index):
        current_widget = self.widget(index)  # 获取当前索引对应的控件
        if self._previous_index < index:
            # 如果前一个索引小于当前索引，设置显示动画的起始位置和目标对象，并开始动画
            self._to_show_pos_ani.setStartValue(QtCore.QPoint(self.width(), 0))
            self._to_show_pos_ani.setTargetObject(current_widget)
            self._to_show_pos_ani.start()
        else:
            # 否则设置隐藏动画的起始位置和目标对象，并开始动画
            self._to_hide_pos_ani.setStartValue(QtCore.QPoint(-self.width(), 0))
            self._to_hide_pos_ani.setTargetObject(current_widget)
            self._to_hide_pos_ani.start()
        
        # 将当前控件设置为透明度效果的目标对象，并启用透明度效果
        current_widget.setGraphicsEffect(self._opacity_eff)
        current_widget.graphicsEffect().setEnabled(True)
        
        # 启动透明度动画
        self._opacity_ani.start()
        
        # 更新前一个索引为当前索引
        self._previous_index = index

    # 禁用透明度的方法
    def _disable_opacity(self):
        # 如果不关掉effect，会跟子控件的 effect 或 paintEvent 冲突引起 crash
        # 关闭当前控件的透明度效果
        self.currentWidget().graphicsEffect().setEnabled(False)

    # 将定义的方法绑定到类上的对应名称
    setattr(cls, "__init__", _new_init)
    setattr(cls, "_play_anim", _play_anim)
    setattr(cls, "_disable_opacity", _disable_opacity)
    return cls
```