# `.\comic-translate\app\ui\dayu_widgets\button_group.py`

```py
# 指定 Python 解释器路径和文件编码
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# 作者: Mu yanru
# 日期: 2019.2
# 邮箱: muyanru345@163.com
###################################################################

# 导入 future 模块，确保代码在不同 Python 版本中兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 functools 模块，提供高阶函数和操作工具
import functools

# 导入 PySide6 中的核心、图形和窗口部件模块
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# 导入 six 模块，用于 Python 2 和 Python 3 兼容
import six

# 从本地模块中导入所需内容
from . import dayu_theme         # 导入 dayu_theme 模块
from .check_box import MCheckBox    # 导入 MCheckBox 类
from .menu import MMenu           # 导入 MMenu 类
from .push_button import MPushButton   # 导入 MPushButton 类
from .qt import get_scale_factor     # 导入 get_scale_factor 函数
from .radio_button import MRadioButton   # 导入 MRadioButton 类
from .tool_button import MToolButton    # 导入 MToolButton 类


class MButtonGroupBase(QtWidgets.QWidget):
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        # 初始化 MButtonGroupBase 类的实例
        super(MButtonGroupBase, self).__init__(parent=parent)

        # 创建主布局对象，水平或垂直布局取决于 orientation 参数
        self._main_layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.LeftToRight   # 如果 orientation 是水平方向，则从左到右布局
            if orientation == QtCore.Qt.Horizontal
            else QtWidgets.QBoxLayout.TopToBottom   # 如果 orientation 是垂直方向，则从上到下布局
        )

        # 设置主布局的边距
        self._main_layout.setContentsMargins(0, 0, 0, 0)

        # 将主布局设置为当前窗口部件的布局
        self.setLayout(self._main_layout)

        # 设置窗口部件的大小策略为最小化
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        # 创建按钮组对象
        self._button_group = QtWidgets.QButtonGroup()

        # 根据 orientation 参数设置布局方向的字符串表示
        self._orientation = "horizontal" if orientation == QtCore.Qt.Horizontal else "vertical"

    def set_spacing(self, value):
        # 设置主布局的间距
        self._main_layout.setSpacing(value)

    def get_button_group(self):
        # 返回按钮组对象
        return self._button_group

    def create_button(self, data_dict):
        # 抽象方法，子类需要实现，用于创建按钮
        raise NotImplementedError()
    # 添加按钮到界面上，根据传入的数据字典来设置按钮的属性和事件处理
    def add_button(self, data_dict, index=None):
        # 如果传入的数据字典是字符串类型，转换为包含"text"键的字典
        if isinstance(data_dict, six.string_types):
            data_dict = {"text": data_dict}
        # 如果传入的数据字典是QtGui.QIcon类型，转换为包含"icon"键的字典
        elif isinstance(data_dict, QtGui.QIcon):
            data_dict = {"icon": data_dict}
        
        # 根据数据字典创建按钮对象
        button = self.create_button(data_dict)
        
        # 设置按钮的"combine"属性为当前对象的方向属性
        button.setProperty("combine", self._orientation)
        
        # 根据数据字典中的各项设置按钮的属性
        if data_dict.get("text"):
            button.setProperty("text", data_dict.get("text"))
        if data_dict.get("icon"):
            button.setProperty("icon", data_dict.get("icon"))
        if data_dict.get("data"):
            button.setProperty("data", data_dict.get("data"))
        if data_dict.get("checked"):
            button.setProperty("checked", data_dict.get("checked"))
        if data_dict.get("shortcut"):
            button.setProperty("shortcut", data_dict.get("shortcut"))
        if data_dict.get("tooltip"):
            button.setProperty("toolTip", data_dict.get("tooltip"))
        if data_dict.get("checkable"):
            button.setProperty("checkable", data_dict.get("checkable"))
        if data_dict.get("enabled") is not None:
            button.setEnabled(data_dict.get("enabled"))
        if data_dict.get("clicked"):
            # 连接按钮的clicked信号到传入的clicked函数
            button.clicked.connect(data_dict.get("clicked"))
        if data_dict.get("toggled"):
            # 连接按钮的toggled信号到传入的toggled函数
            button.toggled.connect(data_dict.get("toggled"))
        
        # 将按钮添加到按钮组中，如果提供了index，则插入到指定位置
        if index is None:
            self._button_group.addButton(button)
        else:
            self._button_group.addButton(button, index)
        
        # 将按钮添加到主布局中
        self._main_layout.insertWidget(self._main_layout.count(), button)
        
        # 返回创建的按钮对象
        return button

    # 设置按钮列表，根据传入的按钮列表重新设置界面上的按钮
    def set_button_list(self, button_list):
        # 移除当前按钮组中的所有按钮，并从主布局中移除这些按钮
        for button in self._button_group.buttons():
            self._button_group.removeButton(button)
            self._main_layout.removeWidget(button)
            button.setVisible(False)
        
        # 根据传入的按钮列表重新创建和添加按钮
        for index, data_dict in enumerate(button_list):
            button = self.add_button(data_dict, index)
# 创建一个自定义的按钮组类 MPushButtonGroup，继承自 MButtonGroupBase
class MPushButtonGroup(MButtonGroupBase):
    # 初始化方法，设置按钮组的方向和父级对象
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        # 调用父类的初始化方法
        super(MPushButtonGroup, self).__init__(orientation=orientation, parent=parent)
        # 设置按钮之间的间距为 1 像素
        self.set_spacing(1)
        # 设置按钮的类型为主要类型
        self._dayu_type = MPushButton.PrimaryType
        # 设置按钮的大小为默认大小
        self._dayu_size = dayu_theme.default_size
        # 设置按钮组为非互斥模式，即允许同时选中多个按钮
        self._button_group.setExclusive(False)

    # 创建按钮的方法，根据传入的数据字典创建按钮并返回
    def create_button(self, data_dict):
        button = MPushButton()
        # 设置按钮的大小为传入数据字典中的 dayu_size 或者使用默认大小
        button.set_dayu_size(data_dict.get("dayu_size", self._dayu_size))
        # 设置按钮的类型为传入数据字典中的 dayu_type 或者使用默认类型
        button.set_dayu_type(data_dict.get("dayu_type", self._dayu_type))
        return button

    # 获取当前按钮组的按钮大小
    def get_dayu_size(self):
        return self._dayu_size

    # 获取当前按钮组的按钮类型
    def get_dayu_type(self):
        return self._dayu_type

    # 设置当前按钮组的按钮大小
    def set_dayu_size(self, value):
        self._dayu_size = value

    # 设置当前按钮组的按钮类型
    def set_dayu_type(self, value):
        self._dayu_type = value

    # 定义 Qt 属性 dayu_size，用于管理按钮组的按钮大小
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)
    # 定义 Qt 属性 dayu_type，用于管理按钮组的按钮类型
    dayu_type = QtCore.Property(str, get_dayu_type, set_dayu_type)



# 创建一个自定义的复选框组类 MCheckBoxGroup，继承自 MButtonGroupBase
class MCheckBoxGroup(MButtonGroupBase):
    # 定义信号，用于在复选框状态变化时发出
    sig_checked_changed = QtCore.Signal(list)

    # 初始化方法，设置复选框组的方向和父级对象
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        # 调用父类的初始化方法
        super(MCheckBoxGroup, self).__init__(orientation=orientation, parent=parent)
        # 获取当前系统的缩放因子
        scale_x, _ = get_scale_factor()
        # 设置复选框之间的间距为 15 像素乘以缩放因子
        self.set_spacing(15 * scale_x)
        # 设置复选框组为非互斥模式，即允许同时选中多个复选框
        self._button_group.setExclusive(False)

        # 设置自定义上下文菜单策略，允许自定义上下文菜单
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # 连接自定义上下文菜单请求信号到槽函数 _slot_context_menu
        self.customContextMenuRequested.connect(self._slot_context_menu)

        # 连接按钮组的按钮点击信号到槽函数 _slot_map_signal
        self._button_group.buttonClicked.connect(self._slot_map_signal)
        # 初始化已选中的复选框列表为空
        self._dayu_checked = []

    # 创建复选框的方法，根据传入的数据字典创建复选框并返回
    def create_button(self, data_dict):
        return MCheckBox()

    # 槽函数，处理上下文菜单请求信号，显示自定义上下文菜单并连接相关动作到对应的槽函数
    @QtCore.Slot(QtCore.QPoint)
    def _slot_context_menu(self, point):
        context_menu = MMenu(parent=self)
        action_select_all = context_menu.addAction("Select All")
        action_select_none = context_menu.addAction("Select None")
        action_select_invert = context_menu.addAction("Select Invert")
        # 连接动作的触发信号到槽函数 _slot_set_select，并传递相应的参数
        action_select_all.triggered.connect(functools.partial(self._slot_set_select, True))
        action_select_none.triggered.connect(functools.partial(self._slot_set_select, False))
        action_select_invert.triggered.connect(functools.partial(self._slot_set_select, None))
        # 在光标位置弹出上下文菜单
        context_menu.exec_(QtGui.QCursor.pos() + QtCore.QPoint(10, 10))

    # 槽函数，根据传入的状态设置所有复选框的选中状态，并触发状态映射信号
    @QtCore.Slot(bool)
    def _slot_set_select(self, state):
        for check_box in self._button_group.buttons():
            if state is None:
                old_state = check_box.isChecked()
                check_box.setChecked(not old_state)
            else:
                check_box.setChecked(state)
        self._slot_map_signal()

    # 槽函数，处理按钮点击信号，发射复选框状态变化信号，传递当前选中的复选框文本列表
    @QtCore.Slot(QtWidgets.QAbstractButton)
    def _slot_map_signal(self, button=None):
        self.sig_checked_changed.emit(
            [check_box.text() for check_box in self._button_group.buttons() if check_box.isChecked()]
        )
    # 设置“大于”选中的项目，当传入值不是列表时，转换为列表
    def set_dayu_checked(self, value):
        if not isinstance(value, list):
            value = [value]
        # 如果传入值等于当前选中的项目列表，则不进行更新
        if value == self.get_dayu_checked():
            return

        # 更新内部属性 _dayu_checked 为新的选中项目列表
        self._dayu_checked = value

        # 遍历按钮组中的每个复选框，根据新的选中项目列表更新复选框的状态
        for check_box in self._button_group.buttons():
            flag = QtCore.Qt.Checked if check_box.text() in value else QtCore.Qt.Unchecked
            # 如果复选框的状态与目标状态不一致，则更新复选框的状态
            if flag != check_box.checkState():
                check_box.setCheckState(flag)

        # 发送选中项目列表变更信号
        self.sig_checked_changed.emit(value)

    # 获取当前选中的“大于”项目列表
    def get_dayu_checked(self):
        return [check_box.text() for check_box in self._button_group.buttons() if check_box.isChecked()]

    # 定义名为 dayu_checked 的 Qt 属性，用于获取和设置选中的项目列表，并在变更时发出信号 sig_checked_changed
    dayu_checked = QtCore.Property("QVariantList", get_dayu_checked, set_dayu_checked, notify=sig_checked_changed)
class MRadioButtonGroup(MButtonGroupBase):
    """
    MRadioButtonGroup 类继承自 MButtonGroupBase 类，表示一个单选按钮组件。
    
    Property:
        dayu_checked
            用于设置和获取当前选中按钮的整数 ID。

    sig_checked_changed = QtCore.Signal(int, str)
        定义一个信号，当选中按钮改变时发射，参数为按钮的整数 ID 和按钮的文本。

    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super(MRadioButtonGroup, self).__init__(orientation=orientation, parent=parent)
        # 调用父类构造函数初始化，设置按钮组的方向和父组件。
        
        scale_x, _ = get_scale_factor()
        # 调用 get_scale_factor 函数获取缩放因子。
        
        self.set_spacing(15 * scale_x)
        # 设置按钮之间的间距，根据缩放因子进行调整。
        
        self._button_group.setExclusive(True)
        # 设置按钮组为互斥模式，只能有一个按钮被选中。
        
        self._button_group.buttonClicked.connect(self._on_button_clicked)
        # 连接按钮组的 buttonClicked 信号到 _on_button_clicked 槽函数。

    def _on_button_clicked(self, button):
        # 当按钮被点击时触发的槽函数，发射 sig_checked_changed 信号。
        
        self.sig_checked_changed.emit(self._button_group.id(button), button.text())
        # 发射信号，传递当前按钮的整数 ID 和文本内容。

    def create_button(self, data_dict):
        # 根据数据字典创建一个 MRadioButton 对象的方法，子类需重写。
        
        return MRadioButton()

    def set_dayu_checked(self, value, str):
        # 设置 dayu_checked 属性的 setter 方法，用于设置当前选中的按钮。
        
        if value == self.get_dayu_checked():
            return  # 如果传入的值与当前选中的按钮值相同，直接返回。
        
        button = self._button_group.button(value)
        # 获取指定 ID 的按钮。
        
        if button:
            button.setChecked(True)
            # 设置该按钮为选中状态。
            
            self.sig_checked_changed.emit(value, button.text())
            # 发射信号，传递选中按钮的 ID 和文本内容。
        else:
            print("error")
            # 如果找不到对应 ID 的按钮，打印错误信息。

    def get_dayu_checked(self):
        # 获取 dayu_checked 属性的 getter 方法，返回当前选中按钮的 ID。
        
        return self._button_group.checkedId()

    def get_current_value(self):
        # 获取当前选中按钮的文本内容。
        
        checked_button = self._button_group.checkedButton()
        # 获取当前选中的按钮对象。
        
        if checked_button:
            return checked_button.text()
            # 返回选中按钮的文本内容。
        
        return None
        # 如果没有选中按钮，返回 None。

    dayu_checked = QtCore.Property(int, get_dayu_checked, set_dayu_checked, notify=sig_checked_changed)
        # 定义一个 dayu_checked 的 Qt 属性，使用 get_dayu_checked 和 set_dayu_checked 方法作为其 getter 和 setter，
        # 并且当属性改变时发射 sig_checked_changed 信号。


```    
from PySide6 import QtCore, QtWidgets

class MToolButtonGroup(MButtonGroupBase):
    """
    MToolButtonGroup 类继承自 MButtonGroupBase 类，表示一个工具按钮组件。
    """

    sig_checked_changed = QtCore.Signal(int)
        # 定义一个信号，当工具按钮组中的按钮选中状态改变时发射，传递按钮的整数 ID。

    def __init__(
        self,
        size=None,
        type=None,
        exclusive=False,
        orientation=QtCore.Qt.Horizontal,
        parent=None,
    ):
        super(MToolButtonGroup, self).__init__(orientation=orientation, parent=parent)
        # 调用父类构造函数初始化，设置按钮组的方向和父组件。
        
        self.set_spacing(1)
        # 设置按钮之间的间距。
        
        self._button_group.setExclusive(exclusive)
        # 根据参数设置按钮组的互斥模式。
        
        self._size = size
        self._type = type
        # 设置按钮的尺寸和类型属性。
        
        self._button_group.buttonClicked.connect(self._on_button_clicked)
        # 连接按钮组的 buttonClicked 信号到 _on_button_clicked 槽函数。

    def _on_button_clicked(self, button):
        # 当按钮被点击时触发的槽函数，发射 sig_checked_changed 信号。
        
        button_id = self._button_group.id(button)
        # 获取被点击按钮的整数 ID。
        
        self.sig_checked_changed.emit(button_id)
        # 发射信号，传递按钮的 ID。

    def create_button(self, data_dict):
        # 根据数据字典创建一个 MToolButton 对象的方法，子类需重写。
        
        button = MToolButton()
        # 创建一个 MToolButton 对象。
        
        if data_dict.get("svg"):
            button.svg(data_dict.get("svg"))
            # 如果数据字典中包含 SVG 数据，设置按钮的 SVG 图标。
        
        if data_dict.get("text"):
            if data_dict.get("svg") or data_dict.get("icon"):
                button.text_beside_icon()
                # 如果数据字典中包含文本且同时包含 SVG 或图标数据，设置按钮文本和图标并排显示。
            else:
                button.text_only()
                # 否则，只显示按钮文本。
        else:
            button.icon_only()
            # 如果没有文本，只显示图标。

        return button
        # 返回创建的按钮对象。

    def set_dayu_checked(self, value):
        # 设置 dayu_checked 属性的 setter 方法，用于设置当前选中的按钮。
        
        if value == self.get_dayu_checked():
            return
            # 如果传入的值与当前选中的按钮值相同，直接返回。
        
        button = self._button_group.button(value)
        # 获取指定 ID 的按钮。
        
        if button:
            button.setChecked(True)
            # 设置该按钮为选中状态。
            
            self.sig_checked_changed.emit(value)
            # 发射信号，传递选中按钮的 ID。
        else:
            print(f"Error: No button found with id {value}")
            # 如果找不到对应 ID 的按钮，打印错误信息。

    def get_dayu_checked(self):
        # 获取 dayu_checked 属性的 getter 方法，返回当前选中按钮的 ID。
        
        return self._button_group.checkedId()
        # 返回当前选中按钮的 ID。
    # 创建一个属性 dayu_checked，类型为整数，使用 get_dayu_checked 方法获取属性值，
    # 使用 set_dayu_checked 方法设置属性值，当属性值变化时发出 sig_checked_changed 信号通知。
    dayu_checked = QtCore.Property(int, get_dayu_checked, set_dayu_checked, notify=sig_checked_changed)
```