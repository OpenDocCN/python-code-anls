# `.\comic-translate\app\ui\dayu_widgets\combo_box.py`

```py
# Import future modules
from __future__ import absolute_import  # 导入绝对导入的未来版本支持
from __future__ import division         # 导入除法的未来版本支持
from __future__ import print_function  # 导入打印函数的未来版本支持

# Import third-party modules
from PySide6 import QtCore        # 导入 PySide6 的 QtCore 模块
from PySide6 import QtWidgets     # 导入 PySide6 的 QtWidgets 模块

# Import local modules
from . import dayu_theme         # 从当前包中导入 dayu_theme 模块
from .completer import MCompleter  # 从当前包中导入 MCompleter 类
from .mixin import cursor_mixin    # 从当前包中导入 cursor_mixin 模块
from .mixin import focus_shadow_mixin  # 从当前包中导入 focus_shadow_mixin 模块
from .mixin import property_mixin  # 从当前包中导入 property_mixin 模块
from . import utils as utils      # 从当前包中导入 utils 模块，并重命名为 utils

@property_mixin  # 使用 property_mixin 装饰器修饰 MComboBoxSearchMixin 类
class MComboBoxSearchMixin(object):
    def __init__(self, *args, **kwargs):
        super(MComboBoxSearchMixin, self).__init__(*args, **kwargs)
        # 创建 QSortFilterProxyModel 对象作为过滤模型
        self.filter_model = QtCore.QSortFilterProxyModel(self)
        # 设置过滤模型的大小写不敏感
        self.filter_model.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        # 将原始模型设置为过滤模型的源模型
        self.filter_model.setSourceModel(self.model())
        # 创建 MCompleter 对象作为自动完成器
        self.completer = MCompleter(self)
        # 设置自动完成模式为无过滤的弹出式完成
        self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
        # 将过滤模型设置为自动完成器的模型
        self.completer.setModel(self.filter_model)

    def search(self):
        # 设置焦点策略为强焦点
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        # 设置可编辑性为 True
        self.setEditable(True)

        # 设置自动完成器
        self.setCompleter(self.completer)

        # 获取编辑框对象
        edit = self.lineEdit()
        # 设置编辑框为非只读
        edit.setReadOnly(False)
        # 断开回车键按下的信号连接
        edit.returnPressed.disconnect()
        # 连接编辑文本修改信号到过滤模型的固定字符串过滤器设置方法
        edit.textEdited.connect(self.filter_model.setFilterFixedString)
        # 连接自动完成器的激活信号到 lambda 表达式，设置当前索引为匹配文本的索引
        self.completer.activated.connect(lambda t: t and self.setCurrentIndex(self.findText(t)))

    def _set_searchable(self, value):
        """search property to True then trigger search"""
        # 如果 value 为真，则触发搜索方法
        value and self.search()

    def setModel(self, model):
        # 调用父类的 setModel 方法设置模型
        super(MComboBoxSearchMixin, self).setModel(model)
        # 设置过滤模型的源模型为新设置的模型
        self.filter_model.setSourceModel(model)
        # 设置自动完成器的模型为过滤模型
        self.completer.setModel(self.filter_model)

    def setModelColumn(self, column):
        # 设置自动完成器的完成列为指定列
        self.completer.setCompletionColumn(column)
        # 设置过滤模型的过滤键列为指定列
        self.filter_model.setFilterKeyColumn(column)
        # 调用父类的 setModelColumn 方法设置模型列
        super(MComboBoxSearchMixin, self).setModelColumn(column)


@cursor_mixin  # 使用 cursor_mixin 装饰器修饰 MComboBox 类
@focus_shadow_mixin  # 使用 focus_shadow_mixin 装饰器修饰 MComboBox 类
class MComboBox(MComboBoxSearchMixin, QtWidgets.QComboBox):
    Separator = "/"  # 定义类属性 Separator 为斜杠字符 '/'
    sig_value_changed = QtCore.Signal(object)  # 定义信号 sig_value_changed 为 QtCore.Signal 类型的对象
    # 初始化函数，用于创建 MComboBox 对象
    def __init__(self, parent=None):
        # 调用父类的初始化函数
        super(MComboBox, self).__init__(parent)

        # 根菜单对象，默认为 None
        self._root_menu = None
        # 显示格式化器，使用 utils 模块中的 display_formatter 函数
        self._display_formatter = utils.display_formatter
        # 设置可编辑状态为 True
        self.setEditable(True)
        # 获取并设置 lineEdit 对象
        line_edit = self.lineEdit()
        # 设置 lineEdit 为只读状态
        line_edit.setReadOnly(True)
        # 设置文本边距
        line_edit.setTextMargins(4, 0, 4, 0)
        # 设置 lineEdit 的样式表，使背景透明
        line_edit.setStyleSheet("background-color:transparent")
        # 设置鼠标样式为手型
        line_edit.setCursor(QtCore.Qt.PointingHandCursor)
        # 安装事件过滤器，将事件过滤器设置为当前对象自身
        line_edit.installEventFilter(self)
        # 是否具有自定义视图，默认为 False
        self._has_custom_view = False
        # 设置初始值为空字符串
        self.set_value("")
        # 设置占位文本
        self.set_placeholder(self.tr("Please Select"))
        # 设置大小策略为在水平方向上扩展，在垂直方向上最小化
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # 设置 dayu_size 初始值为 dayu_theme 模块中的 default_size
        self._dayu_size = dayu_theme.default_size

    # 获取当前 dayu_size 属性值的方法
    def get_dayu_size(self):
        """
        Get the push button height
        :return: integer
        """
        return self._dayu_size

    # 设置 dayu_size 属性值的方法
    def set_dayu_size(self, value):
        """
        Set the avatar size.
        :param value: integer
        :return: None
        """
        # 设置 dayu_size 的值为传入的 value
        self._dayu_size = value
        # 设置 lineEdit 的属性 dayu_size 为当前 value 的值
        self.lineEdit().setProperty("dayu_size", value)
        # 更新当前对象的样式
        self.style().polish(self)

    # 定义一个属性 dayu_size，用于 dayu_size 的属性操作
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    # 设置显示格式化器的方法
    def set_formatter(self, func):
        self._display_formatter = func

    # 设置占位文本的方法
    def set_placeholder(self, text):
        """Display the text when no item selected."""
        # 设置 lineEdit 的占位文本为传入的 text
        self.lineEdit().setPlaceholderText(text)

    # 设置值属性的方法
    def set_value(self, value):
        # 设置属性 "value" 为传入的 value
        self.setProperty("value", value)

    # 私有方法，设置值属性的方法
    def _set_value(self, value):
        # 设置 lineEdit 的文本属性为 value 经过 _display_formatter 格式化后的值
        self.lineEdit().setProperty("text", self._display_formatter(value))
        # 如果 _root_menu 存在，则设置 _root_menu 的值为 value
        if self._root_menu:
            self._root_menu.set_value(value)

    # 设置菜单的方法
    def set_menu(self, menu):
        # 设置根菜单为传入的 menu 对象
        self._root_menu = menu
        # 连接菜单的值变化信号到当前对象的值变化信号
        self._root_menu.sig_value_changed.connect(self.sig_value_changed)
        # 连接菜单的值变化信号到设置值方法
        self._root_menu.sig_value_changed.connect(self.set_value)

    # 重写 setView 方法，标记为具有自定义视图
    def setView(self, *args, **kwargs):
        """Override setView to flag _has_custom_view variable."""
        self._has_custom_view = True
        super(MComboBox, self).setView(*args, **kwargs)

    # 重写 showPopup 方法，根据 _has_custom_view 和 _root_menu 显示自定义菜单或默认菜单
    def showPopup(self):
        """Override default showPopup. When set custom menu, show the menu instead."""
        if self._has_custom_view or self._root_menu is None:
            super(MComboBox, self).showPopup()
        else:
            super(MComboBox, self).hidePopup()
            # 将根菜单显示在当前组件的底部
            self._root_menu.popup(self.mapToGlobal(QtCore.QPoint(0, self.height())))

    # 事件过滤器，处理 lineEdit 的鼠标按下事件
    def eventFilter(self, widget, event):
        if widget is self.lineEdit() and widget.isReadOnly():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                # 当鼠标按下时显示弹出菜单
                self.showPopup()
        # 调用父类的事件过滤器处理事件
        return super(MComboBox, self).eventFilter(widget, event)

    # 设置 MComboBox 为巨大尺寸的方法
    def huge(self):
        """Set MComboBox to huge size"""
        # 设置 dayu_size 为 dayu_theme 模块中的 huge 值
        self.set_dayu_size(dayu_theme.huge)
        # 返回当前对象本身
        return self
    def large(self):
        """将 MComboBox 设置为大尺寸"""
        # 调用 set_dayu_size 方法，设置 MComboBox 的尺寸为大尺寸
        self.set_dayu_size(dayu_theme.large)
        # 返回当前对象的引用，以支持链式调用
        return self

    def medium(self):
        """将 MComboBox 设置为中等尺寸"""
        # 调用 set_dayu_size 方法，设置 MComboBox 的尺寸为中等尺寸
        self.set_dayu_size(dayu_theme.medium)
        # 返回当前对象的引用，以支持链式调用
        return self

    def small(self):
        """将 MComboBox 设置为小尺寸"""
        # 调用 set_dayu_size 方法，设置 MComboBox 的尺寸为小尺寸
        self.set_dayu_size(dayu_theme.small)
        # 返回当前对象的引用，以支持链式调用
        return self

    def tiny(self):
        """将 MComboBox 设置为微小尺寸"""
        # 调用 set_dayu_size 方法，设置 MComboBox 的尺寸为微小尺寸
        self.set_dayu_size(dayu_theme.tiny)
        # 返回当前对象的引用，以支持链式调用
        return self
```