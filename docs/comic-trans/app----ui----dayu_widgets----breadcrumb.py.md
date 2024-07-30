# `.\comic-translate\app\ui\dayu_widgets\breadcrumb.py`

```py
# 导入未来模块，确保代码在 Python 2 和 Python 3 中兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 从 PySide6 中导入 QtWidgets 模块
from PySide6 import QtWidgets

# 从本地模块中导入 MLabel 和 MToolButton 类
from .label import MLabel
from .tool_button import MToolButton

# 定义 MBreadcrumb 类，继承自 QtWidgets.QWidget
class MBreadcrumb(QtWidgets.QWidget):
    """
    MBreadcrumb

    A breadcrumb displays the current location within a hierarchy.
    It allows going back to states higher up in the hierarchy.
    """

    # 构造函数，初始化面包屑导航条
    def __init__(self, separator="/", parent=None):
        # 调用父类 QtWidgets.QWidget 的构造函数
        super(MBreadcrumb, self).__init__(parent)
        
        # 设置分隔符属性
        self._separator = separator
        
        # 创建水平布局对象，并设置边距和间距
        self._main_layout = QtWidgets.QHBoxLayout()
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        # 在布局的末尾添加伸展因子，用于布局中的弹性空间
        self._main_layout.addStretch()
        
        # 将水平布局应用到当前的 QWidget
        self.setLayout(self._main_layout)
        
        # 创建按钮组对象
        self._button_group = QtWidgets.QButtonGroup()
        
        # 创建存储 MLabel 对象的列表
        self._label_list = []

    # 设置面包屑导航的整体项目列表
    def set_item_list(self, data_list):
        """Set the whole breadcrumb items. It will clear the old widgets."""
        # 清除旧的小部件
        for button in self._button_group.buttons():
            self._button_group.removeButton(button)
            self._main_layout.removeWidget(button)
            button.setVisible(False)
        
        for sep in self._label_list:
            self._main_layout.removeWidget(sep)
            sep.setVisible(False)
        
        # 遍历数据列表，添加每个数据项
        for index, data_dict in enumerate(data_list):
            self.add_item(data_dict, index)

    # 添加一个导航项
    def add_item(self, data_dict, index=None):
        """Add a item"""
        # 创建 MToolButton 对象
        button = MToolButton()
        
        # 设置按钮文本
        button.setText(data_dict.get("text"))
        
        # 如果有 SVG 图标，则设置按钮的 SVG 图标
        if data_dict.get("svg"):
            button.svg(data_dict.get("svg"))
        
        # 如果有工具提示，则设置按钮的工具提示
        if data_dict.get("tooltip"):
            button.setProperty("toolTip", data_dict.get("tooltip"))
        
        # 如果有点击事件，则连接按钮的点击信号到指定的槽函数
        if data_dict.get("clicked"):
            button.clicked.connect(data_dict.get("clicked"))
        
        # 根据条件设置按钮的显示方式：只有文本、文本旁边有图标或者仅有图标
        if data_dict.get("text"):
            if data_dict.get("svg") or data_dict.get("icon"):
                button.text_beside_icon()
            else:
                button.text_only()
        else:
            button.icon_only()

        # 如果按钮组中已有按钮，则在按钮前插入分隔符 MLabel 对象
        if self._button_group.buttons():
            separator = MLabel(self._separator).secondary()
            self._label_list.append(separator)
            self._main_layout.insertWidget(self._main_layout.count() - 1, separator)
        
        # 将按钮添加到布局中
        self._main_layout.insertWidget(self._main_layout.count() - 1, button)

        # 根据是否有索引，将按钮添加到按钮组中
        if index is None:
            self._button_group.addButton(button)
        else:
            self._button_group.addButton(button, index)
```