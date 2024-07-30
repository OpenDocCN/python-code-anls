# `.\comic-translate\app\ui\dayu_widgets\item_view_set.py`

```py
# 定义一个 Python 脚本，使用 UTF-8 编码
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# 作者: Mu yanru
# 日期: 2018.5
# 邮箱: muyanru345@163.com
###################################################################

# 导入未来的模块特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from .item_model import MSortFilterModel
from .item_model import MTableModel
from .item_view import MBigView
from .item_view import MListView
from .item_view import MTableView
from .item_view import MTreeView
from .line_edit import MLineEdit
from .tool_button import MToolButton

# 定义一个名为 MItemViewSet 的类，继承自 QtWidgets.QWidget
class MItemViewSet(QtWidgets.QWidget):
    # 定义两个信号，分别表示双击和左击事件
    sig_double_clicked = QtCore.Signal(QtCore.QModelIndex)
    sig_left_clicked = QtCore.Signal(QtCore.QModelIndex)
    
    # 默认的 TableViewType 类型为 MTableView
    TableViewType = MTableView
    BigViewType = MBigView
    TreeViewType = MTreeView
    ListViewType = MListView

    # 初始化方法，接受一个可选的视图类型参数 view_type 和一个可选的父对象参数 parent
    def __init__(self, view_type=None, parent=None):
        # 调用父类的初始化方法
        super(MItemViewSet, self).__init__(parent)
        
        # 创建一个垂直布局管理器
        self.main_lay = QtWidgets.QVBoxLayout()
        self.main_lay.setSpacing(5)  # 设置布局内部控件的间距为 5
        self.main_lay.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距为 0
        
        # 创建排序过滤模型和源模型
        self.sort_filter_model = MSortFilterModel()
        self.source_model = MTableModel()
        self.sort_filter_model.setSourceModel(self.source_model)  # 将源模型设置为排序过滤模型的源模型
        
        # 根据传入的视图类型参数选择相应的视图类，默认为 MTableView
        view_class = view_type or MItemViewSet.TableViewType
        self.item_view = view_class()  # 创建视图对象
        self.item_view.doubleClicked.connect(self.sig_double_clicked)  # 连接双击信号
        self.item_view.pressed.connect(self.slot_left_clicked)  # 连接按下信号
        self.item_view.setModel(self.sort_filter_model)  # 将排序过滤模型设置为视图的模型
        
        # 创建搜索框和搜索按钮
        self._search_line_edit = MLineEdit().search().small()
        self._search_attr_button = MToolButton().icon_only().svg("down_fill.svg").small()
        self._search_line_edit.set_prefix_widget(self._search_attr_button)  # 设置搜索框的前缀部件为搜索按钮
        self._search_line_edit.textChanged.connect(self.sort_filter_model.set_search_pattern)  # 连接搜索框文本改变信号
        self._search_line_edit.setVisible(False)  # 设置搜索框不可见
        self._search_lay = QtWidgets.QHBoxLayout()  # 创建水平布局管理器
        self._search_lay.setContentsMargins(0, 0, 0, 0)  # 设置水平布局的边距为 0
        self._search_lay.addStretch()  # 添加一个可伸缩空间
        self._search_lay.addWidget(self._search_line_edit)  # 将搜索框添加到水平布局中

        # 将搜索框布局和视图布局添加到主布局中
        self.main_lay.addLayout(self._search_lay)
        self.main_lay.addWidget(self.item_view)
        self.setLayout(self.main_lay)  # 设置主布局为当前部件的布局

    # 响应左击事件的槽函数，接受一个 QModelIndex 参数 start_index
    @QtCore.Slot(QtCore.QModelIndex)
    def slot_left_clicked(self, start_index):
        button = QtWidgets.QApplication.mouseButtons()  # 获取当前鼠标按钮状态
        if button == QtCore.Qt.LeftButton:  # 如果是左键按下
            real_index = self.sort_filter_model.mapToSource(start_index)  # 将视图索引映射到源模型索引
            self.sig_left_clicked.emit(real_index)  # 发射左击信号，并传递真实的源模型索引
    # 设置表头列表到源模型和排序过滤模型
    self.source_model.set_header_list(header_list)
    self.sort_filter_model.set_header_list(header_list)
    # 将排序过滤模型设置为源模型的代理模型
    self.sort_filter_model.setSourceModel(self.source_model)
    # 清空源模型中的数据
    self.source_model.clear()
    # 在项目视图中设置表头列表
    self.item_view.set_header_list(header_list)

@QtCore.Slot()
def setup_data(self, data_list):
    # 清空源模型中的数据
    self.source_model.clear()
    # 如果数据列表非空，则设置数据列表到源模型
    if data_list:
        self.source_model.set_data_list(data_list)

def get_data(self):
    # 返回源模型中的数据列表
    return self.source_model.get_data_list()

def searchable(self):
    """启用搜索行编辑器并使其可见。"""
    # 设置搜索行编辑器可见
    self._search_line_edit.setVisible(True)
    # 返回当前对象的引用
    return self

def insert_widget(self, widget):
    """允许将额外的小部件插入到搜索布局中。"""
    # 在搜索布局的位置 0 处插入给定的小部件
    self._search_lay.insertWidget(0, widget)
```