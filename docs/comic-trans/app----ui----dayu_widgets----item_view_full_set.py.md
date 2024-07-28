# `.\comic-translate\app\ui\dayu_widgets\item_view_full_set.py`

```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    ###################################################################
    # Author: Mu yanru
    # Date  : 2018.5
    # Email : muyanru345@163.com
    ###################################################################

    # Import future modules
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    # Import third-party modules
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    # Import local modules
    from .button_group import MToolButtonGroup
    from .item_model import MSortFilterModel
    from .item_model import MTableModel
    from .item_view import MBigView
    from .item_view import MTableView
    from .line_edit import MLineEdit
    from .page import MPage
    from .tool_button import MToolButton


    class MItemViewFullSet(QtWidgets.QWidget):
        sig_double_clicked = QtCore.Signal(QtCore.QModelIndex)
        sig_left_clicked = QtCore.Signal(QtCore.QModelIndex)
        sig_current_changed = QtCore.Signal(QtCore.QModelIndex, QtCore.QModelIndex)
        sig_current_row_changed = QtCore.Signal(QtCore.QModelIndex, QtCore.QModelIndex)
        sig_current_column_changed = QtCore.Signal(QtCore.QModelIndex, QtCore.QModelIndex)
        sig_selection_changed = QtCore.Signal(QtCore.QItemSelection, QtCore.QItemSelection)
        sig_context_menu = QtCore.Signal(object)

        # 启用上下文菜单功能
        def enable_context_menu(self):
            for index in range(self.stack_widget.count()):
                view = self.stack_widget.widget(index)
                view.enable_context_menu(True)
                view.sig_context_menu.connect(self.sig_context_menu)

        # 设置无数据时显示的文本
        def set_no_data_text(self, text):
            for index in range(self.stack_widget.count()):
                view = self.stack_widget.widget(index)
                view.set_no_data_text(text)

        # 设置选择模式
        def set_selection_mode(self, mode):
            for index in range(self.stack_widget.count()):
                view = self.stack_widget.widget(index)
                view.setSelectionMode(mode)

        # 控制工具栏的可见性
        def tool_bar_visible(self, flag):
            self.tool_bar.setVisible(flag)

        # 处理左键点击事件的槽函数
        @QtCore.Slot(QtCore.QModelIndex)
        def slot_left_clicked(self, start_index):
            button = QtWidgets.QApplication.mouseButtons()
            if button == QtCore.Qt.LeftButton:
                real_index = self.sort_filter_model.mapToSource(start_index)
                self.sig_left_clicked.emit(real_index)

        # 设置表头的列表
        def set_header_list(self, header_list):
            # 设置源模型的表头列表
            self.source_model.set_header_list(header_list)
            # 设置排序过滤模型的表头列表
            self.sort_filter_model.set_header_list(header_list)
            # 将源模型设置为排序过滤模型的源模型
            self.sort_filter_model.setSourceModel(self.source_model)
            # 清空源模型的数据
            self.source_model.clear()
            # 遍历视图堆栈中的每个视图，设置它们的表头列表
            for index in range(self.stack_widget.count()):
                view = self.stack_widget.widget(index)
                view.set_header_list(header_list)

        # 在工具栏末尾添加小部件
        def tool_bar_append_widget(self, widget):
            self.top_lay.addWidget(widget)

        # 在工具栏开头插入小部件
        def tool_bar_insert_widget(self, widget):
            self.top_lay.insertWidget(0, widget)

        @QtCore.Slot()
    # 设置数据，清空当前的数据模型
    def setup_data(self, data_list):
        self.source_model.clear()
        # 如果传入了数据列表，则设置数据到源模型中
        if data_list:
            self.source_model.set_data_list(data_list)
        # 设置记录总数为数据列表的长度
        self.set_record_count(len(data_list))

    # 用于接收总记录数并传递给页面设置对象
    @QtCore.Slot(int)
    def set_record_count(self, total):
        self.page_set.set_total(total)

    # 获取数据模型中的数据列表
    def get_data(self):
        return self.source_model.get_data_list()

    # 启用搜索功能，使搜索输入框可见，并返回当前对象
    def searchable(self):
        """Enable search line edit visible."""
        self.search_line_edit.setVisible(True)
        return self
```