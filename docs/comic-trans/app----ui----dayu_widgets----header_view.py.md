# `.\comic-translate\app\ui\dayu_widgets\header_view.py`

```py
    @QtCore.Slot(QtCore.QPoint)
    # 定义一个槽函数，用于响应自定义上下文菜单的请求
    def _slot_context_menu(self, pos):
        # 获取上下文菜单的全局位置
        global_pos = self.mapToGlobal(pos)
        # 创建一个右键菜单对象
        menu = MMenu(self)
        # 在指定位置显示右键菜单
        menu.exec_(global_pos)
    # 定义一个槽函数，用于处理右键菜单的显示和操作
    def _slot_context_menu(self, point):
        # 创建一个右键菜单对象
        context_menu = MMenu(parent=self)
        # 获取点击位置的逻辑列索引
        logical_column = self.logicalIndexAt(point)
        # 获取真实的数据模型对象
        model = utils.real_model(self.model())
        
        # 如果逻辑列索引有效，并且该列是可勾选的
        if logical_column >= 0 and model.header_list[logical_column].get("checkable", False):
            # 添加“全选”动作到右键菜单，并连接到对应的槽函数
            action_select_all = context_menu.addAction(self.tr("Select All"))
            action_select_all.triggered.connect(
                functools.partial(self._slot_set_select, logical_column, QtCore.Qt.Checked)
            )
            # 添加“全不选”动作到右键菜单，并连接到对应的槽函数
            action_select_none = context_menu.addAction(self.tr("Select None"))
            action_select_none.triggered.connect(
                functools.partial(self._slot_set_select, logical_column, QtCore.Qt.Unchecked)
            )
            # 添加“反选”动作到右键菜单，并连接到对应的槽函数
            action_select_invert = context_menu.addAction(self.tr("Select Invert"))
            action_select_invert.triggered.connect(
                functools.partial(self._slot_set_select, logical_column, None)
            )
            # 添加分隔线到右键菜单
            context_menu.addSeparator()

        # 添加“适应大小”动作到右键菜单，并连接到对应的槽函数
        fit_action = context_menu.addAction(self.tr("Fit Size"))
        fit_action.triggered.connect(functools.partial(self._slot_set_resize_mode, True))
        # 添加分隔线到右键菜单
        context_menu.addSeparator()

        # 遍历所有列，并添加每列的显示名称到右键菜单，设置可勾选状态，并连接到对应的槽函数
        for column in range(self.count()):
            action = context_menu.addAction(model.headerData(column, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole))
            action.setCheckable(True)
            action.setChecked(not self.isSectionHidden(column))
            action.toggled.connect(functools.partial(self._slot_set_section_visible, column))
        
        # 在鼠标右键点击位置弹出右键菜单
        context_menu.exec_(QtGui.QCursor.pos() + QtCore.QPoint(10, 10))

    # 槽函数：设置指定列的选择状态
    @QtCore.Slot(int, int)
    def _slot_set_select(self, column, state):
        # 获取当前的数据模型
        current_model = self.model()
        # 获取真实的数据模型对象
        source_model = utils.real_model(current_model)
        # 开始重置数据模型
        source_model.beginResetModel()
        # 构建属性名，根据列的关键信息
        attr = "{}_checked".format(source_model.header_list[column].get("key"))
        # 遍历每一行数据
        for row in range(current_model.rowCount()):
            # 获取真实索引
            real_index = utils.real_index(current_model.index(row, column))
            # 获取数据对象
            data_obj = real_index.internalPointer()
            # 如果状态为 None，则切换选中状态；否则直接设置为指定状态
            if state is None:
                old_state = utils.get_obj_value(data_obj, attr)
                utils.set_obj_value(
                    data_obj,
                    attr,
                    QtCore.Qt.Unchecked if old_state == QtCore.Qt.Checked else QtCore.Qt.Checked,
                )
            else:
                utils.set_obj_value(data_obj, attr, state)
        
        # 结束数据模型重置
        source_model.endResetModel()
        # 发送数据变更信号
        source_model.dataChanged.emit(None, None)

    # 槽函数：设置指定索引列的可见性
    @QtCore.Slot(QtCore.QModelIndex, int)
    def _slot_set_section_visible(self, index, flag):
        # 设置对应索引列的可见性
        self.setSectionHidden(index, not flag)

    # 槽函数：设置列的大小调整模式
    @QtCore.Slot(bool)
    def _slot_set_resize_mode(self, flag):
        # 如果 flag 为 True，则设置为自适应内容大小；否则设置为交互式调整大小
        if flag:
            self.resizeSections(QtWidgets.QHeaderView.ResizeToContents)
        else:
            self.resizeSections(QtWidgets.QHeaderView.Interactive)
    # 设置表头视图的可点击状态，如果支持设置具体节的可点击性，则调用该方法
    def setClickable(self, flag):
        try:
            # 尝试调用 QtWidgets.QHeaderView.setSectionsClickable 方法设置可点击性
            QtWidgets.QHeaderView.setSectionsClickable(self, flag)
        except AttributeError:
            # 如果方法不存在，则调用 QtWidgets.QHeaderView.setClickable 方法
            QtWidgets.QHeaderView.setClickable(self, flag)

    # 设置表头视图的可移动状态，如果支持设置具体节的可移动性，则调用该方法
    def setMovable(self, flag):
        try:
            # 尝试调用 QtWidgets.QHeaderView.setSectionsMovable 方法设置可移动性
            QtWidgets.QHeaderView.setSectionsMovable(self, flag)
        except AttributeError:
            # 如果方法不存在，则调用 QtWidgets.QHeaderView.setMovable 方法
            QtWidgets.QHeaderView.setMovable(self, flag)

    # 设置表头视图特定索引处的调整大小模式，如果支持设置具体节的调整大小模式，则调用该方法
    def resizeMode(self, index):
        try:
            # 尝试调用 QtWidgets.QHeaderView.sectionResizeMode 方法设置调整大小模式
            QtWidgets.QHeaderView.sectionResizeMode(self, index)
        except AttributeError:
            # 如果方法不存在，则调用 QtWidgets.QHeaderView.resizeMode 方法
            QtWidgets.QHeaderView.resizeMode(self, index)

    # 设置表头视图的整体调整大小模式
    def setResizeMode(self, mode):
        try:
            # 尝试调用 QtWidgets.QHeaderView.setResizeMode 方法设置整体调整大小模式
            QtWidgets.QHeaderView.setResizeMode(self, mode)
        except AttributeError:
            # 如果方法不存在，则调用 QtWidgets.QHeaderView.setSectionResizeMode 方法
            QtWidgets.QHeaderView.setSectionResizeMode(self, mode)
```