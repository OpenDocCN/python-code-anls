# `.\comic-translate\app\ui\dayu_widgets\collapse.py`

```py
# 声明一个类 MCollapse，继承自 QtWidgets.QWidget
class MCollapse(QtWidgets.QWidget):
    # 定义信号 sig_context_menu，用于发射上下文菜单相关的信号
    sig_context_menu = QtCore.Signal(object)

    # 构造函数，初始化界面和数据
    def __init__(self, parent=None):
        # 调用父类 QtWidgets.QWidget 的构造函数
        super(MCollapse, self).__init__(parent)
        # 初始化内部变量 _central_widget 为 None
        self._central_widget = None

        # 创建布局管理器 self.content_layout，用于管理内容部件的布局
        self.content_layout = QtWidgets.QVBoxLayout(self)
        
        # 设置 self.content_layout 的间距为 0
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        # 创建内容部件 self.content_widget，初始为 None
        self.content_widget = None

        # 创建扩展图标部件 self.expand_icon，初始为 None
        self.expand_icon = None

        # 创建标题标签 self.title_label，初始为 None
        self.title_label = None

        # 创建描述标签 self.desc_label，初始为 None
        self.desc_label = None

        # 创建文本布局管理器 self.text_lay，用于管理描述标签和标题标签的布局
        self.text_lay = None

        # 创建选中值标签 self.selected_value_label，初始为 None
        self.selected_value_label = None

        # 创建头部部件 self.header_widget，初始为 None
        self.header_widget = None

        # 创建关闭按钮部件 self._close_button，初始为 None
        self._close_button = None

        # 初始化界面
        self._init_ui()

    # 初始化界面布局和信号连接
    def _init_ui(self):
        # 创建标题标签 self.title_label
        self.title_label = MLabel()

        # 创建描述标签 self.desc_label
        self.desc_label = MLabel()

        # 创建选中值标签 self.selected_value_label
        self.selected_value_label = MLabel()

        # 创建关闭按钮 self._close_button
        self._close_button = MToolButton()

        # 设置关闭按钮图标
        self._close_button.setIcon(MPixmap("delete_fill.svg"))

        # 设置关闭按钮大小
        self._close_button.setIconSize(QtCore.QSize(12, 12))

        # 为关闭按钮绑定点击事件处理函数 self._close_button_clicked
        self._close_button.clicked.connect(self._close_button_clicked)

        # 创建头部部件 self.header_widget
        self.header_widget = QtWidgets.QWidget()

        # 创建头部布局 self.header_lay
        self.header_lay = QtWidgets.QHBoxLayout(self.header_widget)

        # 将标题标签添加到头部布局中
        self.header_lay.addWidget(self.title_label)

        # 将关闭按钮添加到头部布局中
        self.header_lay.addWidget(self._close_button)

        # 设置头部部件的布局
        self.header_widget.setLayout(self.header_lay)

        # 创建文本布局管理器 self.text_lay
        self.text_lay = QtWidgets.QVBoxLayout()

        # 设置布局间距为 0
        self.text_lay.setContentsMargins(0, 0, 0, 0)

        # 将描述标签添加到文本布局管理器中
        self.text_lay.addWidget(self.desc_label)

        # 创建内容部件 self.content_widget
        self.content_widget = QtWidgets.QWidget()

        # 创建内容部件布局管理器 self.content_layout
        self.content_layout.addWidget(self.content_widget)

        # 将文本布局管理器和内容部件添加到内容布局管理器中
        self.content_layout.addLayout(self.text_lay)

        # 设置布局管理器
        self.setLayout(self.content_layout)

        # 安装事件过滤器，处理鼠标释放事件
        self.installEventFilter(self)

    # 处理关闭按钮点击事件
    def _close_button_clicked(self):
        # 发送关闭信号
        self.close()

    # 事件过滤器，处理鼠标事件，主要是处理标题和头部部件的点击事件
    def eventFilter(self, widget, event):
        # 判断事件触发的部件是否是头部部件或标题标签
        if widget in [self.header_widget, self.title_label]:
            # 如果是鼠标释放事件
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                # 切换展开状态
                self.set_expand(not self.property("expand"))
        # 调用父类的事件过滤器处理其他事件
        return super(QtWidgets.QWidget, self).eventFilter(widget, event)

    # 设置内容部件
    def set_content(self, widget):
        # 如果已经有中心部件
        if self._central_widget:
            # 从内容布局中移除中心部件
            self.content_layout.removeWidget(self._central_widget)
            # 关闭中心部件
            self._central_widget.close()
        # 将新的部件添加到内容布局中
        self.content_layout.addWidget(widget)
        # 更新中心部件为新添加的部件
        self._central_widget = widget

    # 获取当前的中心部件
    def get_content(self):
        return self._central_widget

    # 设置是否可关闭
    def set_closable(self, value):
        self.setProperty("closable", value)

    # 内部方法，设置是否可关闭
    def _set_closable(self, value):
        # 设置内容部件的可见性
        self.content_widget.setVisible(value)
        # 设置关闭按钮的可见性
        self._close_button.setVisible(value)

    # 设置是否可展开
    def set_expand(self, value):
        self.setProperty("expand", value)

    # 内部方法，设置是否可展开
    def _set_expand(self, value):
        # 设置内容部件的可见性
        self.content_widget.setVisible(value)
        # 设置展开图标的图片，根据展开状态选择不同的图标
        self.expand_icon.setPixmap(MPixmap("up_line.svg" if value else "down_line.svg").scaledToHeight(12))

    # 设置标题
    def set_title(self, value):
        self.setProperty("title", value)

    # 内部方法，设置标题
    def _set_title(self, value):
        # 设置标题标签的文本内容
        self.title_label.setText(value)

    # 设置描述
    def set_description(self, value):
        self.setProperty("description", value)

    # 内部方法，设置描述
    def _set_description(self, value):
        # 设置描述标签的文本内容
        self.desc_label.setText(value)
        # 如果有描述内容，则将描述标签添加到文本布局管理器中
        if value:
            self._add_description_to_layout()
        else:
            # 否则从文本布局管理器中移除描述标签
            self._remove_description_from_layout()

    # 将描述标签添加到文本布局管理器中
    def _add_description_to_layout(self):
        # 如果描述标签不在文本布局管理器的子部件中，则添加到文本布局管理器中
        if self.desc_label not in self.text_lay.children():
            self.text_lay.addWidget(self.desc_label)

    # 从文本布局管理器中移除描述标签
    def _remove_description_from_layout(self):
        # 如果描述标签在文本布局管理器的子部件中，则从文本布局管理器中移除，并将其父部件设置为 None
        if self.desc_label in self.text_lay.children():
            self.text_lay.removeWidget(self.desc_label)
            self.desc_label.setParent(None)

    # 设置选中的值
    def set_selected_value(self, value):
        # 设置选中值标签的文本内容
        self.selected_value_label.setText(value)
    # 初始化函数，设置父级窗口，初始化一个空的部分列表和垂直布局
    def __init__(self, parent=None):
        super(MCollapse, self).__init__(parent)
        self._section_list = []  # 初始化一个空列表，用于存储所有添加的部分
        self._main_layout = QtWidgets.QVBoxLayout()  # 创建一个垂直布局管理器
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)  # 设置部件的大小策略
        self._main_layout.setSpacing(1)  # 设置布局中部件之间的间距
        self._main_layout.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距
        self.setLayout(self._main_layout)  # 将布局应用到当前窗口中

    # 添加一个新的部分到折叠控件中
    def add_section(self, section_data):
        # 根据提供的部分数据创建部分小部件
        section_widget = MSectionItem(
            title=section_data.get("title"),
            expand=section_data.get("expand", False),
            widget=section_data.get("widget"),
            closable=section_data.get("closable", False),
            icon=section_data.get("icon"),
            description=section_data.get("description")
        )
        # 将部分小部件插入到主布局中
        self._main_layout.insertWidget(self._main_layout.count(), section_widget)

        # 如果部件是 MRadioButtonGroup 类型，连接其选中状态改变的信号到更新部分值的函数
        if isinstance(section_data.get("widget"), MRadioButtonGroup):
            radio_group = section_data["widget"]
            radio_group.sig_checked_changed.connect(
                lambda index, text: self.update_section_value(section_widget, text)
            )
            
            # 如果提供了默认选择项，则设置默认选择
            default_selection = section_data.get("default_selection")
            if default_selection is not None:
                id, string = default_selection
                radio_group.set_dayu_checked(id, string)
                # 更新部分小部件的值为默认选择项的文本
                default_button = radio_group.get_button_group().button(id)
                if default_button:
                    self.update_section_value(section_widget, default_button.text())

        return section_widget  # 返回新添加的部分小部件对象

    # 更新部分小部件的值
    def update_section_value(self, section_widget, text):
        section_widget.set_selected_value(text)

    # 批量添加部分到折叠控件中
    def add_section_list(self, section_list):
        for section_data in section_list:
            section_widget = self.add_section(section_data)
            # 连接部分小部件的关闭按钮点击信号到移除部分函数，并将部分小部件添加到部分列表中
            section_widget._close_button.clicked.connect(functools.partial(self.remove_section, section_widget))
            self._section_list.append(section_widget)

    # 从部分列表中移除指定的部分小部件
    def remove_section(self, widget):
        self._section_list.remove(widget)

    # 返回当前折叠控件中所有部分小部件的列表
    def sections(self):
        return self._section_list

    # 清空折叠控件中所有的部分小部件
    def clear(self):
        for widget in self._section_list:
            self._main_layout.removeWidget(widget)  # 从布局中移除部件
            del widget  # 删除部件对象
```