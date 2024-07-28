# `.\comic-translate\app\ui\dayu_widgets\db_path_buttons.py`

```py
# 指定 Python 解释器的路径和字符编码
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# 作者: Mu yanru
# 日期: 2018.5
# 邮箱: muyanru345@163.com
###################################################################

# 导入未来版本兼容的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
from functools import partial  # 导入 functools 模块中的 partial 函数
from itertools import izip_longest  # 导入 itertools 模块中的 izip_longest 函数

# 导入第三方模块
from qt import *  # 导入 qt 模块中的所有内容（通常应避免使用通配符导入）
import utils  # 导入自定义的 utils 模块


def parse_db_orm(orm):
    orm_map = {"view": "items", "search": "items", "folder": "children"}
    return {
        "name": "ROOT" if hasattr(orm, "parent") and orm.parent is None else orm.name,
        "icon": utils.icon_formatter(orm),  # 使用 utils 模块中的 icon_formatter 函数处理 orm 对象
        "get_children": lambda x: [
            parse_db_orm(orm) for orm in getattr(x, orm_map.get(x.__tablename__, None)) if orm.active
        ],  # 返回该 ORM 对象的子对象列表，根据 orm_map 中的映射获取子对象
        "has_children": lambda x: hasattr(x, orm_map.get(x.__tablename__, None)),  # 判断该 ORM 对象是否有子对象
        "data": orm,  # 将 ORM 对象本身作为数据存储在字典中
    }


def parse_path(path):
    # 导入内置模块
    import os

    # 导入第三方模块
    from static import request_file  # 从 static 模块中导入 request_file 函数

    return {
        "name": os.path.basename(path) or path,  # 获取路径的基本名称作为显示名称，如果为空则使用路径本身
        "icon": utils.icon_formatter(request_file("icon-browser.png")),  # 使用 static 模块中的图标请求处理函数处理图标路径
        "get_children": lambda x: [
            parse_path(os.path.join(path, i)) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))
        ],  # 返回路径的子项列表，仅包括目录
        "has_children": lambda x: next(
            (True for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))),
            False,
        ),  # 判断路径是否有子项（目录）
        "data": path,  # 将路径本身作为数据存储在字典中
    }


class MBaseButton(QWidget):
    sig_name_button_clicked = Signal(int)  # 创建按钮名称点击信号，参数为整数
    sig_menu_action_clicked = Signal(int, dict)  # 创建菜单动作点击信号，参数为整数和字典

    def __init__(self, data_dict, parent=None):
        super(MBaseButton, self).__init__(parent)  # 调用父类的初始化方法
        self.data_dict = data_dict  # 保存传入的数据字典
        name_button = QToolButton(parent=self)  # 创建名称按钮对象，父对象为当前窗口
        name_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # 设置按钮样式为图标在文字旁边
        name_button.setIcon(data_dict.get("icon"))  # 设置按钮图标为数据字典中的图标
        name_button.clicked.connect(self.slot_button_clicked)  # 将按钮点击事件连接到槽函数 slot_button_clicked
        name_button.setText(data_dict.get("name"))  # 设置按钮显示的文本为数据字典中的名称

        self.menu_button = QToolButton(parent=self)  # 创建菜单按钮对象，父对象为当前窗口
        self.menu_button.setAutoRaise(False)  # 设置菜单按钮不自动升起
        self.menu_button.setArrowType(Qt.RightArrow)  # 设置菜单按钮箭头类型为向右箭头
        self.menu_button.setPopupMode(QToolButton.InstantPopup)  # 设置菜单按钮的弹出模式为即时弹出
        self.menu_button.setIconSize(QSize(10, 10))  # 设置菜单按钮图标大小为 10x10 像素
        self.menu_button.clicked.connect(self.slot_show_menu)  # 将按钮点击事件连接到槽函数 slot_show_menu
        self.menu_button.setVisible(data_dict.get("has_children")(data_dict.get("data")))  # 根据数据字典中是否有子项设置菜单按钮可见性
        main_lay = QHBoxLayout()  # 创建水平布局对象
        main_lay.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距
        main_lay.setSpacing(0)  # 设置布局中控件的间距为 0
        main_lay.addWidget(name_button)  # 将名称按钮添加到布局中
        main_lay.addWidget(self.menu_button)  # 将菜单按钮添加到布局中
        self.setLayout(main_lay)  # 将布局应用到当前窗口

    @Slot()
    def slot_button_clicked(self):
        self.sig_name_button_clicked.emit(self.data_dict.get("index"))  # 发送名称按钮点击信号，传递参数为数据字典中的索引值

    @Slot()
    def slot_show_menu(self):
        # 空的槽函数，用于处理菜单按钮点击事件
        pass
    # 当槽动作被点击时，发射信号 sig_menu_action_clicked，并传递当前按钮的索引和子对象作为参数
    def slot_action_clicked(self, sub_obj):
        self.sig_menu_action_clicked.emit(self.data_dict.get("index"), sub_obj)

    # 槽函数，用于显示菜单
    @Slot()
    def slot_show_menu(self):
        # 创建一个新的菜单对象
        menu = QMenu(self)
        # 获取子对象列表，调用 data_dict 中指定的函数来获取子对象列表
        data_list = self.data_dict.get("get_children")(self.data_dict.get("data"))
        # 遍历子对象列表
        for sub_obj in data_list:
            # 向菜单中添加动作，显示子对象的图标和名称
            action = menu.addAction(sub_obj.get("icon"), sub_obj.get("name"))
            # 连接动作的触发信号到 slot_action_clicked 槽函数，传递当前子对象作为参数
            action.triggered.connect(partial(self.slot_action_clicked, sub_obj))
        # 将菜单设置为按钮的下拉菜单
        self.menu_button.setMenu(menu)
        # 显示下拉菜单
        self.menu_button.showMenu()

    # 鼠标进入事件处理函数
    def enterEvent(self, *args, **kwargs):
        # 设置菜单按钮的箭头类型为向下箭头
        self.menu_button.setArrowType(Qt.DownArrow)
        # 调用父类的 enterEvent 处理函数
        return super(MBaseButton, self).enterEvent(*args, **kwargs)

    # 鼠标离开事件处理函数
    def leaveEvent(self, *args, **kwargs):
        # 设置菜单按钮的箭头类型为向右箭头
        self.menu_button.setArrowType(Qt.RightArrow)
        # 调用父类的 leaveEvent 处理函数
        return super(MBaseButton, self).leaveEvent(*args, **kwargs)
    # 定义一个名为 MDBPathButtons 的类，继承自 QFrame
class MDBPathButtons(QFrame):
    # 声明一个信号 sig_current_changed，用于通知当前选中的变化
    sig_current_changed = Signal()

    # 使用 dayu_css 装饰器初始化类
    @utils.dayu_css()
    def __init__(self, parent=None):
        # 调用父类 QFrame 的初始化方法
        super(MDBPathButtons, self).__init__(parent)
        # 初始化 parse_function 属性为 None，用于后续设置解析函数
        self.parse_function = None
        # 初始化 data_list 属性为空列表，用于存储数据项
        self.data_list = []

        # 创建水平布局对象 layout
        self.layout = QHBoxLayout()
        # 设置布局的边距为 0
        self.layout.setContentsMargins(0, 0, 0, 0)
        # 设置布局的间距为 0
        self.layout.setSpacing(0)

        # 创建主布局 main_lay，也是水平布局
        main_lay = QHBoxLayout()
        # 设置主布局的边距为 0
        main_lay.setContentsMargins(0, 0, 0, 0)
        # 将 self.layout 添加到 main_lay 中
        main_lay.addLayout(self.layout)
        # 添加一个可伸缩的空间到主布局中
        main_lay.addStretch()
        # 将主布局设置为当前窗口的布局
        self.setLayout(main_lay)

    # 设置解析函数的方法，接收一个函数作为参数
    def set_parse_function(self, func):
        self.parse_function = func

    # 设置数据的方法，接收一个对象 obj 作为参数
    def setup_data(self, obj):
        # 清除之前的下级按钮
        self.clear_downstream(0)
        # 如果传入的对象不为空
        if obj:
            # 使用解析函数处理传入的对象，然后添加到界面中
            self.add_level(self.parse_function(obj))
            # 发射当前选中变化的信号
            self.sig_current_changed.emit()

    # 添加一个层级的按钮方法，接收一个数据字典作为参数
    def add_level(self, data_dict):
        # 获取当前数据列表的长度，作为新数据项的索引
        index = len(self.data_list)
        # 更新数据字典，添加一个 "index" 键，值为当前索引值
        data_dict.update({"index": index})
        # 创建一个 MBaseButton 按钮，使用数据字典作为参数，设置父对象为当前窗口
        button = MBaseButton(data_dict, parent=self)
        # 连接按钮的 sig_name_button_clicked 信号到 slot_button_clicked 槽函数
        button.sig_name_button_clicked.connect(self.slot_button_clicked)
        # 连接按钮的 sig_menu_action_clicked 信号到 slot_menu_button_clicked 槽函数
        button.sig_menu_action_clicked.connect(self.slot_menu_button_clicked)
        # 将按钮添加到布局中
        self.layout.addWidget(button)
        # 更新数据字典，添加一个 "widget" 键，值为当前按钮对象
        data_dict.update({"widget": button})
        # 将数据字典添加到数据列表中
        self.data_list.append(data_dict)

    # 清除下级按钮的方法，接收一个索引值作为参数
    def clear_downstream(self, index):
        # 遍历数据列表中的每个数据字典及其索引
        for i, data_dict in enumerate(self.data_list):
            # 如果索引大于等于指定的索引值
            if i >= index:
                # 获取当前数据字典中的按钮对象
                button = data_dict.get("widget")
                # 从布局中移除按钮
                self.layout.removeWidget(button)
                # 将按钮设为不可见
                button.setVisible(False)
        # 更新数据列表，保留指定索引之前的数据字典
        self.data_list = self.data_list[:index]

    # 槽函数：显示菜单的方法，接收一个菜单按钮和数据字典作为参数
    @Slot(QToolButton, dict)
    def slot_show_menu(self, menu_button, data_dict):
        # 创建一个 QMenu 对象
        menu = QMenu(self)
        # 使用数据字典中的 "get_children" 方法获取子对象列表
        data_list = data_dict.get("get_children")(data_dict.get("data"))
        # 获取数据字典中的索引值
        index = data_dict.get("index")
        # 遍历子对象列表
        for sub_obj in data_list:
            # 创建一个动作，显示子对象的图标和名称
            action = menu.addAction(sub_obj.get("icon"), sub_obj.get("name"))
            # 连接动作的触发信号到 slot_menu_button_clicked 槽函数
            action.triggered.connect(partial(self.slot_menu_button_clicked, index, sub_obj))
        # 将菜单设置为菜单按钮的下拉菜单
        menu_button.setMenu(menu)
        # 显示菜单
        menu_button.showMenu()

    # 槽函数：按钮点击事件的处理方法，接收一个索引值作为参数
    @Slot(object)
    def slot_button_clicked(self, index):
        # 清除当前索引之后的下级按钮
        self.clear_downstream(index + 1)
        # 发射当前选中变化的信号
        self.sig_current_changed.emit()

    # 槽函数：菜单按钮点击事件的处理方法，接收一个索引值和数据字典作为参数
    @Slot(object)
    def slot_menu_button_clicked(self, index, data_dict):
        # 清除当前索引之后的下级按钮
        self.clear_downstream(index + 1)
        # 添加一个新层级的按钮，使用数据字典作为参数
        self.add_level(data_dict)
        # 发射当前选中变化的信号
        self.sig_current_changed.emit()

    # 槽函数：
    # 定义一个方法，用于处理对象列表的更新逻辑
    def slot_go_to(self, obj_list):
        # 使用 izip_longest 函数同时迭代传入的 obj_list 和当前对象的列表，获取索引和对应对象对
        for index, (his_obj, our_obj) in enumerate(izip_longest(obj_list, self.get_obj_list())):
            if his_obj is None:
                # 如果传来的 obj_list 最后一个是 None，则我方的 obj 多，直接清理掉多余的
                self.clear_downstream(index)
                return
            elif our_obj is None:
                # 我方的 obj 不够，则追加
                self.add_level(self.parse_function(his_obj))
            elif his_obj != our_obj:
                # 我方 跟 传来的 obj 不一样，清理掉后面的，并追加传来的 obj
                self.clear_downstream(index)
                self.add_level(self.parse_function(his_obj))
            else:
                # 我方和传来的 obj 完全一样，不做处理
                continue

    # 定义一个方法，返回当前对象的数据列表中的 "data" 字段组成的列表
    def get_obj_list(self):
        return [i.get("data") for i in self.data_list]
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行

    # 导入内置模块 sys
    import sys

    # 创建一个 Qt 应用程序实例
    app = QApplication(sys.argv)

    # 创建一个 MDBPathButtons 的实例
    test = MDBPathButtons()

    # 设置解析路径的函数为 parse_path
    test.set_parse_function(parse_path)

    # 设置初始数据路径为 "d:/"
    test.setup_data("d:/")

    # 显示界面
    test.show()

    # 运行 Qt 应用程序，直到程序结束
    sys.exit(app.exec_())
```