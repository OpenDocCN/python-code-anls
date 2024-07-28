# `.\comic-translate\app\ui\dayu_widgets\item_model.py`

```py
# 指定脚本使用 Python 解释器，并设置字符编码为 UTF-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# Author: Mu yanru
# Date  : 2018.5
# Email : muyanru345@163.com
###################################################################

# 导入未来版本的模块特性，确保代码在 Python 2/3 兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PySide6 的核心和图形界面模块
from PySide6 import QtCore
from PySide6 import QtGui
import six

# 导入本地模块
from .utils import apply_formatter
from .utils import display_formatter
from .utils import font_formatter
from .utils import get_obj_value
from .utils import icon_formatter
from .utils import set_obj_value

# 角色配置映射表，将 Qt 角色映射到对应的配置和格式化函数
SETTING_MAP = {
    QtCore.Qt.BackgroundRole: {"config": "bg_color", "formatter": QtGui.QColor},
    QtCore.Qt.DisplayRole: {"config": "display", "formatter": display_formatter},
    QtCore.Qt.EditRole: {"config": "edit", "formatter": None},
    QtCore.Qt.TextAlignmentRole: {
        "config": "alignment",
        "formatter": {
            "right": QtCore.Qt.AlignRight,
            "left": QtCore.Qt.AlignLeft,
            "center": QtCore.Qt.AlignCenter,
        },
    },
    QtCore.Qt.ForegroundRole: {"config": "color", "formatter": QtGui.QColor},
    QtCore.Qt.FontRole: {"config": "font", "formatter": font_formatter},
    QtCore.Qt.DecorationRole: {"config": "icon", "formatter": icon_formatter},
    QtCore.Qt.ToolTipRole: {"config": "tooltip", "formatter": display_formatter},
    QtCore.Qt.InitialSortOrderRole: {
        "config": "order",
        "formatter": {
            "asc": QtCore.Qt.AscendingOrder,
            "des": QtCore.Qt.DescendingOrder,
        },
    },
    QtCore.Qt.SizeHintRole: {
        "config": "size",
        "formatter": lambda args: QtCore.QSize(*args),
    },
    QtCore.Qt.UserRole: {"config": "data"},  # 用于存储任意数据
}


class MTableModel(QtCore.QAbstractItemModel):
    def __init__(self, parent=None):
        super(MTableModel, self).__init__(parent)
        self.origin_count = 0  # 初始数据条目数
        self.root_item = {"name": "root", "children": []}  # 根节点数据结构
        self.data_generator = None  # 数据生成器对象
        self.header_list = []  # 表头列表
        self.timer = QtCore.QTimer(self)  # 创建定时器对象
        self.timer.timeout.connect(self.fetchMore)  # 绑定定时器超时事件到 fetchMore 方法

    def set_header_list(self, header_list):
        self.header_list = header_list  # 设置表头列表

    def set_data_list(self, data_list):
        if hasattr(data_list, "next"):  # 检查是否具有 next 属性，判断是否为数据生成器
            self.beginResetModel()  # 开始重置模型
            self.root_item["children"] = []  # 清空根节点子项数据
            self.endResetModel()  # 结束重置模型
            self.data_generator = data_list  # 设置数据生成器对象
            self.origin_count = 0  # 重置数据条目计数器
            self.timer.start()  # 启动定时器
        else:
            self.beginResetModel()  # 开始重置模型
            self.root_item["children"] = data_list if data_list is not None else []  # 设置根节点的子项数据
            self.endResetModel()  # 结束重置模型
            self.data_generator = None  # 清空数据生成器

    def clear(self):
        self.beginResetModel()  # 开始重置模型
        self.root_item["children"] = []  # 清空根节点的子项数据
        self.endResetModel()  # 结束重置模型
    # 返回根项目的子项目列表作为数据列表
    def get_data_list(self):
        return self.root_item["children"]

    # 将给定的数据字典添加到根项目的子项目列表末尾，并调用 fetchMore() 方法
    def append(self, data_dict):
        self.root_item["children"].append(data_dict)
        self.fetchMore()

    # 从根项目的子项目列表中移除指定的数据字典
    # 获取要移除数据字典在列表中的索引，并发出 beginRemoveRows 和 endRemoveRows 信号
    def remove(self, data_dict):
        row = self.root_item["children"].index(data_dict)
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self.root_item["children"].remove(data_dict)
        self.endRemoveRows()

    # 返回特定索引处项目的标志，包括是否可编辑、可选择、可拖拽等
    def flags(self, index):
        result = QtCore.QAbstractItemModel.flags(self, index)
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        # 根据头部列表的配置确定项目的各种标志
        if self.header_list[index.column()].get("checkable", False):
            result |= QtCore.Qt.ItemIsUserCheckable
        if self.header_list[index.column()].get("selectable", False):
            result |= QtCore.Qt.ItemIsEditable
        if self.header_list[index.column()].get("editable", False):
            result |= QtCore.Qt.ItemIsEditable
        if self.header_list[index.column()].get("draggable", False):
            result |= QtCore.Qt.ItemIsDragEnabled
        if self.header_list[index.column()].get("droppable", False):
            result |= QtCore.Qt.ItemIsDropEnabled
        return QtCore.Qt.ItemFlags(result)

    # 返回指定部分的头数据，若为垂直方向则调用父类方法，否则返回头部列表中对应部分的标签
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Vertical:
            return super(MTableModel, self).headerData(section, orientation, role)
        if not self.header_list or section >= len(self.header_list):
            return None
        if role == QtCore.Qt.DisplayRole:
            return self.header_list[section]["label"]
        return None

    # 根据行和列索引返回相应的 QModelIndex
    # 如果有父索引且有效，则使用其内部指针；否则使用根项目
    # 获取子项目列表并返回对应行和列的 QModelIndex
    def index(self, row, column, parent_index=None):
        if parent_index and parent_index.isValid():
            parent_item = parent_index.internalPointer()
        else:
            parent_item = self.root_item

        children_list = get_obj_value(parent_item, "children")
        if children_list and len(children_list) > row:
            child_item = children_list[row]
            if child_item:
                set_obj_value(child_item, "_parent", parent_item)
                return self.createIndex(row, column, child_item)
        return QtCore.QModelIndex()

    # 返回给定索引的父索引
    # 若索引无效，返回无效的 QModelIndex
    # 获取索引内部指针对应的子项目，再获取其父项目，依次获取祖父项目
    # 最后返回父项目在父项目列表中的索引
    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()

        child_item = index.internalPointer()
        parent_item = get_obj_value(child_item, "_parent")

        if parent_item is None:
            return QtCore.QModelIndex()

        grand_item = get_obj_value(parent_item, "_parent")
        if grand_item is None:
            return QtCore.QModelIndex()
        parent_list = get_obj_value(grand_item, "children")
        return self.createIndex(parent_list.index(parent_item), 0, parent_item)
    # 返回指定父索引的行数
    def rowCount(self, parent_index=None):
        # 如果有有效的父索引，则获取其内部指针对应的对象作为父项
        if parent_index and parent_index.isValid():
            parent_item = parent_index.internalPointer()
        else:
            # 否则将根项目作为父项
            parent_item = self.root_item
        # 获取父项中的子对象
        children_obj = get_obj_value(parent_item, "children")
        # 如果子对象是一个迭代器或者为空，则返回行数为 0
        if hasattr(children_obj, "next") or (children_obj is None):
            return 0
        else:
            # 否则返回子对象的长度作为行数
            return len(children_obj)

    # 判断指定父索引是否有子项
    def hasChildren(self, parent_index=None):
        # 如果有有效的父索引，则获取其内部指针对应的数据
        if parent_index and parent_index.isValid():
            parent_data = parent_index.internalPointer()
        else:
            # 否则将根数据作为父数据
            parent_data = self.root_item
        # 获取父数据中的子对象
        children_obj = get_obj_value(parent_data, "children")
        # 如果子对象为空，则没有子项
        if children_obj is None:
            return False
        # 如果子对象有 "next" 属性，则有子项
        if hasattr(children_obj, "next"):
            return True
        else:
            # 否则返回子对象的长度来判断是否有子项
            return len(children_obj)

    # 返回列数，即表头列表的长度
    def columnCount(self, parent_index=None):
        return len(self.header_list)

    # 判断是否可以获取更多数据
    def canFetchMore(self, index):
        try:
            # 如果有数据生成器，则尝试获取下一个数据
            if self.data_generator:
                data = self.data_generator.next()
                # 将获取的数据添加到根项目的子项目列表中
                self.root_item["children"].append(data)
                return True
            return False
        except StopIteration:
            # 如果迭代结束，并且计时器是活动状态，则停止计时器
            if self.timer.isActive():
                self.timer.stop()
            return False

    # 获取更多数据，重新设置模型
    def fetchMore(self, index=None):
        self.beginResetModel()  # 开始重置模型
        self.endResetModel()  # 结束重置模型

    # 返回指定索引处的数据，根据指定的角色
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        # 获取该列字段的配置信息
        attr_dict = self.header_list[index.column()]
        # 获取索引对应的内部数据对象
        data_obj = index.internalPointer()
        # 获取列的关键字
        attr = attr_dict.get("key")

        # 如果角色在设置映射中
        if role in SETTING_MAP.keys():
            # 获取角色对应的配置关键字
            role_key = SETTING_MAP[role].get("config")
            # 获取配置中该角色的格式化器
            formatter_from_config = attr_dict.get(role_key)
            # 如果没有找到格式化器，并且角色不是 DisplayRole/EditRole/ToolTipRole，则返回 None
            if not formatter_from_config and role not in [
                QtCore.Qt.DisplayRole,
                QtCore.Qt.EditRole,
                QtCore.Qt.ToolTipRole,
            ]:
                return None
            else:
                # 应用配置中的格式化器到数据对象的属性值上
                value = apply_formatter(formatter_from_config, get_obj_value(data_obj, attr), data_obj)
            # 获取设置映射中角色的转换函数，并应用到值上
            formatter_from_model = SETTING_MAP[role].get("formatter", None)
            result = apply_formatter(formatter_from_model, value)
            return result

        # 如果角色是 CheckStateRole 并且该列是可检查的
        if role == QtCore.Qt.CheckStateRole and attr_dict.get("checkable", False):
            # 获取检查状态并返回相应的 Qt 状态
            state = get_obj_value(data_obj, attr + "_checked")
            return QtCore.Qt.Unchecked if state is None else state
        return None
    # 定义一个方法用于设置数据，接受索引、值和角色参数，默认为编辑角色
    def setData(self, index, value, role=QtCore.Qt.EditRole):
        # 检查索引是否有效，并且角色是 CheckStateRole 或 EditRole 中的一个
        if index.isValid() and role in [QtCore.Qt.CheckStateRole, QtCore.Qt.EditRole]:
            # 获取表头列表中索引列对应的属性字典
            attr_dict = self.header_list[index.column()]
            # 从属性字典中获取键值
            key = attr_dict.get("key")
            # 获取索引中的数据对象
            data_obj = index.internalPointer()
            
            # 如果角色是 CheckStateRole 并且属性字典中指定为可检查
            if role == QtCore.Qt.CheckStateRole and attr_dict.get("checkable", False):
                # 更新键以区分已选中状态
                key += "_checked"
                # 更新数据对象中的值
                set_obj_value(data_obj, key, value)
                # 发送数据变更信号，更新当前索引
                self.dataChanged.emit(index, index, [role])

                # 更新子对象的状态
                for row, sub_obj in enumerate(get_obj_value(data_obj, "children", [])):
                    set_obj_value(sub_obj, key, value)
                    sub_index = self.index(row, index.column(), index)
                    self.dataChanged.emit(sub_index, sub_index, [role])

                # 更新父对象的状态
                parent_index = index.parent()
                if parent_index.isValid():
                    parent_obj = parent_index.internalPointer()
                    new_parent_value = value
                    old_parent_value = get_obj_value(parent_obj, key)
                    for sibling_obj in get_obj_value(get_obj_value(data_obj, "_parent"), "children", []):
                        if value != get_obj_value(sibling_obj, key):
                            new_parent_value = QtCore.Qt.PartiallyChecked
                            break
                    # 如果新的父级值和旧的不同，则更新父对象的值
                    if new_parent_value != old_parent_value:
                        set_obj_value(parent_obj, key, new_parent_value)
                        self.dataChanged.emit(parent_index, parent_index, [role])
            else:
                # 对象中普通编辑角色的数据更新
                set_obj_value(data_obj, key, value)
                self.dataChanged.emit(index, index, [role])
            return True
        else:
            # 索引无效或者角色不匹配时返回 False
            return False
class MSortFilterModel(QtCore.QSortFilterProxyModel):
    # 定义一个自定义的排序和过滤模型，继承自QtCore.QSortFilterProxyModel

    def __init__(self, parent=None):
        # 初始化函数，设置父对象，默认为None
        super(MSortFilterModel, self).__init__(parent)
        # 检查是否存在setRecursiveFilteringEnabled方法，如果有，则启用递归过滤
        if hasattr(self, "setRecursiveFilteringEnabled"):
            self.setRecursiveFilteringEnabled(True)
        # 初始化头部列表为空列表
        self.header_list = []
        # 创建一个空的正则表达式对象用于搜索
        self.search_reg = QtCore.QRegularExpression()
        # 设置正则表达式的匹配选项为不区分大小写
        self.search_reg.setPatternOptions(QtCore.QRegularExpression.CaseInsensitiveOption)
        # 设置默认的搜索模式为".*"，即通配符模式
        self.search_reg.setPattern(".*")  # This sets a wildcard-like pattern in regex

    def set_header_list(self, header_list):
        # 设置头部列表的方法，接受一个头部列表作为参数
        self.header_list = header_list
        # 遍历头部列表中的每个头部
        for head in self.header_list:
            # 创建一个新的正则表达式对象
            reg_exp = QtCore.QRegularExpression()
            # 设置正则表达式的匹配选项为不区分大小写
            reg_exp.setPatternOptions(QtCore.QRegularExpression.CaseInsensitiveOption)
            # 更新当前头部字典，增加一个"reg"键，对应值为新创建的正则表达式对象
            head.update({"reg": reg_exp})

    def filterAcceptsRow(self, source_row, source_parent):
        # 判断是否存在搜索模式
        if self.search_reg.pattern():
            # 如果存在搜索模式，则遍历头部列表中的每个数据字典
            for index, data_dict in enumerate(self.header_list):
                # 检查当前数据字典是否可搜索
                if data_dict.get("searchable", False):
                    # 获取源模型中指定行、列的索引和对应的数据值
                    model_index = self.sourceModel().index(source_row, index, source_parent)
                    value = self.sourceModel().data(model_index)
                    # 尝试匹配数据值和搜索正则表达式
                    match = self.search_reg.match(str(value))
                    if match.hasMatch():
                        # 如果匹配成功，停止搜索并返回True
                        break
            else:
                # 所有搜索完成，未找到匹配项，直接返回False
                return False

        # 开始匹配过滤条件组合
        for index, data_dict in enumerate(self.header_list):
            # 获取源模型中指定行、列的索引和对应的数据值
            model_index = self.sourceModel().index(source_row, index, source_parent)
            value = self.sourceModel().data(model_index)
            # 获取当前数据字典中的正则表达式对象
            reg_exp = data_dict.get("reg", None)
            if reg_exp and reg_exp.pattern():
                # 如果存在有效的正则表达式，则尝试匹配数据值
                match = reg_exp.match(str(value))
                if not match.hasMatch():
                    # 如果匹配失败，直接返回False
                    return False

        # 如果所有条件匹配成功，返回True
        return True

    def set_search_pattern(self, pattern):
        # 设置搜索模式的方法，接受一个字符串模式作为参数
        self.search_reg.setPattern(pattern)
        # 使过滤器失效，触发重新过滤
        self.invalidateFilter()

    def set_filter_attr_pattern(self, attr, pattern):
        # 设置过滤属性的正则表达式模式的方法，接受属性名和字符串模式作为参数
        for data_dict in self.header_list:
            if data_dict.get("key") == attr:
                # 查找匹配的属性名，并设置其对应的正则表达式模式
                data_dict.get("reg").setPattern(pattern)
                break
        # 使过滤器失效，触发重新过滤
        self.invalidateFilter()
```