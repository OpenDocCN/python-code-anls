# `.\comic-translate\app\ui\dayu_widgets\item_view.py`

```py
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
import six

# Import local modules
from . import dayu_theme
from . import utils
from .header_view import MHeaderView
from .item_model import MTableModel
from .menu import MMenu
from .qt import MPixmap
from .qt import get_scale_factor

# 定义用于排序的映射关系，"asc" 对应升序，"desc" 对应降序
HEADER_SORT_MAP = {"asc": QtCore.Qt.AscendingOrder, "desc": QtCore.Qt.DescendingOrder}


def draw_empty_content(view, text=None, pix_map=None):
    # Import local modules
    from . import dayu_theme

    # 如果未提供 pix_map 参数，则使用默认的 "empty.svg" 图像
    pix_map = pix_map or MPixmap("empty.svg")
    # 如果未提供 text 参数，则使用默认的 "No Data" 文本
    text = text or view.tr("No Data")
    # 创建一个 QPainter 对象，用于在视图中绘制内容
    painter = QtGui.QPainter(view)
    # 获取当前字体的度量信息
    font_metrics = painter.fontMetrics()
    # 设置画笔的颜色为主题中定义的次要文本颜色
    painter.setPen(QtGui.QPen(QtGui.QColor(dayu_theme.secondary_text_color)))
    # 计算内容的高度，包括图像和文本的高度
    content_height = pix_map.height() + font_metrics.height()
    # 设定内边距
    padding = 10
    # 计算适当的最小尺寸，限制在视图高度和宽度减去内边距后的最小值
    proper_min_size = min(view.height() - padding * 2, view.width() - padding * 2, content_height)
    # 如果适当的最小尺寸小于内容高度，则按比例缩放图像
    if proper_min_size < content_height:
        pix_map = pix_map.scaledToHeight(proper_min_size - font_metrics.height(), QtCore.Qt.SmoothTransformation)
        content_height = proper_min_size
    # 在视图中心绘制文本，水平居中对齐
    painter.drawText(
        view.width() / 2 - font_metrics.horizontalAdvance(text) / 2,
        view.height() / 2 + content_height / 2 - font_metrics.height() / 2,
        text,
    )
    # 在视图中心绘制图像
    painter.drawPixmap(
        view.width() / 2 - pix_map.width() / 2,
        view.height() / 2 - content_height / 2,
        pix_map,
    )
    # 结束绘制操作
    painter.end()


class MOptionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(MOptionDelegate, self).__init__(parent)
        self.editor = None
        self.showed = False
        self.exclusive = True
        self.parent_widget = None
        self.arrow_space = 20
        self.arrow_height = 6

    # 设置委托是否排他性显示菜单项
    def set_exclusive(self, flag):
        self.exclusive = flag

    # 创建编辑器部件，返回一个 MMenu 对象作为编辑器
    def createEditor(self, parent, option, index):
        self.parent_widget = parent
        # 创建一个 MMenu 对象作为编辑器，设置窗口标志为无边框窗口
        self.editor = MMenu(exclusive=self.exclusive, parent=parent)
        model = utils.real_model(index)
        real_index = utils.real_index(index)
        data_obj = real_index.internalPointer()
        # 获取属性名，用于获取数据
        attr = "{}_list".format(model.header_list[real_index.column()].get("key"))
        # 设置 MMenu 对象显示的数据
        self.editor.set_data(utils.get_obj_value(data_obj, attr, []))
        # 连接信号 sig_value_changed 到槽函数 _slot_finish_edit
        self.editor.sig_value_changed.connect(self._slot_finish_edit)
        return self.editor

    # 将编辑器的数据设置为 index 的编辑角色数据
    def setEditorData(self, editor, index):
        editor.set_value(index.data(QtCore.Qt.EditRole))
    # 设置编辑器数据到指定模型索引处
    def setModelData(self, editor, model, index):
        model.setData(index, editor.property("value"))

    # 更新编辑器的位置和几何信息
    def updateEditorGeometry(self, editor, option, index):
        # 将编辑器移动到父窗口的全局坐标系中指定位置
        editor.move(
            self.parent_widget.mapToGlobal(QtCore.QPoint(option.rect.x(), option.rect.y() + option.rect.height()))
        )

    # 绘制代理项外观
    def paint(self, painter, option, index):
        painter.save()
        icon_color = dayu_theme.icon_color
        # 如果鼠标悬停在选项上，填充背景颜色并更新图标颜色
        if option.state & QtWidgets.QStyle.State_MouseOver:
            painter.fillRect(option.rect, QtGui.QColor(dayu_theme.primary_5))
            icon_color = "#fff"
        # 如果选项被选中，填充背景颜色并更新图标颜色
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor(dayu_theme.primary_6))
            icon_color = "#fff"
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
        # 绘制下拉箭头图标
        pix = MPixmap("down_fill.svg", icon_color)
        h = option.rect.height()
        pix = pix.scaledToWidth(h * 0.5, QtCore.Qt.SmoothTransformation)
        painter.drawPixmap(option.rect.x() + option.rect.width() - h, option.rect.y() + h / 4, pix)
        painter.restore()
        # 调用父类的绘制方法
        super(MOptionDelegate, self).paint(painter, option, index)

    # 处理编辑完成信号的槽函数
    @QtCore.Slot(object)
    def _slot_finish_edit(self, obj):
        self.commitData.emit(self.editor)

    # 返回代理项的推荐大小
    def sizeHint(self, option, index):
        orig = super(MOptionDelegate, self).sizeHint(option, index)
        return QtCore.QSize(orig.width() + self.arrow_space, orig.height())

    # 事件过滤器，用于特定对象的事件处理
    # def eventFilter(self, obj, event):
    #     if obj is self.editor:
    #         print event.type(), obj.size()
    #     return super(MOptionDelegate, self).eventFilter(obj, event)
# 设置表头列表的方法，用给定的表头列表更新实例的表头
def set_header_list(self, header_list):
    # 获取比例因子，用于调整表头宽度
    scale_x, _ = get_scale_factor()
    # 将给定的表头列表赋值给实例变量
    self.header_list = header_list
    # 如果表头视图存在，则更新每个表头项的隐藏状态和宽度
    if self.header_view:
        # 遍历表头列表中的每个表头项
        for index, i in enumerate(header_list):
            # 根据表头项中的 hide 属性设置对应表头的隐藏状态
            self.header_view.setSectionHidden(index, i.get("hide", False))
            # 根据表头项中的 width 属性设置对应表头的宽度，并考虑比例因子
            self.header_view.resizeSection(index, i.get("width", 100) * scale_x)
            # 如果表头项中包含 order 属性，则设置排序指示器
            if "order" in i:
                order = i.get("order")
                # 如果 order 值在 HEADER_SORT_MAP 的值中，则设置排序指示器
                if order in HEADER_SORT_MAP.values():
                    self.header_view.setSortIndicator(index, order)
                # 如果 order 值在 HEADER_SORT_MAP 的键中，则设置排序指示器
                elif order in HEADER_SORT_MAP:
                    self.header_view.setSortIndicator(index, HEADER_SORT_MAP[order])
            # 如果表头项中包含 selectable 属性且为真，则为该列设置选项代理
            if i.get("selectable", False):
                delegate = MOptionDelegate(parent=self)
                delegate.set_exclusive(i.get("exclusive", True))
                self.setItemDelegateForColumn(index, delegate)
            # 否则，如果该列已有选项代理，则移除该代理
            elif self.itemDelegateForColumn(index):
                self.setItemDelegateForColumn(index, None)


# 启用或禁用上下文菜单功能
def enable_context_menu(self, enable):
    # 如果 enable 为真，则设置自定义上下文菜单策略，并连接上下文菜单槽函数
    if enable:
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.slot_context_menu)
    # 否则，禁用上下文菜单功能
    else:
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)


# 处理上下文菜单请求的槽函数，根据点击位置的索引触发相应事件
@QtCore.Slot(QtCore.QPoint)
def slot_context_menu(self, point):
    # 获取点击位置的代理索引
    proxy_index = self.indexAt(point)
    # 如果代理索引有效，则处理选择的行或单元格，并触发上下文菜单事件
    if proxy_index.isValid():
        # 判断当前数据模型是否为 QSortFilterProxyModel 类型，以确定是否需要映射到源模型
        need_map = isinstance(self.model(), QtCore.QSortFilterProxyModel)
        # 收集所有选中行的数据对象
        selection = []
        for index in self.selectionModel().selectedRows() or self.selectionModel().selectedIndexes():
            # 根据需要映射，获取数据对象
            data_obj = self.model().mapToSource(index).internalPointer() if need_map else index.internalPointer()
            selection.append(data_obj)
        # 创建并发送上下文菜单事件对象
        event = utils.ItemViewMenuEvent(view=self, selection=selection, extra={})
        self.sig_context_menu.emit(event)
    # 否则，创建空的上下文菜单事件对象，并触发上下文菜单事件
    else:
        event = utils.ItemViewMenuEvent(view=self, selection=[], extra={})
        self.sig_context_menu.emit(event)


# 处理鼠标移动事件，根据表头属性判断是否显示指针手势
def mouse_move_event(self, event):
    # 获取鼠标当前位置的索引
    index = self.indexAt(event.pos())
    # 获取真实索引，用于获取表头属性
    real_index = utils.real_index(index)
    # 如果表头项设置了 is_link 属性为真，则根据属性 attr 获取数据对象的值，并设置指针手势
    if self.header_list[real_index.column()].get("is_link", False):
        key_name = self.header_list[real_index.column()]["attr"]
        data_obj = utils.real_model(self.model()).data_list[real_index.row()]
        value = utils.get_obj_value(data_obj, key_name)
        if value:
            self.setCursor(QtCore.Qt.PointingHandCursor)
            return
    # 否则，恢复默认箭头指针
    self.setCursor(QtCore.Qt.ArrowCursor)


# 处理鼠标释放事件，处理左键释放事件，并避免处理右键释放事件
def mouse_release_event(self, event):
    # 如果事件不是左键释放事件，则调用父类的鼠标释放事件处理方法并返回
    if event.button() != QtCore.Qt.LeftButton:
        QtWidgets.QTableView.mouseReleaseEvent(self, event)
        return
    # 获取鼠标释放位置的索引
    index = self.indexAt(event.pos())
    # 获取真实索引，用于进一步处理
    real_index = utils.real_index(index)
    # 如果列的头部属性包含 "is_link" 键并且其值为 True
    if self.headerList[real_index.column()].get("is_link", False):
        # 获取列头对应的属性名
        key_name = self.header_list[real_index.column()]["attr"]
        # 获取实际数据对象
        data_obj = utils.real_model(self.model()).data_list[real_index.row()]
        # 根据属性名获取数据对象中对应的值
        value = utils.get_obj_value(data_obj, key_name)
        # 如果值存在
        if value:
            # 如果值是字典类型，发射链接点击信号，并传递该字典作为参数
            if isinstance(value, dict):
                self.sig_link_clicked.emit(value)
            # 如果值是字符串类型，发射链接点击信号，并传递整个数据对象作为参数
            elif isinstance(value, six.string_types):
                self.sig_link_clicked.emit(data_obj)
            # 如果值是列表类型
            elif isinstance(value, list):
                # 对列表中的每个元素，发射链接点击信号
                for i in value:
                    self.sig_link_clicked.emit(i)
class MTableView(QtWidgets.QTableView):
    # 设置表格视图的列头列表
    set_header_list = set_header_list
    # 是否启用上下文菜单功能
    enable_context_menu = enable_context_menu
    # 处理上下文菜单的槽函数
    slot_context_menu = slot_context_menu
    # 定义一个信号，用于发送上下文菜单信号
    sig_context_menu = QtCore.Signal(object)

    def __init__(self, size=None, show_row_count=False, parent=None):
        super(MTableView, self).__init__(parent)
        # 无数据时显示的图像，默认为空
        self._no_data_image = None
        # 无数据时显示的文本，默认为 "No Data"
        self._no_data_text = self.tr("No Data")
        # 如果未指定大小，则使用默认大小
        size = size or dayu_theme.default_size
        # 创建垂直表头视图
        ver_header_view = MHeaderView(QtCore.Qt.Vertical, parent=self)
        ver_header_view.setDefaultSectionSize(size)
        ver_header_view.setSortIndicatorShown(False)
        self.setVerticalHeader(ver_header_view)
        # 初始化表头列表为空列表
        self.header_list = []
        # 创建水平表头视图
        self.header_view = MHeaderView(QtCore.Qt.Horizontal, parent=self)
        self.header_view.setFixedHeight(size)
        # 如果不显示行号，则隐藏垂直表头视图
        if not show_row_count:
            ver_header_view.hide()
        self.setHorizontalHeader(self.header_view)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)

    def set_no_data_text(self, text):
        # 设置无数据时显示的文本
        self._no_data_text = text

    def set_no_data_image(self, image):
        # 设置无数据时显示的图像
        self._no_data_image = image

    def setShowGrid(self, flag):
        # 设置是否显示网格
        self.header_view.setProperty("grid", flag)
        self.verticalHeader().setProperty("grid", flag)
        self.header_view.style().polish(self.header_view)

        return super(MTableView, self).setShowGrid(flag)
    def paintEvent(self, event):
        """
        Override paintEvent when there is no data to show, draw the preset picture and text.
        当没有数据需要展示时，重写paintEvent方法，绘制预设的图片和文本。
        """
        # 获取真实的数据模型
        model = utils.real_model(self.model())
        # 如果模型为空
        if model is None:
            # 绘制空内容
            draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 如果模型是MTableModel类型
        elif isinstance(model, MTableModel):
            # 如果数据列表为空
            if not model.get_data_list():
                # 绘制空内容
                draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 调用父类的paintEvent方法
        return super(MTableView, self).paintEvent(event)

    def save_state(self, name):
        """
        Save the state of the table view's header.
        保存表格视图头部的状态。
        """
        # 创建QSettings对象，用于持久化设置
        settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat,
            QtCore.QSettings.UserScope,
            "DAYU",
            ".",
        )
        # 设置表头状态的值
        settings.setValue("{}/headerState".format(name, self.header_view.saveState()))

    def load_state(self, name):
        """
        Load the state of the table view's header.
        加载表格视图头部的状态。
        """
        # 创建QSettings对象，用于持久化设置
        settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat,
            QtCore.QSettings.UserScope,
            "DAYU",
            ".",
        )
        # 如果存在名为{name}/headerState的设置值
        if settings.value("{}/headerState".format(name)):
            # 恢复表头状态
            self.header_view.restoreState(settings.value("{}/headerState".format(name)))
# 定义一个自定义的树形视图类 MTreeView，继承自 QtWidgets.QTreeView
class MTreeView(QtWidgets.QTreeView):
    # 设置头部列表，默认值来自外部 set_header_list 函数
    set_header_list = set_header_list
    # 是否启用上下文菜单，默认值来自外部 enable_context_menu 函数
    enable_context_menu = enable_context_menu
    # 处理上下文菜单的槽函数，默认值来自外部 slot_context_menu 函数
    slot_context_menu = slot_context_menu
    # 定义一个信号，用于发射上下文菜单相关的信号
    sig_context_menu = QtCore.Signal(object)

    # 构造函数，初始化 MTreeView 实例
    def __init__(self, parent=None):
        # 调用父类 QtWidgets.QTreeView 的构造函数进行初始化
        super(MTreeView, self).__init__(parent)
        # 没有数据时显示的图片，默认为 None
        self._no_data_image = None
        # 没有数据时显示的文本，默认为 "No Data"
        self._no_data_text = self.tr("No Data")
        # 头部列表，初始化为空列表
        self.header_list = []
        # 创建一个自定义的水平头部视图 MHeaderView 实例
        self.header_view = MHeaderView(QtCore.Qt.Horizontal)
        # 将自定义的头部视图设置为当前视图的头部
        self.setHeader(self.header_view)
        # 启用排序功能
        self.setSortingEnabled(True)
        # 设置交替行颜色
        self.setAlternatingRowColors(True)

    # 重写 paintEvent 方法，在没有数据时绘制预设的图片和文本
    def paintEvent(self, event):
        """Override paintEvent when there is no data to show, draw the preset picture and text."""
        # 获取真实的数据模型，utils.real_model 函数用于获取实际的数据模型
        model = utils.real_model(self.model())
        # 如果模型为 None，即没有有效的数据模型
        if model is None:
            # 在视口上绘制空内容，显示指定的文本和图片
            draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 如果模型是 MTableModel 的实例
        elif isinstance(model, MTableModel):
            # 如果数据列表为空
            if not model.get_data_list():
                # 在视口上绘制空内容，显示指定的文本和图片
                draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 调用父类的 paintEvent 方法，继续处理事件
        return super(MTreeView, self).paintEvent(event)

    # 设置没有数据时显示的文本
    def set_no_data_text(self, text):
        self._no_data_text = text



# 定义一个自定义的大图标视图类 MBigView，继承自 QtWidgets.QListView
class MBigView(QtWidgets.QListView):
    # 设置头部列表，默认值来自外部 set_header_list 函数
    set_header_list = set_header_list
    # 是否启用上下文菜单，默认值来自外部 enable_context_menu 函数
    enable_context_menu = enable_context_menu
    # 处理上下文菜单的槽函数，默认值来自外部 slot_context_menu 函数
    slot_context_menu = slot_context_menu
    # 定义一个信号，用于发射上下文菜单相关的信号
    sig_context_menu = QtCore.Signal(object)

    # 构造函数，初始化 MBigView 实例
    def __init__(self, parent=None):
        # 调用父类 QtWidgets.QListView 的构造函数进行初始化
        super(MBigView, self).__init__(parent)
        # 没有数据时显示的图片，默认为 None
        self._no_data_image = None
        # 没有数据时显示的文本，默认为 "No Data"
        self._no_data_text = self.tr("No Data")
        # 头部列表，初始化为空列表
        self.header_list = []
        # 头部视图，默认为 None
        self.header_view = None
        # 设置视图模式为图标模式
        self.setViewMode(QtWidgets.QListView.IconMode)
        # 设置调整模式为自适应
        self.setResizeMode(QtWidgets.QListView.Adjust)
        # 设置移动模式为静态
        self.setMovement(QtWidgets.QListView.Static)
        # 设置图标之间的间距
        self.setSpacing(10)
        # 获取大图标视图的默认大小
        default_size = dayu_theme.big_view_default_size
        # 设置图标的大小
        self.setIconSize(QtCore.QSize(default_size, default_size))

    # 缩放图标的大小
    def scale_size(self, factor):
        """Scale the icon size."""
        # 计算新的图标大小
        new_size = self.iconSize() * factor
        # 获取大图标视图的最大和最小大小
        max_size = dayu_theme.big_view_max_size
        min_size = dayu_theme.big_view_min_size
        # 如果新的宽度超过最大大小
        if new_size.width() > max_size:
            new_size = QtCore.QSize(max_size, max_size)
        # 如果新的宽度小于最小大小
        elif new_size.width() < min_size:
            new_size = QtCore.QSize(min_size, min_size)
        # 设置新的图标大小
        self.setIconSize(new_size)

    # 重写 wheelEvent 方法，当用户按住 Ctrl 键时，缩放列表视图中的图标大小
    def wheelEvent(self, event):
        """Override wheelEvent while user press ctrl, zoom the list view icon size."""
        # 如果用户按下 Ctrl 键
        if event.modifiers() == QtCore.Qt.ControlModifier:
            # 计算滚轮滚动的度数
            num_degrees = event.delta() / 8.0
            num_steps = num_degrees / 15.0
            # 计算缩放因子
            factor = pow(1.125, num_steps)
            # 调用 scale_size 方法缩放图标大小
            self.scale_size(factor)
        else:
            # 否则调用父类的 wheelEvent 方法，继续处理事件
            super(MBigView, self).wheelEvent(event)
    # 重写 paintEvent 方法，在没有数据要显示时绘制预设的图片和文本
    def paintEvent(self, event):
        """Override paintEvent when there is no data to show, draw the preset picture and text."""
        # 获取真实的数据模型
        model = utils.real_model(self.model())
        # 如果模型为 None，则表示没有数据，绘制空内容的图片和文本
        if model is None:
            draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 如果模型是 MTableModel 类的实例
        elif isinstance(model, MTableModel):
            # 如果数据列表为空，则绘制空内容的图片和文本
            if not model.get_data_list():
                draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 调用父类的 paintEvent 方法继续处理事件
        return super(MBigView, self).paintEvent(event)

    # 设置没有数据时显示的文本
    def set_no_data_text(self, text):
        self._no_data_text = text
# 自定义的列表视图类，继承自 QtWidgets.QListView
class MListView(QtWidgets.QListView):
    # 设置表头列表，通过外部传入的 set_header_list 函数设置
    set_header_list = set_header_list
    # 是否启用上下文菜单，通过外部传入的 enable_context_menu 函数设置
    enable_context_menu = enable_context_menu
    # 处理上下文菜单的槽函数，通过外部传入的 slot_context_menu 函数设置
    slot_context_menu = slot_context_menu
    # 定义信号，用于触发上下文菜单操作
    sig_context_menu = QtCore.Signal(object)

    # 初始化函数，可以指定大小和父对象
    def __init__(self, size=None, parent=None):
        super(MListView, self).__init__(parent)
        # 没有数据时显示的图片，默认为 None
        self._no_data_image = None
        # 没有数据时显示的文本，默认为 "No Data"
        self._no_data_text = self.tr("No Data")
        # 设置列表项的属性 "dayu_size"，如果未指定大小则使用默认大小
        self.setProperty("dayu_size", size or dayu_theme.default_size)
        # 表头列表初始化为空列表
        self.header_list = []
        # 表头视图初始化为 None
        self.header_view = None
        # 设置模型列为第 0 列
        self.setModelColumn(0)
        # 启用交替行颜色
        self.setAlternatingRowColors(True)

    # 设置显示特定列的数据
    def set_show_column(self, attr):
        # 遍历表头列表
        for index, attr_dict in enumerate(self.header_list):
            # 如果表头字典中的 "key" 键与 attr 参数相匹配
            if attr_dict.get("key") == attr:
                # 设置模型列为当前索引
                self.setModelColumn(index)
                break
        else:
            # 如果未找到匹配的列，则设置模型列为第 0 列
            self.setModelColumn(0)

    # 重写 paintEvent 方法，在没有数据时绘制预设的图片和文本
    def paintEvent(self, event):
        # 获取真实的数据模型
        model = utils.real_model(self.model())
        # 如果模型为空
        if model is None:
            # 在视口上绘制空内容，显示设定的无数据文本和图片
            draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 如果模型是 MTableModel 类型
        elif isinstance(model, MTableModel):
            # 如果数据列表为空
            if not model.get_data_list():
                # 在视口上绘制空内容，显示设定的无数据文本和图片
                draw_empty_content(self.viewport(), self._no_data_text, self._no_data_image)
        # 调用父类的 paintEvent 方法进行默认绘制
        return super(MListView, self).paintEvent(event)

    # 设置没有数据时显示的文本
    def set_no_data_text(self, text):
        self._no_data_text = text
```