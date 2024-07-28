# `.\comic-translate\app\ui\dayu_widgets\flow_layout.py`

```py
# 导入必要的模块
from PySide6 import QtCore  # 导入 PySide6 的 QtCore 模块
from PySide6 import QtWidgets  # 导入 PySide6 的 QtWidgets 模块

# 创建一个自定义的布局类 MFlowLayout，继承自 QtWidgets.QLayout
class MFlowLayout(QtWidgets.QLayout):
    """
    FlowLayout, the code is come from PySide/examples/layouts/flowlayout.py
    I change the code style and add insertWidget method.
    """

    # 初始化方法，设置布局的边距和间距，默认为0和-1
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(MFlowLayout, self).__init__(parent)

        # 如果有父对象，设置边距
        if parent is not None:
            self.setMargin(margin)

        # 设置布局中组件之间的间距
        self.setSpacing(spacing)

        # 存储添加的小部件的列表
        self.item_list = []

    # 析构函数，删除所有的小部件
    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    # 插入小部件到布局中指定的位置
    def insertWidget(self, index, widget):
        self.addChildWidget(widget)  # 添加子部件
        if index < 0:
            index = self.count()
        item = QtWidgets.QWidgetItem(widget)  # 创建包含部件的 QWidgetItem
        self.item_list.insert(index, item)  # 在指定位置插入部件
        self.update()  # 更新布局

    # 添加 QLayoutItem 到布局中
    def addItem(self, item):
        self.item_list.append(item)

    # 返回布局中的小部件数量
    def count(self):
        return len(self.item_list)

    # 返回指定索引位置的布局项
    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]

        return None

    # 移除并返回指定索引位置的小部件
    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index).widget()

        return None

    # 清空布局，删除所有小部件
    def clear(self):
        while self.item_list:
            widget = self.takeAt(0)
            if widget:
                widget.deleteLater()

    # 返回布局的扩展方向
    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    # 返回是否支持基于宽度的高度计算
    def hasHeightForWidth(self):
        return True

    # 根据指定的宽度计算布局的高度
    def heightForWidth(self, width):
        height = self.do_layout(QtCore.QRect(0, 0, width, 0), True)
        return height

    # 设置布局的几何边界
    def setGeometry(self, rect):
        super(MFlowLayout, self).setGeometry(rect)
        self.do_layout(rect, False)  # 执行布局

    # 返回布局的大小提示
    def sizeHint(self):
        return self.minimumSize()

    # 返回布局的最小尺寸
    def minimumSize(self):
        size = QtCore.QSize()

        # 计算所有小部件的最小尺寸的总和
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())

        # 加上布局的边距
        size += QtCore.QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size
    # 定义一个方法，用于管理布局的位置和大小
    def do_layout(self, rect, test_only):
        # 获取矩形的左上角 x 坐标
        x = rect.x()
        # 获取矩形的左上角 y 坐标
        y = rect.y()
        # 初始化行高为 0
        line_height = 0

        # 遍历存储在 self.item_list 中的每个项目
        for item in self.item_list:
            # 获取项目关联的窗口部件
            wid = item.widget()
            # 计算水平方向上的间距，包括布局中定义的间距
            space_x = self.spacing() + wid.style().layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Horizontal,
            )
            # 计算垂直方向上的间距，包括布局中定义的间距
            space_y = self.spacing() + wid.style().layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Vertical,
            )
            # 计算下一个项目的 x 坐标位置
            next_x = x + item.sizeHint().width() + space_x
            # 如果下一个项目超出了矩形的右侧，并且当前行已经有项目存在
            if next_x - space_x > rect.right() and line_height > 0:
                # 将 x 坐标重置为矩形的左上角 x 坐标
                x = rect.x()
                # 调整 y 坐标，将其设置为当前行的底部位置，包括行高和垂直间距
                y = y + line_height + space_y
                # 重新计算下一个项目的 x 坐标位置
                next_x = x + item.sizeHint().width() + space_x
                # 重置行高为 0，因为开始新的一行
                line_height = 0

            # 如果不是仅测试模式，则设置项目的几何位置
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            # 更新 x 坐标为下一个项目的 x 坐标位置
            x = next_x
            # 更新行高为当前项目高度与现有行高中的较大值
            line_height = max(line_height, item.sizeHint().height())

        # 返回整个布局所占据的高度，即最后一个项目的底部位置减去矩形的 y 坐标
        return y + line_height - rect.y()
```