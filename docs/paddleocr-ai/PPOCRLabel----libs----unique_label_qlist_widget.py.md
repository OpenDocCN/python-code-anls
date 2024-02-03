# `.\PaddleOCR\PPOCRLabel\libs\unique_label_qlist_widget.py`

```
# -*- encoding: utf-8 -*-

# 导入所需的模块
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtWidgets

# 创建一个自定义的 QListWidget 类，可以通过按下 Esc 键清除选择
class EscapableQListWidget(QtWidgets.QListWidget):
    # 重写 keyPressEvent 方法
    def keyPressEvent(self, event):
        # 调用父类的 keyPressEvent 方法
        super(EscapableQListWidget, self).keyPressEvent(event)
        # 如果按下的是 Esc 键
        if event.key() == Qt.Key_Escape:
            # 清除选择

# 创建一个继承自 EscapableQListWidget 的 UniqueLabelQListWidget 类
class UniqueLabelQListWidget(EscapableQListWidget):
    # 重写 mousePressEvent 方法
    def mousePressEvent(self, event):
        # 调用父类的 mousePressEvent 方法
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        # 如果鼠标点击的位置无效
        if not self.indexAt(event.pos()).isValid():
            # 清除选择

    # 根据标签查找列表中的项
    def findItemsByLabel(self, label, get_row=False):
        items = []
        # 遍历列表中的每一行
        for row in range(self.count()):
            # 获取当前行的项
            item = self.item(row)
            # 如果项的用户角色数据等于给定的标签
            if item.data(Qt.UserRole) == label:
                # 将该项添加到列表中
                items.append(item)
                # 如果需要获取行号，则立即返回行号
                if get_row:
                    return row
        return items

    # 根据标签创建一个新的项
    def createItemFromLabel(self, label):
        # 创建一个 QListWidgetItem 对象
        item = QtWidgets.QListWidgetItem()
        # 设置该项的用户角色数据为给定的标签
        item.setData(Qt.UserRole, label)
        return item

    # 设置项的标签和颜色
    def setItemLabel(self, item, label, color=None):
        # 创建一个 QLabel 对象
        qlabel = QtWidgets.QLabel()
        # 如果颜色为 None
        if color is None:
            # 设置文本为给定的标签
            qlabel.setText(f"{label}")
        else:
            # 设置文本为带有颜色的标签
            qlabel.setText('<font color="#{:02x}{:02x}{:02x}">●</font> {} '.format(*color, label))
        # 设置文本对齐方式
        qlabel.setAlignment(Qt.AlignBottom)

        # 设置项的大小提示
        item.setSizeHint(QSize(25, 25))

        # 将 QLabel 对象设置为项的小部件
        self.setItemWidget(item, qlabel)
```