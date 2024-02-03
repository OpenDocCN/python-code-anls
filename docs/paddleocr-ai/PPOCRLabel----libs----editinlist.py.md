# `.\PaddleOCR\PPOCRLabel\libs\editinlist.py`

```py
# 导入必要的模块
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QListWidget

# 创建 EditInList 类，继承自 QListWidget 类
class EditInList(QListWidget):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super(EditInList, self).__init__()
        # 初始化 edited_item 属性为 None
        self.edited_item = None

    # 处理列表项点击事件的方法
    def item_clicked(self, modelindex: QModelIndex):
        try:
            # 如果 edited_item 不为 None，则关闭之前的编辑器
            if self.edited_item is not None:
                self.closePersistentEditor(self.edited_item)
        except:
            # 捕获异常，将当前项设置为 edited_item
            self.edited_item = self.currentItem()

        # 将 modelindex 对应的项设置为 edited_item
        self.edited_item = self.item(modelindex.row())
        # 打开该项的持久编辑器
        self.openPersistentEditor(self.edited_item)
        # 编辑该项
        self.editItem(self.edited_item)

    # 处理鼠标双击事件的方法
    def mouseDoubleClickEvent(self, event):
        pass

    # 处理鼠标离开事件的方法
    def leaveEvent(self, event):
        # 关闭所有项的持久编辑器
        for i in range(self.count()):
            self.closePersistentEditor(self.item(i))
```