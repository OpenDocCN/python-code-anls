# `.\PaddleOCR\PPOCRLabel\libs\keyDialog.py`

```py
# 导入正则表达式模块
import re

# 导入 PyQt5 模块中的 QtCore、QtGui、QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

# 导入 PyQt5.Qt 模块中的 QT_VERSION_STR
from PyQt5.Qt import QT_VERSION_STR

# 导入 libs.utils 模块中的 newIcon、labelValidator 函数
from libs.utils import newIcon, labelValidator

# 判断 QT 版本是否为 5
QT5 = QT_VERSION_STR[0] == '5'

# 定义 KeyQLineEdit 类，继承自 QtWidgets.QLineEdit
class KeyQLineEdit(QtWidgets.QLineEdit):
    # 设置列表小部件
    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    # 键盘按下事件处理函数
    def keyPressEvent(self, e):
        # 如果按下的键为上下箭头键，则调用列表小部件的键盘按下事件处理函数
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(KeyQLineEdit, self).keyPressEvent(e)

# 定义 KeyDialog 类，继承自 QtWidgets.QDialog
class KeyDialog(QtWidgets.QDialog):
    # 初始化函数
    def __init__(
            self,
            text="Enter object label",
            parent=None,
            labels=None,
            sort_labels=True,
            show_text_field=True,
            completion="startswith",
            fit_to_content=None,
            flags=None,
    # 添加标签历史记录
    def addLabelHistory(self, label):
        # 如果标签列表中已存在该标签，则直接返回
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        # 向标签列表中添加标签
        self.labelList.addItem(label)
        # 如果需要对标签进行排序，则对标签列表进行排序
        if self._sort_labels:
            self.labelList.sortItems()

    # 标签被选中时的处理函数
    def labelSelected(self, item):
        self.edit.setText(item.text())

    # 验证输入的标签
    def validate(self):
        text = self.edit.text()
        # 如果 text 具有 strip 方法，则去除首尾空格
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        # 如果 text 不为空，则接受输入
        if text:
            self.accept()

    # 标签双击时的处理函数
    def labelDoubleClicked(self, item):
        self.validate()

    # 后处理函数
    def postProcess(self):
        text = self.edit.text()
        # 如果 text 具有 strip 方法，则去除首尾空格
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)
    # 更新标签对应的标志位
    def updateFlags(self, label_new):
        # 保存旧的标志位状态
        flags_old = self.getFlags()

        flags_new = {}
        # 遍历所有标志位的正则表达式和对应的键
        for pattern, keys in self._flags.items():
            # 如果标签符合正则表达式
            if re.match(pattern, label_new):
                # 更新新的标志位
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        # 设置新的标志位
        self.setFlags(flags_new)

    # 删除所有标志位
    def deleteFlags(self):
        # 逆序遍历所有标志位布局中的控件
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()
            # 移除控件
            self.flagsLayout.removeWidget(item)
            item.setParent(None)

    # 重置标志位
    def resetFlags(self, label=""):
        flags = {}
        # 遍历所有标志位的正则表达式和对应的键
        for pattern, keys in self._flags.items():
            # 如果标签符合正则表达式
            if re.match(pattern, label):
                # 重置标志位为 False
                for key in keys:
                    flags[key] = False
        # 设置标志位
        self.setFlags(flags)

    # 设置标志位
    def setFlags(self, flags):
        # 删除所有标志位
        self.deleteFlags()
        # 遍历所有标志位，创建复选框并设置状态
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    # 获取标志位
    def getFlags(self):
        flags = {}
        # 遍历所有标志位，获取其状态
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags
    # 弹出窗口方法，接受文本、移动标志和标志参数
    def popUp(self, text=None, move=True, flags=None):
        # 如果需要根据内容调整行高
        if self._fit_to_content["row"]:
            # 设置标签列表的最小高度
            self.labelList.setMinimumHeight(
                self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            )
        # 如果需要根据内容调整列宽
        if self._fit_to_content["column"]:
            # 设置标签列表的最小宽度
            self.labelList.setMinimumWidth(
                self.labelList.sizeHintForColumn(0) + 2
            )
        # 如果文本为空，则保留上一个标签
        if text is None:
            text = self.edit.text()
        # 如果有标志参数，则设置标志
        if flags:
            self.setFlags(flags)
        else:
            # 否则重置标志
            self.resetFlags(text)
        # 设置编辑框文本为传入的文本
        self.edit.setText(text)
        # 选择编辑框中的文本
        self.edit.setSelection(0, len(text))

        # 在标签列表中查找与文本匹配的项
        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            # 如果找到匹配项
            if len(items) != 1:
                # 设置当前项为第一个匹配项
                self.labelList.setCurrentItem(items[0])
            # 获取匹配项所在行
            row = self.labelList.row(items[0])
            # 设置自动完成器的当前行
            self.edit.completer().setCurrentRow(row)
        # 设置编辑框焦点
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        # 如果需要移动窗口
        if move:
            # 移动窗口到鼠标位置
            self.move(QtGui.QCursor.pos())
        # 如果执行成功
        if self.exec_():
            # 返回编辑框文本和标志
            return self.edit.text(), self.getFlags()
        else:
            # 否则返回空
            return None, None
```