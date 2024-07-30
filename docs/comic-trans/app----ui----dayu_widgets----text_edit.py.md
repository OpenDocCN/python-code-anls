# `.\comic-translate\app\ui\dayu_widgets\text_edit.py`

```py
# 导入必要的未来模块，确保兼容性和特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PySide6 的核心和控件模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 自定义可调整大小的控件类，继承自 QtWidgets.QSizeGrip
class MSizeGrip(QtWidgets.QSizeGrip):
    def __init__(self, parent=None):
        # 调用父类构造函数初始化
        super(MSizeGrip, self).__init__(parent)

# 自定义文本编辑控件类，继承自 QtWidgets.QTextEdit
class MTextEdit(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        # 调用父类构造函数初始化
        super(MTextEdit, self).__init__(parent)
        # 设置窗口标志为子窗口，以便作为独立窗口显示
        self.setWindowFlags(QtCore.Qt.SubWindow)
        # 创建一个 MSizeGrip 实例并添加到布局中
        self._size_grip = MSizeGrip(self)
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._size_grip, 0, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
        self.setLayout(layout)
        # 初始时隐藏大小调整手柄
        self._size_grip.setVisible(False)

    def autosize(self):
        # 连接文本内容变化的信号到自动调整大小的槽函数
        self.textChanged.connect(self._autosize_text_edit)
        return self

    def _autosize_text_edit(self):
        # 获取当前文本编辑器的宽度和文档对象
        # w = self.width()
        doc = self.document()
        # 打印当前文本编辑器的宽度、行数和理想宽度
        print(self.width(), doc.lineCount(), doc.idealWidth())

    def resizeable(self):
        """显示右下角的大小调整手柄，允许用户调整 MTextEdit 的大小"""
        # 设置大小调整手柄可见
        self._size_grip.setVisible(True)
        return self
```