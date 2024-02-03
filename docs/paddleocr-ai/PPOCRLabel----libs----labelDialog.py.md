# `.\PaddleOCR\PPOCRLabel\libs\labelDialog.py`

```py
# 版权声明，允许任何人免费获取此软件及相关文档文件的副本，并在不受限制的情况下处理该软件，包括但不限于使用、复制、修改、合并、发布、分发、许可和/或出售该软件的副本，并允许提供该软件的人员这样做，但须遵守以下条件：
# 上述版权声明和此许可声明应包含在所有副本或实质部分的软件中。本软件按“原样”提供，不提供任何明示或暗示的担保，包括但不限于适销性、特定用途适用性和非侵权性的担保。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论是在合同、侵权行为还是其他方面，由于软件或使用软件或其他方式与软件的使用有关而产生。
try:
    # 尝试导入 PyQt5 库中的模块
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # 如果导入失败，则导入 PyQt4 库中的模块
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# 从自定义的 utils 模块中导入 newIcon 和 labelValidator 函数
from libs.utils import newIcon, labelValidator

# 定义 QDialogButtonBox 的别名为 BB
BB = QDialogButtonBox

# 定义 LabelDialog 类，继承自 QDialog 类
class LabelDialog(QDialog):
    # 初始化函数，设置对话框的文本、父对象和列表项
    def __init__(self, text="Enter object label", parent=None, listItem=None):
        # 调用父类的初始化函数
        super(LabelDialog, self).__init__(parent)

        # 创建一个单行文本编辑框
        self.edit = QLineEdit()  # OLD
        # self.edit = QTextEdit()
        # 设置文本编辑框的文本内容
        self.edit.setText(text)
        # 连接文本编辑框的编辑完成信号到槽函数 postProcess

        self.edit.editingFinished.connect(self.postProcess)

        # 创建一个字符串列表模型
        model = QStringListModel()
        # 设置字符串列表模型的字符串列表
        model.setStringList(listItem)
        # 创建一个自动完成器
        completer = QCompleter()
        # 设置自动完成器的模型
        completer.setModel(model)
        # 将自动完成器设置给文本编辑框
        self.edit.setCompleter(completer)

        # 创建一个垂直布局
        layout = QVBoxLayout()
        # 将文本编辑框添加到布局中
        layout.addWidget(self.edit)
        # 创建一个按钮盒子
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        # 设置确定按钮和取消按钮的图标
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        # 连接确定按钮的点击信号到 validate 槽函数
        bb.accepted.connect(self.validate)
        # 连接取消按钮的点击信号到 reject 槽函数
        bb.rejected.connect(self.reject)
        # 将按钮盒子添加到布局中
        layout.addWidget(bb)

        # 如果列表项不为空且长度大于0
        # 则创建一个列表部件，添加列表项，并连接点击和双击信号到相应槽函数
        #     self.listWidget = QListWidget(self)
        #     for item in listItem:
        #         self.listWidget.addItem(item)
        #     self.listWidget.itemClicked.connect(self.listItemClick)
        #     self.listWidget.itemDoubleClicked.connect(self.listItemDoubleClick)
        #     layout.addWidget(self.listWidget)

        # 设置布局
        self.setLayout(layout)

    # 验证函数，用于验证文本编辑框的内容
    def validate(self):
        try:
            # 如果文本编辑框的文本去除空格后不为空，则接受对话框
            if self.edit.text().trimmed():
                self.accept()
        except AttributeError:
            # 处理 PyQt5 中的 AttributeError 异常
            if self.edit.text().strip():
                self.accept()

    # 后处理函数，用于处理文本编辑框的内容
    def postProcess(self):
        try:
            # 尝试去除文本编辑框的文本两端的空格
            self.edit.setText(self.edit.text().trimmed())
            # 打印文本编辑框的文本
            # print(self.edit.text())
        except AttributeError:
            # 处理 PyQt5 中的 AttributeError 异常
            self.edit.setText(self.edit.text())
            # 打印文本编辑框的文本
            print(self.edit.text())
    # 弹出一个文本框，显示指定的文本内容，并设置焦点
    def popUp(self, text='', move=True):
        # 设置文本框内容为指定文本
        self.edit.setText(text)
        # 选择文本框中的所有文本
        self.edit.setSelection(0, len(text))
        # 设置焦点到文本框
        self.edit.setFocus(Qt.PopupFocusReason)
        # 如果需要移动弹出框
        if move:
            # 获取鼠标光标位置
            cursor_pos = QCursor.pos()
            # 获取父窗口的右下角坐标
            parent_bottomRight = self.parentWidget().geometry()
            # 计算弹出框的最大 x 和 y 坐标
            max_x = parent_bottomRight.x() + parent_bottomRight.width() - self.sizeHint().width()
            max_y = parent_bottomRight.y() + parent_bottomRight.height() - self.sizeHint().height()
            # 将最大坐标转换为全局坐标
            max_global = self.parentWidget().mapToGlobal(QPoint(max_x, max_y))
            # 如果鼠标光标的 x 坐标大于最大 x 坐标，则设置为最大 x 坐标
            if cursor_pos.x() > max_global.x():
                cursor_pos.setX(max_global.x())
            # 如果鼠标光标的 y 坐标大于最大 y 坐标，则设置为最大 y 坐标
            if cursor_pos.y() > max_global.y():
                cursor_pos.setY(max_global.y())
            # 移动弹出框到计算出的位置
            self.move(cursor_pos)
        # 如果弹出框执行成功，则返回文本框中的文本，否则返回 None
        return self.edit.text() if self.exec_() else None

    # 处理列表项单击事件
    def listItemClick(self, tQListWidgetItem):
        try:
            # 尝试获取列表项的文本并去除首尾空格
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # 处理 PyQt5 中的异常情况
            text = tQListWidgetItem.text().strip()
        # 设置文本框内容为处理后的文本
        self.edit.setText(text)

    # 处理列表项双击事件
    def listItemDoubleClick(self, tQListWidgetItem):
        # 调用列表项单击事件处理函数
        self.listItemClick(tQListWidgetItem)
        # 执行验证操作
        self.validate()
```