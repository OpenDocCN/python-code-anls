# `.\PaddleOCR\PPOCRLabel\libs\colorDialog.py`

```py
# 导入必要的模块，根据 PyQt 版本选择不同的模块
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import QColorDialog, QDialogButtonBox
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# 导入 QDialogButtonBox 模块
BB = QDialogButtonBox

# 定义 ColorDialog 类，继承自 QColorDialog
class ColorDialog(QColorDialog):
    # 初始化函数，设置父窗口，调用父类的初始化函数
    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        # 设置颜色对话框选项，显示 Alpha 通道
        self.setOption(QColorDialog.ShowAlphaChannel)
        # Mac 原生对话框不支持恢复按钮
        self.setOption(QColorDialog.DontUseNativeDialog)
        # 添加一个恢复默认按钮
        # 默认值在调用时设置，以便在不同元素的对话框中起作用
        self.default = None
        # 获取布局中的按钮组件
        self.bb = self.layout().itemAt(1).widget()
        # 添加恢复默认按钮
        self.bb.addButton(BB.RestoreDefaults)
        # 点击按钮时连接到 checkRestore 函数
        self.bb.clicked.connect(self.checkRestore)

    # 获取颜色函数，设置初始值、标题和默认值
    def getColor(self, value=None, title=None, default=None):
        self.default = default
        # 如果有标题，设置对话框标题
        if title:
            self.setWindowTitle(title)
        # 如果有初始值，设置当前颜色
        if value:
            self.setCurrentColor(value)
        # 执行对话框，返回当前颜色或 None
        return self.currentColor() if self.exec_() else None

    # 检查恢复函数，根据按钮和默认值设置当前颜色
    def checkRestore(self, button):
        if self.bb.buttonRole(button) & BB.ResetRole and self.default:
            self.setCurrentColor(self.default)
```