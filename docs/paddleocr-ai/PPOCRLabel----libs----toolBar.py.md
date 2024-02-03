# `.\PaddleOCR\PPOCRLabel\libs\toolBar.py`

```
# 版权声明，允许任何人免费获取并使用此软件，需包含版权声明和许可声明
try:
    # 尝试导入 PyQt5 库中的模块
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # 如果导入失败，则导入 PyQt4 库中的模块
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# 定义 ToolBar 类，继承自 QToolBar 类
class ToolBar(QToolBar):

    # 初始化方法，接受一个标题参数
    def __init__(self, title):
        # 调用父类的初始化方法
        super(ToolBar, self).__init__(title)
        # 获取工具栏的布局
        layout = self.layout()
        # 设置边距为 0
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        # 设置窗口标志为无边框窗口
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
    # 定义一个方法，用于向工具栏添加动作
    def addAction(self, action):
        # 如果传入的动作是 QWidgetAction 类型的，则调用父类的 addAction 方法添加动作
        if isinstance(action, QWidgetAction):
            return super(ToolBar, self).addAction(action)
        # 如果传入的动作不是 QWidgetAction 类型，则创建一个 ToolButton 对象
        btn = ToolButton()
        # 将传入的动作设置为按钮的默认动作
        btn.setDefaultAction(action)
        # 设置按钮的样式为工具按钮样式
        btn.setToolButtonStyle(self.toolButtonStyle())
        # 将按钮添加到工具栏中
        self.addWidget(btn)
# 定义一个工具栏按钮类，确保所有按钮具有相同的大小
class ToolButton(QToolButton):
    """ToolBar companion class which ensures all buttons have the same size."""
    # 设置按钮的最小大小为 (60, 60)
    minSize = (60, 60)

    # 重写父类的 minimumSizeHint 方法
    def minimumSizeHint(self):
        # 调用父类的 minimumSizeHint 方法获取最小大小
        ms = super(ToolButton, self).minimumSizeHint()
        # 获取当前按钮的宽度和高度
        w1, h1 = ms.width(), ms.height()
        # 获取设定的最小大小
        w2, h2 = self.minSize
        # 更新按钮的最小大小为宽度和高度的最大值
        ToolButton.minSize = max(w1, w2), max(h1, h2)
        # 返回更新后的最小大小
        return QSize(*ToolButton.minSize)
```