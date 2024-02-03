# `.\PaddleOCR\PPOCRLabel\libs\zoomWidget.py`

```
# 版权声明，允许在特定条件下使用和分发软件
# 导入 PyQt5 库中的模块
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
# 如果导入失败，则使用 PyQt4 库中的模块
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# 定义一个继承自 QSpinBox 的 ZoomWidget 类
class ZoomWidget(QSpinBox):

    # 初始化方法，设置默认值为 100
    def __init__(self, value=100):
        # 调用父类的初始化方法
        super(ZoomWidget, self).__init__()
        # 设置按钮符号为无按钮
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # 设置取值范围为 1 到 500
        self.setRange(1, 500)
        # 设置后缀为 '%'
        self.setSuffix(' %')
        # 设置初始值
        self.setValue(value)
        # 设置工具提示
        self.setToolTip(u'Zoom Level')
        # 设置状态提示为工具提示
        self.setStatusTip(self.toolTip())
        # 设置对齐方式为居中
        self.setAlignment(Qt.AlignCenter)

    # 重写最小尺寸提示方法
    def minimumSizeHint(self):
        # 获取父类的最小尺寸高度
        height = super(ZoomWidget, self).minimumSizeHint().height()
        # 获取字体度量
        fm = QFontMetrics(self.font())
        # 计算最大值的宽度
        width = fm.width(str(self.maximum()))
        # 返回计算后的尺寸
        return QSize(width, height)
```