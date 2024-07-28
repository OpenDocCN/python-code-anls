# `.\comic-translate\app\ui\dayu_widgets\dock_widget.py`

```py
"""
MDockWidget
"""
# 导入未来的模块，确保向后兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets


class MDockWidget(QtWidgets.QDockWidget):
    """
    Just apply the qss. No more extend.
    """

    def __init__(self, title="", parent=None, flags=QtCore.Qt.Widget):
        # 调用父类的初始化方法，创建一个 QDockWidget 实例
        super(MDockWidget, self).__init__(title, parent=parent, flags=flags)
```