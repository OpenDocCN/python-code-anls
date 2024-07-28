# `.\comic-translate\app\ui\dayu_widgets\tab_widget.py`

```py
# 设置脚本的解释器为 Python，并声明文件编码为 UTF-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################

# 导入未来版本所需的模块，确保代码在 Python 2 和 Python 3 上兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块 PySide6 中的 QtCore 和 QtWidgets 组件
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块中的 mixin 模块中的 cursor_mixin 和 stacked_animation_mixin
from .mixin import cursor_mixin
from .mixin import stacked_animation_mixin


@cursor_mixin
# 创建 MTabBar 类，继承自 QtWidgets.QTabBar
class MTabBar(QtWidgets.QTabBar):
    def __init__(self, parent=None):
        # 调用父类 QtWidgets.QTabBar 的构造方法
        super(MTabBar, self).__init__(parent=parent)
        # 设置标签页不绘制基底
        self.setDrawBase(False)

    # 重写 tabSizeHint 方法，用于计算标签页的推荐大小
    def tabSizeHint(self, index):
        # 获取指定索引位置标签页的文本内容
        tab_text = self.tabText(index)
        # 如果标签页可关闭，返回包含文本长度和额外空间的 QSize 对象
        if self.tabsClosable():
            return QtCore.QSize(
                self.fontMetrics().horizontalAdvance(tab_text) + 70,
                self.fontMetrics().height() + 20,
            )
        # 如果标签页不可关闭，返回包含文本长度和额外空间的 QSize 对象
        else:
            return QtCore.QSize(
                self.fontMetrics().horizontalAdvance(tab_text) + 50,
                self.fontMetrics().height() + 20,
            )


@stacked_animation_mixin
# 创建 MTabWidget 类，继承自 QtWidgets.QTabWidget
class MTabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        # 调用父类 QtWidgets.QTabWidget 的构造方法
        super(MTabWidget, self).__init__(parent=parent)
        # 创建 MTabBar 实例并设置为当前 QTabWidget 的标签栏
        self.bar = MTabBar()
        self.setTabBar(self.bar)

    # 定义 disable_animation 方法，用于取消当前窗口切换时的动画效果
    def disable_animation(self):
        # 断开当前窗口切换时的信号槽连接，避免播放动画
        self.currentChanged.disconnect(self._play_anim)
```