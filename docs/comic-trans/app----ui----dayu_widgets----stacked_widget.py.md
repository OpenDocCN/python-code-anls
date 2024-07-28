# `.\comic-translate\app\ui\dayu_widgets\stacked_widget.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.4
# Email : muyanru345@163.com
###################################################################
"""MStackedWidget"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
from PySide6 import QtWidgets

# Import local modules
from .mixin import stacked_animation_mixin

# 定义一个装饰器，用于在当前类中添加动画混合功能
@stacked_animation_mixin
class MStackedWidget(QtWidgets.QStackedWidget):
    """Just active animation when current index changed."""

    def __init__(self, parent=None):
        # 调用父类构造函数，初始化 Qt 中的堆栈窗口部件
        super(MStackedWidget, self).__init__(parent)

    def disable_animation(self):
        # 断开当前索引改变信号与播放动画方法的连接，用于禁用动画效果
        self.currentChanged.disconnect(self._play_anim)


这段代码定义了一个名为 `MStackedWidget` 的类，它继承自 `QtWidgets.QStackedWidget`。类中包含一个 `disable_animation` 方法，用于断开当前索引改变信号与动画播放方法的连接，从而禁用动画效果。
```