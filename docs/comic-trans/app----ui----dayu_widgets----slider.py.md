# `.\comic-translate\app\ui\dayu_widgets\slider.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
"""MSlider"""
# 导入未来的模块功能，确保代码向后兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets


class MSlider(QtWidgets.QSlider):
    """
    A Slider component for displaying current value and intervals in range.

    MSlider just apply qss for QSlider.
    """

    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super(MSlider, self).__init__(orientation, parent=parent)
        self._show_text_when_move = True

    def disable_show_text(self):
        # 禁用移动时显示数值文本的功能
        self._show_text_when_move = False

    def mouseMoveEvent(self, event):
        """Override the mouseMoveEvent to show current value as a tooltip."""
        if self._show_text_when_move:
            # 如果启用了显示移动时的数值文本，将当前数值显示为工具提示
            QtWidgets.QToolTip.showText(event.globalPos(), str(self.value()), self)
        return super(MSlider, self).mouseMoveEvent(event)
```