# `.\comic-translate\app\ui\dayu_widgets\page.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################
"""MPage"""
# 导入未来模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import functools

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme
from . import utils
from .combo_box import MComboBox
from .field_mixin import MFieldMixin
from .label import MLabel
from .menu import MMenu
from .spin_box import MSpinBox
from .tool_button import MToolButton


class MPage(QtWidgets.QWidget, MFieldMixin):
    """
    MPage
    一个长列表可以通过MPage分成多个页面，
    每次只加载一个页面。
    """

    sig_page_changed = QtCore.Signal(int, int)

    def set_total(self, value):
        """设置页面组件的总数。"""
        self.set_field("total", value)
        self.set_field("current_page", 1)

    def _slot_change_current_page(self, offset):
        """根据偏移量改变当前页码。"""
        self.set_field("current_page", self.field("current_page") + offset)
        self._emit_page_changed()

    def set_page_config(self, data_list):
        """设置每页的页面组件配置。"""
        self.set_field(
            "page_size_list",
            [{"label": str(data), "value": data} if isinstance(data, int) else data for data in data_list],
        )

    def _emit_page_changed(self):
        """发出页面改变的信号。"""
        self.sig_page_changed.emit(self.field("page_size_selected"), self.field("current_page"))
```