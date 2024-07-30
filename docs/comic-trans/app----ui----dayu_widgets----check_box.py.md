# `.\comic-translate\app\ui\dayu_widgets\check_box.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
"""
MCheckBox
"""
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
# 从 PySide6 模块导入 QtWidgets，用于创建用户界面元素
from PySide6 import QtWidgets

# Import local modules
# 从当前包中的 mixin 模块导入 cursor_mixin
from .mixin import cursor_mixin

# 使用 cursor_mixin 装饰器来扩展 MCheckBox 类的功能
@cursor_mixin
class MCheckBox(QtWidgets.QCheckBox):
    """
    MCheckBox just use stylesheet and set cursor shape when hover. No more extend.
    """

    # 初始化 MCheckBox 类的实例
    def __init__(self, text="", parent=None):
        # 调用父类 QtWidgets.QCheckBox 的初始化方法，设置文本和父级窗口
        super(MCheckBox, self).__init__(text=text, parent=parent)
```