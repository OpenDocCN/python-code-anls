# `.\comic-translate\app\ui\dayu_widgets\splitter.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: TimmyLiang
# Date  : 2021.12
# Email : 820472580@qq.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
from functools import partial

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme
from .mixin import property_mixin

@property_mixin
class MSplitter(QtWidgets.QSplitter):
    # 自定义 QSplitter 类 MSplitter，继承自 QtWidgets.QSplitter
    def __init__(self, Orientation=QtCore.Qt.Horizontal, parent=None):
        super(MSplitter, self).__init__(Orientation, parent=parent)
        # 设置分隔条宽度为 10 像素
        self.setHandleWidth(10)
        # 设置属性 "animatable" 为 True，表示支持动画效果
        self.setProperty("animatable", True)
        # 设置属性 "default_size" 的默认值为 100
        self.setProperty("default_size", 100)
        # 设置属性 "anim_move_duration" 表示动画移动的持续时间为 300 毫秒
        self.setProperty("anim_move_duration", 300)
        # 应用 dayu_theme 模块中的主题样式
        dayu_theme.apply(self)

    # 处理分隔条点击事件的槽函数
    def slot_splitter_click(self, index, first=True):
        # 获取当前各个分隔条的尺寸列表
        size_list = self.sizes()
        # 计算当前分隔条索引的前一个索引
        prev = index - 1
        # 获取前一个分隔条的尺寸
        prev_size = size_list[prev]
        # 获取当前分隔条的尺寸
        next_size = size_list[index]
        # 获取属性 "default_size" 的默认值
        default_size = self.property("default_size")

        # 如果前一个分隔条尺寸为 0，则将当前分隔条的尺寸设置为默认值，并调整前一个分隔条的尺寸
        if not prev_size:
            size_list[prev] = default_size
            size_list[index] -= default_size
        # 如果当前分隔条尺寸为 0，则将前一个分隔条的尺寸设置为默认值，并调整当前分隔条的尺寸
        elif not next_size:
            size_list[index] = default_size
            size_list[prev] -= default_size
        else:
            # 如果需要动画效果
            if self.property("animatable"):
                # 创建 QVariantAnimation 对象
                anim = QtCore.QVariantAnimation(self)

                # 定义动画执行时的回调函数
                def anim_size(index, size_list, v):
                    size_list[index - 1] += size_list[index] - v
                    size_list[index] = v
                    self.setSizes(size_list)

                # 将回调函数绑定到动画的值改变信号
                anim.valueChanged.connect(partial(anim_size, index, size_list))
                # 设置动画持续时间
                anim.setDuration(self.property("anim_move_duration"))
                # 设置动画起始值和结束值
                anim.setStartValue(next_size)
                anim.setEndValue(size_list[index])
                # 启动动画
                anim.start()
            else:
                # 如果不需要动画效果，直接设置分隔条尺寸
                self.setSizes(size_list)
    # 定义一个方法用于创建分隔条的控制句柄
    def createHandle(self):
        # 获取当前分隔器中子控件的数量
        count = self.count()

        # 获取分隔器的方向（水平或垂直）
        orient = self.orientation()
        # 判断分隔器是否为水平方向
        is_horizontal = orient is QtCore.Qt.Horizontal
        # 创建一个分隔条句柄对象
        handle = QtWidgets.QSplitterHandle(orient, self)

        # 设置双击分隔条句柄时的事件处理函数，使所有子控件的大小平均分配
        handle.mouseDoubleClickEvent = lambda e: self.setSizes([1 for i in range(self.count())])

        # 根据分隔器的方向选择合适的布局管理器
        layout = QtWidgets.QVBoxLayout() if is_horizontal else QtWidgets.QHBoxLayout()
        # 设置布局的边距为0，以保证按钮紧贴分隔条
        layout.setContentsMargins(0, 0, 0, 0)
        # 将布局管理器应用到分隔条句柄上
        handle.setLayout(layout)

        # 在分隔条句柄上创建一个左/上箭头按钮
        button = QtWidgets.QToolButton(handle)
        button.setArrowType(QtCore.Qt.LeftArrow if is_horizontal else QtCore.Qt.UpArrow)
        # 绑定按钮点击事件的处理函数，用于处理分隔器的点击事件
        button.clicked.connect(lambda: self.slot_splitter_click(count, True))
        layout.addWidget(button)

        # 在分隔条句柄上创建一个右/下箭头按钮
        button = QtWidgets.QToolButton(handle)
        arrow = QtCore.Qt.RightArrow if is_horizontal else QtCore.Qt.DownArrow
        button.setArrowType(arrow)
        # 绑定按钮点击事件的处理函数，用于处理分隔器的点击事件
        button.clicked.connect(lambda: self.slot_splitter_click(count, False))
        layout.addWidget(button)

        # 返回创建好的分隔条句柄对象
        return handle
```