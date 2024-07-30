# `.\comic-translate\app\ui\dayu_widgets\sequence_file.py`

```py
# 指定脚本解释器为 Python，并声明编码方式为 UTF-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 作者信息
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################

# 导入未来版本模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import functools

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets
from dayu_path import DayuPath

# 导入本地模块
from . import dayu_theme
from .check_box import MCheckBox
from .field_mixin import MFieldMixin
from .label import MLabel
from .line_edit import MLineEdit
from .mixin import property_mixin

# 带属性混合类 MSequenceFile，继承自 QWidget 和 MFieldMixin
@property_mixin
class MSequenceFile(QtWidgets.QWidget, MFieldMixin):
    """
    这个类必须依赖 DayuPath
    props:
        path: six.string_types
        sequence: bool
    """

    # 信号声明：当 sequence 属性改变时发射该信号
    sig_is_sequence_changed = QtCore.Signal(bool)

    # 初始化方法
    def __init__(self, size=None, parent=None):
        # 调用父类初始化方法
        super(MSequenceFile, self).__init__(parent)
        # 初始化成员变量
        self.sequence_obj = None
        size = size or dayu_theme.small
        # 文件标签控件，设置 Dayu 样式大小并设置为只读
        self._file_label = MLineEdit()
        self._file_label.set_dayu_size(size)
        self._file_label.setReadOnly(True)
        # 是否序列复选框控件，连接信号到部分函数和自定义信号
        self._is_sequence_check_box = MCheckBox(self.tr("Sequence"))
        self._is_sequence_check_box.toggled.connect(functools.partial(self.setProperty, "sequence"))
        self._is_sequence_check_box.toggled.connect(self.sig_is_sequence_changed)

        # 信息标签控件，设置为次要样式
        self._info_label = MLabel().secondary()
        # 错误标签控件，设置为次要样式并且标记为错误，设置最小宽度和中间省略模式
        self._error_label = MLabel().secondary()
        self._error_label.setProperty("error", True)
        self._error_label.setMinimumWidth(100)
        self._error_label.set_elide_mode(QtCore.Qt.ElideMiddle)

        # 水平布局对象
        seq_lay = QtWidgets.QHBoxLayout()
        # 将控件添加到水平布局中，并设置拉伸因子
        seq_lay.addWidget(self._is_sequence_check_box)
        seq_lay.addWidget(self._info_label)
        seq_lay.addWidget(self._error_label)
        seq_lay.setStretchFactor(self._is_sequence_check_box, 0)
        seq_lay.setStretchFactor(self._info_label, 0)
        seq_lay.setStretchFactor(self._error_label, 100)

        # 垂直布局对象
        self._main_lay = QtWidgets.QVBoxLayout()
        self._main_lay.setContentsMargins(0, 0, 0, 0)
        # 将文件标签和水平布局添加到垂直布局中
        self._main_lay.addWidget(self._file_label)
        self._main_lay.addLayout(seq_lay)
        self.setLayout(self._main_lay)
        # 默认设置为序列文件
        self.set_sequence(True)

    # 私有方法：设置路径
    def _set_path(self, value):
        # 使用 DayuPath 封装路径值
        path = DayuPath(value)
        # 遍历扫描路径中的序列对象
        for seq_obj in path.scan():
            self.sequence_obj = seq_obj
        # 更新信息标签显示内容
        self._update_info()

    # 公共方法：设置路径，设置属性为路径值
    def set_path(self, value):
        self.setProperty("path", value)

    # 公共方法：设置序列属性
    def set_sequence(self, value):
        assert isinstance(value, bool)
        self.setProperty("sequence", value)
    # 设置序列属性的方法，接受一个布尔值作为参数
    def _set_sequence(self, value):
        # 如果传入的值与当前复选框的状态不同
        if value != self._is_sequence_check_box.isChecked():
            # 更新复选框的状态为传入的值
            self._is_sequence_check_box.setChecked(value)
            # 发射信号，通知序列属性改变
            self.sig_is_sequence_changed.emit(value)
        # 更新信息显示
        self._update_info()

    # 更新信息显示的方法
    def _update_info(self):
        # 设置文件标签的文本属性，根据是否具有序列属性选择显示路径或序列对象的路径
        self._file_label.setProperty(
            "text",
            self.sequence_obj if self.property("sequence") else self.property("path"),
        )
        # 如果存在序列对象
        if self.sequence_obj:
            # 设置信息标签的文本内容，显示格式、总数、起始和结束帧范围
            self._info_label.setText(
                "Format: {ext}  "
                "Total: {count}  "
                "Range: {start}-{end}".format(
                    ext=self.sequence_obj.ext,
                    count=len(self.sequence_obj.frames),
                    start=self.sequence_obj.frames[0] if self.sequence_obj.frames else "/",
                    end=self.sequence_obj.frames[-1] if self.sequence_obj.frames else "/",
                )
            )
            # 如果存在缺失帧信息，则显示缺失的帧数
            error_info = "Missing: {}".format(self.sequence_obj.missing) if self.sequence_obj.missing else ""
            # 设置错误标签的文本内容和工具提示信息
            self._error_label.setText(error_info)
            self._error_label.setToolTip(error_info)
        # 根据是否具有序列属性选择是否显示信息标签和错误标签
        self._info_label.setVisible(self.property("sequence"))
        self._error_label.setVisible(self.property("sequence"))
```