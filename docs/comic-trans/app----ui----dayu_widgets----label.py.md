# `.\comic-translate\app\ui\dayu_widgets\label.py`

```py
# 导入未来版本模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore, QtGui
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme

# 定义一个继承自QtWidgets.QLabel的MLabel类，用于显示不同级别的标题
class MLabel(QtWidgets.QLabel):
    """
    Display title in different level.
    Property:
        dayu_level: integer   # 表示标签的级别
        dayu_type: str        # 表示标签的类型
    """

    # 定义类属性，表示不同的标签类型和级别
    SecondaryType = "secondary"
    WarningType = "warning"
    DangerType = "danger"
    H1Level = 1
    H2Level = 2
    H3Level = 3
    H4Level = 4

    def __init__(self, text="", parent=None, flags=QtCore.Qt.Widget):
        super(MLabel, self).__init__(text, parent, flags)
        # 设置标签的文本交互模式和大小策略
        self.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction | QtCore.Qt.LinksAccessibleByMouse)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # 初始化标签的私有属性
        self._dayu_type = ""
        self._dayu_underline = False
        self._dayu_mark = False
        self._dayu_delete = False
        self._dayu_strong = False
        self._dayu_code = False
        self._dayu_border = False
        self._dayu_level = 0
        self._elide_mode = QtCore.Qt.ElideNone
        # 设置标签的自定义属性"dayu_text"为传入的文本内容
        self.setProperty("dayu_text", text)

    def get_dayu_level(self):
        """Get MLabel level."""
        return self._dayu_level

    def set_dayu_level(self, value):
        """Set MLabel level"""
        self._dayu_level = value
        self.style().polish(self)

    def set_dayu_underline(self, value):
        """Set MLabel underline style."""
        self._dayu_underline = value
        self.style().polish(self)

    def get_dayu_underline(self):
        return self._dayu_underline

    def set_dayu_delete(self, value):
        """Set MLabel a delete line style."""
        self._dayu_delete = value
        self.style().polish(self)

    def get_dayu_delete(self):
        return self._dayu_delete

    def set_dayu_strong(self, value):
        """Set MLabel bold style."""
        self._dayu_strong = value
        self.style().polish(self)

    def get_dayu_strong(self):
        return self._dayu_strong

    def set_dayu_mark(self, value):
        """Set MLabel mark style."""
        self._dayu_mark = value
        self.style().polish(self)

    def get_dayu_mark(self):
        return self._dayu_mark

    def set_dayu_code(self, value):
        """Set MLabel code style."""
        self._dayu_code = value
        self.style().polish(self)

    def get_dayu_code(self):
        return self._dayu_code
    
    def set_dayu_border(self, value):
        """Set MLabel border style."""
        self._dayu_border = value
        self.style().polish(self)
    def get_dayu_border(self):
        return self._dayu_border

# 返回当前对象的 _dayu_border 属性值


    def get_elide_mode(self):
        return self._elide_mode

# 返回当前对象的 _elide_mode 属性值


    def set_elide_mode(self, value):
        """Set MLabel elide mode.
        Only accepted Qt.ElideLeft/Qt.ElideMiddle/Qt.ElideRight/Qt.ElideNone"""
        # 设置 MLabel 的文本省略模式，仅接受 Qt.ElideLeft/Qt.ElideMiddle/Qt.ElideRight/Qt.ElideNone
        self._elide_mode = value
        # 更新标签的省略文本
        self._update_elided_text()



    def get_dayu_type(self):
        return self._dayu_type

# 返回当前对象的 _dayu_type 属性值


    def set_dayu_type(self, value):
        self._dayu_type = value
        # 刷新当前对象的样式
        self.style().polish(self)



    dayu_level = QtCore.Property(int, get_dayu_level, set_dayu_level)
    dayu_type = QtCore.Property(str, get_dayu_type, set_dayu_type)
    dayu_underline = QtCore.Property(bool, get_dayu_underline, set_dayu_underline)
    dayu_delete = QtCore.Property(bool, get_dayu_delete, set_dayu_delete)
    dayu_strong = QtCore.Property(bool, get_dayu_strong, set_dayu_strong)
    dayu_mark = QtCore.Property(bool, get_dayu_mark, set_dayu_mark)
    dayu_code = QtCore.Property(bool, get_dayu_code, set_dayu_code)
    dayu_border = QtCore.Property(bool, get_dayu_border, set_dayu_border)
    dayu_elide_mod = QtCore.Property(QtCore.Qt.TextElideMode, get_dayu_code, set_dayu_code)

# 定义多个 QtCore 属性，分别绑定到不同的 getter 和 setter 方法，用于 MLabel 类的不同属性


    def minimumSizeHint(self):
        return QtCore.QSize(1, self.fontMetrics().height())

# 返回一个建议的最小尺寸，高度基于当前字体的字体度量值


    def text(self):
        """
        Overridden base method to return the original unmodified text

        :returns:   The original unmodified text
        """
        return self.property("text")

# 重写基类方法，返回未修改的原始文本内容


    def setText(self, text):
        """
        Overridden base method to set the text on the label

        :param text:    The text to set on the label
        """
        self.setProperty("text", text)
        # 更新省略文本和工具提示
        self._update_elided_text()
        self.setToolTip(text)

# 重写基类方法，设置标签的文本内容，并更新省略文本和工具提示


    def set_link(self, href, text=None):
        """
        :param href: The href attr of a tag
        :param text: The a tag text content
        """
        # 设置富文本超链接样式
        link_style = dayu_theme.hyperlink_style
        self.setText('{style}<a href="{href}">{text}</a>'.format(style=link_style, href=href, text=text or href))
        self.setOpenExternalLinks(True)

# 设置标签的文本内容为包含超链接的富文本，使用给定的 href 和 text 参数，同时启用外部链接打开功能


    def _update_elided_text(self):
        """
        Update the elided text on the label
        """
        _font_metrics = self.fontMetrics()
        text = self.property("text")
        text = text if text else ""
        # 计算并更新省略文本
        _elided_text = _font_metrics.elidedText(text, self._elide_mode, self.width() - 2 * 2)
        super(MLabel, self).setText(_elided_text)

# 更新标签上的省略文本，根据当前的省略模式和宽度计算


    def resizeEvent(self, event):
        """
        Overridden base method called when the widget is resized.

        :param event:    The resize event
        """
        # 在小部件大小改变时调用，更新省略文本
        self._update_elided_text()

# 重写基类方法，处理小部件大小改变事件，更新标签上的省略文本


    def h1(self):
        """Set QLabel with h1 type."""
        # 设置标签为 h1 类型
        self.set_dayu_level(MLabel.H1Level)
        return self

# 设置标签为 h1 类型，返回当前对象以支持链式调用


    def h2(self):
        """Set QLabel with h2 type."""
        # 设置标签为 h2 类型
        self.set_dayu_level(MLabel.H2Level)
        return self

# 设置标签为 h2 类型，返回当前对象以支持链式调用
    def h3(self):
        """
        设置 QLabel 的样式为 h3 类型。
        """
        self.set_dayu_level(MLabel.H3Level)
        return self

    def h4(self):
        """
        设置 QLabel 的样式为 h4 类型。
        """
        self.set_dayu_level(MLabel.H4Level)
        return self

    def secondary(self):
        """
        设置 QLabel 的样式为 secondary 类型。
        """
        self.set_dayu_type(MLabel.SecondaryType)
        return self

    def warning(self):
        """
        设置 QLabel 的样式为 warning 类型。
        """
        self.set_dayu_type(MLabel.WarningType)
        return self

    def danger(self):
        """
        设置 QLabel 的样式为 danger 类型。
        """
        self.set_dayu_type(MLabel.DangerType)
        return self

    def strong(self):
        """
        设置 QLabel 的样式为 strong 样式。
        """
        self.set_dayu_strong(True)
        return self

    def mark(self):
        """
        设置 QLabel 的样式为 mark 样式。
        """
        self.set_dayu_mark(True)
        return self

    def code(self):
        """
        设置 QLabel 的样式为 code 样式。
        """
        self.set_dayu_code(True)
        return self
    
    def border(self):
        """
        设置 QLabel 的样式为 border 样式。
        """
        self.set_dayu_border(True)
        return self

    def delete(self):
        """
        设置 QLabel 的样式为 delete 样式。
        """
        self.set_dayu_delete(True)
        return self

    def underline(self):
        """
        设置 QLabel 的样式为 underline 样式。
        """
        self.set_dayu_underline(True)
        return self

    def event(self, event):
        """
        处理 QLabel 的事件，更新文本内容。
        """
        if event.type() == QtCore.QEvent.DynamicPropertyChange and event.propertyName() == "dayu_text":
            self.setText(self.property("dayu_text"))
        return super(MLabel, self).event(event)
```