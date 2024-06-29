# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\qt_editor\_formlayout.py`

```py
# 创建一个模块用于生成 Qt 表单对话框/布局，以编辑各种类型的参数

# formLayout 许可协议 (MIT 许可证) 的版权声明和条款

# 版本历史:
# 1.0.10: 添加了浮点数验证器（在不合法时禁用 "Ok" 和 "Apply" 按钮）
# 1.0.7: 增加了对 "Apply" 按钮的支持
# 1.0.6: 代码清理

__version__ = '1.0.10'
__license__ = __doc__

# 导入所需的模块和库
from ast import literal_eval  # 导入 ast 模块的 literal_eval 函数，用于字符串到 Python 对象的安全转换
import copy  # 导入 copy 模块，用于对象的浅拷贝和深拷贝操作
import datetime  # 导入 datetime 模块，处理日期和时间的类
import logging  # 导入 logging 模块，用于记录日志消息
from numbers import Integral, Real  # 从 numbers 模块中导入 Integral 和 Real 类型

from matplotlib import _api, colors as mcolors  # 导入 matplotlib 库的 _api 和 colors 模块
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore  # 导入 matplotlib 的 Qt 兼容模块中的部分内容

_log = logging.getLogger(__name__)  # 获取当前模块的 logger 实例

BLACKLIST = {"title", "label"}  # 定义一个黑名单集合，包含不允许的关键字 "title" 和 "label"

class ColorButton(QtWidgets.QPushButton):
    """
    Color choosing push button
    """
    colorChanged = QtCore.Signal(QtGui.QColor)  # 定义一个颜色变化的信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)  # 设置按钮的固定尺寸为 20x20 像素
        self.setIconSize(QtCore.QSize(12, 12))  # 设置按钮图标的尺寸为 12x12 像素
        self.clicked.connect(self.choose_color)  # 绑定按钮点击事件到 choose_color 方法
        self._color = QtGui.QColor()  # 初始化一个 QColor 对象

    def choose_color(self):
        # 打开颜色选择对话框，初始颜色为 self._color，父窗口为 self.parentWidget()
        color = QtWidgets.QColorDialog.getColor(
            self._color, self.parentWidget(), "",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():  # 如果选择的颜色有效
            self.set_color(color)  # 设置颜色为选择的颜色

    def get_color(self):
        return self._color  # 返回当前按钮的颜色

    @QtCore.Slot(QtGui.QColor)
    def set_color(self, color):
        if color != self._color:  # 如果颜色与当前颜色不同
            self._color = color  # 更新颜色
            self.colorChanged.emit(self._color)  # 发送颜色变化信号
            pixmap = QtGui.QPixmap(self.iconSize())  # 创建一个 QPixmap 对象
            pixmap.fill(color)  # 用选择的颜色填充 QPixmap
            self.setIcon(QtGui.QIcon(pixmap))  # 设置按钮图标为填充后的 QPixmap

    color = QtCore.Property(QtGui.QColor, get_color, set_color)  # 定义颜色属性，使用 get_color 和 set_color 方法

def to_qcolor(color):
    """Create a QColor from a matplotlib color"""
    qcolor = QtGui.QColor()  # 创建一个 QColor 对象
    try:
        rgba = mcolors.to_rgba(color)  # 将 matplotlib 颜色转换为 RGBA 格式
        # 设置 QColor 的 RGBA 值
        qcolor.setRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
    except ValueError:
        _log.warning("Unable to set color %s", color)  # 如果转换失败，记录警告日志
    return qcolor  # 返回转换后的 QColor 对象
    # 如果发生 ValueError 异常，则记录警告消息并忽略掉这个无效的颜色
    except ValueError:
        _api.warn_external(f'Ignoring invalid color {color!r}')
        # 返回一个无效的 QColor 对象
        return qcolor  # return invalid QColor
    # 使用给定的 RGBA 值设置 QColor 对象的颜色
    qcolor.setRgbF(*rgba)
    # 返回已设置好颜色的 QColor 对象
    return qcolor
class ColorLayout(QtWidgets.QHBoxLayout):
    """Color-specialized QLineEdit layout"""
    def __init__(self, color, parent=None):
        super().__init__()
        assert isinstance(color, QtGui.QColor)
        # 创建一个带有特定颜色的QLineEdit，并连接编辑完成信号到update_color方法
        self.lineedit = QtWidgets.QLineEdit(
            mcolors.to_hex(color.getRgbF(), keep_alpha=True), parent)
        self.lineedit.editingFinished.connect(self.update_color)
        # 添加LineEdit到布局中
        self.addWidget(self.lineedit)
        # 创建一个ColorButton，并连接颜色变化信号到update_text方法
        self.colorbtn = ColorButton(parent)
        self.colorbtn.color = color
        self.colorbtn.colorChanged.connect(self.update_text)
        # 添加ColorButton到布局中
        self.addWidget(self.colorbtn)

    def update_color(self):
        # 获取当前文本颜色，转换为QColor对象
        color = self.text()
        qcolor = to_qcolor(color)  # 如果不是有效的QColor对象，默认为黑色
        self.colorbtn.color = qcolor

    def update_text(self, color):
        # 更新LineEdit的文本为颜色的十六进制表示
        self.lineedit.setText(mcolors.to_hex(color.getRgbF(), keep_alpha=True))

    def text(self):
        # 获取当前LineEdit的文本内容
        return self.lineedit.text()


def font_is_installed(font):
    """Check if font is installed"""
    # 返回安装的字体家族列表中是否包含给定的字体
    return [fam for fam in QtGui.QFontDatabase().families()
            if str(fam) == font]


def tuple_to_qfont(tup):
    """
    Create a QFont from tuple:
        (family [string], size [int], italic [bool], bold [bool])
    """
    if not (isinstance(tup, tuple) and len(tup) == 4
            and font_is_installed(tup[0])
            and isinstance(tup[1], Integral)
            and isinstance(tup[2], bool)
            and isinstance(tup[3], bool)):
        return None
    # 根据元组创建一个QFont对象，并设置其属性
    font = QtGui.QFont()
    family, size, italic, bold = tup
    font.setFamily(family)
    font.setPointSize(size)
    font.setItalic(italic)
    font.setBold(bold)
    return font


def qfont_to_tuple(font):
    # 将QFont对象转换为元组形式：(family [string], size [int], italic [bool], bold [bool])
    return (str(font.family()), int(font.pointSize()),
            font.italic(), font.bold())


class FontLayout(QtWidgets.QGridLayout):
    """Font selection"""
    def __init__(self, value, parent=None):
        super().__init__()
        # 将元组转换为QFont对象
        font = tuple_to_qfont(value)
        assert font is not None

        # 字体家族选择框
        self.family = QtWidgets.QFontComboBox(parent)
        self.family.setCurrentFont(font)
        self.addWidget(self.family, 0, 0, 1, -1)

        # 字体大小选择框
        self.size = QtWidgets.QComboBox(parent)
        self.size.setEditable(True)
        sizelist = [*range(6, 12), *range(12, 30, 2), 36, 48, 72]
        size = font.pointSize()
        if size not in sizelist:
            sizelist.append(size)
            sizelist.sort()
        self.size.addItems([str(s) for s in sizelist])
        self.size.setCurrentIndex(sizelist.index(size))
        self.addWidget(self.size, 1, 0)

        # 是否斜体选择框
        self.italic = QtWidgets.QCheckBox(self.tr("Italic"), parent)
        self.italic.setChecked(font.italic())
        self.addWidget(self.italic, 1, 1)

        # 是否粗体选择框
        self.bold = QtWidgets.QCheckBox(self.tr("Bold"), parent)
        self.bold.setChecked(font.bold())
        self.addWidget(self.bold, 1, 2)
    # 获取当前选择的字体对象
    def get_font(self):
        font = self.family.currentFont()
        # 根据界面上的选择设置字体是否倾斜
        font.setItalic(self.italic.isChecked())
        # 根据界面上的选择设置字体是否加粗
        font.setBold(self.bold.isChecked())
        # 根据界面上选择的字体大小设置字体的点大小
        font.setPointSize(int(self.size.currentText()))
        # 将 QFont 对象转换为元组形式并返回
        return qfont_to_tuple(font)
# 定义一个函数用于检查编辑器中的文本是否有效
def is_edit_valid(edit):
    # 获取编辑器中的文本内容
    text = edit.text()
    # 获取编辑器的验证器对象，并使用它来验证文本内容，返回状态和验证器对象
    state = edit.validator().validate(text, 0)[0]
    # 返回文本是否为可接受的浮点数状态
    return state == QtGui.QDoubleValidator.State.Acceptable


# 定义一个表单部件类，继承自 QtWidgets.QWidget
class FormWidget(QtWidgets.QWidget):
    # 定义一个信号，用于更新按钮
    update_buttons = QtCore.Signal()

    # 初始化函数
    def __init__(self, data, comment="", with_margin=False, parent=None):
        """
        Parameters
        ----------
        data : list of (label, value) pairs
            要在表单中编辑的数据，每个数据是一个标签-值对
        comment : str, optional
            注释内容，默认为空字符串
        with_margin : bool, default: False
            如果为 False，则表单元素紧贴部件的边框。
            如果 FormWidget 作为一个部件与其他部件（如 QComboBox）一起使用，这是期望的行为。
            如果 FormWidget 是容器中唯一的部件，例如 QTabWidget 中的一个选项卡，则可能希望有边距。
        parent : QWidget or None
            父部件，默认为 None
        """
        # 调用父类的初始化函数
        super().__init__(parent)
        # 深拷贝数据，以防止原始数据被修改
        self.data = copy.deepcopy(data)
        # 初始化一个空的部件列表
        self.widgets = []
        # 创建一个表单布局对象，并将其设置为当前部件的布局
        self.formlayout = QtWidgets.QFormLayout(self)
        
        # 如果不需要边距，则将表单布局的边距设置为 0
        if not with_margin:
            self.formlayout.setContentsMargins(0, 0, 0, 0)
        
        # 如果提供了注释内容，则在表单布局中添加一个标签显示注释
        if comment:
            self.formlayout.addRow(QtWidgets.QLabel(comment))
            # 添加一个空行
            self.formlayout.addRow(QtWidgets.QLabel(" "))

    # 返回 FormDialog 实例的方法
    def get_dialog(self):
        """Return FormDialog instance"""
        # 获取当前部件的父部件
        dialog = self.parent()
        # 循环直到找到一个类型为 QtWidgets.QDialog 的父部件
        while not isinstance(dialog, QtWidgets.QDialog):
            dialog = dialog.parent()
        # 返回找到的 QDialog 实例
        return dialog
    def get(self):
        # 初始化一个空列表，用于存储字段的值
        valuelist = []
        
        # 遍历 self.data 列表中的每个元素，同时获取索引和元组(label, value)
        for index, (label, value) in enumerate(self.data):
            # 获取当前字段的部件（widget）
            field = self.widgets[index]
            
            # 如果标签为 None，则跳过当前循环（分隔符或注释）
            if label is None:
                # 分隔符或注释，跳过处理
                continue
            
            # 如果 value 转换成 QFont 类型不为 None，则获取字段的字体信息
            elif tuple_to_qfont(value) is not None:
                value = field.get_font()
            
            # 如果 value 是字符串类型或类似颜色的对象，则将字段的文本转换成字符串
            elif isinstance(value, str) or mcolors.is_color_like(value):
                value = str(field.text())
            
            # 如果 value 是列表或元组类型
            elif isinstance(value, (list, tuple)):
                # 获取字段当前选定的索引
                index = int(field.currentIndex())
                
                # 如果 value 的第一个元素是列表或元组，则获取指定索引位置的第一个值
                if isinstance(value[0], (list, tuple)):
                    value = value[index][0]
                else:
                    value = value[index]
            
            # 如果 value 是布尔类型，则获取字段的选中状态
            elif isinstance(value, bool):
                value = field.isChecked()
            
            # 如果 value 是整数类型，则获取字段的值并转换成整数
            elif isinstance(value, Integral):
                value = int(field.value())
            
            # 如果 value 是实数类型，则获取字段的文本并转换成浮点数
            elif isinstance(value, Real):
                value = float(str(field.text()))
            
            # 如果 value 是 datetime.datetime 类型，则获取字段的日期时间值
            elif isinstance(value, datetime.datetime):
                datetime_ = field.dateTime()
                if hasattr(datetime_, "toPyDateTime"):
                    value = datetime_.toPyDateTime()
                else:
                    value = datetime_.toPython()
            
            # 如果 value 是 datetime.date 类型，则获取字段的日期值
            elif isinstance(value, datetime.date):
                date_ = field.date()
                if hasattr(date_, "toPyDate"):
                    value = date_.toPyDate()
                else:
                    value = date_.toPython()
            
            # 否则，将字段的文本转换成 Python 字面量的值
            else:
                value = literal_eval(str(field.text()))
            
            # 将处理后的 value 添加到值列表中
            valuelist.append(value)
        
        # 返回处理后的值列表
        return valuelist
class FormComboWidget(QtWidgets.QWidget):
    # 定义信号，用于通知更新按钮状态
    update_buttons = QtCore.Signal()

    def __init__(self, datalist, comment="", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # 创建下拉框组件
        self.combobox = QtWidgets.QComboBox()
        layout.addWidget(self.combobox)

        # 创建堆叠窗口组件
        self.stackwidget = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stackwidget)
        
        # 当下拉框选择项变化时，切换堆叠窗口当前显示的组件
        self.combobox.currentIndexChanged.connect(
            self.stackwidget.setCurrentIndex)

        self.widgetlist = []
        # 遍历数据列表，创建下拉框项和对应的窗口组件
        for data, title, comment in datalist:
            self.combobox.addItem(title)
            widget = FormWidget(data, comment=comment, parent=self)
            self.stackwidget.addWidget(widget)
            self.widgetlist.append(widget)

    # 执行所有窗口组件的设置操作
    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    # 获取所有窗口组件的数据并返回列表
    def get(self):
        return [widget.get() for widget in self.widgetlist]


class FormTabWidget(QtWidgets.QWidget):
    # 定义信号，用于通知更新按钮状态
    update_buttons = QtCore.Signal()

    def __init__(self, datalist, comment="", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.tabwidget = QtWidgets.QTabWidget()
        layout.addWidget(self.tabwidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.widgetlist = []
        # 遍历数据列表，根据数据长度选择创建下拉框组件或普通窗口组件
        for data, title, comment in datalist:
            if len(data[0]) == 3:
                widget = FormComboWidget(data, comment=comment, parent=self)
            else:
                widget = FormWidget(data, with_margin=True, comment=comment,
                                    parent=self)
            # 将创建的组件添加到选项卡中，并设置选项卡的提示信息
            index = self.tabwidget.addTab(widget, title)
            self.tabwidget.setTabToolTip(index, comment)
            self.widgetlist.append(widget)

    # 执行所有窗口组件的设置操作
    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    # 获取所有窗口组件的数据并返回列表
    def get(self):
        return [widget.get() for widget in self.widgetlist]


class FormDialog(QtWidgets.QDialog):
    """Form Dialog"""
    def __init__(self, data, title="", comment="",
                 icon=None, parent=None, apply=None):
        super().__init__(parent)

        self.apply_callback = apply  # 设置对象的回调函数

        # Form
        if isinstance(data[0][0], (list, tuple)):
            self.formwidget = FormTabWidget(data, comment=comment,
                                            parent=self)  # 创建多标签页表单部件
        elif len(data[0]) == 3:
            self.formwidget = FormComboWidget(data, comment=comment,
                                              parent=self)  # 创建下拉框表单部件
        else:
            self.formwidget = FormWidget(data, comment=comment,
                                         parent=self)  # 创建基本表单部件
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.formwidget)  # 将表单部件添加到垂直布局中

        self.float_fields = []  # 初始化浮点数字段列表
        self.formwidget.setup()  # 设置表单部件

        # Button box
        self.bbox = bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton(
                    _to_int(QtWidgets.QDialogButtonBox.StandardButton.Ok) |
                    _to_int(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            ))  # 创建对话框按钮盒子，并设置包含 Ok 和 Cancel 按钮
        self.formwidget.update_buttons.connect(self.update_buttons)  # 连接表单部件的按钮更新信号到 update_buttons 方法
        if self.apply_callback is not None:
            apply_btn = bbox.addButton(
                QtWidgets.QDialogButtonBox.StandardButton.Apply)  # 添加应用按钮到按钮盒子
            apply_btn.clicked.connect(self.apply)  # 连接应用按钮的点击信号到 apply 方法

        bbox.accepted.connect(self.accept)  # 连接按钮盒子的接受信号到 accept 方法
        bbox.rejected.connect(self.reject)  # 连接按钮盒子的拒绝信号到 reject 方法
        layout.addWidget(bbox)  # 将按钮盒子添加到垂直布局中

        self.setLayout(layout)  # 设置窗口布局为垂直布局

        self.setWindowTitle(title)  # 设置窗口标题
        if not isinstance(icon, QtGui.QIcon):
            icon = QtWidgets.QWidget().style().standardIcon(
                QtWidgets.QStyle.SP_MessageBoxQuestion)  # 如果图标不是 QIcon 类型，则使用默认问题图标
        self.setWindowIcon(icon)  # 设置窗口图标

    def register_float_field(self, field):
        self.float_fields.append(field)  # 将浮点数字段注册到浮点数字段列表中

    def update_buttons(self):
        valid = True
        for field in self.float_fields:
            if not is_edit_valid(field):
                valid = False  # 检查所有浮点数字段是否有效
        for btn_type in ["Ok", "Apply"]:
            btn = self.bbox.button(
                getattr(QtWidgets.QDialogButtonBox.StandardButton,
                        btn_type))
            if btn is not None:
                btn.setEnabled(valid)  # 根据有效性设置 Ok 和 Apply 按钮的状态

    def accept(self):
        self.data = self.formwidget.get()  # 获取表单部件的数据
        self.apply_callback(self.data)  # 调用回调函数并传递数据
        super().accept()  # 调用父类的 accept 方法

    def reject(self):
        self.data = None  # 设置数据为 None
        super().reject()  # 调用父类的 reject 方法

    def apply(self):
        self.apply_callback(self.formwidget.get())  # 调用回调函数并传递表单部件的数据

    def get(self):
        """Return form result"""
        return self.data  # 返回表单的结果
def fedit(data, title="", comment="", icon=None, parent=None, apply=None):
    """
    Create form dialog

    data: datalist, datagroup
        - datalist: list/tuple of (field_name, field_value)
        - datagroup: list/tuple of (datalist *or* datagroup, title, comment)

    title: str
        - Title of the form dialog

    comment: str
        - Additional comment or description for the form dialog

    icon: QIcon instance
        - Optional icon for the form dialog

    parent: parent QWidget
        - Parent widget for the form dialog

    apply: apply callback (function)
        - Callback function to be executed when 'Apply' button is pressed

    -> Generates fields and tabs based on the structure of 'data'

    Supported types for field_value:
      - int, float, str, bool
      - colors: in Qt-compatible text form, i.e. in hex format or name
                (red, ...) (automatically detected from a string)
      - list/tuple:
          * the first element will be the selected index (or value)
          * the other elements can be couples (key, value) or only values
    """

    # Create a QApplication instance if no instance currently exists
    # (e.g., if the module is used directly from the interpreter)
    if QtWidgets.QApplication.startingUp():
        _app = QtWidgets.QApplication([])

    # Create a FormDialog instance with the provided parameters
    dialog = FormDialog(data, title, comment, icon, parent, apply)

    # If a parent widget is provided and it has a '_fedit_dialog' attribute,
    # close any existing '_fedit_dialog' associated with the parent
    if parent is not None:
        if hasattr(parent, "_fedit_dialog"):
            parent._fedit_dialog.close()
        # Assign the created dialog to '_fedit_dialog' attribute of the parent
        parent._fedit_dialog = dialog

    # Display the dialog
    dialog.show()


if __name__ == "__main__":

    _app = QtWidgets.QApplication([])

    def create_datalist_example():
        return [('str', 'this is a string'),
                ('list', [0, '1', '3', '4']),
                ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
                           ('-.', 'DashDot'), ('-', 'Solid'),
                           ('steps', 'Steps'), (':', 'Dotted')]),
                ('float', 1.2),
                (None, 'Other:'),
                ('int', 12),
                ('font', ('Arial', 10, False, True)),
                ('color', '#123409'),
                ('bool', True),
                ('date', datetime.date(2010, 10, 10)),
                ('datetime', datetime.datetime(2010, 10, 10)),
                ]

    def create_datagroup_example():
        datalist = create_datalist_example()
        return ((datalist, "Category 1", "Category 1 comment"),
                (datalist, "Category 2", "Category 2 comment"),
                (datalist, "Category 3", "Category 3 comment"))

    # --------- datalist example
    datalist = create_datalist_example()

    def apply_test(data):
        print("data:", data)
    
    # Open a form dialog with 'datalist' data and additional parameters
    fedit(datalist, title="Example",
          comment="This is just an <b>example</b>.",
          apply=apply_test)

    _app.exec()

    # --------- datagroup example
    datagroup = create_datagroup_example()
    
    # Open a form dialog with 'datagroup' data and 'Global title' as title
    fedit(datagroup, "Global title",
          apply=apply_test)
    
    _app.exec()

    # --------- datagroup inside a datagroup example
    # 创建一个数据列表，使用函数 create_datalist_example()
    datalist = create_datalist_example()
    # 创建一个数据组，使用函数 create_datagroup_example()
    datagroup = create_datagroup_example()
    # 调用函数 fedit()，编辑多个数据组及其标题和注释
    fedit(((datagroup, "Title 1", "Tab 1 comment"),
           (datalist, "Title 2", "Tab 2 comment"),
           (datalist, "Title 3", "Tab 3 comment")),
          "Global title",  # 设置全局标题为 "Global title"
          apply=apply_test)  # 传递一个函数 apply_test 作为参数 apply
    # 执行应用程序的主事件循环
    _app.exec()
```