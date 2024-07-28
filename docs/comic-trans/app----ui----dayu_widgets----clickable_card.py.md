# `.\comic-translate\app\ui\dayu_widgets\clickable_card.py`

```py
# 导入未来的模块特性，确保向后兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme  # 导入名为dayu_theme的本地模块
from .avatar import MAvatar  # 从avatar模块导入MAvatar类
from .divider import MDivider  # 从divider模块导入MDivider类
from .label import MLabel  # 从label模块导入MLabel类
from .mixin import cursor_mixin  # 从mixin模块导入cursor_mixin装饰器
from .mixin import hover_shadow_mixin  # 从mixin模块导入hover_shadow_mixin装饰器
from .tool_button import MToolButton  # 从tool_button模块导入MToolButton类

# 使用hover_shadow_mixin和cursor_mixin装饰器装饰的类，提供悬浮效果和光标样式
@hover_shadow_mixin
@cursor_mixin
class ClickCard(QtWidgets.QWidget):
    # 点击信号
    clicked = QtCore.Signal()
    # 额外按钮点击信号
    extra_button_clicked = QtCore.Signal()

    # 初始化方法
    def __init__(self, title=None, image=None, size=None, extra=None, type=None, parent=None):
        # 调用父类的初始化方法
        super(ClickCard, self).__init__(parent=parent)
        # 设置QWidget的样式属性，使其支持样式背景
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        # 设置属性"border"为False，可能是为了取消边框
        self.setProperty("border", False)
        # 如果未提供size，则使用默认的dayu_theme.default_size
        size = size or dayu_theme.default_size
        # 根据不同的size设置标题标签的样式级别和填充
        map_label = {
            dayu_theme.large: (MLabel.H2Level, 20),
            dayu_theme.medium: (MLabel.H3Level, 15),
            dayu_theme.small: (MLabel.H4Level, 10),
        }
        # 创建标题标签，并设置文本为title
        self._title_label = MLabel(text=title)
        # 设置标题标签的dayu级别
        self._title_label.set_dayu_level(map_label.get(size)[0])
        # 设置标题标签对鼠标事件透明
        self._title_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # 根据size确定填充值
        padding = map_label.get(size)[-1]
        # 创建标题布局为水平布局
        self._title_layout = QtWidgets.QHBoxLayout()
        # 设置标题布局的边距
        self._title_layout.setContentsMargins(padding, padding, padding, padding)
        # 如果提供了image，则创建头像控件，并设置头像图像和大小
        if image:
            self._title_icon = MAvatar()
            self._title_icon.set_dayu_image(image)
            self._title_icon.set_dayu_size(size)
            # 设置头像控件对鼠标事件透明
            self._title_icon.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            # 将头像控件添加到标题布局中
            self._title_layout.addWidget(self._title_icon)
        # 将标题标签添加到标题布局中
        self._title_layout.addWidget(self._title_label)
        # 添加伸缩控件到标题布局中，用于布局灵活性

        # 如果提供了extra，则创建额外按钮，并连接点击信号到槽函数_on_extra_button_clicked
        self._extra_button = None
        if extra:
            self._extra_button = MToolButton().icon_only().svg("more.svg")
            self._extra_button.clicked.connect(self._on_extra_button_clicked)
            # 将额外按钮添加到标题布局中
            self._title_layout.addWidget(self._extra_button)

        # 创建内容布局为垂直布局
        self._content_layout = QtWidgets.QVBoxLayout()

        # 创建主布局为垂直布局，并设置间距和内容边距
        self._main_lay = QtWidgets.QVBoxLayout()
        self._main_lay.setSpacing(0)
        self._main_lay.setContentsMargins(1, 1, 1, 1)
        # 如果提供了title，则将标题布局和分隔线添加到主布局中
        if title:
            self._main_lay.addLayout(self._title_layout)
            self._main_lay.addWidget(MDivider())
        # 将内容布局添加到主布局中
        self._main_lay.addLayout(self._content_layout)
        # 设置当前Widget的布局为主布局
        self.setLayout(self._main_lay)

    # 获取额外按钮的方法
    def get_more_button(self):
        return self._extra_button

    # 额外按钮点击时的槽函数，发射extra_button_clicked信号
    def _on_extra_button_clicked(self):
        self.extra_button_clicked.emit()
    # 将给定的 widget 设置为对鼠标事件透明
    def set_widget(self, widget):
        widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # 将新的 widget 对鼠标事件设置为透明
        self._content_layout.addWidget(widget)

    # 设置边框属性并更新样式
    def border(self):
        self.setProperty("border", True)  # 设置属性 "border" 为 True
        self.style().polish(self)  # 更新样式以应用属性更改
        return self  # 返回自身引用

    # 处理鼠标按下事件
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # 检查点击是否在额外按钮上（如果存在）
            if self._extra_button and self._extra_button.geometry().contains(event.pos()):
                # 将事件交给额外按钮处理
                return
            self.clicked.emit()  # 发射 clicked 信号
        super(ClickCard, self).mousePressEvent(event)  # 调用父类的鼠标按下事件处理方法
@hover_shadow_mixin
@cursor_mixin
class ClickMeta(QtWidgets.QWidget):
    # 定义信号，当主按钮被点击时触发
    clicked = QtCore.Signal()
    # 定义信号，当额外按钮被点击时触发
    extra_button_clicked = QtCore.Signal()

    def __init__(
        self,
        cover=None,
        avatar=None,
        title=None,
        description=None,
        extra=False,
        parent=None,
        avatar_size=()
    ):
        # 调用父类的初始化方法
        super(ClickMeta, self).__init__(parent)
        # 设置该窗口的样式属性，使其支持样式背景
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        # 初始化各个控件
        self._cover_label = QtWidgets.QLabel()
        self._avatar = MAvatar()
        self._title_label = MLabel().secondary()
        self._description_label = MLabel().secondary()
        self._description_label.setWordWrap(True)  # 设置描述标签支持换行显示
        self._description_label.set_elide_mode(QtCore.Qt.ElideRight)  # 设置描述标签的省略模式

        self._title_layout = QtWidgets.QHBoxLayout()
        self._title_layout.addWidget(self._title_label)
        self._title_layout.addStretch()  # 在标题布局中添加伸展因子，用于调整布局

        self._extra = extra
        self._extra_button = MToolButton(parent=self).icon_only().svg("more.svg")  # 创建一个额外按钮，使用SVG图标
        self._title_layout.addWidget(self._extra_button)
        self._extra_button.setVisible(extra)  # 根据额外按钮的显示状态设置其可见性
        if self._extra:
            self._extra_button.clicked.connect(self._on_extra_button_clicked)  # 如果额外按钮存在，则连接其点击事件到处理函数

        content_lay = QtWidgets.QVBoxLayout()
        content_lay.addLayout(self._title_layout)
        content_lay.addWidget(self._description_label)

        avatar_layout = QtWidgets.QVBoxLayout()
        avatar_layout.addStretch()
        avatar_layout.addWidget(self._avatar)
        avatar_layout.addStretch()

        avatar_content_layout = QtWidgets.QHBoxLayout()
        avatar_content_layout.addSpacing(2)
        avatar_content_layout.addLayout(avatar_layout)
        avatar_content_layout.addSpacing(3)
        avatar_content_layout.addLayout(content_lay)

        self._button_layout = QtWidgets.QHBoxLayout()

        main_lay = QtWidgets.QVBoxLayout()
        main_lay.setSpacing(0)
        main_lay.setContentsMargins(1, 1, 1, 1)
        main_lay.addWidget(self._cover_label)
        main_lay.addLayout(avatar_content_layout)
        main_lay.addLayout(self._button_layout)
        self.setLayout(main_lay)
        self._cover_label.setFixedSize(QtCore.QSize(200, 200))  # 设置封面标签的固定大小
        # 设置头像的固定大小
        if avatar_size:
            w, h = avatar_size
            self._avatar.setFixedSize(QtCore.QSize(w, h))
        # 使得所有控件对于鼠标事件透明（除了额外按钮）
        self._make_widgets_transparent()

    def _make_widgets_transparent(self):
        # 设置所有子控件对于鼠标事件的透明度，除了额外按钮
        for widget in self.findChildren(QtWidgets.QWidget):
            if widget != self._extra_button:
                widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def _on_extra_button_clicked(self):
        # 当额外按钮被点击时，触发额外按钮点击信号
        self.extra_button_clicked.emit()
    # 检查数据字典中是否包含键 "title"
    has_title = data_dict.get("title") is not None
    # 检查数据字典中是否包含键 "description"
    has_description = data_dict.get("description") is not None

    # 如果存在标题，则设置标题标签的文本并显示
    if has_title:
        self._title_label.setText(data_dict.get("title"))
        self._title_label.setVisible(True)
    else:
        # 否则隐藏标题标签
        self._title_label.setVisible(False)

    # 如果存在描述，则设置描述标签的文本并显示
    if has_description:
        self._description_label.setText(data_dict.get("description"))
        self._description_label.setVisible(True)
    else:
        # 否则隐藏描述标签
        self._description_label.setVisible(False)
    
    # 如果数据字典中存在 "avatar" 键，则设置 avatar 图像并显示
    if data_dict.get("avatar"):
        self._avatar.set_dayu_image(data_dict.get("avatar"))
        self._avatar.setVisible(True)
    else:
        # 否则隐藏 avatar 图像
        self._avatar.setVisible(False)

    # 如果数据字典中存在 "cover" 键，则设置封面图片，并根据固定宽度调整大小并显示
    if data_dict.get("cover"):
        fixed_height = self._cover_label.width()
        self._cover_label.setPixmap(
            data_dict.get("cover").scaledToWidth(fixed_height, QtCore.Qt.SmoothTransformation)
        )
        self._cover_label.setVisible(True)
    else:
        # 否则隐藏封面图片
        self._cover_label.setVisible(False)

    # 如果数据字典中存在 "clicked" 键，并且其值为可调用对象，则连接到 clicked 信号
    if "clicked" in data_dict and callable(data_dict["clicked"]):
        self.connect_clicked(data_dict["clicked"])

    # 如果数据字典中存在 "extra_clicked" 键，并且其值为可调用对象，则连接到 extra_clicked 信号
    if "extra_clicked" in data_dict and callable(data_dict["extra_clicked"]):
        self.connect_extra_clicked(data_dict["extra_clicked"])

def connect_clicked(self, func):
    # 连接 clicked 信号到给定的函数
    self.clicked.connect(func)

def connect_extra_clicked(self, func):
    # 如果存在额外按钮，则将其 clicked 信号连接到给定的函数，并显示额外按钮
    if self._extra:
        self._extra_button.clicked.connect(func)
        self._extra_button.setVisible(True)

def mousePressEvent(self, event):
    # 检查鼠标事件是否是左键点击
    if event.button() == QtCore.Qt.LeftButton:
        # 如果额外按钮存在且可见，并且点击位置在额外按钮范围内，则让额外按钮处理事件
        if self._extra_button.isVisible() and self._extra_button.geometry().contains(event.pos()):
            return
        # 否则发射 clicked 信号
        self.clicked.emit()
    # 调用父类的鼠标点击事件处理函数
    super(ClickMeta, self).mousePressEvent(event)

def set_highlight(self, highlighted):
    # 获取当前背景色
    current_color = self.palette().color(self.backgroundRole())
    
    # 创建稍微较暗的高亮颜色
    highlight_color = current_color.darker(130)
    
    # 如果需要高亮，则设置样式表为高亮颜色背景；否则清空样式表
    if highlighted:
        self.setStyleSheet(f"background-color: {highlight_color.name()};")
    else:
        self.setStyleSheet("")
    # 更新界面显示
    self.update()
```