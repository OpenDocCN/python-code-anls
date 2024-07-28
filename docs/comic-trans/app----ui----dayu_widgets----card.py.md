# `.\comic-translate\app\ui\dayu_widgets\card.py`

```py
# 导入未来模块，确保代码在Python 2和Python 3上的兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入PySide6的核心和控件模块
from PySide6 import QtCore
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme          # 导入自定义的dayu主题模块
from .avatar import MAvatar       # 导入MAvatar类
from .divider import MDivider     # 导入MDivider类
from .label import MLabel         # 导入MLabel类
from .mixin import cursor_mixin   # 导入cursor_mixin混合类
from .mixin import hover_shadow_mixin  # 导入hover_shadow_mixin混合类
from .tool_button import MToolButton  # 导入MToolButton类

# MCard类，继承自QtWidgets.QWidget类，具有阴影效果和鼠标指针特性
@hover_shadow_mixin               # 使用hover_shadow_mixin混合类装饰器
@cursor_mixin                     # 使用cursor_mixin混合类装饰器
class MCard(QtWidgets.QWidget):   # 定义MCard类，表示卡片窗口部件
    def __init__(self, title=None, image=None, size=None, extra=None, type=None, parent=None):
        super(MCard, self).__init__(parent=parent)  # 调用父类QWidget的构造函数
        self.setAttribute(QtCore.Qt.WA_StyledBackground)  # 设置样式背景属性
        self.setProperty("border", False)          # 设置属性"border"为False，不显示边框
        size = size or dayu_theme.default_size     # 如果size为None，则使用默认尺寸

        # 根据不同的size选择合适的标题标签和内边距
        map_label = {
            dayu_theme.large: (MLabel.H2Level, 20),
            dayu_theme.medium: (MLabel.H3Level, 15),
            dayu_theme.small: (MLabel.H4Level, 10),
        }
        self._title_label = MLabel(text=title)     # 创建标题标签，并设置文本
        self._title_label.set_dayu_level(map_label.get(size)[0])  # 设置标题标签的dayu级别

        padding = map_label.get(size)[-1]          # 获取内边距
        self._title_layout = QtWidgets.QHBoxLayout()  # 创建水平布局对象_title_layout
        self._title_layout.setContentsMargins(padding, padding, padding, padding)  # 设置布局的边距
        if image:
            self._title_icon = MAvatar()           # 创建头像部件_title_icon
            self._title_icon.set_dayu_image(image)  # 设置头像的dayu图片
            self._title_icon.set_dayu_size(size)    # 设置头像的dayu大小
            self._title_layout.addWidget(self._title_icon)  # 将头像部件添加到标题布局中
        self._title_layout.addWidget(self._title_label)   # 将标题标签添加到标题布局中
        self._title_layout.addStretch()         # 添加伸缩空间
        if extra:
            self._extra_button = MToolButton().icon_only().svg("more.svg")  # 创建图标按钮_extra_button
            self._title_layout.addWidget(self._extra_button)   # 将图标按钮添加到标题布局中

        self._content_layout = QtWidgets.QVBoxLayout()   # 创建垂直布局对象_content_layout

        self._main_lay = QtWidgets.QVBoxLayout()   # 创建垂直布局对象_main_lay
        self._main_lay.setSpacing(0)               # 设置布局间距为0
        self._main_lay.setContentsMargins(1, 1, 1, 1)   # 设置布局的边距
        if title:
            self._main_lay.addLayout(self._title_layout)  # 如果有标题，则将标题布局添加到主布局中
            self._main_lay.addWidget(MDivider())    # 添加分隔线部件
        self._main_lay.addLayout(self._content_layout)   # 将内容布局添加到主布局中
        self.setLayout(self._main_lay)         # 设置当前窗口部件的布局为_main_lay

    def get_more_button(self):
        return self._extra_button         # 返回更多按钮_extra_button

    def set_widget(self, widget):
        self._content_layout.addWidget(widget)   # 向内容布局中添加指定的部件widget

    def border(self):
        self.setProperty("border", True)    # 设置属性"border"为True，显示边框
        self.style().polish(self)           # 更新样式
        return self
    ):
        # 调用父类的构造函数初始化窗口部件
        super(MMeta, self).__init__(parent)
        # 设置窗口部件的样式背景属性
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        # 创建一个 QLabel 作为封面图标
        self._cover_label = QtWidgets.QLabel()
        # 创建一个 MAvatar 对象作为头像
        self._avatar = MAvatar()
        # 创建一个 MLabel 对象作为标题标签，并设置为 H4 大小
        self._title_label = MLabel().h4()
        # 创建一个 MLabel 对象作为描述标签，并设置为次要文本样式
        self._description_label = MLabel().secondary()
        # 设置描述标签可以换行显示
        self._description_label.setWordWrap(True)
        # 设置描述标签的文本裁剪模式为右端省略
        self._description_label.set_elide_mode(QtCore.Qt.ElideRight)
        # 创建水平布局管理器用于标题部分
        self._title_layout = QtWidgets.QHBoxLayout()
        # 将标题标签添加到标题布局中
        self._title_layout.addWidget(self._title_label)
        # 添加一个伸缩空间到标题布局中，使得标题可以居中显示
        self._title_layout.addStretch()
        # 创建一个 MToolButton 作为额外按钮，默认隐藏
        self._extra_button = MToolButton(parent=self).icon_only().svg("more.svg")
        self._title_layout.addWidget(self._extra_button)
        # 根据参数决定是否显示额外按钮
        self._extra_button.setVisible(extra)

        # 创建表单布局管理器用于内容部分
        content_lay = QtWidgets.QFormLayout()
        content_lay.setContentsMargins(5, 5, 5, 5)
        # 将头像和标题布局添加到内容布局中
        content_lay.addRow(self._avatar, self._title_layout)
        # 将描述标签添加到内容布局中
        content_lay.addRow(self._description_label)

        # 创建水平布局管理器用于按钮部分
        self._button_layout = QtWidgets.QHBoxLayout()

        # 创建垂直布局管理器用于主要布局
        main_lay = QtWidgets.QVBoxLayout()
        main_lay.setSpacing(0)
        main_lay.setContentsMargins(1, 1, 1, 1)
        # 将封面标签添加到主要布局中
        main_lay.addWidget(self._cover_label)
        # 将内容布局添加到主要布局中
        main_lay.addLayout(content_lay)
        # 将按钮布局添加到主要布局中
        main_lay.addLayout(self._button_layout)
        # 添加伸缩空间到主要布局中，使得界面可以自适应大小
        main_lay.addStretch()
        # 设置窗口部件的整体布局为主要布局
        self.setLayout(main_lay)
        # 设置封面标签的固定大小为正方形 200x200
        self._cover_label.setFixedSize(QtCore.QSize(200, 200))
        # self.setFixedWidth(200)

    def get_more_button(self):
        # 返回额外按钮对象的引用
        return self._extra_button

    def setup_data(self, data_dict):
        # 根据传入的数据字典设置界面的显示内容
        if data_dict.get("title"):
            # 如果数据字典中包含标题信息，则设置标题文本并显示标题标签
            self._title_label.setText(data_dict.get("title"))
            self._title_label.setVisible(True)
        else:
            # 否则隐藏标题标签
            self._title_label.setVisible(False)

        if data_dict.get("description"):
            # 如果数据字典中包含描述信息，则设置描述文本并显示描述标签
            self._description_label.setText(data_dict.get("description"))
            self._description_label.setVisible(True)
        else:
            # 否则隐藏描述标签
            self._description_label.setVisible(False)

        if data_dict.get("avatar"):
            # 如果数据字典中包含头像信息，则设置头像图片并显示头像
            self._avatar.set_dayu_image(data_dict.get("avatar"))
            self._avatar.setVisible(True)
        else:
            # 否则隐藏头像
            self._avatar.setVisible(False)

        if data_dict.get("cover"):
            # 如果数据字典中包含封面信息，则根据封面图片调整封面标签的显示
            fixed_height = self._cover_label.width()
            self._cover_label.setPixmap(
                data_dict.get("cover").scaledToWidth(fixed_height, QtCore.Qt.SmoothTransformation)
            )
            self._cover_label.setVisible(True)
        else:
            # 否则隐藏封面标签
            self._cover_label.setVisible(False)
```