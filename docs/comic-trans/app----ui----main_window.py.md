# `.\comic-translate\app\ui\main_window.py`

```py
# 导入必要的模块
import os  # 操作系统功能模块
from PySide6 import QtWidgets  # PySide6 中的图形用户界面组件
from PySide6 import QtCore  # PySide6 中的核心功能模块

# 导入自定义的 Dayu Widgets 组件
from .dayu_widgets import dayu_theme  # Dayu 主题设置
from .dayu_widgets.divider import MDivider  # 分隔线组件
from .dayu_widgets.combo_box import MComboBox  # 下拉框组件
from .dayu_widgets.text_edit import MTextEdit  # 文本编辑框组件
from .dayu_widgets.browser import MDragFileButton, MClickBrowserFileToolButton, MClickSaveFileToolButton  # 文件浏览按钮组件
from .dayu_widgets.push_button import MPushButton  # 按钮组件
from .dayu_widgets.tool_button import MToolButton  # 工具按钮组件
from .dayu_widgets.radio_button import MRadioButton  # 单选按钮组件
from .dayu_widgets.button_group import MPushButtonGroup, MToolButtonGroup  # 按钮组及工具按钮组
from .dayu_widgets.slider import MSlider  # 滑块组件
from .dayu_widgets.label import MLabel  # 标签组件
from .dayu_widgets.qt import MPixmap  # Qt 图像像素映射组件
from .dayu_widgets.progress_bar import MProgressBar  # 进度条组件
from .dayu_widgets.loading import MLoading  # 加载动画组件
from .dayu_widgets.theme import MTheme  # 主题组件

# 导入自定义的应用程序相关模块
from .image_viewer import ImageViewer  # 图像查看器
from .settings.settings_page import SettingsPage  # 设置页面

# 支持的源语言列表
supported_source_languages = [
    "Korean", "Japanese", "French", "Chinese", "English",
    "Russian", "German", "Dutch", "Spanish", "Italian"
]

# 支持的目标语言列表
supported_target_languages = [
    "English", "Korean", "Japanese", "French", "Simplified Chinese",
    "Traditional Chinese", "Russian", "German", "Dutch", "Spanish", 
    "Italian", "Turkish", "Polish", "Portuguese", "Brazilian Portuguese"
]

# 定义 ComicTranslateUI 类，继承自 QtWidgets.QMainWindow
class ComicTranslateUI(QtWidgets.QMainWindow):
    # 初始化函数，设置界面的标题为 "Comic Translate"
    def __init__(self, parent=None):
        # 调用父类的初始化方法
        super(ComicTranslateUI, self).__init__(parent)
        # 设置窗口的标题为 "Comic Translate"
        self.setWindowTitle("Comic Translate")
        
        # 获取主屏幕对象
        screen = QtWidgets.QApplication.primaryScreen()
        # 获取主屏幕的几何信息
        geo = screen.geometry()
        
        # 获取主屏幕的宽度和高度，并转换为浮点数
        width = float(geo.width())
        height = float(geo.height())
        # 设置窗口的初始位置为 (50, 50)
        x = 50
        y = 50
        # 计算窗口的宽度和高度，分别为主屏幕宽度和高度的 1.2 倍
        w = int(width / 1.2)
        h = int(height / 1.2)
        # 设置窗口的几何形状
        self.setGeometry(x, y, w, h)

        # 创建一个图片查看器对象
        self.image_viewer = ImageViewer(self)
        # 创建一个设置页面对象
        self.settings_page = SettingsPage(self)
        # 连接主题改变信号到对应的槽函数 apply_theme
        self.settings_page.theme_changed.connect(self.apply_theme)
        # 初始化主内容部件为 None
        self.main_content_widget = None
        # 初始化工具按钮字典，用于存储互斥工具名称和其对应的按钮
        self.tool_buttons = {}

        # 启用手势识别功能：捕捉平移手势和缩放手势
        self.grabGesture(QtCore.Qt.GestureType.PanGesture)
        self.grabGesture(QtCore.Qt.GestureType.PinchGesture)

        # 设置语言映射字典，将界面的翻译显示名称映射到实际语言字符串
        self.lang_mapping = {
            self.tr("English"): "English",
            self.tr("Korean"): "Korean",
            self.tr("Japanese"): "Japanese",
            self.tr("French"): "French",
            self.tr("Simplified Chinese"): "Simplified Chinese",
            self.tr("Traditional Chinese"): "Traditional Chinese",
            self.tr("Chinese"): "Chinese",
            self.tr("Russian"): "Russian",
            self.tr("German"): "German",
            self.tr("Dutch"): "Dutch",
            self.tr("Spanish"): "Spanish",
            self.tr("Italian"): "Italian",
            self.tr("Turkish"): "Turkish",
            self.tr("Polish"): "Polish",
            self.tr("Portuguese"): "Portuguese",
            self.tr("Brazilian Portuguese"): "Brazilian Portuguese"
        }
        # 创建反向映射，将语言字符串映射到翻译显示名称
        self.reverse_lang_mapping = {v: k for k, v in self.lang_mapping.items()}

        # 调用初始化界面的私有方法
        self._init_ui()

    # 初始化界面布局
    def _init_ui(self):
        # 创建主部件对象
        main_widget = QtWidgets.QWidget(self)
        # 创建主布局为水平布局
        self.main_layout = QtWidgets.QHBoxLayout()
        # 将主布局设置给主部件
        main_widget.setLayout(self.main_layout)
        # 将主部件设置为中央部件
        self.setCentralWidget(main_widget)

        # 创建导航栏布局
        nav_rail_layout = self._create_nav_rail()
        # 将导航栏布局添加到主布局中
        self.main_layout.addLayout(nav_rail_layout)
        # 添加一个分割线部件到主布局中，作为导航栏和主内容之间的分割
        self.main_layout.addWidget(MDivider(orientation=QtCore.Qt.Vertical))

        # 创建主内容部件
        self.main_content_widget = self._create_main_content()
        # 将主内容部件添加到主布局中
        self.main_layout.addWidget(self.main_content_widget)
    # 创建导航栏布局的方法
    def _create_nav_rail(self):
        # 创建垂直布局管理器
        nav_rail_layout = QtWidgets.QVBoxLayout()
        
        # 创建一个分隔线控件并设置固定宽度为30
        nav_divider = MDivider()
        nav_divider.setFixedWidth(30)

        # 创建用于浏览文件的多选工具按钮
        self.tool_browser = MClickBrowserFileToolButton(multiple=True)
        self.tool_browser.set_dayu_svg("upload-file.svg")  # 设置按钮的SVG图标
        self.tool_browser.set_dayu_filters([".png", ".jpg", ".jpeg", ".webp", ".bmp",
                                            ".zip", ".cbz", ".cbr", ".cb7", ".cbt",
                                            ".pdf", ".epub"])  # 设置文件过滤器
        self.tool_browser.setToolTip(self.tr("Import Images, PDFs, Epubs or Comic Book Archive Files(cbr, cbz, etc)"))  # 设置工具提示信息

        # 创建用于保存文件的工具按钮
        self.save_browser = MClickSaveFileToolButton()
        save_file_types = [("Images", ["png", "jpg", "jpeg", "webp", "bmp"])]
        self.save_browser.set_file_types(save_file_types)  # 设置保存文件类型
        self.save_browser.set_dayu_svg("save.svg")  # 设置按钮的SVG图标
        self.save_browser.setToolTip(self.tr("Save Currently Loaded Image"))  # 设置工具提示信息

        # 创建用于保存所有文件的工具按钮
        save_all_file_types = [
            ("ZIP files", "zip"),
            ("CBZ files", "cbz"),
            ("CB7 files", "cb7"),
            ("PDF files", "pdf"),
            ("EPUB files", "epub"),
        ]
        self.save_all_browser = MClickSaveFileToolButton()
        self.save_all_browser.set_dayu_svg("save-all.svg")  # 设置按钮的SVG图标
        self.save_all_browser.set_file_types(save_all_file_types)  # 设置保存文件类型
        self.save_all_browser.setToolTip(self.tr("Save all Images"))  # 设置工具提示信息

        # 创建垂直排列的工具按钮组
        nav_tool_group = MToolButtonGroup(orientation=QtCore.Qt.Vertical, exclusive=True)
        nav_tools = [
            {"svg": "home_line.svg", "checkable": True, "tooltip": self.tr("Home"), "clicked": self.show_main_page},  # 主页按钮设置
            {"svg": "settings.svg", "checkable": True, "tooltip": self.tr("Settings"), "clicked": self.show_settings_page},  # 设置按钮设置
        ]
        nav_tool_group.set_button_list(nav_tools)  # 将按钮列表设置到工具按钮组中
        nav_tool_group.get_button_group().buttons()[0].setChecked(True)  # 设置默认选中第一个按钮

        # 将各个控件添加到导航栏布局中
        nav_rail_layout.addWidget(self.tool_browser)
        nav_rail_layout.addWidget(self.save_browser)
        nav_rail_layout.addWidget(self.save_all_browser)
        nav_rail_layout.addWidget(nav_divider)
        nav_rail_layout.addWidget(nav_tool_group)
        nav_rail_layout.addStretch()  # 添加伸展控件，使布局在垂直方向上扩展

        nav_rail_layout.setContentsMargins(0, 0, 0, 0)  # 设置布局的边距为0

        return nav_rail_layout  # 返回创建的导航栏布局

    # 创建普通的PushButton按钮的方法
    def create_push_button(self, text: str, clicked = None):
        button = MPushButton(text)  # 创建按钮，并设置按钮文本
        button.set_dayu_size(dayu_theme.small)  # 设置按钮尺寸
        button.set_dayu_type(MPushButton.DefaultType)  # 设置按钮类型为默认类型

        if clicked:
            button.clicked.connect(clicked)  # 如果传入了点击事件，连接按钮的点击信号到指定的槽函数

        return button  # 返回创建的PushButton按钮对象

    # 创建工具按钮的方法
    def create_tool_button(self, text: str = "", svg: str = "", checkable: bool = False):
        if text:
            button = MToolButton().svg(svg).text_beside_icon()  # 创建带文本和SVG图标的工具按钮
            button.setText(text)  # 设置按钮的文本内容
        else:
            button = MToolButton().svg(svg)  # 创建只带SVG图标的工具按钮

        if checkable:
            button.setCheckable(True)  # 如果设置了可选中属性，设置按钮为可选中状态
        else:
            button.setCheckable(False)  # 否则设置按钮为不可选中状态

        return button  # 返回创建的工具按钮对象
    # 显示设置页面方法
    def show_settings_page(self):
        # 如果设置页面不存在，则创建一个新的设置页面对象
        if not self.settings_page:
            self.settings_page = SettingsPage(self)
        
        # 从主布局中移除主内容部件
        self.main_layout.removeWidget(self.main_content_widget)
        self.main_content_widget.hide()
        
        # 将设置页面添加到主布局中，并显示
        self.main_layout.addWidget(self.settings_page)
        self.settings_page.show()

    # 显示主页面方法
    def show_main_page(self):
        # 如果设置页面存在，则从主布局中移除它并隐藏
        if self.settings_page:
            self.main_layout.removeWidget(self.settings_page)
            self.settings_page.hide()
        
        # 将主内容部件重新添加到主布局中，并显示
        self.main_layout.addWidget(self.main_content_widget)
        self.main_content_widget.show()

    # 应用主题方法
    def apply_theme(self, theme: str):
        # 根据主题名创建新的主题对象
        if theme == self.settings_page.ui.tr("Light"):
            new_theme = MTheme("light", primary_color=MTheme.blue)
        else:
            new_theme = MTheme("dark", primary_color=MTheme.yellow)
        
        # 应用新主题到当前界面
        new_theme.apply(self)

        # 刷新界面以应用新主题
        self.repaint()

    # 切换平移工具方法
    def toggle_pan_tool(self):
        # 如果平移按钮被选中，则设置工具为平移
        if self.pan_button.isChecked():
            self.set_tool('pan')
        else:
            # 否则取消当前工具的选择
            self.set_tool(None)

    # 切换框选工具方法
    def toggle_box_tool(self):
        # 如果框选按钮被选中，则设置工具为框选
        if self.box_button.isChecked():
            self.set_tool('box')
        else:
            # 否则取消当前工具的选择
            self.set_tool(None)

    # 切换画笔工具方法
    def toggle_brush_tool(self):
        # 如果画笔按钮被选中，则设置工具为画笔
        if self.brush_button.isChecked():
            self.set_tool('brush')
        else:
            # 否则取消当前工具的选择
            self.set_tool(None)

    # 切换橡皮擦工具方法
    def toggle_eraser_tool(self):
        # 如果橡皮擦按钮被选中，则设置工具为橡皮擦
        if self.eraser_button.isChecked():
            self.set_tool('eraser')
        else:
            # 否则取消当前工具的选择
            self.set_tool(None)

    # 设置工具方法，根据给定的工具名设置工具
    def set_tool(self, tool_name: str):
        # 清除图像查看器的当前光标设置
        self.image_viewer.unsetCursor()
        # 设置图像查看器的工具
        self.image_viewer.set_tool(tool_name)

        # 遍历工具按钮字典，根据工具名设置对应按钮的选中状态
        for name, button in self.tool_buttons.items():
            if name != tool_name:
                button.setChecked(False)
            elif tool_name is not None:
                button.setChecked(True)

        # 如果工具名为None，则取消所有按钮的选中状态
        if not tool_name:
            for button in self.tool_buttons.values():
                button.setChecked(False)

    # 设置画笔大小方法，根据给定的大小设置画笔在图像上的实际大小
    def set_brush_size(self, size: int):
        # 如果图像查看器中存在图像
        if self.image_viewer.hasPhoto():
            # 获取当前图像的OpenCV格式
            image = self.image_viewer.get_cv2_image()
            h, w, c = image.shape
            # 根据图像实际尺寸缩放给定的画笔大小
            scaled_size = self.scale_size(size, w, h)
            self.image_viewer.set_brush_size(scaled_size)

    # 设置橡皮擦大小方法，根据给定的大小设置橡皮擦在图像上的实际大小
    def set_eraser_size(self, size: int):
        # 如果图像查看器中存在图像
        if self.image_viewer.hasPhoto():
            # 获取当前图像的OpenCV格式
            image = self.image_viewer.get_cv2_image()
            h, w, c = image.shape
            # 根据图像实际尺寸缩放给定的橡皮擦大小
            scaled_size = self.scale_size(size, w, h)
            self.image_viewer.set_eraser_size(scaled_size)
    # 根据给定的基准尺寸和图像宽高计算图像的对角线长度
    def scale_size(self, base_size, image_width, image_height):
        # 计算图像的对角线长度
        image_diagonal = (image_width**2 + image_height**2)**0.5
        
        # 使用参考对角线长度（例如 1000 像素）计算缩放因子
        reference_diagonal = 1000
        scaling_factor = image_diagonal / reference_diagonal
        
        # 根据缩放因子调整基准尺寸
        scaled_size = base_size * scaling_factor
        
        return scaled_size

    # 撤销上一笔画操作
    def brush_undo(self):
        self.image_viewer.undo_brush_stroke()

    # 重做上一笔画操作
    def brush_redo(self):
        self.image_viewer.redo_brush_stroke()

    # 删除选中的矩形框
    def delete_selected_box(self):
        self.image_viewer.delete_selected_rectangle()
```