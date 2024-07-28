# `.\comic-translate\app\ui\settings\settings_ui.py`

```py
import os  # 导入操作系统相关的模块
from typing import List  # 导入用于类型提示的 List 类型
from PySide6 import QtWidgets  # 导入 PySide6 中的 QtWidgets 模块
from PySide6 import QtCore  # 导入 PySide6 中的 QtCore 模块
from PySide6.QtGui import QFontMetrics  # 导入 PySide6 中的 QFontMetrics 类

from ..dayu_widgets.label import MLabel  # 导入自定义模块中的 MLabel 类
from ..dayu_widgets.line_edit import MLineEdit  # 导入自定义模块中的 MLineEdit 类
from ..dayu_widgets.text_edit import MTextEdit  # 导入自定义模块中的 MTextEdit 类
from ..dayu_widgets.check_box import MCheckBox  # 导入自定义模块中的 MCheckBox 类
from ..dayu_widgets.clickable_card import ClickMeta  # 导入自定义模块中的 ClickMeta 类
from ..dayu_widgets.divider import MDivider  # 导入自定义模块中的 MDivider 类
from ..dayu_widgets.qt import MPixmap  # 导入自定义模块中的 MPixmap 类
from ..dayu_widgets.combo_box import MComboBox  # 导入自定义模块中的 MComboBox 类
from ..dayu_widgets.spin_box import MSpinBox  # 导入自定义模块中的 MSpinBox 类
from ..dayu_widgets.browser import MClickBrowserFileToolButton  # 导入自定义模块中的 MClickBrowserFileToolButton 类

class SettingsPageUI(QtWidgets.QWidget):
    def _init_ui(self):
        self.stacked_widget = QtWidgets.QStackedWidget()  # 创建一个 QStackedWidget 对象，用于管理多个子页面

        navbar_layout = self._create_navbar()  # 调用 _create_navbar 方法创建顶部导航栏布局
        personalization_layout = self._create_personalization_layout()  # 调用 _create_personalization_layout 方法创建个性化设置布局
        tools_layout = self._create_tools_layout()  # 调用 _create_tools_layout 方法创建工具设置布局
        credentials_layout = self._create_credentials_layout()  # 调用 _create_credentials_layout 方法创建凭据设置布局
        llms_layout = self._create_llms_layout()  # 调用 _create_llms_layout 方法创建LLMS设置布局
        text_rendering_layout = self._create_text_rendering_layout()  # 调用 _create_text_rendering_layout 方法创建文本渲染设置布局
        export_layout = self._create_export_layout()  # 调用 _create_export_layout 方法创建导出设置布局

        personalization_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于个性化设置
        personalization_widget.setLayout(personalization_layout)  # 设置个性化设置页面的布局为 personalization_layout
        self.stacked_widget.addWidget(personalization_widget)  # 将个性化设置页面添加到 stacked_widget 中管理

        tools_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于工具设置
        tools_widget.setLayout(tools_layout)  # 设置工具设置页面的布局为 tools_layout
        self.stacked_widget.addWidget(tools_widget)  # 将工具设置页面添加到 stacked_widget 中管理

        credentials_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于凭据设置
        credentials_widget.setLayout(credentials_layout)  # 设置凭据设置页面的布局为 credentials_layout
        self.stacked_widget.addWidget(credentials_widget)  # 将凭据设置页面添加到 stacked_widget 中管理

        llms_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于LLMS设置
        llms_widget.setLayout(llms_layout)  # 设置LLMS设置页面的布局为 llms_layout
        self.stacked_widget.addWidget(llms_widget)  # 将LLMS设置页面添加到 stacked_widget 中管理

        text_rendering_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于文本渲染设置
        text_rendering_widget.setLayout(text_rendering_layout)  # 设置文本渲染设置页面的布局为 text_rendering_layout
        self.stacked_widget.addWidget(text_rendering_widget)  # 将文本渲染设置页面添加到 stacked_widget 中管理

        export_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象用于导出设置
        export_widget.setLayout(export_layout)  # 设置导出设置页面的布局为 export_layout
        self.stacked_widget.addWidget(export_widget)  # 将导出设置页面添加到 stacked_widget 中管理

        settings_layout = QtWidgets.QHBoxLayout()  # 创建一个 QHBoxLayout 对象，用于整体设置页面的布局
        settings_layout.addLayout(navbar_layout)  # 将顶部导航栏布局添加到整体布局中
        settings_layout.addWidget(MDivider(orientation=QtCore.Qt.Orientation.Vertical))  # 在整体布局中添加一个垂直分割线
        settings_layout.addWidget(self.stacked_widget, 1)  # 在整体布局中添加 stacked_widget，并占据剩余空间
        settings_layout.setContentsMargins(3, 3, 3, 3)  # 设置整体布局的边距

        self.setLayout(settings_layout)  # 设置当前 QWidget 的布局为 settings_layout

    def _create_title_and_combo(self, title: str, options: List[str]):
        combo_widget = QtWidgets.QWidget()  # 创建一个 QWidget 对象作为包含标题和下拉框的容器
        combo_layout = QtWidgets.QVBoxLayout()  # 创建一个垂直布局管理器用于标题和下拉框的布局

        if title in [self.tr("Inpainter"), self.tr("HD Strategy")]:
            label = MLabel(title)  # 创建标题标签 MLabel 对象
        else:
            label = MLabel(title).h4()  # 创建标题为 h4 格式的 MLabel 对象
        combo = MComboBox().small()  # 创建一个小尺寸的 MComboBox 对象
        combo.addItems(options)  # 向下拉框中添加选项

        combo_layout.addWidget(label)  # 将标题标签添加到垂直布局中
        combo_layout.addWidget(combo)  # 将下拉框添加到垂直布局中

        combo_widget.setLayout(combo_layout)  # 设置 combo_widget 的布局为 combo_layout

        return combo_widget, combo  # 返回包含标题和下拉框的 QWidget 对象和 MComboBox 对象
    # 创建导航栏布局，使用垂直布局管理器
    def _create_navbar(self):
        navbar_layout = QtWidgets.QVBoxLayout()

        # 遍历设置列表，每个设置项包含标题和图标
        for index, setting in enumerate([
            {"title": self.tr("Personalization"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Tools"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Credentials"), "avatar": MPixmap(".svg")},
            {"title": self.tr("LLMs"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Text Rendering"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Export"), "avatar": MPixmap(".svg")},
        ]):
            # 创建可点击的导航卡片
            nav_card = ClickMeta(extra=False)
            # 设置卡片的数据
            nav_card.setup_data(setting)
            # 连接点击信号到槽函数，传递当前索引和卡片对象
            nav_card.clicked.connect(lambda i=index, c=nav_card: self.on_nav_clicked(i, c))
            # 将卡片添加到导航栏布局中
            navbar_layout.addWidget(nav_card)
            # 将卡片对象添加到实例变量中，用于后续引用
            self.nav_cards.append(nav_card)

        # 添加伸展因子，保证导航栏布局在垂直方向上填充父容器
        navbar_layout.addStretch(1)
        # 返回导航栏布局
        return navbar_layout

    # 处理导航栏卡片的点击事件
    def on_nav_clicked(self, index: int, clicked_nav: ClickMeta):
        # 取消之前高亮显示的导航项
        if self.current_highlighted_nav:
            self.current_highlighted_nav.set_highlight(False)

        # 高亮显示当前点击的导航项
        clicked_nav.set_highlight(True)
        self.current_highlighted_nav = clicked_nav

        # 设置堆叠窗口当前显示的页面索引
        self.stacked_widget.setCurrentIndex(index)

    # 创建个性化设置布局
    def _create_personalization_layout(self):
        personalization_layout = QtWidgets.QVBoxLayout()

        # 创建语言选择部件，并关联下拉框
        language_widget, self.lang_combo = self._create_title_and_combo(self.tr("Language"), self.languages)
        self.set_combo_box_width(self.lang_combo, self.languages)
        
        # 创建主题选择部件，并关联下拉框
        theme_widget, self.theme_combo = self._create_title_and_combo(self.tr("Theme"), self.themes)
        self.set_combo_box_width(self.theme_combo, self.themes)

        # 将语言和主题部件添加到个性化设置布局中
        personalization_layout.addWidget(language_widget) 
        personalization_layout.addWidget(theme_widget) 
        personalization_layout.addStretch()

        # 返回个性化设置布局
        return personalization_layout

    # 创建LLMs布局
    def _create_llms_layout(self):
        llms_layout = QtWidgets.QVBoxLayout()

        # 创建额外上下文的标签和文本编辑器部件
        prompt_label = MLabel(self.tr("Extra Context:"))
        self.llm_widgets['extra_context'] = MTextEdit()

        # 创建复选框用于指定是否提供图像输入给多模态LLMs
        image_checkbox = MCheckBox(self.tr("Provide Image as input to multimodal LLMs"))
        image_checkbox.setChecked(True)
        self.llm_widgets['image_input'] = image_checkbox

        # 将标签、文本编辑器和复选框部件添加到LLMs布局中
        llms_layout.addWidget(prompt_label)
        llms_layout.addWidget(self.llm_widgets['extra_context'])
        llms_layout.addWidget(image_checkbox)
        llms_layout.addStretch(1)

        # 返回LLMs布局
        return llms_layout
    # 创建文本渲染布局的方法
    def _create_text_rendering_layout(self):
        # 创建垂直布局对象
        text_rendering_layout = QtWidgets.QVBoxLayout()

        # 文本对齐部分
        alignment_layout = QtWidgets.QVBoxLayout()
        # 创建文本对齐标签并设置样式
        alignment_label = MLabel(self.tr("Text Alignment")).h4()
        # 创建文本对齐下拉框并设置样式
        alignment_combo = MComboBox().small()
        alignment_combo.addItems(self.alignment)
        # 设置下拉框宽度以匹配最长的对齐选项
        self.set_combo_box_width(alignment_combo, self.alignment)
        # 默认选择居中对齐
        alignment_combo.setCurrentText(self.tr("Center"))
        # 将标签和下拉框添加到布局中
        alignment_layout.addWidget(alignment_label)
        alignment_layout.addWidget(alignment_combo)
        # 将对齐布局添加到主布局中
        text_rendering_layout.addLayout(alignment_layout)

        # 字体选择部分
        font_layout = QtWidgets.QVBoxLayout()
        combo_layout = QtWidgets.QHBoxLayout()

        # 创建字体标签并设置样式
        font_label = MLabel(self.tr("Font")).h4()
        # 创建字体下拉框并设置样式
        self.font_combo = MComboBox().small()
        # 获取当前工作目录下的字体文件列表
        font_folder_path = os.path.join(os.getcwd(), "fonts")
        font_files = [f for f in os.listdir(font_folder_path) if f.endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))]
        # 将字体文件添加到字体下拉框中
        self.font_combo.addItems(font_files)
        # 根据字体文件列表设置下拉框的宽度
        self.set_combo_box_width(self.font_combo, font_files)

        # 创建字体浏览按钮
        self.font_browser = MClickBrowserFileToolButton(multiple=True)
        # 设置文件过滤器以仅显示特定字体文件类型
        self.font_browser.set_dayu_filters([".ttf", ".ttc", ".otf", ".woff", ".woff2"])
        # 设置工具提示信息
        self.font_browser.setToolTip(self.tr("Import the Font to use for Rendering Text on Images"))

        # 将字体下拉框和字体浏览按钮添加到水平布局中
        combo_layout.addWidget(self.font_combo)
        combo_layout.addWidget(self.font_browser)
        combo_layout.addStretch()

        # 将字体标签和水平布局添加到字体布局中
        font_layout.addWidget(font_label)
        font_layout.addLayout(combo_layout)

        # 添加间距到主布局中
        text_rendering_layout.addSpacing(10)
        # 将字体布局添加到主布局中
        text_rendering_layout.addLayout(font_layout)

        # 字体颜色部分
        color_layout = QtWidgets.QVBoxLayout()
        # 创建颜色标签
        color_label = MLabel(self.tr("Color"))
        # 创建颜色按钮并设置固定大小和样式
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setFixedSize(30, 30)
        self.color_button.setStyleSheet(
            "background-color: black; border: none; border-radius: 5px;"
        )
        # 设置按钮的属性，用于存储选定的颜色值
        self.color_button.setProperty('selected_color', "#000000")
        
        # 将颜色标签和颜色按钮添加到颜色布局中
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()
        # 将颜色布局添加到主布局中
        text_rendering_layout.addLayout(color_layout)
        text_rendering_layout.addSpacing(10)

        # 创建复选框用于控制是否将文本渲染为大写
        uppercase_checkbox = MCheckBox(self.tr("Render Text in UpperCase"))
        # 将复选框添加到主布局中
        text_rendering_layout.addWidget(uppercase_checkbox)

        # 存储各个部件，以便后续访问
        self.text_rendering_widgets['alignment'] = alignment_combo
        self.text_rendering_widgets['font'] = self.font_combo
        self.text_rendering_widgets['color_button'] = self.color_button
        self.text_rendering_widgets['upper_case'] = uppercase_checkbox

        # 添加伸展因子以确保布局能够占据额外的空间
        text_rendering_layout.addStretch(1)
        # 返回文本渲染布局对象
        return text_rendering_layout
    # 创建导出布局的函数，返回一个垂直布局对象
    def _create_export_layout(self):
        # 创建一个垂直布局对象
        export_layout = QtWidgets.QVBoxLayout()

        # 创建一个标签显示"Automatic Mode"文本，设置为h4大小
        batch_label = MLabel(self.tr("Automatic Mode")).h4()

        # 创建三个复选框，分别用于导出原始文本、翻译后文本和修复后图像
        raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))

        # 将三个复选框添加到导出窗口小部件字典中，以便稍后访问
        self.export_widgets['raw_text'] = raw_text_checkbox
        self.export_widgets['translated_text'] = translated_text_checkbox
        self.export_widgets['inpainted_image'] = inpainted_image_checkbox

        # 将"Automatic Mode"标签和三个复选框依次添加到导出布局中
        export_layout.addWidget(batch_label)
        export_layout.addWidget(raw_text_checkbox)
        export_layout.addWidget(translated_text_checkbox)
        export_layout.addWidget(inpainted_image_checkbox)

        # 定义一组文件类型
        file_types = ['pdf', 'epub', 'cbr', 'cbz', 'cb7', 'cbt']
        # 定义一个有效的文件类型列表，排除'cbr'并添加其他类型
        available_file_types = ['pdf', 'epub', 'cbz', 'cb7']  # Exclude 'CBR' and add other types

        # 遍历每种文件类型
        for file_type in file_types:
            # 创建一个水平布局对象
            save_layout = QtWidgets.QHBoxLayout()
            # 创建一个标签，显示"Save {file_type} as:"，其中{file_type}是当前文件类型
            save_label = MLabel(self.tr("Save {file_type} as:").format(file_type=file_type))
            # 创建一个下拉框小部件
            save_combo = MComboBox().small()
            # 从有效文件类型列表中排除'cbr'，并将其添加到下拉框中
            save_items = [ft for ft in available_file_types if ft != 'cbr']
            save_combo.addItems(save_items)  # Exclude 'CBR'
            # 根据下拉框中的项目设置其宽度
            self.set_combo_box_width(save_combo, save_items)

            # 如果文件类型是'cbr'，则将下拉框默认选择设置为'cbz'，否则选择当前文件类型
            if file_type == 'cbr':
                save_combo.setCurrentText('cbz')
            elif file_type in available_file_types:
                save_combo.setCurrentText(file_type)

            # 将下拉框添加到导出窗口小部件字典中，使用文件类型作为键
            self.export_widgets[f'.{file_type.lower()}_save_as'] = save_combo

            # 将保存标签和下拉框添加到水平布局中
            save_layout.addWidget(save_label)
            save_layout.addWidget(save_combo)
            save_layout.addStretch(1)

            # 将水平布局添加到导出布局中
            export_layout.addLayout(save_layout)

        # 在导出布局的最后添加一个伸展因子，用于布局调整
        export_layout.addStretch(1)

        # 返回创建的导出布局对象
        return export_layout

    # 更新高清策略窗口小部件的显示状态，根据传入的索引值选择性显示部件
    def update_hd_strategy_widgets(self, index: int):
        # 获取当前选择的修复策略
        strategy = self.inpaint_strategy_combo.itemText(index)
        # 根据选择的策略显示或隐藏调整窗口和裁剪窗口小部件
        self.resize_widget.setVisible(strategy == self.tr("Resize"))
        self.crop_widget.setVisible(strategy == self.tr("Crop"))

        # 当选择"Original"策略时，调整布局以移除空白空间
        if strategy == self.tr("Original"):
            self.hd_strategy_widgets.setFixedHeight(0)
        else:
            # 否则，根据窗口小部件的推荐高度设置固定高度
            self.hd_strategy_widgets.setFixedHeight(self.hd_strategy_widgets.sizeHint().height())

    # 设置下拉框的宽度，根据传入的项目列表和填充值进行调整
    def set_combo_box_width(self, combo_box: MComboBox, items: List[str], padding: int = 40):
        # 获取当前字体的字体度量对象
        font_metrics = QFontMetrics(combo_box.font())
        # 计算项目列表中各项的最大水平宽度
        max_width = max(font_metrics.horizontalAdvance(item) for item in items)
        # 设置下拉框的固定宽度，考虑到填充值
        combo_box.setFixedWidth(max_width + padding)
    # 定义一个方法，用于设置标签控件的宽度，接受一个标签对象和一个可选的填充值作为参数
    def set_label_width(self, label: MLabel, padding: int = 20):
        # 获取标签当前字体的字体度量信息
        font_metrics = QFontMetrics(label.font())
        # 计算标签文本的水平宽度
        text_width = font_metrics.horizontalAdvance(label.text())
        # 根据文本宽度和填充值设置标签的固定宽度
        label.setFixedWidth(text_width + padding)
```