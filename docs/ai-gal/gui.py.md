# `.\AI-GAL\gui.py`

```py
import configparser
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLineEdit, QLabel,
    QStackedWidget, QTabWidget, QScrollArea, QToolBar,
    QAction, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from qt_material import apply_stylesheet

class ToggleSwitch(QPushButton):
    # 自定义信号，用于表示开关状态改变
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置按钮为可勾选状态
        self.setCheckable(True)
        # 设置按钮的最小和最大尺寸
        self.setMinimumSize(50, 25)
        self.setMaximumSize(50, 25)
        # 初始化按钮样式
        self.updateStyle()
        # 连接按钮点击信号到更新样式的槽函数
        self.clicked.connect(self.updateStyle)

    def updateStyle(self):
        # 根据按钮是否被勾选设置不同的样式
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    border-radius: 12px;
                }
                QPushButton::before {
                    content: '';
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    background-color: white;
                    position: absolute;
                    margin-left: 25px;
                    margin-top: 2px;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #CCCCCC;
                    border-radius: 12px;
                }
                QPushButton::before {
                    content: '';
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    background-color: white;
                    position: absolute;
                    margin-left: 2px;
                    margin-top: 2px;
                }
            """)
        # 发射开关状态改变信号
        self.toggled.emit(self.isChecked())

class ConfigTab(QWidget):
    def __init__(self, tab_name, fields, checkbox_fields, config=None):
        super().__init__()
        # 初始化标签页名称、字段列表和复选框字段列表，以及可选的配置对象
        self.tab_name = tab_name
        self.fields = fields
        self.checkbox_fields = checkbox_fields
        self.config = config
        # 初始化用户界面
        self.initUI()
    # 初始化用户界面布局
    def initUI(self):
        # 创建垂直布局
        layout = QVBoxLayout()
        # 初始化输入字段字典
        self.inputs = {}

        # 遍历每个字段名
        for field_name in self.fields:
            # 创建垂直布局用于每个输入项
            input_layout = QVBoxLayout()

            # 如果字段名在复选框字段列表中
            if field_name in self.checkbox_fields:
                # 创建标签和开关按钮
                label = QLabel(f"{field_name}:")
                toggle_switch = ToggleSwitch()
                toggle_switch.setObjectName(field_name)
                input_layout.addWidget(label)
                input_layout.addWidget(toggle_switch)
                self.inputs[field_name] = toggle_switch
                # 将开关按钮的切换事件连接到更新字段的方法
                toggle_switch.toggled.connect(self.updateFields)

                # 根据配置初始化开关按钮状态
                if self.config and self.config.has_option(self.tab_name, field_name):
                    toggle_switch.setChecked(self.config.getboolean(self.tab_name, field_name))
            else:
                # 对于非复选框字段，创建标签和文本输入框
                label = QLabel(f"{field_name}:")
                input_edit = QLineEdit()
                input_edit.setObjectName(field_name)
                input_layout.addWidget(label)
                input_layout.addWidget(input_edit)
                self.inputs[field_name] = input_edit

                # 根据配置初始化文本输入框内容
                if self.config and self.config.has_option(self.tab_name, field_name):
                    input_edit.setText(self.config.get(self.tab_name, field_name))

            # 将每个输入布局添加到主布局中，并增加间距
            layout.addLayout(input_layout)
            layout.addSpacing(10)

        # 添加伸缩空间使布局更加灵活
        layout.addStretch()
        # 将整体布局应用到当前窗口
        self.setLayout(layout)

    # 更新字段状态的方法
    def updateFields(self, checked):
        # 如果当前标签页是 "SOVITS"
        if self.tab_name == "SOVITS":
            # 根据复选框状态设置 "语音key" 输入框的可用状态
            self.inputs["语音key"].setEnabled(checked)
            # 对于指定的字段列表，根据复选框状态设置输入框的可用状态
            for field in ["gpt_model_path", "sovits_model_path", "sovits_url1", "sovits_url2", "sovits_url3",
                          "sovits_url4", "sovits_url5", "sovits_url6"]:
                self.inputs[field].setEnabled(not checked)
        # 如果当前标签页是 "AI绘画"
        elif self.tab_name == "AI绘画":
            # 根据复选框状态设置 "绘画key" 输入框的可用状态
            self.inputs["绘画key"].setEnabled(checked)
            # 设置 "人物绘画模型ID(本地模式不填)" 输入框的可用状态
            self.inputs["人物绘画模型ID(本地模式不填)"].setEnabled(checked)
            # 设置 "背景绘画模型ID(本地模式不填)" 输入框的可用状态
            self.inputs["背景绘画模型ID(本地模式不填)"].setEnabled(checked)
# 定义名为 ConfigPage 的 QWidget 子类，用于配置页面的显示和交互
class ConfigPage(QWidget):

    # 初始化方法，接收页面标题、选项卡内容、复选框字段、是否显示按钮和配置对象
    def __init__(self, title, tabs, checkbox_fields, show_buttons=False, config=None):
        super().__init__()
        self.tabs = tabs  # 存储传入的选项卡内容
        self.checkbox_fields = checkbox_fields  # 存储传入的复选框字段
        self.config = config  # 接收传入的配置对象
        self.initUI(title, tabs, checkbox_fields, show_buttons)  # 初始化界面

    # 初始化界面的方法，设置页面标题、选项卡、复选框字段和按钮的显示
    def initUI(self, title, tabs, checkbox_fields, show_buttons):
        main_layout = QVBoxLayout()  # 创建垂直布局管理器
        titleLabel = QLabel(title)  # 创建标题标签
        titleLabel.setStyleSheet("font-size: 18px; font-weight: bold;")  # 设置标题样式
        main_layout.addWidget(titleLabel)  # 将标题标签添加到布局中

        self.tab_widget = QTabWidget()  # 创建选项卡部件
        self.tab_contents = {}  # 存储每个选项卡对应的内容

        # 遍历选项卡字典，创建选项卡和对应的滚动区域
        for tab_name, fields in tabs.items():
            tab_content = ConfigTab(tab_name, fields, checkbox_fields.get(tab_name, []), self.config)
            scroll_area = QScrollArea()  # 创建滚动区域
            scroll_area.setWidgetResizable(True)  # 设置滚动区域大小自适应
            scroll_area.setWidget(tab_content)  # 在滚动区域中添加选项卡内容
            self.tab_widget.addTab(scroll_area, tab_name)  # 在选项卡部件中添加选项卡
            self.tab_contents[tab_name] = tab_content  # 将选项卡内容添加到内容字典中

        main_layout.addWidget(self.tab_widget)  # 将选项卡部件添加到主布局中

        if show_buttons:
            button_layout = QHBoxLayout()  # 创建水平布局管理器
            button_layout.addStretch()  # 添加伸缩空间

            save_button = QPushButton("保存")  # 创建保存按钮
            start_game_button = QPushButton("开始游戏")  # 创建开始游戏按钮
            button_layout.addWidget(save_button)  # 将保存按钮添加到按钮布局中
            button_layout.addWidget(start_game_button)  # 将开始游戏按钮添加到按钮布局中
            main_layout.addLayout(button_layout)  # 将按钮布局添加到主布局中

            save_button.clicked.connect(self.saveConfig)  # 绑定保存按钮的点击事件到 saveConfig 方法
            start_game_button.clicked.connect(self.startGame)  # 绑定开始游戏按钮的点击事件到 startGame 方法

        self.setLayout(main_layout)  # 设置主布局

    # 保存配置的方法
    def saveConfig(self):
        config = configparser.ConfigParser()  # 创建配置解析器对象

        # 遍历所有选项卡内容，并将用户输入的配置信息添加到配置对象中
        for tab_name, tab_content in self.tab_contents.items():
            config[tab_name] = {}
            for field_name, widget in tab_content.inputs.items():
                if isinstance(widget, QLineEdit):
                    config[tab_name][field_name] = widget.text()
                elif isinstance(widget, ToggleSwitch):
                    config[tab_name][field_name] = str(widget.isChecked())

        # 将配置写入到文件中
        with open('game/config.ini', 'w', encoding="utf-8") as configfile:
            config.write(configfile)

        self.showSuccessMessage()  # 调用显示保存成功消息的方法

    # 显示保存成功消息的方法
    def showSuccessMessage(self):
        msg_box = QMessageBox(self)  # 创建消息框
        msg_box.setWindowTitle("保存成功")  # 设置消息框标题
        msg_box.setText("配置已成功保存!")  # 设置消息框文本内容
        msg_box.setIcon(QMessageBox.Information)  # 设置消息框图标为信息图标
        msg_box.show()  # 显示消息框
        QTimer.singleShot(3000, msg_box.close)  # 定时3秒后关闭消息框

    # 启动游戏的方法
    def startGame(self):
        exe_path = "AI GAL.exe"  # 游戏可执行文件路径
        try:
            subprocess.Popen(exe_path)  # 启动游戏进程
            QApplication.quit()  # 退出应用程序
        except Exception as e:
            print(f"无法启动游戏: {e}")  # 捕获并打印启动游戏异常信息
    def __init__(self):
        super().__init__()
        self.setWindowTitle("配置界面")  # 设置窗口标题为"配置界面"
        self.setGeometry(100, 100, 1000, 600)  # 设置窗口初始位置和大小
        self.setFixedSize(1000, 600)  # 固定窗口大小为 1000x600
        self.config = self.loadConfig()  # 载入配置文件并将结果存储在 self.config 中
        self.initUI()  # 初始化用户界面

    def loadConfig(self):
        config = configparser.ConfigParser()  # 创建一个 configparser 对象
        config.read('game/config.ini', encoding='utf-8')  # 读取指定路径下的配置文件
        return config  # 返回读取后的配置对象

    def initUI(self):
        main_layout = QHBoxLayout()  # 创建一个水平布局对象

        # 创建并设置导航栏工具条
        self.toolbar = QToolBar("导航栏")
        self.toolbar.setOrientation(Qt.Vertical)  # 设置工具条为垂直方向
        self.toolbar.setMovable(False)  # 禁止工具条移动
        self.toolbar.setFixedWidth(150)  # 固定工具条宽度为 150 像素
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)  # 将工具条添加到左侧工具栏区域
        self.basic_action = QAction("基础配置", self)  # 创建基础配置的动作
        self.advanced_action = QAction("高级配置", self)  # 创建高级配置的动作
        self.toolbar.addAction(self.basic_action)  # 将基础配置动作添加到工具条
        self.toolbar.addAction(self.advanced_action)  # 将高级配置动作添加到工具条
        self.toolbar.setStyleSheet("QToolBar { border: none; }")  # 设置工具条样式为无边框

        self.stack_widget = QStackedWidget()  # 创建一个堆叠窗口部件

        # 定义基础配置页面的选项卡和复选框字段
        basic_tabs = {
            "CHATGPT": ["GPT_KEY", "BASE_URL", "model"],
            "SOVITS": ["云端模式", "语音key", "gpt_model_path", "sovits_model_path", "sovits_url1", "sovits_url2",
                       "sovits_url3", "sovits_url4", "sovits_url5", "sovits_url6"],
            "AI绘画": ["云端模式", "绘画key", "人物绘画模型ID(本地模式不填)", "背景绘画模型ID(本地模式不填)"],
            "剧情": ["剧本的主题"]
        }

        # 定义复选框字段的初始化设置
        checkbox_fields = {
            "CHATGPT": [],
            "SOVITS": ["云端模式"],
            "AI绘画": ["云端模式"],
            "剧情": []
        }

        # 创建基础配置页面对象并添加到堆叠窗口中
        self.basic_page = ConfigPage("基础配置", basic_tabs, checkbox_fields, show_buttons=True, config=self.config)

        # 创建空的高级配置页面对象并添加到堆叠窗口中
        self.advanced_page = ConfigPage("高级配置", {"高级配置": []}, {}, config=self.config)

        self.stack_widget.addWidget(self.basic_page)  # 将基础配置页面添加到堆叠窗口中
        self.stack_widget.addWidget(self.advanced_page)  # 将高级配置页面添加到堆叠窗口中

        main_layout.addWidget(self.stack_widget)  # 将堆叠窗口添加到主布局中

        central_widget = QWidget()  # 创建中心部件
        central_widget.setLayout(main_layout)  # 将主布局设置为中心部件的布局
        self.setCentralWidget(central_widget)  # 将中心部件设置为主窗口的中心部件

        self.basic_action.triggered.connect(lambda: self.stack_widget.setCurrentIndex(0))  # 连接基础配置动作触发的信号与切换页面的槽函数
        self.advanced_action.triggered.connect(lambda: self.stack_widget.setCurrentIndex(1))  # 连接高级配置动作触发的信号与切换页面的槽函数

        self.switchToBasic()  # 初始化时显示基础配置页面

    def switchToBasic(self):
        self.stack_widget.setCurrentIndex(0)  # 切换当前显示页面为基础配置页面
if __name__ == "__main__":
    # 检查当前模块是否作为主程序执行

    import sys
    # 导入系统模块

    app = QApplication(sys.argv)
    # 创建一个Qt应用程序对象，传入命令行参数

    apply_stylesheet(app, theme='light_blue.xml')  # Assuming you have a custom stylesheet
    # 应用自定义样式表到Qt应用程序中，这里假设有一个名为'light_blue.xml'的样式表文件

    window = UserInterface()
    # 创建用户界面窗口对象

    window.show()
    # 显示用户界面窗口

    sys.exit(app.exec_())
    # 运行Qt应用程序的事件循环，直到应用程序结束，然后返回退出状态码
```