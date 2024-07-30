# `.\comic-translate\app\ui\dayu_widgets\color_palette.py`

```py
# 导入未来模块的特性，确保代码在不同 Python 版本中兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import functools

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# 导入本地模块
from .divider import MDivider
from .label import MLabel
from .message import MMessage
from . import utils as utils

# 定义一个名为 MColorChart 的自定义 QWidget 类
class MColorChart(QtWidgets.QWidget):
    def __init__(self, parent=None):
        # 调用父类的初始化方法
        super(MColorChart, self).__init__(parent)
        
        # 创建垂直布局管理器
        main_lay = QtWidgets.QVBoxLayout()
        main_lay.setSpacing(0)
        
        # 初始化按钮列表为空列表
        self.button_list = []
        
        # 循环创建 10 个按钮
        for index in range(10):
            button = QtWidgets.QPushButton()
            button.setCursor(QtCore.Qt.PointingHandCursor)
            # 设置按钮的工具提示
            button.setToolTip(self.tr("Click to Copy Color"))
            # 连接按钮的点击事件到 slot_copy_color 方法，使用 functools.partial 传递额外参数 button
            button.clicked.connect(functools.partial(self.slot_copy_color, button))
            button.setFixedSize(QtCore.QSize(250, 45))
            button.setText("color-{}".format(index + 1))
            main_lay.addWidget(button)
            # 将创建的按钮添加到按钮列表中
            self.button_list.append(button)
        
        # 设置当前窗口的布局为 main_lay
        self.setLayout(main_lay)

    # 设置颜色方法，用于更新按钮的文本和样式
    def set_colors(self, color_list):
        for index, button in enumerate(self.button_list):
            target = color_list[index]
            button.setText("color-{}\t{}".format(index + 1, target))
            button.setProperty("color", target)
            button.setStyleSheet(
                "QPushButton{{background-color:{};color:{};border: 0 solid black}}"
                "QPushButton:hover{{font-weight:bold;}}".format(target, "#000" if index < 5 else "#fff")
            )

    # 复制颜色槽函数，复制按钮关联的颜色到剪贴板，并显示成功消息
    def slot_copy_color(self, button):
        color = button.property("color")
        QtWidgets.QApplication.clipboard().setText(color)
        MMessage.success("copied: {}".format(color), parent=self)


class MColorPaletteDialog(QtWidgets.QDialog):
    # 省略未完的类定义，需要进一步完善和注释
    # 初始化函数，用于创建颜色调色板对话框实例
    def __init__(self, init_color, parent=None):
        # 调用父类的初始化函数
        super(MColorPaletteDialog, self).__init__(parent)
        # 设置对话框标题
        self.setWindowTitle("DAYU Color Palette")
        # 初始化主色彩对象，并使用传入的颜色值
        self.primary_color = QtGui.QColor(init_color)
        # 创建颜色图表对象
        self.color_chart = MColorChart()
        # 创建选择颜色的按钮
        self.choose_color_button = QtWidgets.QPushButton()
        # 设置选择颜色按钮的固定尺寸
        self.choose_color_button.setFixedSize(QtCore.QSize(100, 30))
        # 创建显示颜色的标签
        self.color_label = QtWidgets.QLabel()
        # 创建信息显示标签，并设置属性为错误信息
        self.info_label = MLabel()
        self.info_label.setProperty("error", True)
        # 创建水平布局对象用于排列颜色相关的控件
        color_lay = QtWidgets.QHBoxLayout()
        # 添加标签显示"Primary Color:"
        color_lay.addWidget(MLabel("Primary Color:"))
        # 添加选择颜色按钮
        color_lay.addWidget(self.choose_color_button)
        # 添加显示颜色的标签
        color_lay.addWidget(self.color_label)
        # 添加显示信息的标签
        color_lay.addWidget(self.info_label)
        # 添加伸缩空间
        color_lay.addStretch()
        
        # 创建颜色选择对话框实例，初始颜色为主色彩，父对象为当前对话框
        dialog = QtWidgets.QColorDialog(self.primary_color, parent=self)
        # 设置对话框为窗口级别
        dialog.setWindowFlags(QtCore.Qt.Widget)
        # 设置对话框选项为无按钮模式
        dialog.setOption(QtWidgets.QColorDialog.NoButtons)
        # 连接当前颜色改变事件到槽函数slot_color_changed
        dialog.currentColorChanged.connect(self.slot_color_changed)
        
        # 创建垂直布局对象用于排列设置相关的控件
        setting_lay = QtWidgets.QVBoxLayout()
        # 添加颜色相关的水平布局
        setting_lay.addLayout(color_lay)
        # 添加分隔线
        setting_lay.addWidget(MDivider())
        # 添加颜色选择对话框
        setting_lay.addWidget(dialog)

        # 创建水平布局对象用于排列主颜色图表和设置布局
        main_lay = QtWidgets.QHBoxLayout()
        # 添加颜色图表对象
        main_lay.addWidget(self.color_chart)
        # 添加设置布局对象
        main_lay.addLayout(setting_lay)
        # 设置当前对话框的整体布局
        self.setLayout(main_lay)
        # 更新显示主颜色信息
        self.update_color()

    @QtCore.Slot(QtGui.QColor)
    # 颜色改变槽函数，接收新颜色并更新相关界面
    def slot_color_changed(self, color):
        # 更新主颜色为新的颜色
        self.primary_color = color
        # 计算主颜色的亮度
        light = self.primary_color.lightness()
        # 计算主颜色的饱和度
        saturation = self.primary_color.saturation()
        # 清空信息标签的文本
        self.info_label.setText("")
        # 如果亮度低于或等于70，显示建议信息
        if light <= 70:
            self.info_label.setText("亮度建议不低于70（现在 {}）".format(light))
        # 如果饱和度低于或等于70，显示建议信息
        if saturation <= 70:
            self.info_label.setText("饱和度建议不低于70（现在 {}）".format(saturation))

        # 更新颜色显示
        self.update_color()

    # 更新颜色相关界面
    def update_color(self):
        # 设置选择颜色按钮的样式，使用主颜色作为背景色
        self.choose_color_button.setStyleSheet(
            "border-radius: 0;border: none;border:1px solid gray;"
            "background-color:{};".format(self.primary_color.name())
        )
        # 设置颜色标签显示主颜色的名称
        self.color_label.setText(self.primary_color.name())
        # 设置颜色图表显示一系列基于主颜色生成的颜色
        self.color_chart.set_colors([utils.generate_color(self.primary_color, index + 1) for index in range(10)])
if __name__ == "__main__":
    # 如果当前模块被直接执行（而非被导入到其他模块中执行），则执行以下代码块

    # 导入内置模块
    import sys

    # 导入本地模块中的 dayu_theme
    from . import dayu_theme

    # 创建一个 Qt 应用程序对象
    app = QtWidgets.QApplication(sys.argv)

    # 创建一个 MColorPaletteDialog 对象，初始化颜色为 "#1890ff"
    test = MColorPaletteDialog(init_color="#1890ff")

    # 应用 dayu_theme 主题样式到 test 对象上
    dayu_theme.apply(test)

    # 显示 test 对象
    test.show()

    # 运行 Qt 应用程序的主事件循环，直到应用程序终止
    sys.exit(app.exec_())
```