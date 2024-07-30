# `.\comic-translate\app\ui\dayu_widgets\theme.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
import string

# Import local modules
from . import DEFAULT_STATIC_FOLDER  # 导入当前目录下的 DEFAULT_STATIC_FOLDER
from . import utils  # 导入当前目录下的 utils 模块
from .qt import get_scale_factor  # 导入当前目录下的 qt 模块中的 get_scale_factor 函数


def get_theme_size():
    scale_factor_x, scale_factor_y = get_scale_factor()  # 调用 get_scale_factor 函数获取比例因子


class QssTemplate(string.Template):
    delimiter = "@"  # 设置模板变量的分隔符为 "@"
    idpattern = r"[_a-z][_a-z0-9]*"  # 设置模板变量的命名规则正则表达式


class MTheme(object):
    blue = "#1890ff"  # 主题颜色常量定义
    purple = "#722ed1"
    cyan = "#13c2c2"
    green = "#52c41a"
    magenta = "#eb2f96"
    pink = "#ef5b97"
    red = "#f5222d"
    orange = "#fa8c16"
    yellow = "#fadb14"
    volcano = "#fa541c"
    geekblue = "#2f54eb"
    lime = "#a0d911"
    gold = "#faad14"
    female_color = "#ef5b97"
    male_color = "#4ebbff"

    def __init__(self, theme="light", primary_color=None):
        super(MTheme, self).__init__()  # 调用父类初始化方法
        default_qss_file = utils.get_static_file("main.qss")  # 获取静态文件 main.qss 的完整路径
        with open(default_qss_file, "r") as f:
            self.default_qss = QssTemplate(f.read())  # 读取 main.qss 文件内容，并使用 QssTemplate 进行模板化处理
        self.primary_color, self.item_hover_bg = (None, None)  # 初始化主题颜色和项目悬停背景色为 None
        (
            self.primary_1,
            self.primary_2,
            self.primary_3,
            self.primary_4,
            self.primary_5,
            self.primary_6,
            self.primary_7,
            self.primary_8,
            self.primary_9,
            self.primary_10,
        ) = (None,) * 10  # 初始化十个主题色变量为 None
        self.hyperlink_style = ""  # 初始化超链接样式为空字符串
        self._init_color()  # 调用私有方法 _init_color 初始化颜色相关属性
        self.set_primary_color(primary_color or MTheme.blue)  # 设置主题的主色调，若未提供则使用默认蓝色
        self.set_theme(theme)  # 根据指定主题设置主题样式
        self._init_font()  # 调用私有方法 _init_font 初始化字体相关属性
        # self._init_size()  # 注释掉的代码，暂时未使用，可能为未实现的功能
        self.unit = "px"  # 设置单位为像素
        self.font_unit = "pt"  # 设置字体单位为磅

        self.text_error_color = self.error_7  # 设置错误文本颜色为 error_7 的值
        self.text_color_inverse = "#fff"  # 设置反色文本颜色为白色
        self.text_warning_color = self.warning_7  # 设置警告文本颜色为 warning_7 的值

    def set_theme(self, theme):
        if theme == "light":
            self._light()  # 根据主题设置为浅色主题
        else:
            self._dark()  # 根据主题设置为深色主题
        self._init_icon(theme)  # 初始化主题图标
    def set_primary_color(self, color):
        # 设置主要颜色
        self.primary_color = color
        # 使用主要颜色生成不同程度的颜色变化
        self.primary_1 = utils.generate_color(color, 1)
        self.primary_2 = utils.generate_color(color, 2)
        self.primary_3 = utils.generate_color(color, 3)
        self.primary_4 = utils.generate_color(color, 4)
        self.primary_5 = utils.generate_color(color, 5)
        self.primary_6 = utils.generate_color(color, 6)
        self.primary_7 = utils.generate_color(color, 7)
        self.primary_8 = utils.generate_color(color, 8)
        self.primary_9 = utils.generate_color(color, 9)
        self.primary_10 = utils.generate_color(color, 10)
        # 设置项目悬停背景色为主要颜色的第一种变化色
        self.item_hover_bg = self.primary_1
        # 设置富文本超链接样式，使用主要颜色
        self.hyperlink_style = """
        <style>
         a {{
            text-decoration: none;
            color: {0};
        }}
        </style>""".format(
            self.primary_color
        )

    def _init_icon(self, theme):
        # 初始化图标路径前缀，替换默认静态文件夹路径中的反斜杠为斜杠
        pre_str = DEFAULT_STATIC_FOLDER.replace("\\", "/")
        # 根据主题选择图标后缀
        suf_str = "" if theme == "light" else "_dark"
        # 设置图标路径模板，包含前缀和后缀
        url_prefix = "{pre}/{{}}{suf}.png".format(pre=pre_str, suf=suf_str)
        url_prefix_2 = "{pre}/{{}}.svg".format(pre=pre_str)
        # 设置各个具体图标的路径
        self.icon_down = url_prefix.format("down_line")
        self.icon_up = url_prefix.format("up_line")
        self.icon_left = url_prefix.format("left_line")
        self.icon_right = url_prefix.format("right_line")
        self.icon_close = url_prefix.format("close_line")
        self.icon_calender = url_prefix.format("calendar_fill")
        self.icon_splitter = url_prefix.format("splitter")
        self.icon_float = url_prefix.format("float")
        self.icon_size_grip = url_prefix.format("size_grip")

        self.icon_check = url_prefix_2.format("check")
        self.icon_minus = url_prefix_2.format("minus")
        self.icon_circle = url_prefix_2.format("circle")
        self.icon_sphere = url_prefix_2.format("sphere")
    # 初始化颜色变量，将不同类型的消息颜色赋值为预设的颜色值
    self.info_color = self.blue
    self.success_color = self.green
    self.processing_color = self.blue
    self.error_color = self.red
    self.warning_color = self.gold

    # 根据预设的颜色值生成不同透明度和变化程度的信息颜色
    self.info_1 = utils.fade_color(self.info_color, "15%")
    self.info_2 = utils.generate_color(self.info_color, 2)
    self.info_3 = utils.fade_color(self.info_color, "35%")
    self.info_4 = utils.generate_color(self.info_color, 4)
    self.info_5 = utils.generate_color(self.info_color, 5)
    self.info_6 = utils.generate_color(self.info_color, 6)
    self.info_7 = utils.generate_color(self.info_color, 7)
    self.info_8 = utils.generate_color(self.info_color, 8)
    self.info_9 = utils.generate_color(self.info_color, 9)
    self.info_10 = utils.generate_color(self.info_color, 10)

    # 根据预设的颜色值生成不同透明度和变化程度的成功消息颜色
    self.success_1 = utils.fade_color(self.success_color, "15%")
    self.success_2 = utils.generate_color(self.success_color, 2)
    self.success_3 = utils.fade_color(self.success_color, "35%")
    self.success_4 = utils.generate_color(self.success_color, 4)
    self.success_5 = utils.generate_color(self.success_color, 5)
    self.success_6 = utils.generate_color(self.success_color, 6)
    self.success_7 = utils.generate_color(self.success_color, 7)
    self.success_8 = utils.generate_color(self.success_color, 8)
    self.success_9 = utils.generate_color(self.success_color, 9)
    self.success_10 = utils.generate_color(self.success_color, 10)

    # 根据预设的颜色值生成不同透明度和变化程度的警告消息颜色
    self.warning_1 = utils.fade_color(self.warning_color, "15%")
    self.warning_2 = utils.generate_color(self.warning_color, 2)
    self.warning_3 = utils.fade_color(self.warning_color, "35%")
    self.warning_4 = utils.generate_color(self.warning_color, 4)
    self.warning_5 = utils.generate_color(self.warning_color, 5)
    self.warning_6 = utils.generate_color(self.warning_color, 6)
    self.warning_7 = utils.generate_color(self.warning_color, 7)
    self.warning_8 = utils.generate_color(self.warning_color, 8)
    self.warning_9 = utils.generate_color(self.warning_color, 9)
    self.warning_10 = utils.generate_color(self.warning_color, 10)

    # 根据预设的颜色值生成不同透明度和变化程度的错误消息颜色
    self.error_1 = utils.fade_color(self.error_color, "15%")
    self.error_2 = utils.generate_color(self.error_color, 2)
    self.error_3 = utils.fade_color(self.error_color, "35%")
    self.error_4 = utils.generate_color(self.error_color, 4)
    self.error_5 = utils.generate_color(self.error_color, 5)
    self.error_6 = utils.generate_color(self.error_color, 6)
    self.error_7 = utils.generate_color(self.error_color, 7)
    self.error_8 = utils.generate_color(self.error_color, 8)
    self.error_9 = utils.generate_color(self.error_color, 9)
    self.error_10 = utils.generate_color(self.error_color, 10)
    # 初始化字体相关属性
    def _init_font(self):
        # 设置默认字体系列
        self.font_family = (
            'BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",'
            '"Helvetica Neue",Helvetica,Arial,sans-serif'
        )
        # 设置基础字体大小
        self.font_size_base = 9
        # 计算大号字体大小
        self.font_size_large = self.font_size_base + 2
        # 计算小号字体大小
        self.font_size_small = self.font_size_base - 2
        # 设置标题1的字体大小
        self.h1_size = int(self.font_size_base * 2.71)
        # 设置标题2的字体大小
        self.h2_size = int(self.font_size_base * 2.12)
        # 设置标题3的字体大小
        self.h3_size = int(self.font_size_base * 1.71)
        # 设置标题4的字体大小
        self.h4_size = int(self.font_size_base * 1.41)

    # 获取对象属性的方法，当属性不存在时，返回0
    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return get_theme_size().get(item, 0)

    # 设置暗色主题的颜色值
    def _dark(self):
        self.title_color = "#ffffff"
        self.primary_text_color = "#d9d9d9"
        self.secondary_text_color = "#a6a6a6"
        self.disable_color = "#737373"
        self.border_color = "#1e1e1e"
        self.divider_color = "#262626"
        self.header_color = "#0a0a0a"
        self.icon_color = "#a6a6a6"

        self.background_color = "#323232"
        self.background_selected_color = "#292929"
        self.background_in_color = "#3a3a3a"
        self.background_out_color = "#494949"
        # 计算遮罩颜色，透明度为90%
        self.mask_color = utils.fade_color(self.background_color, "90%")
        self.toast_color = "#555555"

    # 设置亮色主题的颜色值
    def _light(self):
        self.title_color = "#262626"
        self.primary_text_color = "#595959"
        self.secondary_text_color = "#8c8c8c"
        self.disable_color = "#e5e5e5"
        self.border_color = "#d9d9d9"
        self.divider_color = "#e8e8e8"
        self.header_color = "#fafafa"
        self.icon_color = "#8c8c8c"

        self.background_color = "#f8f8f9"
        self.background_selected_color = "#bfbfbf"
        self.background_in_color = "#ffffff"
        self.background_out_color = "#eeeeee"
        # 计算遮罩颜色，透明度为90%
        self.mask_color = utils.fade_color(self.background_color, "90%")
        self.toast_color = "#333333"

    # 应用当前主题设置到指定的widget上
    def apply(self, widget):
        size_dict = get_theme_size()
        size_dict.update(vars(self))
        # 使用主题设置对应的样式表
        widget.setStyleSheet(self.default_qss.substitute(size_dict))

    # 装饰器函数，用于将主题设置应用到类实例的样式上
    def deco(self, cls):
        original_init__ = cls.__init__

        # 修改类的初始化方法，以应用主题设置
        def my__init__(instance, *args, **kwargs):
            original_init__(instance, *args, **kwargs)
            size_dict = get_theme_size()
            size_dict.update(vars(self))
            instance.setStyleSheet(self.default_qss.substitute(size_dict))

        # 定义一个方法，用于刷新样式
        def polish(instance):
            instance.style().polish(instance)

        # 将修改后的初始化方法和刷新样式方法设置到类中
        setattr(cls, "__init__", my__init__)
        setattr(cls, "polish", polish)
        return cls
```