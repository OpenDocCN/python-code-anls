# `.\PaddleOCR\StyleText\engine\text_drawers.py`

```
# 导入所需的库
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from utils.logging import get_logger

# 定义一个标准文本绘制器类
class StdTextDrawer(object):
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 获取日志记录器
        self.logger = get_logger()
        # 获取全局配置中的图片宽度和高度
        self.max_width = config["Global"]["image_width"]
        self.height = config["Global"]["image_height"]
        # 定义字符列表
        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        # 初始化字体字典
        self.font_dict = {}
        # 加载字体
        self.load_fonts(config["TextDrawer"]["fonts"])
        # 支持的语言列表
        self.support_languages = list(self.font_dict)

    # 加载字体方法
    def load_fonts(self, fonts_config):
        # 遍历字体配置
        for language in fonts_config:
            # 获取字体路径
            font_path = fonts_config[language]
            # 获取有效的字体高度
            font_height = self.get_valid_height(font_path)
            # 使用PIL库加载字体文件
            font = ImageFont.truetype(font_path, font_height)
            # 将字体对象存入字体字典
            self.font_dict[language] = font

    # 获取有效的字体高度方法
    def get_valid_height(self, font_path):
        # 使用PIL库加载字体文件
        font = ImageFont.truetype(font_path, self.height - 4)
        # 获取字体包围框
        left, top, right, bottom = font.getbbox(self.char_list)
        # 计算字体高度
        _, font_height = right - left, bottom - top
        # 如果字体高度小于等于指定高度-4，则返回指定高度-4
        if font_height <= self.height - 4:
            return self.height - 4
        # 否则返回根据比例计算的高度
        else:
            return int((self.height - 4)**2 / font_height)
```